"""Batch text feature generation with optional JSON-backed caching."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .generate import (
    _build_prompt_for_generation,
    _extract_feature_names,
    load_discovered_features,
    parse_json_from_markdown,
)
from .prompts import text_generation_prompt
from .providers.openai_provider import OpenAIProvider

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


class BatchTextCache:
    """JSON-backed cache keyed by input text and discovered feature schema."""

    def __init__(self, cache_file: Union[str, Path] = "outputs/feature_cache.json") -> None:
        self.cache_file = Path(cache_file)
        self._store: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_file.exists():
            return

        try:
            self._store = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._store = {}

    def save(self) -> None:
        """Persist all pending cache updates to disk."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _make_key(self, text: str, features_hash: str) -> str:
        return f"{self._hash(text)}_{features_hash}"

    def get(self, text: str, features_hash: str) -> Optional[Dict[str, Any]]:
        """Return a cached provider response, or ``None`` when not cached."""
        return self._store.get(self._make_key(text, features_hash))

    def set(
        self,
        text: str,
        features_hash: str,
        value: Dict[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        """Store a response and optionally persist immediately."""
        self._store[self._make_key(text, features_hash)] = value
        if persist:
            self.save()

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Remove all cached entries and delete the cache file if present."""
        self._store = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


def _call_provider_batch(
    provider: OpenAIProvider,
    texts: List[str],
    prompt: str,
) -> List[Dict[str, Any]]:
    return provider.text_features(texts, prompt=prompt)


def _normalise_provider_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict) and "features" in response and isinstance(response["features"], str):
        response = {"features": parse_json_from_markdown(response["features"])}

    if isinstance(response, dict):
        inner = response.get("features", response)
        return inner if isinstance(inner, dict) else {}

    return {}


def generate_features_batch(
    texts: Sequence[str],
    labels: Sequence[str],
    discovered_features: Union[str, Path, Dict[str, Any]],
    provider: Optional[OpenAIProvider] = None,
    batch_size: int = 8,
    output_csv: Optional[Union[str, Path]] = None,
    cache: Optional[BatchTextCache] = None,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """Generate text feature values in batches, with optional caching.

    Unlike the folder-based helpers, this function works on in-memory texts
    and returns a DataFrame instead of writing per-class CSVs. Provider calls
    are grouped into batches; a failed batch is retried once and then skipped,
    leaving ``"not given by LLM"`` in the affected rows.

    Args:
        texts: Raw input texts.
        labels: One class label per text, written to the ``Class`` column.
        discovered_features: Schema to score against — a path to a discovery
            JSON artifact or an already-loaded schema dictionary.
        provider: Provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        batch_size: Number of texts per provider call batch.
        output_csv: Optional path; when given, the DataFrame is also saved
            there.
        cache: Optional [BatchTextCache][llm_feature_gen.batch.BatchTextCache].
            Cached texts are skipped and new responses are stored, so repeated
            runs only pay for new inputs.
        retry_delay: Seconds to wait before retrying a failed batch.

    Returns:
        A DataFrame with columns ``File``, ``Class``, one column per
        discovered feature, and ``raw_llm_output``.

    Raises:
        ValueError: If ``batch_size`` is not positive, ``texts`` and
            ``labels`` differ in length, or the schema contains no features.

    Example:
        ```python
        df = generate_features_batch(
            texts=["I am sad", "I am scared"],
            labels=["sadness", "fear"],
            discovered_features="outputs/discovered_text_features.json",
            cache=BatchTextCache(),
        )
        ```
    """
    provider = provider or OpenAIProvider()
    texts = list(texts)
    labels = list(labels)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if len(texts) != len(labels):
        raise ValueError(f"texts and labels must have the same length ({len(texts)} vs {len(labels)})")

    if isinstance(discovered_features, (str, Path)):
        discovered_features = load_discovered_features(discovered_features)

    feature_names = _extract_feature_names(discovered_features)
    if not feature_names:
        raise ValueError("No feature names found in discovered_features.")

    features_hash = BatchTextCache._hash(json.dumps(discovered_features, sort_keys=True))
    full_prompt = _build_prompt_for_generation(text_generation_prompt, discovered_features)
    all_columns = ["File", "Class"] + feature_names + ["raw_llm_output"]

    indices_to_process: List[int] = []
    cached_results: Dict[int, Dict[str, Any]] = {}

    for index, text in enumerate(texts):
        if cache is not None:
            cached = cache.get(text, features_hash)
            if cached is not None:
                cached_results[index] = cached
                continue
        indices_to_process.append(index)

    if cached_results:
        print(f"Cache hits: {len(cached_results)} / {len(texts)}")

    total_batches = (len(indices_to_process) + batch_size - 1) // batch_size
    iterator = range(0, len(indices_to_process), batch_size)
    if _tqdm is not None:
        iterator = _tqdm(list(iterator), desc="Batch generation", unit="batch", total=total_batches)

    batch_responses: Dict[int, Dict[str, Any]] = {}

    for batch_start in iterator:
        batch_indices = indices_to_process[batch_start: batch_start + batch_size]
        batch_texts = [texts[index] for index in batch_indices]

        try:
            responses = _call_provider_batch(provider, batch_texts, full_prompt)
        except Exception as exc:
            print(f"Batch error ({exc}), retrying after {retry_delay}s...")
            time.sleep(retry_delay)
            try:
                responses = _call_provider_batch(provider, batch_texts, full_prompt)
            except Exception as retry_exc:
                print(f"Batch failed again: {retry_exc}. Skipping batch.")
                responses = [{}] * len(batch_texts)

        cache_changed = False
        for local_pos, global_index in enumerate(batch_indices):
            parsed = responses[local_pos] if local_pos < len(responses) else {}
            inner = _normalise_provider_response(parsed)
            batch_responses[global_index] = inner

            if cache is not None:
                cache.set(texts[global_index], features_hash, inner, persist=False)
                cache_changed = True

        if cache is not None and cache_changed:
            cache.save()

    rows: List[Dict[str, Any]] = []
    for index in range(len(texts)):
        inner = cached_results.get(index, batch_responses.get(index, {}))
        row: Dict[str, Any] = {
            "File": f"text_{index}",
            "Class": labels[index],
            "raw_llm_output": json.dumps(inner, ensure_ascii=False),
        }
        for feature in feature_names:
            row[feature] = inner.get(feature, "not given by LLM")
        rows.append(row)

    df = pd.DataFrame(rows, columns=all_columns)

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved batch results to {output_path}")

    return df


def generate_features_from_texts_cached(
    root_folder: Union[str, Path],
    discovered_features_path: Union[str, Path],
    provider: Optional[OpenAIProvider] = None,
    classes: Optional[List[str]] = None,
    output_dir: Union[str, Path] = "outputs",
    merge_to_single_csv: bool = True,
    merged_csv_name: str = "all_feature_values.csv",
    batch_size: int = 8,
    cache_file: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Generate text feature CSVs using batched provider calls and caching.

    A drop-in alternative to
    [generate_features_from_texts][llm_feature_gen.generate.generate_features_from_texts]
    for larger text datasets: files are scored in batches through
    [generate_features_batch][llm_feature_gen.batch.generate_features_batch],
    and responses are cached on disk so interrupted or repeated runs do not
    re-pay for texts that were already scored.

    Args:
        root_folder: Dataset root containing one subfolder per class with
            ``.txt`` files inside.
        discovered_features_path: Path to the discovered schema JSON.
        provider: Provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        classes: Optional subset of class-folder names. Defaults to all
            immediate subdirectories of ``root_folder``.
        output_dir: Directory where CSV outputs and the cache file are
            written.
        merge_to_single_csv: Also write one concatenated CSV across classes.
        merged_csv_name: Filename of the merged CSV artifact.
        batch_size: Number of texts per provider call batch.
        cache_file: Cache location. Defaults to
            ``<output_dir>/feature_cache.json``.

    Returns:
        Mapping from class name to generated CSV path; the merged CSV is
        stored under the ``"__merged__"`` key when enabled.

    Raises:
        FileNotFoundError: If a requested class folder does not exist.

    Example:
        ```python
        csv_paths = generate_features_from_texts_cached(
            root_folder="texts",
            discovered_features_path="outputs/discovered_text_features.json",
            batch_size=8,
        )
        ```
    """
    root_folder = Path(root_folder)
    output_dir = Path(output_dir)
    provider = provider or OpenAIProvider()

    if classes is None:
        classes = [path.name for path in root_folder.iterdir() if path.is_dir()]

    discovered_features = load_discovered_features(discovered_features_path)
    cache = BatchTextCache(cache_file=cache_file or (output_dir / "feature_cache.json"))

    csv_paths: Dict[str, str] = {}
    all_dfs: List[pd.DataFrame] = []

    for class_name in classes:
        class_folder = root_folder / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Class folder not found: {class_folder}")

        files = sorted(class_folder.glob("*.txt"))
        texts = [file.read_text(encoding="utf-8").strip() for file in files]
        labels = [class_name] * len(texts)

        class_csv = output_dir / f"{class_name}_feature_values.csv"
        df = generate_features_batch(
            texts=texts,
            labels=labels,
            discovered_features=discovered_features,
            provider=provider,
            batch_size=batch_size,
            output_csv=class_csv,
            cache=cache,
        )
        df["File"] = [file.name for file in files]
        df.to_csv(class_csv, index=False)

        csv_paths[class_name] = str(class_csv)
        all_dfs.append(df)

    if merge_to_single_csv and all_dfs:
        merged_path = output_dir / merged_csv_name
        pd.concat(all_dfs, ignore_index=True).to_csv(merged_path, index=False)
        csv_paths["__merged__"] = str(merged_path)

    return csv_paths
