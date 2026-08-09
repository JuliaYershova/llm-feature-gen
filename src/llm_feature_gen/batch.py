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
    _build_generation_response_schema,
    _extract_feature_names,
    _provider_call_kwargs,
    _validate_generation_features,
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

    def delete(self, text: str, features_hash: str, *, persist: bool = True) -> None:
        """Remove a cached response when it no longer matches the schema."""
        self._store.pop(self._make_key(text, features_hash), None)
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
    response_schema: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return provider.text_features(
        texts,
        prompt=prompt,
        **_provider_call_kwargs(provider, None, response_schema),
    )


def _normalise_provider_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict) and "features" in response and isinstance(response["features"], str):
        response = {"features": parse_json_from_markdown(response["features"])}

    if isinstance(response, dict):
        inner = response.get("features", response)
        return inner if isinstance(inner, dict) else {}

    return {}


def _validated_provider_response(response: Any, discovered_features: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one provider response and ensure it matches the discovered schema."""
    inner = _normalise_provider_response(response)
    _validate_generation_features(inner, discovered_features)
    return inner


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
    """Generate text feature values in batches."""
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
    response_schema = _build_generation_response_schema(discovered_features)
    all_columns = ["File", "Class"] + feature_names + ["raw_llm_output"]

    indices_to_process: List[int] = []
    cached_results: Dict[int, Dict[str, Any]] = {}
    cache_changed = False

    for index, text in enumerate(texts):
        if cache is not None:
            cached = cache.get(text, features_hash)
            if cached is not None:
                try:
                    _validate_generation_features(cached, discovered_features)
                except ValueError:
                    cache.delete(text, features_hash, persist=False)
                    cache_changed = True
                else:
                    cached_results[index] = cached
                    continue
        indices_to_process.append(index)

    if cache is not None and cache_changed:
        cache.save()

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

        batch_failed = False
        try:
            responses = _call_provider_batch(provider, batch_texts, full_prompt, response_schema)
        except Exception as exc:
            print(f"Batch error ({exc}), retrying after {retry_delay}s...")
            time.sleep(retry_delay)
            try:
                responses = _call_provider_batch(provider, batch_texts, full_prompt, response_schema)
            except Exception as retry_exc:
                print(f"Batch failed again: {retry_exc}. Skipping batch.")
                responses = [{}] * len(batch_texts)
                batch_failed = True

        cache_changed = False
        for local_pos, global_index in enumerate(batch_indices):
            parsed = responses[local_pos] if local_pos < len(responses) else {}
            try:
                inner = _validated_provider_response(parsed, discovered_features)
            except ValueError as exc:
                print(f"Invalid batch response for text_{global_index}: {exc}")
                if batch_failed:
                    inner = None
                else:
                    try:
                        retry_response = _call_provider_batch(
                            provider,
                            [texts[global_index]],
                            full_prompt,
                            response_schema,
                        )[0]
                        inner = _validated_provider_response(retry_response, discovered_features)
                    except Exception as retry_exc:
                        print(f"Retry failed for text_{global_index}: {retry_exc}")
                        inner = None

            if inner is None:
                batch_responses[global_index] = {}
                continue

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
    """Generate text feature CSVs using batched provider calls and caching."""
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
