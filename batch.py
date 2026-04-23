"""Batch processing for llm-feature-gen.

This module adds efficient batch processing for text collections, with optional
caching so that LLM responses are not re-requested for documents that were
already processed in a previous run.

Key additions
-------------
- ``generate_features_batch``   : process texts in configurable batches
- ``BatchTextCache``            : simple JSON-backed cache keyed by (text_hash, features_hash)
- ``generate_features_from_texts_cached`` : drop-in replacement that skips cached items

Usage example::

    from llm_feature_gen.batch import generate_features_batch, BatchTextCache
    from llm_feature_gen.providers.local_provider import LocalProvider

    provider = LocalProvider(
        base_url="https://litellm.vse.cz/",
        api_key="sk-...",
        default_text_model="qwen3.6-35b",
    )

    cache = BatchTextCache(cache_file="outputs/feature_cache.json")

    result_df = generate_features_batch(
        texts=my_texts,
        labels=my_labels,
        discovered_features=discovered,
        provider=provider,
        batch_size=8,
        cache=cache,
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from llm_feature_gen.generate import (
    _build_prompt_for_generation,
    _extract_feature_names,
    _infer_feature_names_from_llm,
    load_discovered_features,
    parse_json_from_markdown,
)
from llm_feature_gen.providers.openai_provider import OpenAIProvider
from llm_feature_gen.prompts import text_generation_prompt

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────

class BatchTextCache:
    """Simple JSON-backed cache for LLM feature generation responses.

    Cache entries are keyed by a SHA-256 hash of the input text combined with
    a hash of the discovered feature schema, so the cache is automatically
    invalidated whenever the feature schema changes.

    Parameters
    ----------
    cache_file : str or Path
        Path to the JSON cache file. Created on first write if it does not
        exist.
    """

    def __init__(self, cache_file: Union[str, Path] = "outputs/feature_cache.json") -> None:
        self.cache_file = Path(cache_file)
        self._store: Dict[str, Any] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._store = json.loads(
                    self.cache_file.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                self._store = {}

    def _save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── key helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    def _make_key(self, text: str, features_hash: str) -> str:
        return f"{self._hash(text)}_{features_hash}"

    # ── public API ───────────────────────────────────────────────────────────

    def get(self, text: str, features_hash: str) -> Optional[Dict[str, Any]]:
        """Return the cached LLM response dict or ``None`` if not cached."""
        return self._store.get(self._make_key(text, features_hash))

    def set(self, text: str, features_hash: str, value: Dict[str, Any]) -> None:
        """Store an LLM response and persist the cache to disk."""
        self._store[self._make_key(text, features_hash)] = value
        self._save()

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Wipe all cached entries and delete the file."""
        self._store = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Batch generation
# ─────────────────────────────────────────────────────────────────────────────

def _call_provider_batch(
    provider: OpenAIProvider,
    texts: List[str],
    prompt: str,
) -> List[Dict[str, Any]]:
    """Send a batch of texts to the provider and return their parsed responses.

    Uses ``provider.text_features`` which already iterates internally, but
    calling it with multiple texts at once avoids per-call overhead when the
    provider supports true batching.
    """
    return provider.text_features(texts, prompt=prompt)


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
    """Generate feature values for a list of texts in configurable batches.

    Parameters
    ----------
    texts : sequence of str
        Raw text documents to process.
    labels : sequence of str
        Class label for each text (same length as *texts*).
    discovered_features : str, Path, or dict
        Either a path to a ``discovered_text_features.json`` artifact or a
        dict already loaded by ``load_discovered_features``.
    provider : OpenAIProvider, optional
        LLM provider. Defaults to ``OpenAIProvider()``.
    batch_size : int, default 8
        Number of texts to send to the provider in each iteration.
        Larger batches reduce round-trip overhead; smaller batches are more
        fault-tolerant and easier to cache.
    output_csv : str or Path, optional
        If given, the resulting DataFrame is also written to this CSV file.
    cache : BatchTextCache, optional
        When provided, already-processed texts are skipped and their cached
        responses are reused. This is especially useful for resuming
        interrupted runs.
    retry_delay : float, default 1.0
        Seconds to wait before retrying a failed batch (single retry).

    Returns
    -------
    pd.DataFrame
        One row per input text with columns: ``File``, ``Class``,
        one column per discovered feature, ``raw_llm_output``.
    """
    from llm_feature_gen.providers.openai_provider import OpenAIProvider as _OAI

    provider = provider or _OAI()
    texts = list(texts)
    labels = list(labels)

    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have the same length "
            f"({len(texts)} vs {len(labels)})"
        )

    # ── Load / normalise discovered features ─────────────────────────────────
    if isinstance(discovered_features, (str, Path)):
        discovered_features = load_discovered_features(discovered_features)

    feature_names = _extract_feature_names(discovered_features)
    if not feature_names:
        raise ValueError("No feature names found in discovered_features.")

    features_hash = BatchTextCache._hash(
        json.dumps(discovered_features, sort_keys=True)
    )
    full_prompt = _build_prompt_for_generation(text_generation_prompt, discovered_features)

    all_columns = ["File", "Class"] + feature_names + ["raw_llm_output"]
    rows: List[Dict[str, Any]] = []

    # ── Separate cached vs uncached ───────────────────────────────────────────
    indices_to_process: List[int] = []
    cached_results: Dict[int, Dict[str, Any]] = {}

    for i, text in enumerate(texts):
        if cache is not None:
            hit = cache.get(text, features_hash)
            if hit is not None:
                cached_results[i] = hit
                continue
        indices_to_process.append(i)

    if cached_results:
        print(f"Cache hits: {len(cached_results)} / {len(texts)}")

    # ── Process uncached texts in batches ────────────────────────────────────
    total_batches = (len(indices_to_process) + batch_size - 1) // batch_size

    iterator = range(0, len(indices_to_process), batch_size)
    if _tqdm is not None:
        iterator = _tqdm(
            list(iterator),
            desc="Batch generation",
            unit="batch",
            total=total_batches,
        )

    batch_responses: Dict[int, Dict[str, Any]] = {}

    for batch_start in iterator:
        batch_idx = indices_to_process[batch_start : batch_start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]

        try:
            responses = _call_provider_batch(provider, batch_texts, full_prompt)
        except Exception as exc:
            print(f"Batch error ({exc}), retrying after {retry_delay}s …")
            time.sleep(retry_delay)
            try:
                responses = _call_provider_batch(provider, batch_texts, full_prompt)
            except Exception as exc2:
                print(f"Batch failed again: {exc2}. Skipping batch.")
                responses = [{}] * len(batch_texts)

        for local_pos, global_idx in enumerate(batch_idx):
            parsed = responses[local_pos] if local_pos < len(responses) else {}

            # Normalise nested structure produced by some models
            if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
                parsed = {"features": parse_json_from_markdown(parsed["features"])}

            inner = parsed.get("features", parsed) if isinstance(parsed, dict) else {}

            batch_responses[global_idx] = inner

            if cache is not None:
                cache.set(texts[global_idx], features_hash, inner)

    # ── Assemble final rows in original order ─────────────────────────────────
    for i in range(len(texts)):
        if i in cached_results:
            inner = cached_results[i]
        else:
            inner = batch_responses.get(i, {})

        row: Dict[str, Any] = {
            "File": f"text_{i}",
            "Class": labels[i],
            "raw_llm_output": json.dumps(inner, ensure_ascii=False),
        }
        for feat in feature_names:
            row[feat] = inner.get(feat, "not given by LLM")

        rows.append(row)

    df = pd.DataFrame(rows, columns=all_columns)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Saved batch results to {output_csv}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper matching the existing generate_features_from_texts API
# ─────────────────────────────────────────────────────────────────────────────

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
    """Drop-in replacement for ``generate_features_from_texts`` with batch + cache.

    Parameters
    ----------
    root_folder : str or Path
        Dataset root with one sub-folder per class.
    discovered_features_path : str or Path
        Path to the discovered features JSON artifact.
    provider : OpenAIProvider, optional
        LLM provider instance.
    classes : list of str, optional
        Subset of class-folder names to process.
    output_dir : str or Path
        Directory for CSV outputs.
    merge_to_single_csv : bool, default True
        Whether to write a merged CSV.
    merged_csv_name : str
        Filename for the merged CSV.
    batch_size : int, default 8
        Texts per provider call.
    cache_file : str or Path, optional
        Path for the JSON cache. Defaults to ``<output_dir>/feature_cache.json``.

    Returns
    -------
    dict mapping class name → CSV path, plus ``"__merged__"`` when applicable.
    """
    from llm_feature_gen.providers.openai_provider import OpenAIProvider as _OAI

    root_folder = Path(root_folder)
    output_dir = Path(output_dir)
    provider = provider or _OAI()

    if classes is None:
        classes = [p.name for p in root_folder.iterdir() if p.is_dir()]

    discovered_features = load_discovered_features(discovered_features_path)
    cache = BatchTextCache(
        cache_file=cache_file or (output_dir / "feature_cache.json")
    )

    csv_paths: Dict[str, str] = {}
    all_dfs: List[pd.DataFrame] = []

    for cls in classes:
        cls_folder = root_folder / cls
        if not cls_folder.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_folder}")

        # Collect .txt files
        files = sorted(cls_folder.glob("*.txt"))
        texts = [f.read_text(encoding="utf-8").strip() for f in files]
        labels = [cls] * len(texts)

        cls_csv = output_dir / f"{cls}_feature_values.csv"
        df = generate_features_batch(
            texts=texts,
            labels=labels,
            discovered_features=discovered_features,
            provider=provider,
            batch_size=batch_size,
            output_csv=cls_csv,
            cache=cache,
        )
        # Restore original filenames
        df["File"] = [f.name for f in files]
        df.to_csv(cls_csv, index=False)

        csv_paths[cls] = str(cls_csv)
        all_dfs.append(df)

    if merge_to_single_csv and all_dfs:
        merged_path = output_dir / merged_csv_name
        pd.concat(all_dfs, ignore_index=True).to_csv(merged_path, index=False)
        csv_paths["__merged__"] = str(merged_path)

    return csv_paths
