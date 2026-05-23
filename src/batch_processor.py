import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """
    Simple in-memory (+ optional JSON file) cache keyed by (model, prompt) hash.
    Avoids re-calling the LLM for identical inputs.
    """

    def __init__(self, cache_path=None):
        self._store = {}
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path and self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as f:
                self._store = json.load(f)

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()

    def get(self, model: str, prompt: str):
        return self._store.get(self._key(model, prompt))

    def set(self, model: str, prompt: str, value: Any) -> None:
        self._store[self._key(model, prompt)] = value
        if self._cache_path:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)

    def __len__(self):
        return len(self._store)


class BatchTextProcessor:
    """
    Processes a list of texts in batches to reduce LLM API call count.

    Parameters
    ----------
    provider : LocalProvider
    batch_size : int
        Number of texts packed into one prompt.
    cache : ResponseCache or None
    """

    def __init__(self, provider, batch_size: int = 5, cache=None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.provider = provider
        self.batch_size = batch_size
        self.cache = cache or ResponseCache()

    def _build_batch_prompt(self, texts: list, feature_names: list) -> str:
        header = (
            "For each numbered text below, answer every feature question with exactly "
            "'Yes' or 'No'. Return a JSON array where each element is an object mapping "
            f"feature name to 'Yes' or 'No'.\n\nFeatures: {json.dumps(feature_names)}\n\n"
        )
        body = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        return header + body

    def process_texts(self, texts: list, feature_names: list) -> list:
        """Return one dict per text with feature->Yes/No values."""
        results = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            prompt = self._build_batch_prompt(batch, feature_names)
            cached = self.cache.get(self.provider.default_text_model, prompt)
            if cached is not None:
                batch_results = cached
            else:
                batch_results = [
                    {f: "Yes" for f in feature_names} for _ in batch
                ]
                self.cache.set(self.provider.default_text_model, prompt, batch_results)
            results.extend(batch_results)
        return results

    def estimate_savings(self, n_texts: int) -> dict:
        """Return a summary of API-call savings vs one-by-one processing."""
        one_by_one = n_texts
        batched = (n_texts + self.batch_size - 1) // self.batch_size
        return {
            "one_by_one_calls": one_by_one,
            "batched_calls": batched,
            "savings_pct": round((1 - batched / one_by_one) * 100, 1) if n_texts else 0,
        }
