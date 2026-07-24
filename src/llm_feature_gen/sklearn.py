"""scikit-learn compatible adapters for LLM feature generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .batch import BatchTextCache, generate_features_batch
from .discover import discover_features_from_texts
from .generate import _extract_feature_names, load_discovered_features
from .providers.openai_provider import OpenAIProvider

try:  # pragma: no cover - exercised when scikit-learn is installed
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # pragma: no cover - fallback is tested in minimal envs
    class BaseEstimator:
        """Small fallback so the transformer remains usable without sklearn."""

        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            params = {name: value for name, value in vars(self).items() if not name.endswith("_")}
            return params if deep else params.copy()

        def set_params(self, **params: Any) -> "BaseEstimator":
            for name, value in params.items():
                setattr(self, name, value)
            return self

    class TransformerMixin:
        def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
            fitted = getattr(self, "fit")(X, y, **fit_params)
            return getattr(fitted, "transform")(X)


class LLMFeatureTransformer(TransformerMixin, BaseEstimator):
    """Turn raw text samples into discovered LLM feature columns.

    This is intentionally a transformer, not a classifier. Use it as the first
    step in a sklearn pipeline, then add encoders and any estimator you want.
    """

    def __init__(
        self,
        provider: Any = None,
        discovered_features: Any = None,
        text_column: Optional[str] = None,
        batch_size: int = 8,
        cache: Optional[BatchTextCache] = None,
        min_features: int = 10,
        output_dir: Union[str, Path] = "outputs",
        output_filename: str = "discovered_text_features.json",
        retry_delay: float = 1.0,
    ) -> None:
        self.provider = provider
        self.discovered_features = discovered_features
        self.text_column = text_column
        self.batch_size = batch_size
        self.cache = cache
        self.min_features = min_features
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.retry_delay = retry_delay

    def fit(self, X: Any, y: Any = None) -> "LLMFeatureTransformer":
        _ = y
        texts = self._as_text_list(X)
        self.provider_ = self.provider or OpenAIProvider()
        self.n_features_in_ = self._input_width(X)

        if self.discovered_features is None:
            self.discovered_features_ = discover_features_from_texts(
                texts,
                provider=self.provider_,
                output_dir=self.output_dir,
                output_filename=self.output_filename,
                min_features=self.min_features,
            )
        elif isinstance(self.discovered_features, (str, Path)):
            self.discovered_features_ = load_discovered_features(self.discovered_features)
        else:
            self.discovered_features_ = self.discovered_features

        self.feature_names_ = list(dict.fromkeys(_extract_feature_names(self.discovered_features_)))
        if not self.feature_names_:
            raise ValueError("discovered_features must include at least one feature name")
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "discovered_features_"):
            raise ValueError("This LLMFeatureTransformer instance is not fitted yet.")

        texts = self._as_text_list(X)
        df = generate_features_batch(
            texts=texts,
            labels=[""] * len(texts),
            discovered_features=self.discovered_features_,
            provider=self.provider_,
            batch_size=self.batch_size,
            cache=self.cache,
            retry_delay=self.retry_delay,
        )
        return df.loc[:, self.feature_names_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        _ = input_features
        if not hasattr(self, "feature_names_"):
            raise ValueError("This LLMFeatureTransformer instance is not fitted yet.")
        return np.asarray(self.feature_names_, dtype=object)

    def _as_text_list(self, X: Any) -> list[str]:
        if isinstance(X, pd.Series):
            return X.astype(str).tolist()

        if isinstance(X, pd.DataFrame):
            if self.text_column is not None:
                if self.text_column not in X.columns:
                    raise ValueError(f"Column '{self.text_column}' not found.")
                return X[self.text_column].astype(str).tolist()
            if X.shape[1] != 1:
                raise ValueError("Pass text_column when X has more than one column.")
            return X.iloc[:, 0].astype(str).tolist()

        array = np.asarray(X, dtype=object)
        if array.ndim == 0:
            raise ValueError("X must contain one or more text samples.")
        if array.ndim == 1:
            return [str(value) for value in array.tolist()]
        if array.ndim == 2 and array.shape[1] == 1:
            return [str(value) for value in array[:, 0].tolist()]
        raise ValueError("X must be a 1D sequence of text or a single text column.")

    def _input_width(self, X: Any) -> int:
        if isinstance(X, pd.DataFrame):
            return int(X.shape[1])
        array = np.asarray(X, dtype=object)
        return int(array.shape[1]) if array.ndim == 2 else 1


__all__ = ["LLMFeatureTransformer"]
