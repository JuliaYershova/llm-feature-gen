"""scikit-learn compatible adapters for LLM feature generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .batch import BatchTextCache, generate_features_batch
from .generate import (
    _extract_feature_names,
    load_discovered_features,
)
from .multiclass import discover_features_multiclass
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
    Discovery happens during :meth:`fit`; validated feature generation happens
    during :meth:`transform`. Separate discovery and generation prompts let
    callers customize both phases without bypassing output validation.

    ``discovery_prompt`` and ``generation_prompt`` control the task performed
    in each phase. Their corresponding ``*_system_prompt`` values control the
    model's role and behavior. Generation always appends the discovered feature
    specification to the selected task prompt.
    """

    def __init__(
        self,
        provider: Any = None,
        discovered_features: Any = None,
        text_column: Optional[str] = None,
        use_batch: bool = True,
        batch_size: int = 8,
        cache: Optional[BatchTextCache] = None,
        min_features: int = 10,
        classes: Optional[Sequence[str]] = None,
        output_dir: Union[str, Path] = "outputs",
        output_filename: str = "discovered_text_features.json",
        retry_delay: float = 1.0,
        discovery_prompt: Optional[str] = None,
        discovery_system_prompt: Optional[str] = None,
        generation_system_prompt: Optional[str] = None,
        generation_prompt: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.discovered_features = discovered_features
        self.text_column = text_column
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.cache = cache
        self.min_features = min_features
        self.classes = classes
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.retry_delay = retry_delay
        self.discovery_prompt = discovery_prompt
        self.discovery_system_prompt = discovery_system_prompt
        self.generation_system_prompt = generation_system_prompt
        self.generation_prompt = generation_prompt

    def fit(self, X: Any, y: Any = None) -> "LLMFeatureTransformer":
        texts = self._as_text_list(X)
        self.provider_ = self.provider or OpenAIProvider()
        self.n_features_in_ = self._input_width(X)

        if self.discovered_features is None:
            self.discovered_features_ = discover_features_multiclass(
                texts_or_file=texts,
                classes=self._classes_from_y(y),
                provider=self.provider_,
                prompt=self.discovery_prompt,
                output_dir=self.output_dir,
                output_filename=self.output_filename,
                min_features=self.min_features,
                system_prompt=self.discovery_system_prompt,
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
        if not self.use_batch:
            return self._transform_one_by_one(texts)

        df = generate_features_batch(
            texts=texts,
            labels=[""] * len(texts),
            discovered_features=self.discovered_features_,
            provider=self.provider_,
            batch_size=self.batch_size,
            cache=self.cache,
            retry_delay=self.retry_delay,
            system_prompt=self.generation_system_prompt,
            prompt=self.generation_prompt,
        )
        return df.loc[:, self.feature_names_]

    def _transform_one_by_one(self, texts: list[str]) -> pd.DataFrame:
        df = generate_features_batch(
            texts=texts,
            labels=[""] * len(texts),
            discovered_features=self.discovered_features_,
            provider=self.provider_,
            batch_size=1,
            cache=self.cache,
            retry_delay=self.retry_delay,
            system_prompt=self.generation_system_prompt,
            prompt=self.generation_prompt,
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

    def _classes_from_y(self, y: Any) -> list[str]:
        if self.classes is not None:
            return [str(label) for label in self.classes]
        if y is None:
            return ["category_1", "category_2"]
        labels = pd.unique(np.asarray(y, dtype=object).ravel())
        return [str(label) for label in labels]


__all__ = ["LLMFeatureTransformer"]
