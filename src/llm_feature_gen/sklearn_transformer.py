"""Scikit-learn compatible transformer for LLM-generated features."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union
from uuid import uuid4

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .discover import discover_features_from_texts
from .generate import generate_features_from_texts
from .providers.openai_provider import OpenAIProvider


class LLMFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform raw texts into LLM-generated feature DataFrames.

    This transformer integrates with scikit-learn pipelines. It first
    discovers features from the training data, then generates feature
    values for any input texts.

    Args:
        provider: LLM provider instance. Defaults to OpenAIProvider.
        classes: Optional list of class names. The first class name is used as
            the temporary class folder/label during transformation.
        output_dir: Directory to store intermediate files.
        n_discovery_samples: Number of samples to use for feature discovery.
        random_state: Seed or random state for reproducible discovery sampling.

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> transformer = LLMFeatureTransformer(provider=my_provider, classes=["sad", "fear"])
        >>> pipeline = Pipeline([("features", transformer), ("clf", DecisionTreeClassifier())])
    """

    def __init__(
        self,
        provider=None,
        classes: Optional[list[str]] = None,
        output_dir: Union[str, Path] = "outputs",
        n_discovery_samples: int = 10,
        random_state=None,
    ):
        self.provider = provider
        self.classes = classes
        self.output_dir = output_dir
        self.n_discovery_samples = n_discovery_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Discover features from training texts.

        Args:
            X: Iterable of raw text strings.
            y: Ignored. Present for scikit-learn compatibility.

        Returns:
            self
        """
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        provider = self.provider or OpenAIProvider()

        texts = self._as_text_list(X)
        samples = self._sample_texts(texts)
        output_filename = f"discovered_text_features_{uuid4().hex}.json"

        discover_features_from_texts(
            texts_or_file=samples,
            provider=provider,
            as_set=True,
            output_dir=output_dir,
            output_filename=output_filename,
        )

        self.discovered_features_path_ = output_dir / output_filename
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Generate feature values for input texts.

        Args:
            X: Iterable of raw text strings.
            y: Ignored. Present for scikit-learn compatibility.

        Returns:
            DataFrame with one row per text and one column per feature.

        Raises:
            NotFittedError: If fit() has not been called yet.
        """
        try:
            check_is_fitted(self, "discovered_features_path_")
        except NotFittedError as exc:
            raise NotFittedError("Call fit() before transform().") from exc

        provider = self.provider or OpenAIProvider()
        texts = self._as_text_list(X)
        class_name = self._transform_class_name()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory() as tmp_root, TemporaryDirectory(dir=output_dir) as tmp_output:
            tmp_dir = Path(tmp_root)
            tmp_class_dir = tmp_dir / class_name
            tmp_class_dir.mkdir()
            input_files = [f"text_{i:05d}.txt" for i in range(len(texts))]
            for filename, text in zip(input_files, texts):
                (tmp_class_dir / filename).write_text(str(text), encoding="utf-8")

            csv_paths = generate_features_from_texts(
                root_folder=tmp_dir,
                discovered_features_path=self.discovered_features_path_,
                provider=provider,
                classes=[class_name],
                output_dir=tmp_output,
                merge_to_single_csv=True,
                merged_csv_name="transformed_features.csv",
            )

            merged_path = Path(csv_paths["__merged__"])
            df = pd.read_csv(merged_path)
            df = self._align_generated_rows(df, input_files)

            drop_cols = [c for c in ["File", "Class", "raw_llm_output"] if c in df.columns]
            return df.drop(columns=drop_cols)

    def _transform_class_name(self) -> str:
        if not self.classes:
            return "texts"
        if isinstance(self.classes, str):
            return self.classes
        return self.classes[0]

    @staticmethod
    def _as_text_list(X) -> list:
        if isinstance(X, str):
            return [X]
        return list(X)

    def _sample_texts(self, texts: list) -> list:
        sample_count = min(self.n_discovery_samples, len(texts))
        if sample_count >= len(texts):
            return texts
        rng = check_random_state(self.random_state)
        indices = rng.choice(len(texts), size=sample_count, replace=False)
        return [texts[i] for i in indices]

    @staticmethod
    def _align_generated_rows(df: pd.DataFrame, input_files: list[str]) -> pd.DataFrame:
        if "File" not in df.columns:
            if len(df) != len(input_files):
                raise ValueError(
                    f"Generated {len(df)} rows for {len(input_files)} input texts."
                )
            return df

        aligned = df.set_index("File", drop=False).reindex(input_files)
        if aligned.isna().all(axis=1).any():
            raise ValueError(
                f"Generated {len(df)} rows for {len(input_files)} input texts."
            )
        return aligned.reset_index(drop=True)

    def _more_tags(self):
        return {"X_types": ["string"], "requires_y": False}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = False
        return tags
