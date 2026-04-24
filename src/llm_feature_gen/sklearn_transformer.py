"""Scikit-learn compatible transformer for LLM-generated features."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
        classes: List of class names for generation.
        output_dir: Directory to store intermediate files.
        n_discovery_samples: Number of samples to use for feature discovery.

    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> transformer = LLMFeatureTransformer(provider=my_provider, classes=["sad", "fear"])
        >>> pipeline = Pipeline([("features", transformer), ("clf", DecisionTreeClassifier())])
    """

    def __init__(
        self,
        provider=None,
        classes: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "outputs",
        n_discovery_samples: int = 10,
    ):
        self.provider = provider
        self.classes = classes
        self.output_dir = Path(output_dir)
        self.n_discovery_samples = n_discovery_samples
        self.discovered_features_path_: Optional[Path] = None

    def fit(self, X: List[str], y=None):
        """Discover features from training texts.

        Args:
            X: List of raw text strings.
            y: Ignored. Present for scikit-learn compatibility.

        Returns:
            self
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        provider = self.provider or OpenAIProvider()

        # Sample texts for discovery
        samples = X[: self.n_discovery_samples]

        discovered = discover_features_from_texts(
            texts_or_file=samples,
            provider=provider,
            as_set=True,
            output_dir=self.output_dir,
            output_filename="discovered_text_features.json",
        )

        self.discovered_features_path_ = self.output_dir / "discovered_text_features.json"
        return self

    def transform(self, X: List[str], y=None) -> pd.DataFrame:
        """Generate feature values for input texts.

        Args:
            X: List of raw text strings.
            y: Ignored. Present for scikit-learn compatibility.

        Returns:
            DataFrame with one row per text and one column per feature.

        Raises:
            RuntimeError: If fit() has not been called yet.
        """
        if self.discovered_features_path_ is None:
            raise RuntimeError("Call fit() before transform().")

        provider = self.provider or OpenAIProvider()

        # Write texts to temp files
        import tempfile
        import shutil

        tmp_dir = Path(tempfile.mkdtemp())
        tmp_class_dir = tmp_dir / "texts"
        tmp_class_dir.mkdir()

        try:
            for i, text in enumerate(X):
                (tmp_class_dir / f"text_{i:05d}.txt").write_text(text, encoding="utf-8")

            csv_paths = generate_features_from_texts(
                root_folder=tmp_dir,
                discovered_features_path=self.discovered_features_path_,
                provider=provider,
                classes=["texts"],
                output_dir=self.output_dir / "transformed",
                merge_to_single_csv=True,
                merged_csv_name="transformed_features.csv",
            )

            merged_path = Path(csv_paths["__merged__"])
            df = pd.read_csv(merged_path)

            drop_cols = [c for c in ["File", "Class", "raw_llm_output"] if c in df.columns]
            return df.drop(columns=drop_cols)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)