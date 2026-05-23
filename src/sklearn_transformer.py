cat << 'EOF'
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class LLMFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible transformer that uses llm-feature-gen to
    discover binary LLM features and return a dense NumPy feature matrix.

    Parameters
    ----------
    provider : LocalProvider or compatible
        LLM provider to use for discovery and generation.
    classes : list[str]
        The two class names expected in the input.
    output_dir : str or Path, optional
        Directory to cache intermediate outputs.
    temperature : float, optional
        Sampling temperature forwarded to the provider.
    """

    def __init__(self, provider, classes, output_dir="llm_transformer_cache", temperature=0.2):
        self.provider = provider
        self.classes = classes
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self._features_ = None

    def _texts_to_dir(self, X, y, root):
        """Write (text, label) pairs into root/<class>/<n>.txt layout."""
        import shutil
        if root.exists():
            shutil.rmtree(root)
        for cn in self.classes:
            (root / cn).mkdir(parents=True, exist_ok=True)
        counters = {cn: 0 for cn in self.classes}
        for text, label in zip(X, y):
            counters[label] += 1
            (root / label / f"{label}_{counters[label]}.txt").write_text(str(text), encoding="utf-8")
        return root

    def _discover_dir(self, X, y):
        """Create flat discovery directory from a small sample."""
        import shutil
        from collections import defaultdict
        disc = self.output_dir / "_discover"
        if disc.exists():
            shutil.rmtree(disc)
        disc.mkdir(parents=True, exist_ok=True)
        counters = {cn: 0 for cn in self.classes}
        per_class = defaultdict(list)
        for text, label in zip(X, y):
            if label in self.classes:
                per_class[label].append(text)
        for cn in self.classes:
            for txt in per_class[cn][:10]:
                counters[cn] += 1
                (disc / f"{cn}_discover_{counters[cn]}.txt").write_text(str(txt), encoding="utf-8")
        return disc

    def fit(self, X, y):
        """Discover features from (X, y) training data."""
        from llm_feature_gen.discover import discover_features_from_texts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        disc_dir = self._discover_dir(X, y)
        self._features_ = discover_features_from_texts(
            texts_or_file=disc_dir,
            provider=self.provider,
            as_set=True,
            output_dir=self.output_dir,
            output_filename="sklearn_discovered_features.json",
        )
        self._train_root_ = self._texts_to_dir(X, y, self.output_dir / "_train_texts")
        return self

    def transform(self, X, y=None):
        """Generate feature values for X; returns a dense NumPy array."""
        import shutil
        from llm_feature_gen.generate import generate_features_from_texts

        if self._features_ is None:
            raise RuntimeError("Call fit() before transform().")

        labels = y if y is not None else ["unknown"] * len(X)
        trans_root = self.output_dir / "_transform_texts"
        self._texts_to_dir(X, labels, trans_root)

        trans_out = self.output_dir / "_transform_out"
        if trans_out.exists():
            shutil.rmtree(trans_out)

        csv_paths = generate_features_from_texts(
            root_folder=trans_root,
            discovered_features_path=self.output_dir / "sklearn_discovered_features.json",
            provider=self.provider,
            classes=self.classes,
            output_dir=trans_out,
            merge_to_single_csv=True,
            merged_csv_name="transform_feature_values.csv",
        )
        merged = Path(csv_paths["__merged__"])
        df = pd.read_csv(merged)
        drop_cols = [c for c in ["File", "Class", "raw_llm_output", "split"] if c in df.columns]
        feature_df = pd.get_dummies(df.drop(columns=drop_cols), dtype=int)
        return feature_df.values.astype(np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)
EOF
Output

import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class LLMFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    scikit-learn compatible transformer that uses llm-feature-gen to
    discover binary LLM features and return a dense NumPy feature matrix.

    Parameters
    ----------
    provider : LocalProvider or compatible
        LLM provider to use for discovery and generation.
    classes : list[str]
        The two class names expected in the input.
    output_dir : str or Path, optional
        Directory to cache intermediate outputs.
    temperature : float, optional
        Sampling temperature forwarded to the provider.
    """

    def __init__(self, provider, classes, output_dir="llm_transformer_cache", temperature=0.2):
        self.provider = provider
        self.classes = classes
        self.output_dir = Path(output_dir)
        self.temperature = temperature
        self._features_ = None

    def _texts_to_dir(self, X, y, root):
        """Write (text, label) pairs into root/<class>/<n>.txt layout."""
        import shutil
        if root.exists():
            shutil.rmtree(root)
        for cn in self.classes:
            (root / cn).mkdir(parents=True, exist_ok=True)
        counters = {cn: 0 for cn in self.classes}
        for text, label in zip(X, y):
            counters[label] += 1
            (root / label / f"{label}_{counters[label]}.txt").write_text(str(text), encoding="utf-8")
        return root

    def _discover_dir(self, X, y):
        """Create flat discovery directory from a small sample."""
        import shutil
        from collections import defaultdict
        disc = self.output_dir / "_discover"
        if disc.exists():
            shutil.rmtree(disc)
        disc.mkdir(parents=True, exist_ok=True)
        counters = {cn: 0 for cn in self.classes}
        per_class = defaultdict(list)
        for text, label in zip(X, y):
            if label in self.classes:
                per_class[label].append(text)
        for cn in self.classes:
            for txt in per_class[cn][:10]:
                counters[cn] += 1
                (disc / f"{cn}_discover_{counters[cn]}.txt").write_text(str(txt), encoding="utf-8")
        return disc

    def fit(self, X, y):
        """Discover features from (X, y) training data."""
        from llm_feature_gen.discover import discover_features_from_texts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        disc_dir = self._discover_dir(X, y)
        self._features_ = discover_features_from_texts(
            texts_or_file=disc_dir,
            provider=self.provider,
            as_set=True,
            output_dir=self.output_dir,
            output_filename="sklearn_discovered_features.json",
        )
        self._train_root_ = self._texts_to_dir(X, y, self.output_dir / "_train_texts")
        return self

    def transform(self, X, y=None):
        """Generate feature values for X; returns a dense NumPy array."""
        import shutil
        from llm_feature_gen.generate import generate_features_from_texts

        if self._features_ is None:
            raise RuntimeError("Call fit() before transform().")

        labels = y if y is not None else ["unknown"] * len(X)
        trans_root = self.output_dir / "_transform_texts"
        self._texts_to_dir(X, labels, trans_root)

        trans_out = self.output_dir / "_transform_out"
        if trans_out.exists():
            shutil.rmtree(trans_out)

        csv_paths = generate_features_from_texts(
            root_folder=trans_root,
            discovered_features_path=self.output_dir / "sklearn_discovered_features.json",
            provider=self.provider,
            classes=self.classes,
            output_dir=trans_out,
            merge_to_single_csv=True,
            merged_csv_name="transform_feature_values.csv",
        )
        merged = Path(csv_paths["__merged__"])
        df = pd.read_csv(merged)
        drop_cols = [c for c in ["File", "Class", "raw_llm_output", "split"] if c in df.columns]
        feature_df = pd.get_dummies(df.drop(columns=drop_cols), dtype=int)
        return feature_df.values.astype(np.float32)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)
