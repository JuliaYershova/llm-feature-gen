import unittest
from unittest.mock import MagicMock
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sklearn_transformer import LLMFeatureTransformer


class TestLLMFeatureTransformer(unittest.TestCase):

    def _make_transformer(self):
        mock_provider = MagicMock()
        return LLMFeatureTransformer(provider=mock_provider, classes=["class_a", "class_b"])

    def test_init_attributes(self):
        t = self._make_transformer()
        self.assertEqual(t.classes, ["class_a", "class_b"])
        self.assertIsNone(t._features_)

    def test_is_sklearn_compatible(self):
        from sklearn.base import BaseEstimator, TransformerMixin
        t = self._make_transformer()
        self.assertIsInstance(t, BaseEstimator)
        self.assertIsInstance(t, TransformerMixin)

    def test_get_params(self):
        t = self._make_transformer()
        params = t.get_params()
        self.assertIn("classes", params)
        self.assertIn("output_dir", params)

    def test_not_fitted_raises(self):
        t = self._make_transformer()
        with self.assertRaises(RuntimeError):
            t.transform(["some text"])

    def test_texts_to_dir(self):
        import tempfile
        t = self._make_transformer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "test_root"
            t._texts_to_dir(["hello", "world"], ["class_a", "class_b"], root)
            files_a = list((root / "class_a").glob("*.txt"))
            files_b = list((root / "class_b").glob("*.txt"))
            self.assertEqual(len(files_a), 1)
            self.assertEqual(len(files_b), 1)
            self.assertEqual(files_a[0].read_text(), "hello")


if __name__ == "__main__":
    unittest.main()
