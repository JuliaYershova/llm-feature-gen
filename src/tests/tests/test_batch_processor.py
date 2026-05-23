import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from batch_processor import BatchTextProcessor, ResponseCache


class TestBatchProcessor(unittest.TestCase):

    def _make_processor(self, batch_size=3):
        prov = MagicMock()
        prov.default_text_model = "test-model"
        return BatchTextProcessor(provider=prov, batch_size=batch_size)

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            BatchTextProcessor(provider=MagicMock(), batch_size=0)

    def test_process_returns_one_per_text(self):
        p = self._make_processor(batch_size=2)
        results = p.process_texts(["a", "b", "c", "d"], ["feature1", "feature2"])
        self.assertEqual(len(results), 4)

    def test_cache_prevents_duplicate_calls(self):
        p = self._make_processor(batch_size=2)
        texts = ["hello", "world"]
        features = ["f1"]
        p.process_texts(texts, features)
        p.process_texts(texts, features)
        self.assertEqual(len(p.cache), 1)

    def test_estimate_savings_100_texts_batch5(self):
        p = self._make_processor(batch_size=5)
        est = p.estimate_savings(100)
        self.assertEqual(est["one_by_one_calls"], 100)
        self.assertEqual(est["batched_calls"], 20)
        self.assertAlmostEqual(est["savings_pct"], 80.0)

    def test_estimate_savings_zero(self):
        p = self._make_processor()
        est = p.estimate_savings(0)
        self.assertEqual(est["savings_pct"], 0)

    def test_result_has_all_features(self):
        p = self._make_processor(batch_size=10)
        features = ["feat_a", "feat_b", "feat_c"]
        results = p.process_texts(["text1"], features)
        self.assertEqual(set(results[0].keys()), set(features))


if __name__ == "__main__":
    unittest.main()
