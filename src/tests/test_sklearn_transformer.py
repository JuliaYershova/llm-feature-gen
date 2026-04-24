"""Tests for the scikit-learn compatible LLMFeatureTransformer."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

from llm_feature_gen.sklearn_transformer import LLMFeatureTransformer


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.text_features.return_value = [
        {
            "proposed_features": [
                {
                    "feature": "emotional_tone",
                    "description": "Overall emotional tone of the text",
                    "possible_values": ["Positive", "Negative", "Neutral"]
                }
            ]
        }
    ]
    provider.text_feature_values.return_value = [
        {"emotional_tone": "Negative"}
    ]
    return provider


def test_transformer_init():
    """Test that transformer initializes with correct defaults."""
    transformer = LLMFeatureTransformer()
    assert transformer.provider is None
    assert transformer.classes is None
    assert transformer.n_discovery_samples == 10
    assert transformer.discovered_features_path_ is None


def test_transformer_fit_sets_features_path(mock_provider, tmp_path):
    """Test that fit() sets discovered_features_path_."""
    transformer = LLMFeatureTransformer(
        provider=mock_provider,
        output_dir=tmp_path,
    )

    texts = ["I am very sad today", "I feel scared and anxious"]

    with patch("llm_feature_gen.sklearn_transformer.discover_features_from_texts") as mock_discover:
        mock_discover.return_value = {"proposed_features": []}
        transformer.fit(texts)

    assert transformer.discovered_features_path_ is not None
    assert transformer.discovered_features_path_ == tmp_path / "discovered_text_features.json"


def test_transformer_transform_raises_before_fit():
    """Test that transform() raises RuntimeError if fit() not called."""
    transformer = LLMFeatureTransformer()
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before transform\\(\\)"):
        transformer.transform(["some text"])


def test_transformer_is_sklearn_compatible():
    """Test that transformer has required sklearn interface."""
    transformer = LLMFeatureTransformer()
    assert hasattr(transformer, "fit")
    assert hasattr(transformer, "transform")
    assert hasattr(transformer, "fit_transform")
    assert hasattr(transformer, "get_params")
    assert hasattr(transformer, "set_params")


def test_transformer_get_params():
    """Test that get_params returns correct parameters."""
    transformer = LLMFeatureTransformer(
        classes=["sad", "fear"],
        n_discovery_samples=5,
    )
    params = transformer.get_params()
    assert params["classes"] == ["sad", "fear"]
    assert params["n_discovery_samples"] == 5