"""Tests for the scikit-learn compatible LLMFeatureTransformer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator

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
                    "possible_values": ["Positive", "Negative", "Neutral"],
                }
            ]
        }
    ]
    provider.text_feature_values.return_value = [{"emotional_tone": "Negative"}]
    return provider


def _fake_generate_features_from_texts(
    root_folder,
    discovered_features_path,
    provider,
    classes,
    output_dir,
    merge_to_single_csv,
    merged_csv_name,
):
    del discovered_features_path, provider, merge_to_single_csv

    root_folder = Path(root_folder)
    output_dir = Path(output_dir)
    class_name = classes[0]
    files = sorted((root_folder / class_name).glob("*.txt"))
    output_path = output_dir / merged_csv_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "File": [file.name for file in files],
            "Class": [class_name] * len(files),
            "emotional_tone": [0.0] * len(files),
            "raw_llm_output": ["{}"] * len(files),
        }
    ).to_csv(output_path, index=False)
    return {"__merged__": str(output_path)}


def _fake_discover_features_from_texts(texts_or_file, provider, as_set, output_dir, output_filename):
    del texts_or_file, provider, as_set

    output_dir = Path(output_dir)
    output_path = output_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "proposed_features": [
                    {
                        "feature": "emotional_tone",
                        "description": "Overall emotional tone of the text",
                        "possible_values": ["Positive", "Negative", "Neutral"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_transformer_init():
    """Test that transformer initializes with correct defaults."""
    transformer = LLMFeatureTransformer()
    assert transformer.provider is None
    assert transformer.classes is None
    assert transformer.output_dir == "outputs"
    assert transformer.n_discovery_samples == 10
    assert not hasattr(transformer, "discovered_features_path_")


def test_transformer_fit_sets_features_path(mock_provider, tmp_path):
    """Test that fit() sets discovered_features_path_."""
    transformer = LLMFeatureTransformer(
        provider=mock_provider,
        output_dir=tmp_path,
    )

    texts = pd.Series(["I am very sad today", "I feel scared and anxious"])

    with patch(
        "llm_feature_gen.sklearn_transformer.discover_features_from_texts",
        side_effect=_fake_discover_features_from_texts,
    ) as mock_discover:
        transformer.fit(texts)

    assert transformer.discovered_features_path_ == tmp_path / "discovered_text_features.json"
    assert mock_discover.call_args.kwargs["texts_or_file"] == list(texts)


def test_transformer_transform_raises_before_fit():
    """Test that transform() raises NotFittedError if fit() has not been called."""
    transformer = LLMFeatureTransformer()
    with pytest.raises(NotFittedError, match="Call fit\\(\\) before transform\\(\\)"):
        transformer.transform(["some text"])


def test_transformer_transform_uses_configured_class_name(mock_provider, tmp_path):
    """Test that transform() does not hardcode the temporary class name."""
    transformer = LLMFeatureTransformer(
        provider=mock_provider,
        classes=["support_tickets"],
        output_dir=tmp_path,
    )

    with patch(
        "llm_feature_gen.sklearn_transformer.discover_features_from_texts",
        side_effect=_fake_discover_features_from_texts,
    ), patch(
        "llm_feature_gen.sklearn_transformer.generate_features_from_texts",
        side_effect=_fake_generate_features_from_texts,
    ) as mock_generate:
        result = transformer.fit_transform(np.array(["first text", "second text"]))

    assert list(result.columns) == ["emotional_tone"]
    assert result.shape == (2, 1)
    assert mock_generate.call_args.kwargs["classes"] == ["support_tickets"]


def test_transformer_treats_raw_string_as_single_sample(mock_provider, tmp_path):
    """Test that one raw string is not split into characters."""
    transformer = LLMFeatureTransformer(provider=mock_provider, output_dir=tmp_path)

    with patch(
        "llm_feature_gen.sklearn_transformer.discover_features_from_texts",
        side_effect=_fake_discover_features_from_texts,
    ), patch(
        "llm_feature_gen.sklearn_transformer.generate_features_from_texts",
        side_effect=_fake_generate_features_from_texts,
    ):
        result = transformer.fit_transform("single text")

    assert result.shape == (1, 1)


def test_transformer_accepts_string_class_name(mock_provider, tmp_path):
    """Test that a string class name is accepted as a convenience."""
    transformer = LLMFeatureTransformer(
        provider=mock_provider,
        classes="support_tickets",
        output_dir=tmp_path,
    )

    with patch(
        "llm_feature_gen.sklearn_transformer.discover_features_from_texts",
        side_effect=_fake_discover_features_from_texts,
    ), patch(
        "llm_feature_gen.sklearn_transformer.generate_features_from_texts",
        side_effect=_fake_generate_features_from_texts,
    ) as mock_generate:
        transformer.fit_transform(["first text"])

    assert mock_generate.call_args.kwargs["classes"] == ["support_tickets"]


def test_transformer_get_params():
    """Test that get_params returns constructor parameters without mutation."""
    classes = ["sad", "fear"]
    transformer = LLMFeatureTransformer(
        classes=classes,
        n_discovery_samples=5,
    )
    params = transformer.get_params()
    assert params["classes"] is classes
    assert params["output_dir"] == "outputs"
    assert params["n_discovery_samples"] == 5


def test_transformer_tags_support_string_inputs():
    """Test sklearn compatibility tags for current and older sklearn versions."""
    transformer = LLMFeatureTransformer()

    assert transformer._more_tags() == {"X_types": ["string"], "requires_y": False}

    tags = transformer.__sklearn_tags__()
    assert tags.input_tags.string is True
    assert tags.input_tags.one_d_array is True
    assert tags.input_tags.two_d_array is False
    assert tags.target_tags.required is False


def test_transformer_passes_sklearn_estimator_checks(tmp_path, monkeypatch):
    """Run sklearn's built-in estimator compatibility checks with mocked LLM calls."""
    monkeypatch.setattr(
        "llm_feature_gen.sklearn_transformer.discover_features_from_texts",
        _fake_discover_features_from_texts,
    )
    monkeypatch.setattr(
        "llm_feature_gen.sklearn_transformer.generate_features_from_texts",
        _fake_generate_features_from_texts,
    )
    monkeypatch.setattr("llm_feature_gen.sklearn_transformer.OpenAIProvider", MagicMock)

    check_estimator(LLMFeatureTransformer(output_dir=tmp_path))
