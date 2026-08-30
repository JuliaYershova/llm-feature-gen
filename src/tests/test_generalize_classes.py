"""Tests for generalized class handling in discover.py."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.text_features.return_value = [
        {
            "proposed_features": [
                {
                    "feature": "emotional_tone",
                    "description": "Overall emotional tone",
                    "possible_values": ["Positive", "Negative", "Neutral"]
                }
            ]
        }
    ]
    return provider


def test_discover_accepts_num_classes_param():
    """Test that discover_features_from_texts accepts num_classes parameter."""
    import inspect
    from llm_feature_gen.discover import discover_features_from_texts

    sig = inspect.signature(discover_features_from_texts)
    assert "num_classes" in sig.parameters
    assert sig.parameters["num_classes"].default is None


def test_discover_default_num_classes_is_none():
    """Test that default num_classes is None (backward compatible)."""
    import inspect
    from llm_feature_gen.discover import discover_features_from_texts

    sig = inspect.signature(discover_features_from_texts)
    assert sig.parameters["num_classes"].default is None


def test_discover_prompt_updated_for_multiple_classes(mock_provider, tmp_path):
    """Test that prompt is updated when num_classes > 2."""
    from llm_feature_gen.discover import discover_features_from_texts

    texts = ["I am sad", "I am happy", "I am angry"]
    captured_prompts = []

    def capture_prompt(texts, prompt):
        captured_prompts.append(prompt)
        return [{"proposed_features": []}]

    mock_provider.text_features.side_effect = capture_prompt

    discover_features_from_texts(
        texts_or_file=texts,
        provider=mock_provider,
        as_set=True,
        output_dir=tmp_path,
        num_classes=3,
    )

    assert len(captured_prompts) > 0
    assert "3 text categories" in captured_prompts[0]
    assert "  - category_3" in captured_prompts[0]


def test_discover_prompt_unchanged_for_two_classes(mock_provider, tmp_path):
    """Test that prompt is unchanged when num_classes=2 (default behavior)."""
    from llm_feature_gen.discover import discover_features_from_texts
    from llm_feature_gen.prompts import DiscoveryPromptBuilder

    texts = ["I am sad", "I am happy"]
    captured_prompts = []

    def capture_prompt(texts, prompt):
        captured_prompts.append(prompt)
        return [{"proposed_features": []}]

    mock_provider.text_features.side_effect = capture_prompt

    discover_features_from_texts(
        texts_or_file=texts,
        provider=mock_provider,
        as_set=True,
        output_dir=tmp_path,
        num_classes=2,
    )

    assert len(captured_prompts) > 0
    assert captured_prompts[0] == DiscoveryPromptBuilder(n_classes=2, modality="text").build()
    assert "2 text categories" in captured_prompts[0]