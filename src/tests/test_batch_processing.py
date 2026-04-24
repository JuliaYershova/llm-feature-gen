"""Tests for batch processing support in generate.py."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import json


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.text_features.return_value = [
        {"emotional_tone": "Negative", "anxiety_level": "High"}
    ]
    return provider


@pytest.fixture
def sample_discovered_features():
    return {
        "proposed_features": [
            {
                "feature": "emotional_tone",
                "description": "Overall emotional tone",
                "possible_values": ["Positive", "Negative", "Neutral"]
            },
            {
                "feature": "anxiety_level",
                "description": "Level of anxiety expressed",
                "possible_values": ["High", "Medium", "Low"]
            }
        ]
    }


@pytest.fixture
def class_folder(tmp_path, sample_discovered_features):
    """Create a temporary class folder with sample text files."""
    class_dir = tmp_path / "sad"
    class_dir.mkdir()

    for i in range(3):
        (class_dir / f"text_{i}.txt").write_text(
            f"I feel very sad today number {i}", encoding="utf-8"
        )

    features_path = tmp_path / "discovered_text_features.json"
    features_path.write_text(
        json.dumps(sample_discovered_features), encoding="utf-8"
    )

    return tmp_path


def test_generate_features_accepts_batch_size_param(class_folder, mock_provider):
    """Test that generate_features accepts batch_size parameter."""
    from llm_feature_gen.generate import generate_features

    with patch("llm_feature_gen.generate.assign_feature_values_from_folder") as mock_assign:
        mock_assign.return_value = class_folder / "sad_feature_values.csv"

        pd.DataFrame(columns=["File", "Class", "emotional_tone", "anxiety_level", "raw_llm_output"]).to_csv(
            class_folder / "sad_feature_values.csv", index=False
        )

        generate_features(
            root_folder=class_folder,
            discovered_features_path=class_folder / "discovered_text_features.json",
            classes=["sad"],
            provider=mock_provider,
            output_dir=class_folder / "outputs",
            batch_size=2,
        )

        mock_assign.assert_called_once()
        _, kwargs = mock_assign.call_args
        assert kwargs["batch_size"] == 2


def test_assign_folder_accepts_batch_size_param(class_folder, mock_provider, sample_discovered_features):
    """Test that assign_feature_values_from_folder accepts batch_size parameter."""
    from llm_feature_gen.generate import assign_feature_values_from_folder

    result = assign_feature_values_from_folder(
        folder_path=class_folder,
        class_name="sad",
        discovered_features=sample_discovered_features,
        provider=mock_provider,
        output_dir=class_folder / "outputs",
        batch_size=2,
    )

    assert result.exists()


def test_batch_size_default_is_one(class_folder, mock_provider, sample_discovered_features):
    """Test that default batch_size is 1 (backward compatible)."""
    from llm_feature_gen.generate import assign_feature_values_from_folder
    import inspect

    sig = inspect.signature(assign_feature_values_from_folder)
    assert sig.parameters["batch_size"].default == 1