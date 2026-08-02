from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_feature_gen.multiclass as multiclass_mod


def test_multiclass_prompt_builder_includes_classes_and_configurable_min_features():
    prompt = multiclass_mod.MultiClassDiscoveryPromptBuilder(
        classes=["travel_alert", "book_flight", "flight_status"],
        min_features=7,
    ).build()

    assert "3 text categories" in prompt
    assert "  - travel_alert" in prompt
    assert "  - book_flight" in prompt
    assert "  - flight_status" in prompt
    assert "at least 7 distinct features" in prompt
    assert "{n_classes}" not in prompt
    assert "{class_list}" not in prompt
    assert "{min_features}" not in prompt


def test_multiclass_prompt_builder_defaults_and_validation():
    prompt = multiclass_mod.MultiClassDiscoveryPromptBuilder(classes=["A", "B", "C", "D"]).build()
    assert "at least 12 distinct features" in prompt

    prompt = multiclass_mod.MultiClassDiscoveryPromptBuilder(classes=["A", "B"]).build()
    assert "at least 10 distinct features" in prompt

    with pytest.raises(ValueError, match="At least 2 classes"):
        multiclass_mod.MultiClassDiscoveryPromptBuilder(classes=["only"])

    with pytest.raises(ValueError, match="min_features"):
        multiclass_mod.MultiClassDiscoveryPromptBuilder(classes=["A", "B"], min_features=0)


def test_discover_features_multiclass_delegates_with_formatted_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_discover_features_from_texts(**kwargs):
        captured.update(kwargs)
        return {"proposed_features": [{"feature": "intent"}]}

    monkeypatch.setattr(
        multiclass_mod,
        "discover_features_from_texts",
        fake_discover_features_from_texts,
    )

    result = multiclass_mod.discover_features_multiclass(
        texts_or_file=["one", "two"],
        classes=["A", "B", "C"],
        provider="provider",
        output_dir=tmp_path,
        output_filename="features.json",
        as_set=False,
        min_features=5,
    )

    assert result == {"proposed_features": [{"feature": "intent"}]}
    assert captured["texts_or_file"] == ["one", "two"]
    assert captured["provider"] == "provider"
    assert captured["output_dir"] == tmp_path
    assert captured["output_filename"] == "features.json"
    assert captured["as_set"] is False
    assert "3 text categories" in captured["prompt"]
    assert "at least 5 distinct features" in captured["prompt"]


def test_generate_features_multiclass_validates_classes_and_delegates_path_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir()
    discovered_path = tmp_path / "features.json"
    discovered_path.write_text(json.dumps({"proposed_features": [{"feature": "intent"}]}), encoding="utf-8")

    captured = {}

    def fake_generate_features(**kwargs):
        captured.update(kwargs)
        return {"A": "a.csv", "B": "b.csv"}

    monkeypatch.setattr(multiclass_mod, "generate_features", fake_generate_features)

    result = multiclass_mod.generate_features_multiclass(
        root_folder=root,
        discovered_features=discovered_path,
        classes=["A", "B"],
        provider="provider",
        output_dir=tmp_path / "out",
        merge_to_single_csv=False,
        merged_csv_name="merged.csv",
    )

    assert result == {"A": "a.csv", "B": "b.csv"}
    assert captured["root_folder"] == root
    assert captured["discovered_features_path"] == discovered_path
    assert captured["classes"] == ["A", "B"]
    assert captured["provider"] == "provider"
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["merge_to_single_csv"] is False
    assert captured["merged_csv_name"] == "merged.csv"


def test_generate_features_multiclass_writes_dict_schema_to_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir()
    schema = {"proposed_features": [{"feature": "intent"}]}
    captured = {}

    def fake_generate_features(**kwargs):
        captured.update(kwargs)
        return {"ok": "1"}

    monkeypatch.setattr(multiclass_mod, "generate_features", fake_generate_features)

    result = multiclass_mod.generate_features_multiclass(
        root_folder=root,
        discovered_features=schema,
        classes=["A", "B"],
        output_dir=tmp_path / "out",
    )

    assert result == {"ok": "1"}
    temp_path = captured["discovered_features_path"]
    assert temp_path == tmp_path / "out" / "_tmp_discovered_features.json"
    assert json.loads(temp_path.read_text(encoding="utf-8")) == schema


def test_generate_features_multiclass_rejects_invalid_class_inputs(tmp_path: Path):
    root = tmp_path / "root"
    (root / "A").mkdir(parents=True)

    with pytest.raises(ValueError, match="At least 2 classes"):
        multiclass_mod.generate_features_multiclass(
            root_folder=root,
            discovered_features=tmp_path / "features.json",
            classes=["A"],
        )

    with pytest.raises(FileNotFoundError, match="Class sub-folder not found"):
        multiclass_mod.generate_features_multiclass(
            root_folder=root,
            discovered_features=tmp_path / "features.json",
            classes=["A", "missing"],
        )


def test_run_multiclass_pipeline_orchestrates_discovery_train_and_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_discover_features_multiclass(**kwargs):
        calls.append(("discover", kwargs))
        return {"proposed_features": [{"feature": "intent"}]}

    def fake_generate_features_multiclass(**kwargs):
        calls.append(("generate", kwargs))
        return {"__merged__": str(kwargs["output_dir"] / kwargs["merged_csv_name"])}

    monkeypatch.setattr(
        multiclass_mod,
        "discover_features_multiclass",
        fake_discover_features_multiclass,
    )
    monkeypatch.setattr(
        multiclass_mod,
        "generate_features_multiclass",
        fake_generate_features_multiclass,
    )

    result = multiclass_mod.run_multiclass_pipeline(
        discover_folder=tmp_path / "discover",
        train_folder=tmp_path / "train",
        test_folder=tmp_path / "test",
        classes=["A", "B", "C"],
        provider="provider",
        output_dir=tmp_path / "out",
        min_features=9,
    )

    assert result["discovered_features"] == {"proposed_features": [{"feature": "intent"}]}
    assert result["train_csv_paths"]["__merged__"].endswith("train_feature_values.csv")
    assert result["test_csv_paths"]["__merged__"].endswith("test_feature_values.csv")

    assert calls[0][0] == "discover"
    assert calls[0][1]["min_features"] == 9
    assert calls[0][1]["output_filename"] == "discovered_text_features.json"

    assert calls[1][0] == "generate"
    assert calls[1][1]["root_folder"] == tmp_path / "train"
    assert calls[1][1]["discovered_features"] == tmp_path / "out" / "discovered_text_features.json"
    assert calls[1][1]["merged_csv_name"] == "train_feature_values.csv"

    assert calls[2][0] == "generate"
    assert calls[2][1]["root_folder"] == tmp_path / "test"
    assert calls[2][1]["merged_csv_name"] == "test_feature_values.csv"


def test_multiclass_image_prompt_builder_uses_visual_template():
    prompt = multiclass_mod.MultiClassDiscoveryPromptBuilder(
        classes=["American_Crow", "Fish_Crow", "Common_Raven", "Shiny_Cowbird"],
        min_features=12,
        template=multiclass_mod.multiclass_image_discovery_prompt,
    ).build()

    assert "4 visual categories" in prompt
    assert "  - Fish_Crow" in prompt
    assert "at least 12 distinct features" in prompt
    assert "{n_classes}" not in prompt


def test_discover_features_multiclass_images_delegates_with_formatted_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_discover_features_from_images(**kwargs):
        captured.update(kwargs)
        return {"proposed_features": [{"feature": "beak_shape"}]}

    monkeypatch.setattr(
        multiclass_mod,
        "discover_features_from_images",
        fake_discover_features_from_images,
    )

    result = multiclass_mod.discover_features_multiclass_images(
        image_paths_or_folder=["a.jpg", "b.jpg"],
        classes=["A", "B", "C"],
        provider="provider",
        output_dir=tmp_path,
        output_filename="features.json",
        as_set=False,
        min_features=5,
    )

    assert result == {"proposed_features": [{"feature": "beak_shape"}]}
    assert captured["image_paths_or_folder"] == ["a.jpg", "b.jpg"]
    assert captured["output_filename"] == "features.json"
    assert captured["as_set"] is False
    assert "3 visual categories" in captured["prompt"]
    assert "at least 5 distinct features" in captured["prompt"]


def test_discover_features_multiclass_videos_forwards_video_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_discover_features_from_videos(**kwargs):
        captured.update(kwargs)
        return {"proposed_features": [{"feature": "arm_stroke_pattern"}]}

    monkeypatch.setattr(
        multiclass_mod,
        "discover_features_from_videos",
        fake_discover_features_from_videos,
    )

    multiclass_mod.discover_features_multiclass_videos(
        videos_or_folder=tmp_path,
        classes=["Diving", "FrontCrawl", "BreastStroke"],
        provider="provider",
        output_dir=tmp_path,
        num_frames=8,
        use_audio=False,
        max_videos_to_sample=4,
        max_total_frames_payload=12,
        random_seed=42,
        min_features=11,
    )

    assert captured["num_frames"] == 8
    assert captured["use_audio"] is False
    assert captured["max_videos_to_sample"] == 4
    assert captured["max_total_frames_payload"] == 12
    assert captured["random_seed"] == 42
    assert "3 visual categories" in captured["prompt"]
    assert "at least 11 distinct features" in captured["prompt"]