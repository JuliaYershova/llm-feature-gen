from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_text_to_tabular_pipeline_example_matches_expected(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "examples" / "text_to_tabular_pipeline.py"
    output_dir = tmp_path / "text_to_tabular_pipeline"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--provider",
            "replay",
            "--output-dir",
            str(output_dir),
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert (output_dir / "discovered_text_features.json").exists()
    assert (output_dir / "routine_feature_values.csv").exists()
    assert (output_dir / "urgent_feature_values.csv").exists()
    assert (output_dir / "all_feature_values.csv").exists()
    assert (output_dir / "classifier_predictions.csv").exists()
    assert (output_dir / "classifier_report.json").exists()


def test_map_reduce_example_retains_final_input_and_reuses_checkpoints(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "map_reduce_discovery.py"
    for run in range(2):
        result = subprocess.run(
            [sys.executable, str(script), "--provider", "offline", "--output-dir", str(tmp_path)],
            cwd=repo_root, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        schema = json.loads(result.stdout)
        assert [spec["feature"] for spec in schema["proposed_features"]] == ["topic", "rare_detail"]
        report = json.loads((tmp_path / "discovered_text_features_discovery_report.json").read_text())
        assert report["processed_item_ids"] == [f"text:{i}" for i in range(5)]
        assert report["map_batches"] == 3
        assert report["features"]["rare_detail"]["example_item_ids"] == ["text:4"]
        if run:
            assert report["provider_calls"] == 0
            assert report["cached_steps"] == 4
