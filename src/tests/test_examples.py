from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


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


def test_sklearn_pipeline_example_matches_expected(tmp_path: Path):
    pytest.importorskip("sklearn")
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "examples" / "sklearn_pipeline.py"
    output_dir = tmp_path / "sklearn_pipeline"

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
    assert (output_dir / "feature_table.csv").exists()
    assert (output_dir / "result.json").exists()
