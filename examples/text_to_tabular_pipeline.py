#!/usr/bin/env python3
"""Canonical end-to-end example for turning raw text into tabular features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_feature_gen.discover import discover_features_from_texts
from llm_feature_gen.generate import generate_features_from_texts
from llm_feature_gen.providers import LocalProvider, OpenAIProvider

EXAMPLE_ROOT = REPO_ROOT / "examples"
DATA_ROOT = EXAMPLE_ROOT / "data" / "text_to_tabular"
DISCOVERY_DIR = DATA_ROOT / "discovery"
GENERATION_DIR = DATA_ROOT / "tickets"
EXPECTED_DIR = EXAMPLE_ROOT / "expected" / "text_to_tabular_pipeline"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "text_to_tabular_pipeline"

DISCOVERED_JSON_NAME = "discovered_text_features.json"
MERGED_CSV_NAME = "all_feature_values.csv"
PREDICTIONS_CSV_NAME = "classifier_predictions.csv"
REPORT_JSON_NAME = "classifier_report.json"

class ReplayTextProvider:
    """Replay checked-in provider outputs for offline tests and demos."""

    def __init__(self) -> None:
        self.discovery_payload = json.loads(
            (EXPECTED_DIR / DISCOVERED_JSON_NAME).read_text(encoding="utf-8")
        )
        merged_df = pd.read_csv(EXPECTED_DIR / MERGED_CSV_NAME)
        self.feature_lookup = self._load_feature_lookup(merged_df)

    def text_features(self, text_list: List[str], prompt: str | None = None) -> List[Dict[str, Any]]:
        if prompt and "DISCOVERED_FEATURES_SPEC" in prompt:
            outputs = []
            for text in text_list:
                key = self._normalize_text(text)
                if key not in self.feature_lookup:
                    raise KeyError("Replay provider has no fixture for the requested text input.")
                outputs.append({"features": self.feature_lookup[key]})
            return outputs
        return self.discovery_payload

    def _load_feature_lookup(self, merged_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        lookup: Dict[str, Dict[str, str]] = {}
        for _, row in merged_df.iterrows():
            class_name = row["Class"]
            file_name = row["File"]
            text_path = GENERATION_DIR / class_name / file_name
            text = text_path.read_text(encoding="utf-8")
            lookup[self._normalize_text(text)] = {
                "urgency_level": str(row["urgency_level"]),
                "customer_tone": str(row["customer_tone"]),
                "requested_action": str(row["requested_action"]),
                "operational_impact": str(row["operational_impact"]),
                "mentions_deadline": str(row["mentions_deadline"]),
            }
        return lookup

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "local", "replay"],
        default="auto",
        help=(
            "Provider backend. 'auto' selects OpenAI/Azure when configured, "
            "otherwise a local OpenAI-compatible endpoint when configured."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated artifacts should be written.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated artifacts against the checked-in expected outputs.",
    )
    return parser.parse_args()


def select_provider(provider_name: str) -> Any:
    if provider_name == "replay":
        return ReplayTextProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "local":
        return LocalProvider()

    if (
        "AZURE_OPENAI_ENDPOINT" in os.environ
        or "OPENAI_API_KEY" in os.environ
    ):
        return OpenAIProvider()
    if (
        "LOCAL_OPENAI_BASE_URL" in os.environ
        or "LOCAL_MODEL_TEXT" in os.environ
        or "LOCAL_MODEL_VISION" in os.environ
    ):
        return LocalProvider()

    raise EnvironmentError(
        "No provider configuration found. Set OpenAI/Azure credentials, set local "
        "OpenAI-compatible server variables, or run with --provider replay."
    )


def build_classifier_outputs(merged_csv_path: Path, output_dir: Path) -> Dict[str, Path]:
    df = pd.read_csv(merged_csv_path)
    feature_columns = [
        "urgency_level",
        "customer_tone",
        "requested_action",
        "operational_impact",
        "mentions_deadline",
    ]

    encoded = pd.get_dummies(df[feature_columns], dtype=float)
    labels = df["Class"].astype(str)

    predictions: List[Dict[str, Any]] = []
    for row_idx in range(len(df)):
        x_test = encoded.iloc[row_idx]
        x_train = encoded.drop(index=row_idx)
        y_train = labels.drop(index=row_idx)

        centroids = {
            label: x_train.loc[y_train == label].mean(axis=0)
            for label in sorted(y_train.unique())
        }
        distances = {
            label: float(((x_test - centroid) ** 2).sum())
            for label, centroid in centroids.items()
        }
        predicted = min(distances, key=lambda label: (distances[label], label))
        actual = labels.iloc[row_idx]
        predictions.append(
            {
                "File": df.iloc[row_idx]["File"],
                "actual_class": actual,
                "predicted_class": predicted,
                "correct": predicted == actual,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    predictions_path = output_dir / PREDICTIONS_CSV_NAME
    predictions_df.to_csv(predictions_path, index=False)

    accuracy = float(predictions_df["correct"].mean())
    confusion = (
        pd.crosstab(
            predictions_df["actual_class"],
            predictions_df["predicted_class"],
            rownames=["actual_class"],
            colnames=["predicted_class"],
            dropna=False,
        )
        .sort_index()
        .sort_index(axis=1)
    )

    report = {
        "model": "leave_one_out_nearest_centroid",
        "feature_columns": feature_columns,
        "num_examples": int(len(df)),
        "accuracy": accuracy,
        "confusion_matrix": {
            row_label: {col_label: int(value) for col_label, value in row.items()}
            for row_label, row in confusion.to_dict(orient="index").items()
        },
    }

    report_path = output_dir / REPORT_JSON_NAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "predictions": predictions_path,
        "report": report_path,
    }


def compare_with_expected(output_dir: Path) -> None:
    for relative_name in [
        DISCOVERED_JSON_NAME,
        "routine_feature_values.csv",
        "urgent_feature_values.csv",
        MERGED_CSV_NAME,
        PREDICTIONS_CSV_NAME,
        REPORT_JSON_NAME,
    ]:
        generated = output_dir / relative_name
        expected = EXPECTED_DIR / relative_name
        if relative_name.endswith(".csv"):
            generated_df = pd.read_csv(generated)
            expected_df = pd.read_csv(expected)
            if not generated_df.equals(expected_df):
                raise AssertionError(f"Generated CSV does not match expected output: {relative_name}")
        else:
            generated_json = json.loads(generated.read_text(encoding="utf-8"))
            expected_json = json.loads(expected.read_text(encoding="utf-8"))
            if generated_json != expected_json:
                raise AssertionError(f"Generated JSON does not match expected output: {relative_name}")


def clear_previous_outputs(output_dir: Path) -> None:
    for relative_name in [
        DISCOVERED_JSON_NAME,
        "routine_feature_values.csv",
        "urgent_feature_values.csv",
        MERGED_CSV_NAME,
        PREDICTIONS_CSV_NAME,
        REPORT_JSON_NAME,
    ]:
        artifact_path = output_dir / relative_name
        if artifact_path.exists():
            artifact_path.unlink()


def run_pipeline(output_dir: Path, check_expected: bool, provider_name: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(output_dir)
    provider = select_provider(provider_name)

    discover_features_from_texts(
        texts_or_file=DISCOVERY_DIR,
        provider=provider,
        output_dir=output_dir,
        output_filename=DISCOVERED_JSON_NAME,
    )

    discovered_path = output_dir / DISCOVERED_JSON_NAME
    csv_paths = generate_features_from_texts(
        root_folder=GENERATION_DIR,
        discovered_features_path=discovered_path,
        provider=provider,
        classes=["routine", "urgent"],
        output_dir=output_dir,
        merge_to_single_csv=True,
        merged_csv_name=MERGED_CSV_NAME,
    )

    classifier_paths = build_classifier_outputs(Path(csv_paths["__merged__"]), output_dir)

    if check_expected:
        compare_with_expected(output_dir)

    return {
        "discovered": discovered_path,
        "routine_csv": Path(csv_paths["routine"]),
        "urgent_csv": Path(csv_paths["urgent"]),
        "merged_csv": Path(csv_paths["__merged__"]),
        "predictions_csv": classifier_paths["predictions"],
        "classifier_report": classifier_paths["report"],
    }


def main() -> int:
    args = parse_args()
    paths = run_pipeline(args.output_dir, args.check, args.provider)

    print(f"provider: {args.provider}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    if args.check:
        print("all generated artifacts match the checked-in expected outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
