# src/LLM_feature_gen/generate.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from PIL import Image
import numpy as np

from .providers.openai_provider import OpenAIProvider
from .utils.image import image_to_base64
from .prompts import image_generation_prompt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ----------------------------
# helpers
# ----------------------------
def load_discovered_features(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Discovered features file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # normalize
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict) and "proposed_features" in data[0]:
            data = data[0]
        else:
            data = {"proposed_features": data}

    return data


def parse_json_from_markdown(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    txt = text.strip()
    if txt.startswith("```"):
        lines = txt.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        txt = "\n".join(lines).strip()
    try:
        return json.loads(txt)
    except Exception:
        return {}


def _build_prompt_for_generation(base_prompt: str, discovered_features: Dict[str, Any]) -> str:
    return (
        base_prompt.rstrip()
        + "\n\nDISOVERED_FEATURES_SPEC:\n"
        + json.dumps(discovered_features, ensure_ascii=False, indent=2)
    )


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_feature_names(discovered_features: Any) -> List[str]:
    """
    Try to get feature names from discovered_features.
    Supports:
      - {"proposed_features": [ {"feature": "..."}, ... ]}
      - [{"feature": "..."}, ...]
      - ["feature a", "feature b"]
    """
    if isinstance(discovered_features, list):
        discovered_features = {"proposed_features": discovered_features}

    feats = discovered_features.get("proposed_features", [])
    names: List[str] = []
    for f in feats:
        if isinstance(f, dict) and "feature" in f:
            names.append(f["feature"])
        elif isinstance(f, str):
            names.append(f)
    return names


def _infer_feature_names_from_llm(parsed: Any) -> List[str]:
    """
    Your LLM sometimes returns:
        [ { "presence of liquid broth": "...", ... } ]
    or
        { "features": { ... } }
    This tries to infer feature names from that.
    """
    # case: list with single dict
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return list(parsed[0].keys())

    # case: {"features": {...}}
    if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], dict):
        return list(parsed["features"].keys())

    # case: flat dict
    if isinstance(parsed, dict):
        return list(parsed.keys())

    return []


# ----------------------------
# per-class generation
# ----------------------------
def assign_feature_values_from_folder(
    folder_path: Union[str, Path],
    class_name: str,
    discovered_features: Dict[str, Any],
    prompt_text: str,
    provider: Optional["OpenAIProvider"] = None,
    output_dir: Union[str, Path] = "outputs",
) -> Path:
    """
    For each image in <folder_path>/<class_name>:
    - send image + discovered-features prompt to LLM
    - parse response
    - APPEND to outputs/<class_name>_feature_values.csv

    Output CSV columns (fixed):
        Image, Class, <feature1>, <feature2>, ..., raw_llm_output
    Each feature cell is formatted as: "<feature name> = <feature value>"
    """
    provider = provider or OpenAIProvider()

    folder_path = Path(folder_path)
    class_folder = folder_path / class_name
    if not class_folder.exists():
        raise FileNotFoundError(f"Class folder not found: {class_folder}")

    # 1) try to get feature names from discovered_features
    feature_names = _extract_feature_names(discovered_features)

    full_prompt = _build_prompt_for_generation(prompt_text, discovered_features)

    image_files = [
        f for f in os.listdir(class_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    output_dir = _ensure_output_dir(output_dir)
    csv_path = output_dir / f"{class_name}_feature_values.csv"

    # We'll need to maybe infer feature names from the FIRST image
    iterator = image_files
    if tqdm is not None:
        iterator = tqdm(image_files, desc=f"{class_name}", unit="img")

    first_row_written = csv_path.exists()

    for idx, filename in enumerate(iterator):
        img_path = class_folder / filename
        img = Image.open(img_path).convert("RGB")
        img_b64 = image_to_base64(np.array(img))

        llm_resp = provider.image_features(
            image_base64_list=[img_b64],
            prompt=full_prompt,
        )
        parsed = llm_resp

        # sometimes model returns {"features": "<json str>"}
        if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
            maybe_json = parse_json_from_markdown(parsed["features"])
            if isinstance(maybe_json, dict):
                parsed = {"features": maybe_json}

        # If we STILL have no feature names, infer from this first LLM output
        if not feature_names:
            feature_names = _infer_feature_names_from_llm(parsed)

        # Now we know all columns
        all_columns = ["Image", "Class"] + feature_names + ["raw_llm_output"]

        # If CSV doesn't exist yet, create with header now (we now know feature names!)
        if not csv_path.exists():
            header_df = pd.DataFrame(columns=all_columns)
            header_df.to_csv(csv_path, index=False, encoding="utf-8")

        # build row
        row: Dict[str, Any] = {
            "Image": filename,
            "Class": class_name,
        }

        raw_dump = json.dumps(parsed, ensure_ascii=False)

        # ---- fill feature values ----
        # 1) parsed is list with single dict
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
            inner = parsed[0]
            for feat in feature_names:
                value = inner.get(feat, "not given by LLM")
                row[feat] = f"{feat} = {value}"

        # 2) parsed is {"features": {...}}
        elif isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], dict):
            inner = parsed["features"]
            for feat in feature_names:
                value = inner.get(feat, "not given by LLM")
                row[feat] = f"{feat} = {value}"

        # 3) parsed is flat dict
        elif isinstance(parsed, dict):
            for feat in feature_names:
                value = parsed.get(feat, "not given by LLM")
                row[feat] = f"{feat} = {value}"

        # 4) unknown
        else:
            for feat in feature_names:
                row[feat] = f"{feat} = not given by LLM"

        row["raw_llm_output"] = raw_dump

        # ensure all columns
        for col in all_columns:
            row.setdefault(col, "")

        df = pd.DataFrame([row], columns=all_columns)
        df.to_csv(
            csv_path,
            mode="a",
            header=False,  # header already created
            index=False,
            encoding="utf-8",
        )

    return csv_path


# ----------------------------
# high-level orchestrator
# ----------------------------
def generate_features(
    root_folder: Union[str, Path],
    discovered_features: Optional[Dict[str, Any]] = None,
    discovered_features_path: Union[str, Path] = "outputs/discovered_features.json",
    prompt: Optional[str] = None,
    output_dir: Union[str, Path] = "outputs",
    classes: Optional[List[str]] = None,
    provider: Optional[OpenAIProvider] = None,
    merge_to_single_csv: bool = False,
    merged_csv_name: str = "all_feature_values.csv",
) -> Dict[str, str]:
    root_folder = Path(root_folder)
    provider = provider or OpenAIProvider()

    if discovered_features is None:
        discovered_features = load_discovered_features(discovered_features_path)
        print(f"Loaded discovered features from {discovered_features_path}")

    if prompt is None:
        prompt = image_generation_prompt

    if classes is None:
        classes = [p.name for p in root_folder.iterdir() if p.is_dir()]

    csv_paths: Dict[str, str] = {}
    per_class_dfs: List[pd.DataFrame] = []

    for cls in classes:
        csv_path = assign_feature_values_from_folder(
            folder_path=root_folder,
            class_name=cls,
            discovered_features=discovered_features,
            prompt_text=prompt,
            provider=provider,
            output_dir=output_dir,
        )
        csv_paths[cls] = str(csv_path)

        if merge_to_single_csv:
            per_class_dfs.append(pd.read_csv(csv_path))

    if merge_to_single_csv and per_class_dfs:
        output_dir = _ensure_output_dir(output_dir)
        merged_path = Path(output_dir) / merged_csv_name
        merged_df = pd.concat(per_class_dfs, ignore_index=True)
        merged_df.to_csv(merged_path, index=False, encoding="utf-8")
        csv_paths["__merged__"] = str(merged_path)

    return csv_paths


def generate_features_from_images(*args, **kwargs) -> Dict[str, str]:
    return generate_features(*args, **kwargs)