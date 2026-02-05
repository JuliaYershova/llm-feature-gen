# src/LLM_feature_gen/generate.py
from __future__ import annotations
from .utils.video import extract_key_frames, transcribe_video
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd
from PIL import Image
import numpy as np

from .providers.openai_provider import OpenAIProvider
from .utils.image import image_to_base64
from .prompts import image_generation_prompt, text_generation_prompt

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ----------------------------
# helpers
# ----------------------------

def _prepare_text_inputs(file_path: Path) -> List[str]:
    """
    Load text from a file (txt, pdf, docx, etc.)
    Returns a list of text chunks.
    """
    from .utils.text import extract_text_from_file
    return extract_text_from_file(file_path)


def _prepare_video_inputs(file_path: Path, use_audio: bool) -> Tuple[List[str], Optional[str]]:
    transcript_context = None

    # 1. Audio / Transcript
    if use_audio:
        raw_transcript = transcribe_video(str(file_path))
        if raw_transcript and len(raw_transcript) > 10:
            transcript_context = raw_transcript
        else:
            transcript_context = "No distinct speech detected."
    else:
        transcript_context = None

    # 2. Visuals / Frames
    b64_list = extract_key_frames(str(file_path), frame_limit=6)

    if not b64_list:
        print(f"Skipping video {file_path.name}: No frames extracted.")
        return [], None

    return b64_list, transcript_context


def _prepare_image_inputs(file_path: Path) -> Tuple[List[str], Optional[str]]:
    img = Image.open(file_path).convert("RGB")
    b64_list = [image_to_base64(np.array(img))]
    return b64_list, None


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
        provider: Optional[OpenAIProvider] = None,
        output_dir: Union[str, Path] = "outputs",
        use_audio: bool = True,
) -> Path:
    provider = provider or OpenAIProvider()
    folder_path = Path(folder_path)
    class_folder = folder_path / class_name

    if not class_folder.exists():
        raise FileNotFoundError(f"Class folder not found: {class_folder}")

    feature_names = _extract_feature_names(discovered_features)

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png"}
    text_exts = {".txt", ".pdf", ".docx", ".md", ".html"}

    all_exts = video_exts | image_exts | text_exts

    files = sorted(
        f for f in os.listdir(class_folder)
        if Path(f).suffix.lower() in all_exts
    )

    output_dir = _ensure_output_dir(output_dir)
    csv_path = output_dir / f"{class_name}_feature_values.csv"

    iterator = tqdm(files, desc=class_name, unit="file") if tqdm else files

    for filename in iterator:
        file_path = class_folder / filename
        ext = file_path.suffix.lower()

        try:
            # ----------------------------
            # prompt selection (single place)
            # ----------------------------
            if ext in image_exts or ext in video_exts:
                base_prompt = image_generation_prompt
            elif ext in text_exts:
                base_prompt = text_generation_prompt
            else:
                continue

            full_prompt = _build_prompt_for_generation(base_prompt, discovered_features)

            # ----------------------------
            # modality-specific execution
            # ----------------------------
            if ext in video_exts:
                b64_list, transcript_context = _prepare_video_inputs(file_path, use_audio)
                llm_resp = provider.image_features(
                    image_base64_list=b64_list,
                    prompt=full_prompt,
                    extra_context=transcript_context,
                )

            elif ext in image_exts:
                b64_list, _ = _prepare_image_inputs(file_path)
                llm_resp = provider.image_features(
                    image_base64_list=b64_list,
                    prompt=full_prompt,
                )

            elif ext in text_exts:
                texts = _prepare_text_inputs(file_path)
                combined_text = "\n\n---\n\n".join(texts)
                llm_resp = provider.text_features(
                    [combined_text],
                    prompt=full_prompt,
                )

            parsed = llm_resp[0]

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # normalize JSON
        if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
            parsed = {"features": parse_json_from_markdown(parsed["features"])}

        if not feature_names:
            feature_names = _infer_feature_names_from_llm(parsed)

        all_columns = ["File", "Class"] + feature_names + ["raw_llm_output"]

        if not csv_path.exists():
            pd.DataFrame(columns=all_columns).to_csv(csv_path, index=False)

        row: Dict[str, Any] = {
            "File": filename,
            "Class": class_name,
            "raw_llm_output": json.dumps(parsed, ensure_ascii=False),
        }

        inner = parsed.get("features", parsed) if isinstance(parsed, dict) else {}

        for feat in feature_names:
            value = inner.get(feat, "not given by LLM")
            row[feat] = f"{feat} = {value}"

        df = pd.DataFrame([row], columns=all_columns)
        df.to_csv(csv_path, mode="a", header=False, index=False)

    return csv_path


# ----------------------------
# high-level orchestrator
# ----------------------------
def generate_features(
        root_folder: Union[str, Path],
        discovered_features_path: Union[str, Path],
        output_dir: Union[str, Path] = "outputs",
        classes: Optional[List[str]] = None,
        provider: Optional[OpenAIProvider] = None,
        merge_to_single_csv: bool = False,
        merged_csv_name: str = "all_feature_values.csv",
        use_audio: bool = True,
) -> Dict[str, str]:
    root_folder = Path(root_folder)
    provider = provider or OpenAIProvider()
    discovered_features = load_discovered_features(discovered_features_path)

    if classes is None:
        classes = [p.name for p in root_folder.iterdir() if p.is_dir()]

    csv_paths: Dict[str, str] = {}
    dfs: List[pd.DataFrame] = []

    for cls in classes:
        csv_path = assign_feature_values_from_folder(
            folder_path=root_folder,
            class_name=cls,
            discovered_features=discovered_features,
            provider=provider,
            output_dir=output_dir,
            use_audio=use_audio,
        )
        csv_paths[cls] = str(csv_path)

        if merge_to_single_csv:
            dfs.append(pd.read_csv(csv_path))

    if merge_to_single_csv and dfs:
        merged_path = Path(output_dir) / merged_csv_name
        pd.concat(dfs, ignore_index=True).to_csv(merged_path, index=False)
        csv_paths["__merged__"] = str(merged_path)

    return csv_paths


# ----------------------------
# modality-specific wrappers
# ----------------------------
def generate_features_from_texts(*args, **kwargs) -> Dict[str, str]:
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_text_features.json"
    return generate_features(*args, **kwargs)


def generate_features_from_images(*args, **kwargs) -> Dict[str, str]:
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_image_features.json"
    return generate_features(*args, **kwargs)


def generate_features_from_videos(*args, **kwargs) -> Dict[str, str]:
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_image_features.json"

    kwargs.setdefault("use_audio", True)
    return generate_features(*args, **kwargs)