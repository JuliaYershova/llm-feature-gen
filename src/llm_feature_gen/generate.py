"""Feature value generation helpers built on top of discovered schemas.

The functions in this module take discovery artifacts produced by
``llm_feature_gen.discover`` and apply them to class-organized folders of raw
inputs, producing CSV files that are ready for downstream analysis.
"""

from __future__ import annotations
from .utils.video import extract_key_frames, extract_audio_track
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

def _prepare_tabular_inputs(
        file_path: Path,
        text_column: str,
        label_column: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load a tabular file and normalize rows into text-generation inputs.

    Args:
        file_path: Source CSV, Excel, Parquet, or JSON file.
        text_column: Column containing the free-form text to send to the LLM.
        label_column: Optional column whose values should override the class
            label in the generated output rows.

    Returns:
        A list of dictionaries with at least a ``text`` key and, when
        available, a ``label`` key.

    Raises:
        ValueError: If the file extension is unsupported or ``text_column`` is
            missing from the dataset.
    """

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=";")

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)

    elif suffix == ".json":
        df = pd.read_json(file_path)

    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {file_path.name}")

    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        entry = {
            "text": str(row[text_column])
        }

        if label_column and label_column in df.columns:
            entry["label"] = row[label_column]

        results.append(entry)

    return results


def _prepare_text_inputs(file_path: Path) -> List[str]:
    """Extract text chunks from a supported document file."""
    from .utils.text import extract_text_from_file
    return extract_text_from_file(file_path)


def _prepare_video_inputs(
        file_path: Path,
        use_audio: bool,
        provider: Any
) -> Tuple[List[str], Optional[str]]:
    """Convert a video file into frame payloads plus optional transcript text.

    Args:
        file_path: Video file to process.
        use_audio: Whether to extract audio and request transcription.
        provider: Provider instance that may optionally implement
            ``transcribe_audio``.

    Returns:
        A tuple of ``(base64_frames, transcript_context)``.
    """

    transcript_context = None

    # -------------------------------------------------
    # 1) Audio → transcript (optional)
    # -------------------------------------------------
    if use_audio:
        audio_file = None
        try:
            audio_file = extract_audio_track(str(file_path))

            if audio_file and os.path.exists(audio_file):
                if hasattr(provider, "transcribe_audio"):
                    transcript_context = provider.transcribe_audio(audio_file)
                else:
                    transcript_context = "(Audio transcription not supported by provider)"

        except Exception as e:
            print(f"Warning: Audio extraction failed for {file_path.name}: {e}")

        finally:
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)

    # -------------------------------------------------
    # 2) Visual frames
    # -------------------------------------------------
    b64_list = extract_key_frames(str(file_path), frame_limit=6)

    if not b64_list:
        print(f"Skipping video {file_path.name}: No frames extracted.")
        return [], None

    return b64_list, transcript_context


def _prepare_image_inputs(file_path: Path) -> Tuple[List[str], Optional[str]]:
    """Convert a single image file into a one-item base64 payload."""
    img = Image.open(file_path).convert("RGB")
    b64_list = [image_to_base64(np.array(img))]
    return b64_list, None


def load_discovered_features(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and normalize a discovered-features JSON artifact.

    The helper accepts both the list-oriented structure written by the
    discovery functions and a direct dictionary payload. It normalizes both into
    a dictionary with a ``proposed_features`` key.
    """
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
    """Parse JSON content that may be wrapped in a fenced Markdown block."""
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
    """Append the discovered schema to a base generation prompt."""
    return (
            base_prompt.rstrip()
            + "\n\nDISOVERED_FEATURES_SPEC:\n"
            + json.dumps(discovered_features, ensure_ascii=False, indent=2)
    )


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    """Create an output directory if needed and return it as a ``Path``."""
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
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
) -> Path:
    """Generate feature values for every supported file in one class folder.

    Args:
        folder_path: Root dataset folder containing one subdirectory per class.
        class_name: Name of the class subdirectory to process.
        discovered_features: Discovery payload containing the feature schema.
        provider: Provider instance used to generate values. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        output_dir: Directory where the per-class CSV should be written.
        use_audio: Whether video files should include audio transcription
            context when supported by the provider.
        text_column: Required when processing tabular files. Identifies the
            column sent to the LLM.
        label_column: Optional tabular column whose values override the class
            label for row-level outputs.

    Returns:
        The path to the generated per-class CSV file.

    Raises:
        FileNotFoundError: If the requested class folder does not exist.
        ValueError: If tabular generation is attempted without ``text_column``.
    """
    provider = provider or OpenAIProvider()
    folder_path = Path(folder_path)
    class_folder = folder_path / class_name

    if not class_folder.exists():
        raise FileNotFoundError(f"Class folder not found: {class_folder}")

    raw_names = _extract_feature_names(discovered_features)
    feature_names = list(dict.fromkeys(raw_names))

    video_exts = {".mp4", ".mov", ".avi", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png"}
    text_exts = {".txt", ".pdf", ".docx", ".md", ".html"}
    tabular_exts = {".csv", ".xlsx", ".xls", ".parquet", ".json"}

    all_exts = video_exts | image_exts | text_exts | tabular_exts

    files = sorted(
        f for f in os.listdir(class_folder)
        if Path(f).suffix.lower() in all_exts
    )

    output_dir = _ensure_output_dir(output_dir)
    csv_path = output_dir / f"{class_name}_feature_values.csv"

    iterator = tqdm(files, desc=class_name, unit="file") if tqdm else files

    all_columns = ["File", "Class"] + feature_names + ["raw_llm_output"]

    if not csv_path.exists():
        pd.DataFrame(columns=all_columns).to_csv(csv_path, index=False)

    for filename in iterator:
        file_path = class_folder / filename
        ext = file_path.suffix.lower()

        try:
            # =========================================================
            # TABULAR HANDLING (row-level processing)
            # =========================================================
            if ext in tabular_exts:

                if not text_column:
                    raise ValueError("For tabular generation, text_column must be provided.")

                rows = _prepare_tabular_inputs(
                    file_path=file_path,
                    text_column=text_column,
                    label_column=label_column,
                )

                full_prompt = _build_prompt_for_generation(
                    text_generation_prompt,
                    discovered_features
                )

                for idx, row_data in enumerate(rows):

                    text_value = row_data["text"]

                    llm_resp = provider.text_features(
                        [text_value],
                        prompt=full_prompt,
                    )

                    parsed = llm_resp[0]

                    if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
                        parsed = {"features": parse_json_from_markdown(parsed["features"])}

                    inner = parsed.get("features", parsed) if isinstance(parsed, dict) else {}

                    row_dict: Dict[str, Any] = {
                        "File": f"{filename}__row_{idx}",
                        "Class": row_data.get("label", class_name),
                        "raw_llm_output": json.dumps(parsed, ensure_ascii=False),
                    }

                    for feat in feature_names:
                        value = inner.get(feat, "not given by LLM")
                        row_dict[feat] = value

                    df_out = pd.DataFrame([row_dict], columns=all_columns)
                    df_out.to_csv(csv_path, mode="a", header=False, index=False)

                continue  # skip default file-level logic

            # =========================================================
            # STANDARD FILE-LEVEL HANDLING
            # =========================================================
            if ext in video_exts:
                full_prompt = _build_prompt_for_generation(image_generation_prompt, discovered_features)
                b64_list, transcript_context = _prepare_video_inputs(
                    file_path,
                    use_audio,
                    provider
                )

                if not b64_list:
                    continue

                llm_resp = provider.image_features(
                    image_base64_list=b64_list,
                    prompt=full_prompt,
                    as_set=True,
                    extra_context=transcript_context,
                )

            elif ext in image_exts:
                full_prompt = _build_prompt_for_generation(image_generation_prompt, discovered_features)
                b64_list, _ = _prepare_image_inputs(file_path)
                llm_resp = provider.image_features(
                    image_base64_list=b64_list,
                    prompt=full_prompt,
                )

            elif ext in text_exts:
                full_prompt = _build_prompt_for_generation(text_generation_prompt, discovered_features)
                texts = _prepare_text_inputs(file_path)
                combined_text = "\n\n---\n\n".join(texts)
                llm_resp = provider.text_features(
                    [combined_text],
                    prompt=full_prompt,
                )

            else:
                continue

            parsed = llm_resp[0]

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        # =========================================================
        # FILE-LEVEL RESULT WRITING
        # =========================================================
        if isinstance(parsed, dict) and "features" in parsed and isinstance(parsed["features"], str):
            parsed = {"features": parse_json_from_markdown(parsed["features"])}

        if not feature_names:
            feature_names = _infer_feature_names_from_llm(parsed)

        inner = parsed.get("features", parsed) if isinstance(parsed, dict) else {}

        row = {
            "File": filename,
            "Class": class_name,
            "raw_llm_output": json.dumps(parsed, ensure_ascii=False),
        }

        for feat in feature_names:
            value = inner.get(feat, "not given by LLM")
            row[feat] = value

        pd.DataFrame([row], columns=all_columns).to_csv(
            csv_path,
            mode="a",
            header=False,
            index=False
        )

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
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
) -> Dict[str, str]:
    """Run the full feature-generation pipeline for a class-organized dataset.

    Args:
        root_folder: Dataset root containing one subfolder per class.
        discovered_features_path: Path to a JSON artifact produced by one of
            the discovery helpers.
        output_dir: Directory where CSV outputs should be written.
        classes: Optional subset of class-folder names to process. When omitted,
            all immediate subdirectories are used.
        provider: Provider instance used to generate values.
        merge_to_single_csv: Whether to concatenate per-class CSVs into one
            additional file.
        merged_csv_name: Filename to use for the merged CSV artifact.
        use_audio: Whether video generation should include transcript context.
        text_column: Required for tabular generation.
        label_column: Optional row-level label override for tabular generation.

    Returns:
        A mapping from class name to generated CSV path. When
        ``merge_to_single_csv`` is enabled, the merged output is returned under
        the ``"__merged__"`` key.
    """
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
            text_column=text_column,
            label_column=label_column,
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
def generate_features_from_tabular(*args, **kwargs) -> Dict[str, str]:
    """Generate features using ``outputs/discovered_tabular_features.json`` by default."""
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_tabular_features.json"
    return generate_features(*args, **kwargs)


def generate_features_from_texts(*args, **kwargs) -> Dict[str, str]:
    """Generate features using ``outputs/discovered_text_features.json`` by default."""
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_text_features.json"
    return generate_features(*args, **kwargs)


def generate_features_from_images(*args, **kwargs) -> Dict[str, str]:
    """Generate features using ``outputs/discovered_image_features.json`` by default."""
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_image_features.json"
    return generate_features(*args, **kwargs)


def generate_features_from_videos(*args, **kwargs) -> Dict[str, str]:
    """Generate features using ``outputs/discovered_video_features.json`` by default."""
    if "discovered_features_path" not in kwargs:
        kwargs["discovered_features_path"] = "outputs/discovered_video_features.json"

    kwargs.setdefault("use_audio", True)
    return generate_features(*args, **kwargs)
