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
from .prompts import (
    image_generation_prompt,
    tabular_generation_prompt,
    text_generation_prompt,
    video_generation_prompt,
)
from .contracts import ProviderResponseError, normalize_feature_values_response, parse_json_object_from_markdown

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ----------------------------
# helpers
# ----------------------------

class GenerationCircuitBreakerError(RuntimeError):
    """Raised when repeated provider/output failures make a run unlikely to recover."""


def _format_generation_failure(source: str, reason: str) -> str:
    """Build a compact, user-facing failure message."""
    return f"{source}: {reason}"


def _provider_error_from_payload(payload: Any) -> Optional[str]:
    """Return a provider error message from a returned payload, if present."""
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return None


def _check_failure_threshold(
        failure_count: int,
        failure_threshold: Optional[int],
        last_failure: str,
) -> None:
    """Raise when the configured consecutive failure threshold is reached."""
    if failure_threshold is not None and failure_threshold > 0 and failure_count >= failure_threshold:
        raise GenerationCircuitBreakerError(
            f"Stopping generation after {failure_count} consecutive provider/output "
            f"failures. Last failure: {last_failure}"
        )

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
        provider: Any,
        num_frames: int = 6,
) -> Tuple[List[str], Optional[str]]:
    """Convert a video file into frame payloads plus optional transcript text.

    Args:
        file_path: Video file to process.
        use_audio: Whether to extract audio and request transcription.
        provider: Provider instance that may optionally implement
            ``transcribe_audio``.
        num_frames: Maximum number of key frames extracted per video.

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
    b64_list = extract_key_frames(str(file_path), frame_limit=num_frames)

    if not b64_list:
        print(f"Skipping video {file_path.name}: No frames extracted.")
        return [], None

    return b64_list, transcript_context


def _prepare_image_inputs(file_path: Path) -> Tuple[List[str], Optional[str]]:
    """Convert a single image file into a one-item base64 payload."""
    img = Image.open(file_path).convert("RGB")
    b64_list = [image_to_base64(np.array(img))]
    return b64_list, None


def _validate_discovered_schema(data: Any) -> None:
    """
    Raises ValueError if the discovery artifact is not a usable schema.
    Catches error payloads, missing proposed_features, and empty feature lists.
    """
    # error payload from provider
    if isinstance(data, dict) and "error" in data:
        raise ValueError(
            f"Discovery artifact contains a provider error instead of a schema: "
            f"{data['error']}"
        )

    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict) and "error" in data[0]:
            raise ValueError(
                f"Discovery artifact contains a provider error instead of a schema: "
                f"{data[0]['error']}"
            )
        # normalize for further checks
        if len(data) == 1 and isinstance(data[0], dict) and "proposed_features" in data[0]:
            data = data[0]
        else:
            data = {"proposed_features": data}

    proposed = data.get("proposed_features") if isinstance(data, dict) else None

    if not proposed:
        raise ValueError(
            "Discovery artifact has no 'proposed_features'. "
            "Discovery likely failed. Check your provider and try again."
        )

    if not isinstance(proposed, list):
        raise ValueError(
            f"'proposed_features' must be a list, got {type(proposed).__name__}."
        )

    valid = [
        f for f in proposed
        if (isinstance(f, dict) and f.get("feature"))
        or (isinstance(f, str) and f)
    ]

    if not valid:
        raise ValueError(
            "Discovery artifact has 'proposed_features' but no valid entries. "
            "Each entry must have a non-empty 'feature' key."
        )


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

    _validate_discovered_schema(data)

    # normalize
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict) and "proposed_features" in data[0]:
            data = data[0]
        else:
            data = {"proposed_features": data}

    return data


def parse_json_from_markdown(text: str) -> Dict[str, Any]:
    """Parse JSON content that may be wrapped in a fenced Markdown block."""
    try:
        return parse_json_object_from_markdown(text)
    except ProviderResponseError:
        return {}


def _build_prompt_for_generation(base_prompt: str, discovered_features: Dict[str, Any]) -> str:
    """Append the discovered schema to a base generation prompt."""
    return (
            base_prompt.rstrip()
            + "\n\nDISCOVERED_FEATURES_SPEC:\n"
            + json.dumps(discovered_features, ensure_ascii=False, indent=2)
    )


def _ensure_output_dir(path: Union[str, Path]) -> Path:
    """Create an output directory if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _provider_call_kwargs(
        provider: Any,
        system_prompt: Optional[str],
        response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if response_schema is not None and getattr(provider, "supports_response_schema", False) is True:
        kwargs["response_schema"] = response_schema
    return kwargs


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


def _iter_feature_specs(discovered_features: Any) -> List[Dict[str, Any]]:
    if isinstance(discovered_features, list):
        discovered_features = {"proposed_features": discovered_features}

    specs: List[Dict[str, Any]] = []
    seen = set()
    for item in discovered_features.get("proposed_features", []):
        if isinstance(item, str):
            item = {"feature": item}
        if not isinstance(item, dict) or not item.get("feature"):
            continue

        name = item["feature"]
        if name in seen:
            continue
        seen.add(name)
        specs.append(item)
    return specs


def _enum_values(feature_spec: Dict[str, Any]) -> List[str]:
    values = feature_spec.get("allowed_values") or feature_spec.get("possible_values") or []
    return [value for value in values if isinstance(value, str)]


def _build_generation_response_schema(discovered_features: Any) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for feature_spec in _iter_feature_specs(discovered_features):
        name = feature_spec["feature"]
        schema: Dict[str, Any] = {"type": "string"}
        values = _enum_values(feature_spec)
        if values:
            schema["enum"] = values
        properties[name] = schema
        required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _validate_generation_features(features: Dict[str, Any], discovered_features: Any) -> None:
    specs = _iter_feature_specs(discovered_features)
    expected = [spec["feature"] for spec in specs]
    expected_set = set(expected)
    actual_set = set(features)

    missing = expected_set - actual_set
    if missing:
        raise ValueError(f"Generation response is missing feature keys: {sorted(missing)}")

    extra = actual_set - expected_set
    if extra:
        raise ValueError(f"Generation response has unexpected feature keys: {sorted(extra)}")

    for spec in specs:
        name = spec["feature"]
        value = features[name]
        if not isinstance(value, str):
            raise ValueError(f"Generation value for '{name}' must be a string.")

        values = _enum_values(spec)
        if values and value not in values:
            raise ValueError(
                f"Generation value for '{name}' must be one of {values}, got {value!r}."
            )


def _normalize_generation_payload(payload: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize a provider response and validate that it has feature values.

    Returns the normalized raw payload and the inner feature dictionary used for
    CSV columns. Provider errors, empty responses, and invalid feature payloads
    raise ``ValueError`` so the generation circuit breaker can count them.
    """
    provider_error = _provider_error_from_payload(payload)
    if provider_error:
        raise ValueError(f"provider_error: {provider_error}")

    try:
        normalized = normalize_feature_values_response(payload)
    except ProviderResponseError as exc:
        raise ValueError(str(exc)) from exc

    return normalized, normalized["features"]

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
        num_frames: int = 6,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        failure_threshold: Optional[int] = 3,
        system_prompt: Optional[str] = None,
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
        num_frames: Maximum number of key frames extracted per video.
        text_column: Required when processing tabular files. Identifies the
            column sent to the LLM.
        label_column: Optional tabular column whose values override the class
            label for row-level outputs.
        failure_threshold: Number of consecutive provider/output failures after
            which generation aborts. Pass ``None`` or ``0`` to disable.
        system_prompt: Optional custom system instruction for the provider.

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
    if not feature_names:
        raise ValueError("discovered_features must include at least one feature name")
    response_schema = _build_generation_response_schema(discovered_features)

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

    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"Overwriting {csv_path.name} left by a previous run.")
    pd.DataFrame(columns=all_columns).to_csv(csv_path, index=False)

    consecutive_failures = 0

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
                    tabular_generation_prompt,
                    discovered_features
                )

                for idx, row_data in enumerate(rows):

                    text_value = row_data["text"]

                    try:
                        llm_resp = provider.text_features(
                            [text_value],
                            prompt=full_prompt,
                            **_provider_call_kwargs(provider, system_prompt, response_schema),
                        )

                        parsed, inner = _normalize_generation_payload(llm_resp[0])
                        _validate_generation_features(inner, discovered_features)

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
                        consecutive_failures = 0

                    except Exception as e:
                        consecutive_failures += 1
                        row_source = f"{filename}__row_{idx}"
                        last_failure = _format_generation_failure(row_source, str(e))
                        print(f"Error processing {last_failure}")
                        _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
                        continue

                continue  # skip default file-level logic

            # =========================================================
            # STANDARD FILE-LEVEL HANDLING
            # =========================================================
            if ext in video_exts:
                full_prompt = _build_prompt_for_generation(video_generation_prompt, discovered_features)
                b64_list, transcript_context = _prepare_video_inputs(
                    file_path,
                    use_audio,
                    provider,
                    num_frames=num_frames,
                )

                if not b64_list:
                    continue

                try:
                    llm_resp = provider.image_features(
                        image_base64_list=b64_list,
                        prompt=full_prompt,
                        as_set=True,
                        extra_context=transcript_context,
                        **_provider_call_kwargs(provider, system_prompt, response_schema),
                    )
                except Exception as e:
                    consecutive_failures += 1
                    last_failure = _format_generation_failure(filename, str(e))
                    print(f"Error processing {last_failure}")
                    _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
                    continue

            elif ext in image_exts:
                full_prompt = _build_prompt_for_generation(image_generation_prompt, discovered_features)
                b64_list, _ = _prepare_image_inputs(file_path)
                try:
                    llm_resp = provider.image_features(
                        image_base64_list=b64_list,
                        prompt=full_prompt,
                        **_provider_call_kwargs(provider, system_prompt, response_schema),
                    )
                except Exception as e:
                    consecutive_failures += 1
                    last_failure = _format_generation_failure(filename, str(e))
                    print(f"Error processing {last_failure}")
                    _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
                    continue

            elif ext in text_exts:
                full_prompt = _build_prompt_for_generation(text_generation_prompt, discovered_features)
                texts = _prepare_text_inputs(file_path)
                combined_text = "\n\n---\n\n".join(texts)
                try:
                    llm_resp = provider.text_features(
                        [combined_text],
                        prompt=full_prompt,
                        **_provider_call_kwargs(provider, system_prompt, response_schema),
                    )
                except Exception as e:
                    consecutive_failures += 1
                    last_failure = _format_generation_failure(filename, str(e))
                    print(f"Error processing {last_failure}")
                    _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
                    continue

            else:
                continue

            try:
                parsed = llm_resp[0]
            except Exception:
                consecutive_failures += 1
                last_failure = _format_generation_failure(filename, "empty_output: provider returned no output")
                print(f"Error processing {last_failure}")
                _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
                continue

        except GenerationCircuitBreakerError:
            raise
        except Exception as e:
            consecutive_failures += 1
            last_failure = _format_generation_failure(filename, f"input_preparation_error: {e}")
            print(f"Error processing {last_failure}")
            _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
            continue

        # =========================================================
        # FILE-LEVEL RESULT WRITING
        # =========================================================
        try:
            parsed, inner = _normalize_generation_payload(parsed)
            _validate_generation_features(inner, discovered_features)
        except Exception as e:
            consecutive_failures += 1
            last_failure = _format_generation_failure(filename, str(e))
            print(f"Error processing {last_failure}")
            _check_failure_threshold(consecutive_failures, failure_threshold, last_failure)
            continue
            
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
        consecutive_failures = 0

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
        num_frames: int = 6,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        failure_threshold: Optional[int] = 3,
        system_prompt: Optional[str] = None,
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
        num_frames: Maximum number of key frames extracted per video.
        text_column: Required for tabular generation.
        label_column: Optional row-level label override for tabular generation.
        failure_threshold: Number of consecutive provider/output failures after
            which generation aborts. Pass ``None`` or ``0`` to disable.
        system_prompt: Optional custom system instruction for the provider.

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
            failure_threshold=failure_threshold,
            num_frames=num_frames,
            system_prompt=system_prompt,
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
