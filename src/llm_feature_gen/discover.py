"""Public discovery helpers for multimodal feature schema generation.

The functions in this module accept raw inputs or folders on disk, delegate the
actual reasoning to a provider, and persist the discovered schema as JSON in an
output directory. Discovery is intentionally folder-oriented so the same API can
be used from notebooks, scripts, and batch pipelines.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime

from .utils.image import image_to_base64
from .utils.text import extract_text_from_file
from dotenv import load_dotenv
from .utils.video import extract_key_frames, extract_audio_track, downsample_batch
from .providers.openai_provider import OpenAIProvider
from .prompts import image_discovery_prompt, text_discovery_prompt

# Load environment variables automatically
load_dotenv()

DiscoveryPayload = Dict[str, Any]
DiscoveryResult = Union[DiscoveryPayload, List[DiscoveryPayload]]


def discover_features_from_images(
        image_paths_or_folder: str | List[str],
        prompt: str = image_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # <- default TRUE for discovery
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
) -> DiscoveryResult:
    """Discover features from image files and persist the provider response.

    Args:
        image_paths_or_folder: A single image path, a folder containing images,
            or a list of image file paths.
        prompt: System-style prompt passed through to the provider.
        provider: Optional provider instance. When omitted, an
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider] is
            created from environment variables.
        as_set: When ``True``, all images are analyzed together and a single
            shared feature schema is produced. When ``False``, each image is
            sent independently and the result contains one entry per image.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_image_features.json``.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        per-image mode. The on-disk JSON always preserves the raw provider
        result list.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no supported image files are found.
        RuntimeError: If image decoding fails for every candidate input.
    """
    # 1) init provider
    provider = provider or OpenAIProvider()

    # 2) collect image paths
    if isinstance(image_paths_or_folder, (str, Path)):
        folder_path = Path(image_paths_or_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Path not found: {folder_path}")

        if folder_path.is_dir():
            image_paths = [
                str(p)
                for p in folder_path.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        else:
            image_paths = [str(folder_path)]
    else:
        image_paths = list(image_paths_or_folder)

    if not image_paths:
        raise ValueError("No image files found to process.")

    # 3) to base64
    b64_list: List[str] = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            b64_list.append(image_to_base64(np.array(img)))
        except Exception as e:
            print(f"Could not load {path}: {e}")

    if not b64_list:
        raise RuntimeError("Failed to load any valid images from input.")

    # 4) CALL PROVIDER
    if as_set:
        # send ALL images in ONE request – this uses your new provider logic
        result_list = provider.image_features(
            b64_list,
            prompt=prompt,
            as_set=True,
        )
    else:
        # per-image behavior
        result_list = provider.image_features(
            b64_list,
            prompt=prompt,
            as_set=False,
        )

    # - joint mode: result_list is like: [ { "proposed_features": [...] } ]
    # - per-image mode: result_list is list of dicts

    # 5) save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = "discovered_image_features.json"

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Features saved to {output_path}")

    # return the FIRST (and only) element in joint mode to keep downstream simple
    if as_set and isinstance(result_list, list) and len(result_list) == 1:
        return result_list[0]

    return result_list


def discover_features_from_videos(
        videos_or_folder: str | List[str],
        prompt: str = image_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # stejné chování jako image/text
        num_frames: int = 5,
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        use_audio: bool = True,
        max_videos_to_sample: int = 5,
        max_total_frames_payload: int = 15
) -> DiscoveryResult:
    """Discover features from one or more videos.

    Each video is converted into representative frames and, optionally, an
    audio transcript. The resulting multimodal payload is sent to the provider
    and the raw response is written to JSON.

    Args:
        videos_or_folder: A single video path, a folder containing videos, or a
            list of video file paths.
        prompt: Prompt passed through to the provider.
        provider: Optional provider instance implementing ``image_features``
            and, when ``use_audio=True``, optionally ``transcribe_audio``.
        as_set: When ``True``, all extracted frames are analyzed together to
            produce one shared schema. When ``False``, frames are analyzed in
            per-item mode.
        num_frames: Target number of key frames to extract per video before
            downsampling across the batch.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_video_features.json``.
        use_audio: Whether to extract an audio track and include a transcript
            as extra context when the provider supports transcription.
        max_videos_to_sample: Upper bound on how many videos are sampled from a
            folder input to control cost and payload size.
        max_total_frames_payload: Upper bound on the total number of frames sent
            to the provider across the batch.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        per-item mode.

    Raises:
        FileNotFoundError: If the input path is missing or a folder contains no
            supported video files.
        ValueError: If no frames can be extracted from the provided videos.
    """

    # -------------------------------------------------
    # 1) init provider
    # -------------------------------------------------
    provider = provider or OpenAIProvider()

    # -------------------------------------------------
    # 2) collect video paths
    # -------------------------------------------------
    if isinstance(videos_or_folder, (str, Path)):
        path_obj = Path(videos_or_folder)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path_obj}")

        if path_obj.is_dir():
            valid_exts = {".mp4", ".mov", ".avi", ".mkv"}

            video_paths = [
                p for p in path_obj.iterdir()
                if p.suffix.lower() in valid_exts
            ]

            if not video_paths:
                raise FileNotFoundError(f"No videos found in folder: {path_obj}")

            # optional sampling (same logic as before)
            if len(video_paths) > max_videos_to_sample:
                video_paths = random.sample(video_paths, max_videos_to_sample)

        else:
            video_paths = [path_obj]

    else:
        video_paths = [Path(p) for p in videos_or_folder]

    # -------------------------------------------------
    # 3) extract frames + transcripts
    # -------------------------------------------------
    all_frames_b64: List[str] = []
    combined_transcripts: List[str] = []

    for video_p in video_paths:

        # ---- A) extract visual frames
        try:
            frames = extract_key_frames(str(video_p), frame_limit=num_frames)
            if frames:
                all_frames_b64.extend(frames)
        except Exception as e:
            print(f"Error extracting frames from {video_p.name}: {e}")
            continue

        # ---- B) extract + transcribe audio (optional)
        if use_audio:
            audio_file_path = None
            try:
                audio_file_path = extract_audio_track(str(video_p))

                if audio_file_path and os.path.exists(audio_file_path):
                    if hasattr(provider, "transcribe_audio"):
                        transcript = provider.transcribe_audio(audio_file_path)

                        if transcript and len(transcript) > 10:
                            combined_transcripts.append(
                                f"TRANSCRIPT ({video_p.name}):\n{transcript}"
                            )
                    else:
                        print("Warning: Provider does not support transcribe_audio.")

            except Exception as e:
                print(f"Audio processing failed for {video_p.name}: {e}")

            finally:
                # cleanup temporary audio file
                if audio_file_path and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)

    if not all_frames_b64:
        raise ValueError("No frames extracted from input videos.")

    if len(all_frames_b64) > max_total_frames_payload:
        all_frames_b64 = downsample_batch(all_frames_b64, max_total_frames_payload)

    # join transcripts into a single context block
    final_context = "\n\n".join(combined_transcripts) if combined_transcripts else None

    # -------------------------------------------------
    # 4) CALL PROVIDER
    # -------------------------------------------------
    if as_set:
        # joint discovery (ALL frames in one request)
        result_list = provider.image_features(
            all_frames_b64,
            prompt=prompt,
            as_set=True,
            extra_context=final_context,
        )
    else:
        # per-frame discovery
        result_list = provider.image_features(
            all_frames_b64,
            prompt=prompt,
            as_set=False,
            extra_context=final_context,
        )

    # -------------------------------------------------
    # 5) save results
    # -------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = "discovered_video_features.json"

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Features saved to {output_path}")

    # -------------------------------------------------
    # 6) return behavior (same as image/text)
    # -------------------------------------------------
    if as_set and isinstance(result_list, list) and len(result_list) == 1:
        return result_list[0]

    return result_list


def discover_features_from_texts(
        texts_or_file: str | List[str],  # input is text(s)
        prompt: str = text_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # same semantics as image version
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
) -> DiscoveryResult:
    """Discover features from text strings, files, or folders of documents.

    Args:
        texts_or_file: Either raw text strings, a single supported document
            path, or a directory containing supported text documents.
        prompt: Prompt passed through to the provider.
        provider: Optional provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        as_set: When ``True``, all extracted text is combined into a single
            request so the provider can discover a shared schema. When
            ``False``, each text chunk is processed independently.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_text_features.json``.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        per-text mode.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the path is invalid or no supported text input can be
            extracted.
    """

    # 1) init provider
    provider = provider or OpenAIProvider()

    # -------------------------------------------------
    # 2) collect texts
    # -------------------------------------------------
    texts: List[str] = []

    if isinstance(texts_or_file, (str, Path)):
        path = Path(texts_or_file)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            # single file of ANY supported text type
            texts = extract_text_from_file(path)

        elif path.is_dir():
            # folder with mixed document types
            for file in sorted(path.iterdir()):
                if file.is_file():
                    try:
                        texts.extend(extract_text_from_file(file))
                    except ValueError:
                        pass  # skip unsupported files silently

        else:
            raise ValueError("Invalid path provided.")

    else:
        # list of raw text strings
        texts = list(texts_or_file)

    if not texts:
        raise ValueError("No text inputs found to process.")
    # -------------------------------------------------
    # 3) CALL PROVIDER
    # -------------------------------------------------
    if as_set:
        #  JOINT DISCOVERY MODE
        combined_text = "\n\n---\n\n".join(texts)

        result_list = provider.text_features(
            [combined_text],  # ONE request
            prompt=prompt,
        )
    else:
        # PER-TEXT DESCRIPTION MODE
        result_list = provider.text_features(
            texts,  # MANY requests
            prompt=prompt,
        )

    # -------------------------------------------------
    # 4) save
    # -------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = "discovered_text_features.json"

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Features saved to {output_path}")

    # -------------------------------------------------
    # 5) return behavior
    # -------------------------------------------------
    if as_set and isinstance(result_list, list) and len(result_list) == 1:
        return result_list[0]

    return result_list


def discover_features_from_tabular(
        file_or_folder: str | Path,
        text_column: str,
        provider: Optional[OpenAIProvider] = None,
        prompt: str = text_discovery_prompt,
        as_set: bool = True,
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        max_rows: Optional[int] = None,
        ) -> DiscoveryResult:
    """Discover features from tabular datasets by projecting a text column.

    Supported files are loaded into a single DataFrame, the selected text
    column is extracted, and the resulting list of strings is delegated to
    [discover_features_from_texts][llm_feature_gen.discover.discover_features_from_texts].

    Args:
        file_or_folder: A single tabular file or a directory containing
            supported tabular files.
        text_column: Column name whose values should be used as textual input
            for discovery.
        provider: Optional provider instance.
        prompt: Prompt passed through to the provider.
        as_set: Whether to discover one shared schema across all sampled rows or
            process rows independently.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_tabular_features.json``.
        max_rows: Optional cap on how many rows are used from the concatenated
            dataset.

    Returns:
        The same return shape as
        [discover_features_from_texts][llm_feature_gen.discover.discover_features_from_texts].

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no supported tabular files are found or ``text_column``
            is missing.
    """
    import pandas as pd
    provider = provider or OpenAIProvider()
    path = Path(file_or_folder)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    def load_file(file_path: Path):
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            try:
                return pd.read_csv(file_path)
            except:
                return pd.read_csv(file_path, sep=";")
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif suffix == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    dfs = []

    if path.is_file():
        dfs.append(load_file(path))
    elif path.is_dir():
        for f in sorted(path.iterdir()):
            if f.is_file():
                try:
                    dfs.append(load_file(f))
                except Exception as e:
                    print(f"Skipping {f.name}: {e}")

    if not dfs:
        raise ValueError("No valid tabular files found.")

    df = pd.concat(dfs, ignore_index=True)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found.")

    texts = df[text_column].dropna().astype(str).tolist()

    if max_rows:
        texts = texts[:max_rows]

    return discover_features_from_texts(
        texts_or_file=texts,
        prompt=prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename or "discovered_tabular_features.json",
    )
