# src/LLM_feature_gen/discover.py
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime

from .utils.image import image_to_base64
from .utils.text import extract_text_from_file
from dotenv import load_dotenv
from .utils.video import extract_key_frames, transcribe_video
from .providers.openai_provider import OpenAIProvider
from .prompts import image_discovery_prompt, text_discovery_prompt
import random

# Load environment variables automatically
load_dotenv()


def discover_features_from_images(
        image_paths_or_folder: str | List[str],
        prompt: str = image_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # <- default TRUE for discovery
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper: takes a list of image file paths OR a folder path,
    converts images to base64, calls the provider, and saves the JSON result.
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
        video_path: str,
        prompt: str = image_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        num_frames: int = 5,
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        use_audio: bool = True,
        max_videos_to_sample: int = 5,
) -> Dict[str, Any]:
    """
    Extracts key frames AND audio transcript from a video (or sample of videos) to discover features.
    """

    # Initialize provider
    provider = provider or OpenAIProvider()

    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {video_path}")

    # 0. Determine videos to process (Single file vs Folder sample) ---
    videos_to_process = []

    if path_obj.is_dir():
        # Find all videos
        valid_exts = {".mp4", ".mov", ".avi", ".mkv"}
        all_videos = [p for p in path_obj.iterdir() if p.suffix.lower() in valid_exts]

        if not all_videos:
            raise FileNotFoundError(f"No videos found in folder: {video_path}")

        # Sample if too many
        if len(all_videos) > max_videos_to_sample:
            print(f"Sampling {max_videos_to_sample} random videos from folder '{path_obj.name}'...")
            videos_to_process = random.sample(all_videos, max_videos_to_sample)
        else:
            print(f"Processing all {len(all_videos)} videos from folder '{path_obj.name}'...")
            videos_to_process = all_videos
    else:
        # Single file
        print(f"Processing single video for discovery: {path_obj.name}")
        videos_to_process = [path_obj]

    # Accumulators for the loop
    all_frames_b64 = []
    combined_transcripts = []

    # Loop through selected videos ---
    for video_p in videos_to_process:
        print(f"Analyzing: {video_p.name}")

        # 1. Extract Visuals (Frames)
        try:
            frames = extract_key_frames(str(video_p), frame_limit=num_frames)
            if frames:
                all_frames_b64.extend(frames)
        except Exception as e:
            print(f"    Error extracting frames from {video_p.name}: {e}")
            continue

        # 2. Extract Audio (Transcript)
        if use_audio:
            try:
                transcript_text = transcribe_video(str(video_p))

                if transcript_text and len(transcript_text) > 20:
                    # Append with identifier to distinguish sources
                    combined_transcripts.append(f"TRANSCRIPT ({video_p.name}):\n{transcript_text}")

            except Exception as e:
                print(f"    Audio transcription failed for {video_p.name}: {e}")

    if not all_frames_b64:
        raise ValueError("No frames extracted from any processed videos.")

    print(f" -> Prepared {len(all_frames_b64)} frames and {len(combined_transcripts)} transcripts. Sending to LLM...")

    # Join transcripts into one context string
    final_context = "\n\n".join(combined_transcripts) if combined_transcripts else None

    # 3. Call Provider
    result_list = provider.image_features(
        all_frames_b64,
        prompt=prompt,
        as_set=True,
        extra_context=final_context
    )

    # 4. Save Results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        # Name based on folder or file
        base_name = path_obj.name if path_obj.is_dir() else path_obj.stem
        output_filename = f"features_{base_name}.json"

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Features saved to {output_path}")

    if isinstance(result_list, list) and len(result_list) == 1:
        return result_list[0]

    return result_list


def discover_features_from_texts(
        texts_or_file: str | List[str],  # input is text(s)
        prompt: str = text_discovery_prompt,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # same semantics as image version
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Feature discovery / description for texts.

    - as_set=True  → joint feature discovery (ALL texts together)
    - as_set=False → per-text description (texts sent separately)
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