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
import warnings
from itertools import zip_longest
from datetime import datetime

from .utils.image import image_to_base64
from .utils.text import extract_text_from_file
from dotenv import load_dotenv
from .utils.video import extract_key_frames, extract_audio_track, downsample_batch
from .generate import _provider_call_kwargs
from .providers.openai_provider import FEATURE_DISCOVERY_SCHEMA, OpenAIProvider
from .prompts import DiscoveryPromptBuilder, map_discovery_template
from .discovery_map_reduce import discover_map_reduce

# Load environment variables automatically
load_dotenv()

DiscoveryPayload = Dict[str, Any]
DiscoveryResult = Union[DiscoveryPayload, List[DiscoveryPayload]]
SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".html"}


def _load_tabular_file(file_path: Path):
    """Read a supported table for discovery without losing source row positions."""
    import pandas as pd
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep=";")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported format: {suffix}")


def _collect_exhaustive_inputs(modality, source, provider, text_column, num_frames, use_audio):
    import pandas as pd

    suffixes = {
        "text": SUPPORTED_TEXT_SUFFIXES,
        "tabular": {".csv", ".xlsx", ".xls", ".parquet", ".json"},
        "image": {".jpg", ".jpeg", ".png"},
        "video": {".mp4", ".mov", ".avi", ".mkv"},
    }[modality]
    items, exclusions, groups = [], [], {}

    def add(item_id, text, group, images=None):
        if images is None and not text.strip():
            exclusions.append({"item_id": item_id, "reason": "empty"})
            return
        item = {"item_id": item_id, "text": text}
        if images is not None:
            item["images"] = images
        groups.setdefault(group, []).append(item)

    if modality == "text" and isinstance(source, list):
        for index, text in enumerate(source):
            add(f"text:{index}", text, "texts")
    elif modality == "text" and isinstance(source, str) and not Path(source).exists() and not _looks_like_text_path(source):
        add("text:0", source, "texts")
    else:
        if isinstance(source, (str, Path)):
            root = Path(source)
            if not root.exists():
                raise FileNotFoundError(f"Path not found: {root}")
            paths = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
            named_paths = [(path, path.relative_to(root).as_posix() if root.is_dir() else path.name) for path in paths]
        else:
            named_paths = [(Path(path), f"{index}:{Path(path).name}") for index, path in enumerate(source)]
        for path, name in named_paths:
            if path.suffix.lower() not in suffixes:
                exclusions.append({"item_id": name, "reason": "unsupported_format"})
                continue
            try:
                if modality == "text":
                    chunks = extract_text_from_file(path)
                    if not chunks:
                        exclusions.append({"item_id": name, "reason": "empty"})
                    for index, text in enumerate(chunks):
                        add(f"{name}:chunk:{index}", text, name)
                elif modality == "tabular":
                    table = _load_tabular_file(path)
                    if text_column not in table.columns:
                        raise ValueError(f"Column {text_column!r} not found")
                    for index, value in enumerate(table[text_column]):
                        item_id = f"{name}:row:{index}"
                        if pd.isna(value):
                            exclusions.append({"item_id": item_id, "reason": "null"})
                        else:
                            add(item_id, str(value), name)
                elif modality == "image":
                    with Image.open(path) as image:
                        frames = [image_to_base64(np.array(image.convert("RGB")))]
                    add(name, "", name, frames)
                else:
                    frames = extract_key_frames(str(path), frame_limit=num_frames)
                    if not frames:
                        raise ValueError("No frames extracted")
                    transcript, audio_path = "", None
                    try:
                        if use_audio:
                            audio_path = extract_audio_track(str(path))
                            if audio_path and Path(audio_path).exists():
                                transcript = provider.transcribe_audio(audio_path)
                    finally:
                        if audio_path and Path(audio_path).exists():
                            os.remove(audio_path)
                    add(name, transcript, name, frames)
            except Exception as exc:
                raise ValueError(f"Cannot prepare discovery input {name!r}: {exc}") from exc
    # Interleave documents/tables so consecutive batches compare multiple sources.
    for row in zip_longest(*groups.values()):
        items.extend(item for item in row if item is not None)
    return items, exclusions


def _discover_exhaustive(modality, source, provider, prompt, system_prompt, output_dir,
                        output_filename, num_classes, min_features, strategy, as_set,
                        batch_size, reduce_batch_size, max_request_chars, checkpoint_dir,
                        reduce_prompt, text_column=None, max_rows=None, num_frames=5, use_audio=True):
    if strategy != "map_reduce":
        raise ValueError("strategy must be 'single' or 'map_reduce'")
    if not as_set:
        raise ValueError("map_reduce discovery requires as_set=True")
    for name, value in (("batch_size", batch_size), ("reduce_batch_size", reduce_batch_size),
                        ("max_request_chars", max_request_chars), ("num_frames", num_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if max_rows is not None:
        raise ValueError("map_reduce discovery requires max_rows=None")
    builder = DiscoveryPromptBuilder(
        modality=modality, n_classes=num_classes, min_features=min_features,
        template=map_discovery_template.replace("{modality}", modality),
    )
    target = builder.min_features or max(10, builder.n_classes * 3)
    if prompt is None:
        task = builder.build()
    elif any(placeholder in prompt for placeholder in ("{n_classes}", "{class_list}", "{min_features}")):
        task = _resolve_discovery_prompt(modality, prompt, num_classes, min_features)
    else:
        task = prompt
    provider = provider or OpenAIProvider()
    items, exclusions = _collect_exhaustive_inputs(modality, source, provider, text_column, num_frames, use_audio)
    output_path = Path(output_dir) / (output_filename or f"discovered_{modality}_features.json")
    return discover_map_reduce(
        items, provider=provider, prompt=task, system_prompt=system_prompt,
        output_path=output_path, batch_size=1 if modality == "video" else batch_size,
        reduce_batch_size=reduce_batch_size, max_request_chars=max_request_chars,
        checkpoint_dir=checkpoint_dir, reduce_prompt=reduce_prompt, min_features=target,
        modality=modality, exclusions=exclusions,
        extraction_settings={"num_frames": num_frames, "use_audio": use_audio, "text_column": text_column},
    )


def _resolve_discovery_prompt(
    modality: str,
    prompt: Optional[str],
    num_classes: Optional[int],
    min_features: Optional[int],
) -> str:
    """Build the bundled or caller-supplied discovery prompt template."""
    placeholders = ("{n_classes}", "{class_list}", "{min_features}")
    if prompt is not None and not any(name in prompt for name in placeholders):
        unused = [
            name
            for name, value in (("num_classes", num_classes), ("min_features", min_features))
            if value is not None
        ]
        if unused:
            raise ValueError(
                f"{' and '.join(unused)} cannot be applied: the supplied prompt has "
                f"none of the placeholders {', '.join(placeholders)}."
            )
        return prompt

    return DiscoveryPromptBuilder(
        modality=modality,
        n_classes=num_classes,
        min_features=min_features,
        template=prompt,
    ).build()


def _looks_like_text_path(value: str) -> bool:
    """Heuristically distinguish raw text from a missing filesystem path."""
    candidate = Path(value)
    return (
        "\n" not in value
        and (
            value.startswith(("~", ".", "/"))
            or "\\" in value
            or "/" in value
            or candidate.suffix.lower() in SUPPORTED_TEXT_SUFFIXES
        )
    )


def _nonempty_text_chunks(chunks: List[str]) -> List[str]:
    """Return text chunks that contain non-whitespace content."""
    return [chunk for chunk in chunks if chunk.strip()]


def discover_features_from_images(
        image_paths_or_folder: str | List[str],
        prompt: Optional[str] = None,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # <- default TRUE for discovery
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        num_classes: Optional[int] = None,
        min_features: Optional[int] = None,
        system_prompt: Optional[str] = None,
        *,
        strategy: str = "single",
        batch_size: int = 15,
        reduce_batch_size: int = 32,
        max_request_chars: int = 24000,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        reduce_prompt: Optional[str] = None,
) -> DiscoveryResult:
    """Discover features from image files and persist the provider response.

    Args:
        image_paths_or_folder: A single image path, a folder containing images,
            or a list of image file paths.
        prompt: Optional discovery task or template replacing the bundled
            image prompt. Templates may use ``{n_classes}``, ``{class_list}``,
            and ``{min_features}`` placeholders.
        provider: Optional provider instance. When omitted, an
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider] is
            created from environment variables.
        as_set: When ``True``, all images are analyzed together and a single
            shared feature schema is produced. When ``False``, each image is
            sent independently and the result contains one entry per image.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_image_features.json``.
        num_classes: Optional number of hidden classes reflected in the prompt.
        min_features: Minimum number of features to request from the provider.
        system_prompt: Optional high-level instruction controlling the model's
            role, behavior, and response style. It does not replace ``prompt``.

        strategy: ``"single"`` preserves the original discovery behavior.
            ``"map_reduce"`` processes every eligible input in bounded batches
            and consolidates evidenced candidates into one shared schema.
            Requires ``as_set=True``. Video mode visits every file, ignoring
            folder sampling and the global frame cap; ``num_frames`` still
            limits each video's representation. Tabular mode requires
            ``max_rows=None``.
        batch_size: Maximum items per map request. Video mode uses one video
            per request to preserve frame/transcript grouping.
        reduce_batch_size: Maximum candidates per reduce request.
        max_request_chars: Rendered text size guard including instructions
            and retry room; excludes image payloads and is not a token limit.
            Individually oversized items raise instead of being truncated.
        checkpoint_dir: Optional directory for validated, resumable discovery
            checkpoints. Custom providers need a ``cache_identity`` to reuse
            checkpoints. A separate discovery report records provenance.
        reduce_prompt: Optional consolidation task; the equivalence contract
            is appended automatically.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        per-image mode. The on-disk JSON always preserves the raw provider
        result list.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no supported image files are found.
        RuntimeError: If image decoding fails for every candidate input.
    """
    if strategy != "single":
        return _discover_exhaustive(
            "image", image_paths_or_folder, provider, prompt, system_prompt, output_dir,
            output_filename, num_classes, min_features, strategy, as_set,
            batch_size, reduce_batch_size, max_request_chars, checkpoint_dir,
            reduce_prompt,
        )

    # 1) init provider
    provider = provider or OpenAIProvider()
    prompt = _resolve_discovery_prompt("image", prompt, num_classes, min_features)

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
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )
    else:
        # per-image behavior
        result_list = provider.image_features(
            b64_list,
            prompt=prompt,
            as_set=False,
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )

    # validate before saving
    if len(result_list) == 1 and isinstance(result_list[0], dict) and "error" in result_list[0]:
        raise ValueError(
            f"Discovery failed. Provider returned an error: {result_list[0]['error']}\n"
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
        prompt: Optional[str] = None,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # stejné chování jako image/text
        num_frames: int = 5,
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        use_audio: bool = True,
        max_videos_to_sample: int = 5,
        max_total_frames_payload: int = 15,
        random_seed: Optional[int] = None,
        num_classes: Optional[int] = None,
        min_features: Optional[int] = None,
        system_prompt: Optional[str] = None,
        *,
        strategy: str = "single",
        batch_size: int = 15,
        reduce_batch_size: int = 32,
        max_request_chars: int = 24000,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        reduce_prompt: Optional[str] = None,
) -> DiscoveryResult:
    """Discover features from one or more videos.

    Each video is converted into representative frames and, optionally, an
    audio transcript. The resulting multimodal payload is sent to the provider
    and the raw response is written to JSON.

    Args:
        videos_or_folder: A single video path, a folder containing videos, or a
            list of video file paths.
        prompt: Optional discovery task or template replacing the bundled
            video prompt. Templates may use ``{n_classes}``, ``{class_list}``,
            and ``{min_features}`` placeholders.
        provider: Optional provider instance implementing ``image_features``
            and, when ``use_audio=True``, optionally ``transcribe_audio``.
        as_set: When ``True``, all extracted frames are analyzed together to
            produce one shared schema. When ``False``, all extracted frames are
            pooled together and analyzed individually, so the returned list has
            one entry per extracted frame rather than one entry per source
            video.
        num_frames: Target number of key frames to extract per video before
            downsampling across the batch.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_video_features.json``.
        use_audio: Whether to extract an audio track and include a transcript
            as extra context when the provider supports transcription.
        max_videos_to_sample: Upper bound on how many videos are sampled from a
            folder input to control cost and payload size. When a folder
            contains more than this many videos, a subset is sampled before
            frame extraction.
        max_total_frames_payload: Upper bound on the total number of frames sent
            to the provider across the batch.
        random_seed: Optional seed used when folder inputs need to sample a
            subset of videos. Pass a value here to make the sampled subset
            reproducible across runs.
        num_classes: Optional number of hidden classes reflected in the prompt.
        min_features: Minimum number of features to request from the provider.
        system_prompt: Optional high-level instruction controlling the model's
            role, behavior, and response style. It does not replace ``prompt``.

        strategy: ``"single"`` preserves the original discovery behavior.
            ``"map_reduce"`` processes every eligible input in bounded batches
            and consolidates evidenced candidates into one shared schema.
            Requires ``as_set=True``. Video mode visits every file, ignoring
            folder sampling and the global frame cap; ``num_frames`` still
            limits each video's representation. Tabular mode requires
            ``max_rows=None``.
        batch_size: Maximum items per map request. Video mode uses one video
            per request to preserve frame/transcript grouping.
        reduce_batch_size: Maximum candidates per reduce request.
        max_request_chars: Rendered text size guard including instructions
            and retry room; excludes image payloads and is not a token limit.
            Individually oversized items raise instead of being truncated.
        checkpoint_dir: Optional directory for validated, resumable discovery
            checkpoints. Custom providers need a ``cache_identity`` to reuse
            checkpoints. A separate discovery report records provenance.
        reduce_prompt: Optional consolidation task; the equivalence contract
            is appended automatically.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        pooled per-frame mode.

    Raises:
        FileNotFoundError: If the input path is missing or a folder contains no
            supported video files.
        ValueError: If no frames can be extracted from the provided videos.
    """
    if strategy != "single":
        return _discover_exhaustive(
            "video", videos_or_folder, provider, prompt, system_prompt, output_dir,
            output_filename, num_classes, min_features, strategy, as_set,
            batch_size, reduce_batch_size, max_request_chars, checkpoint_dir,
            reduce_prompt, num_frames=num_frames, use_audio=use_audio,
        )


    # -------------------------------------------------
    # 1) init provider
    # -------------------------------------------------
    provider = provider or OpenAIProvider()
    prompt = _resolve_discovery_prompt("video", prompt, num_classes, min_features)

    # -------------------------------------------------
    # 2) collect video paths
    # -------------------------------------------------
    if isinstance(videos_or_folder, (str, Path)):
        path_obj = Path(videos_or_folder)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path_obj}")

        if path_obj.is_dir():
            valid_exts = {".mp4", ".mov", ".avi", ".mkv"}

            video_paths = sorted(
                [
                    p for p in path_obj.iterdir()
                    if p.suffix.lower() in valid_exts
                ]
            )

            if not video_paths:
                raise FileNotFoundError(f"No videos found in folder: {path_obj}")

            if len(video_paths) > max_videos_to_sample:
                sampler = random.Random(random_seed) if random_seed is not None else random
                sampled_indices = sorted(
                    sampler.sample(range(len(video_paths)), max_videos_to_sample)
                )
                video_paths = [video_paths[index] for index in sampled_indices]

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
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )
    else:
        # per-frame discovery
        result_list = provider.image_features(
            all_frames_b64,
            prompt=prompt,
            as_set=False,
            extra_context=final_context,
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )

    # validate before saving
    if len(result_list) == 1 and isinstance(result_list[0], dict) and "error" in result_list[0]:
        raise ValueError(
            f"Discovery failed. Provider returned an error: {result_list[0]['error']}\n"
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
        prompt: Optional[str] = None,
        provider: Optional[OpenAIProvider] = None,
        as_set: bool = True,  # same semantics as image version
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        num_classes: Optional[int] = None,
        min_features: Optional[int] = None,
        system_prompt: Optional[str] = None,
        *,
        strategy: str = "single",
        batch_size: int = 15,
        reduce_batch_size: int = 32,
        max_request_chars: int = 24000,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        reduce_prompt: Optional[str] = None,
) -> DiscoveryResult:
    """Discover features from text strings, files, or folders of documents.

    Args:
        texts_or_file: Either a raw text string, a list of raw text strings, a
            single supported document path, or a directory containing supported
            text documents. String inputs are treated as paths only when they
            already exist on disk or look path-like, such as ``notes/file.txt``.
        prompt: Optional discovery task or template replacing the bundled text
            prompt. Templates may use ``{n_classes}``, ``{class_list}``, and
            ``{min_features}`` placeholders.
        provider: Optional provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        as_set: When ``True``, all extracted text is combined into a single
            request so the provider can discover a shared schema. When
            ``False``, each text chunk is processed independently.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_text_features.json``.
        num_classes: Optional number of hidden classes reflected in the prompt.
        min_features: Minimum number of distinct features to request from the
            provider.
        system_prompt: Optional high-level instruction controlling the model's
            role, behavior, and response style. It does not replace ``prompt``.

        strategy: ``"single"`` preserves the original discovery behavior.
            ``"map_reduce"`` processes every eligible input in bounded batches
            and consolidates evidenced candidates into one shared schema.
            Requires ``as_set=True``. Video mode visits every file, ignoring
            folder sampling and the global frame cap; ``num_frames`` still
            limits each video's representation. Tabular mode requires
            ``max_rows=None``.
        batch_size: Maximum items per map request. Video mode uses one video
            per request to preserve frame/transcript grouping.
        reduce_batch_size: Maximum candidates per reduce request.
        max_request_chars: Rendered text size guard including instructions
            and retry room; excludes image payloads and is not a token limit.
            Individually oversized items raise instead of being truncated.
        checkpoint_dir: Optional directory for validated, resumable discovery
            checkpoints. Custom providers need a ``cache_identity`` to reuse
            checkpoints. A separate discovery report records provenance.
        reduce_prompt: Optional consolidation task; the equivalence contract
            is appended automatically.

    Returns:
        A single discovery payload in joint mode or a list of payloads in
        per-text mode.

    Raises:
        FileNotFoundError: If a path-like input does not exist.
        ValueError: If the path is invalid or no supported text input can be
            extracted.
    """
    if strategy != "single":
        return _discover_exhaustive(
            "text", texts_or_file, provider, prompt, system_prompt, output_dir,
            output_filename, num_classes, min_features, strategy, as_set,
            batch_size, reduce_batch_size, max_request_chars, checkpoint_dir,
            reduce_prompt,
        )


    # 1) init provider
    provider = provider or OpenAIProvider()
    prompt = _resolve_discovery_prompt("text", prompt, num_classes, min_features)

    # -------------------------------------------------
    # 2) collect texts
    # -------------------------------------------------
    texts: List[str] = []

    def add_text_chunks(chunks: List[str], label: str) -> None:
        nonlocal texts
        nonempty_chunks = _nonempty_text_chunks(chunks)
        if nonempty_chunks:
            texts.extend(nonempty_chunks)
        else:
            warnings.warn(
                f"Skipping '{label}' because it is empty or contains only whitespace.",
                UserWarning,
                stacklevel=2,
            )

    if isinstance(texts_or_file, Path):
        path = Path(texts_or_file)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_file():
            # single file of ANY supported text type
            add_text_chunks(extract_text_from_file(path), path.name)

        elif path.is_dir():
            # folder with mixed document types
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    try:
                        add_text_chunks(extract_text_from_file(file), file.name)
                    except ValueError:
                        pass  # skip unsupported files silently

        else:
            raise ValueError("Invalid path provided.")

    elif isinstance(texts_or_file, str):
        path = Path(texts_or_file)

        if path.exists():
            if path.is_file():
                add_text_chunks(extract_text_from_file(path), path.name)
            elif path.is_dir():
                for file in sorted(path.rglob("*")):
                    if file.is_file():
                        try:
                            add_text_chunks(extract_text_from_file(file), file.name)
                        except ValueError:
                            pass
            else:
                raise ValueError("Invalid path provided.")
        elif _looks_like_text_path(texts_or_file):
            raise FileNotFoundError(f"Path not found: {path}")
        else:
            add_text_chunks([texts_or_file], "text input")

    else:
        for index, text in enumerate(texts_or_file):
            add_text_chunks([text], f"text input at index {index}")

    if not texts:
        raise ValueError("No non-empty text inputs found to process.")
    # -------------------------------------------------
    # 3) CALL PROVIDER
    # -------------------------------------------------
    if as_set:
        #  JOINT DISCOVERY MODE
        combined_text = "\n\n---\n\n".join(texts)

        result_list = provider.text_features(
            [combined_text],  # ONE request
            prompt=prompt,
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )
    else:
        # PER-TEXT DESCRIPTION MODE
        result_list = provider.text_features(
            texts,  # MANY requests
            prompt=prompt,
            **_provider_call_kwargs(provider, system_prompt, FEATURE_DISCOVERY_SCHEMA),
        )

    # validate before saving
    if len(result_list) == 1 and isinstance(result_list[0], dict) and "error" in result_list[0]:
        raise ValueError(
            f"Discovery failed. Provider returned an error: {result_list[0]['error']}\n"
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
        prompt: Optional[str] = None,
        as_set: bool = True,
        output_dir: str | Path = "outputs",
        output_filename: Optional[str] = None,
        max_rows: Optional[int] = None,
        num_classes: Optional[int] = None,
        min_features: Optional[int] = None,
        system_prompt: Optional[str] = None,
        *,
        strategy: str = "single",
        batch_size: int = 15,
        reduce_batch_size: int = 32,
        max_request_chars: int = 24000,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        reduce_prompt: Optional[str] = None,
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
        prompt: Optional discovery task or template replacing the bundled
            tabular prompt. Templates may use ``{n_classes}``, ``{class_list}``,
            and ``{min_features}`` placeholders.
        as_set: Whether to discover one shared schema across all sampled rows or
            process rows independently.
        output_dir: Directory where the JSON artifact should be written.
        output_filename: Custom filename for the saved artifact. Defaults to
            ``discovered_tabular_features.json``.
        max_rows: Optional cap on how many rows are used from the concatenated
            dataset.
        num_classes: Optional number of hidden classes reflected in the prompt.
        min_features: Minimum number of distinct features to request from the provider.
        system_prompt: Optional high-level instruction controlling the model's
            role, behavior, and response style. It does not replace ``prompt``.

        strategy: ``"single"`` preserves the original discovery behavior.
            ``"map_reduce"`` processes every eligible input in bounded batches
            and consolidates evidenced candidates into one shared schema.
            Requires ``as_set=True``. Video mode visits every file, ignoring
            folder sampling and the global frame cap; ``num_frames`` still
            limits each video's representation. Tabular mode requires
            ``max_rows=None``.
        batch_size: Maximum items per map request. Video mode uses one video
            per request to preserve frame/transcript grouping.
        reduce_batch_size: Maximum candidates per reduce request.
        max_request_chars: Rendered text size guard including instructions
            and retry room; excludes image payloads and is not a token limit.
            Individually oversized items raise instead of being truncated.
        checkpoint_dir: Optional directory for validated, resumable discovery
            checkpoints. Custom providers need a ``cache_identity`` to reuse
            checkpoints. A separate discovery report records provenance.
        reduce_prompt: Optional consolidation task; the equivalence contract
            is appended automatically.

    Returns:
        The same return shape as
        [discover_features_from_texts][llm_feature_gen.discover.discover_features_from_texts].

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If no supported tabular files are found or ``text_column``
            is missing.
    """
    if strategy != "single":
        return _discover_exhaustive(
            "tabular", file_or_folder, provider, prompt, system_prompt, output_dir,
            output_filename, num_classes, min_features, strategy, as_set,
            batch_size, reduce_batch_size, max_request_chars, checkpoint_dir,
            reduce_prompt, text_column=text_column, max_rows=max_rows,
        )

    import pandas as pd
    provider = provider or OpenAIProvider()
    prompt = _resolve_discovery_prompt("tabular", prompt, num_classes, min_features)
    path = Path(file_or_folder)

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    dfs = []

    if path.is_file():
        dfs.append(_load_tabular_file(path))
    elif path.is_dir():
        for f in sorted(path.iterdir()):
            if f.is_file():
                try:
                    dfs.append(_load_tabular_file(f))
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
        system_prompt=system_prompt,
    )
