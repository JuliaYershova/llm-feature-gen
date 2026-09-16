"""Multi-class feature discovery and generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .discover import (
    discover_features_from_images,
    discover_features_from_tabular,
    discover_features_from_texts,
    discover_features_from_videos,
)
from .generate import generate_features
from .prompts import DiscoveryPromptBuilder
from .providers.openai_provider import OpenAIProvider


def discover_features_multiclass(
    texts_or_file: Union[str, Path, List[str]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_text_features.json",
    as_set: bool = True,
    min_features: Optional[int] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover text features that distinguish all supplied classes.

    ``prompt`` optionally replaces the discovery task template and may use
    ``{n_classes}``, ``{class_list}``, and ``{min_features}`` placeholders.
    ``system_prompt`` controls the model's role and behavior.
    """
    resolved_prompt = DiscoveryPromptBuilder(
        classes=classes,
        min_features=min_features,
        modality="text",
        template=prompt,
    ).build()

    return discover_features_from_texts(
        texts_or_file=texts_or_file,
        prompt=resolved_prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename,
        system_prompt=system_prompt,
    )


def generate_features_multiclass(
    root_folder: Union[str, Path],
    discovered_features: Union[str, Path, Dict[str, Any]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    merge_to_single_csv: bool = True,
    merged_csv_name: str = "all_feature_values.csv",
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, str]:
    """Generate feature-value CSVs for two or more classes.

    ``prompt`` replaces the bundled generation task while ``system_prompt``
    controls model behavior. The discovered feature specification is appended
    to the task automatically.
    """
    classes = list(classes)
    if len(classes) < 2:
        raise ValueError(f"At least 2 classes required for multi-class generation, got {len(classes)}.")

    root_folder = Path(root_folder)
    for class_name in classes:
        class_dir = root_folder / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class sub-folder not found: {class_dir}. Expected one folder per class inside {root_folder}."
            )

    if isinstance(discovered_features, dict):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        discovered_features_path = output_path / "_tmp_discovered_features.json"
        discovered_features_path.write_text(
            json.dumps(discovered_features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        discovered_features_path = Path(discovered_features)

    return generate_features(
        root_folder=root_folder,
        discovered_features_path=discovered_features_path,
        output_dir=output_dir,
        classes=classes,
        provider=provider,
        merge_to_single_csv=merge_to_single_csv,
        merged_csv_name=merged_csv_name,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def run_multiclass_pipeline(
    discover_folder: Union[str, Path],
    train_folder: Union[str, Path],
    test_folder: Union[str, Path],
    classes: Sequence[str],
    provider: OpenAIProvider,
    output_dir: Union[str, Path] = "outputs",
    min_features: Optional[int] = None,
    discovery_prompt: Optional[str] = None,
    discovery_system_prompt: Optional[str] = None,
    generation_prompt: Optional[str] = None,
    generation_system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Run discovery and generation with independently customizable prompts.

    ``*_prompt`` values control each phase's task. ``*_system_prompt`` values
    control the model's role and behavior for that phase.
    """
    output_dir = Path(output_dir)
    features_path = output_dir / "discovered_text_features.json"

    print(f"Discovering features for {len(classes)} classes: {list(classes)}")
    discovered = discover_features_multiclass(
        texts_or_file=discover_folder,
        classes=classes,
        provider=provider,
        output_dir=output_dir,
        output_filename="discovered_text_features.json",
        min_features=min_features,
        prompt=discovery_prompt,
        system_prompt=discovery_system_prompt,
    )

    print("Generating train features...")
    train_csv_paths = generate_features_multiclass(
        root_folder=train_folder,
        discovered_features=features_path,
        classes=classes,
        provider=provider,
        output_dir=output_dir / "train_generated",
        merged_csv_name="train_feature_values.csv",
        prompt=generation_prompt,
        system_prompt=generation_system_prompt,
    )

    print("Generating test features...")
    test_csv_paths = generate_features_multiclass(
        root_folder=test_folder,
        discovered_features=features_path,
        classes=classes,
        provider=provider,
        output_dir=output_dir / "test_generated",
        merged_csv_name="test_feature_values.csv",
        prompt=generation_prompt,
        system_prompt=generation_system_prompt,
    )

    return {
        "discovered_features": discovered,
        "train_csv_paths": train_csv_paths,
        "test_csv_paths": test_csv_paths,
    }

def discover_features_multiclass_images(
    image_paths_or_folder: Union[str, Path, List[str]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_image_features.json",
    as_set: bool = True,
    min_features: Optional[int] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover image features with optional task and system prompts.

    The task ``prompt`` may use the discovery placeholders; ``system_prompt``
    controls model behavior.
    """
    resolved_prompt = DiscoveryPromptBuilder(
        classes=classes,
        min_features=min_features,
        modality="image",
        template=prompt,
    ).build()

    return discover_features_from_images(
        image_paths_or_folder=image_paths_or_folder,
        prompt=resolved_prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename,
        system_prompt=system_prompt,
    )


def discover_features_multiclass_videos(
    videos_or_folder: Union[str, Path, List[str]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_video_features.json",
    as_set: bool = True,
    min_features: Optional[int] = None,
    num_frames: int = 5,
    use_audio: bool = True,
    max_videos_to_sample: int = 5,
    max_total_frames_payload: int = 15,
    random_seed: Optional[int] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover video features with optional task and system prompts.

    The task ``prompt`` may use the discovery placeholders; ``system_prompt``
    controls model behavior.
    """
    resolved_prompt = DiscoveryPromptBuilder(
        classes=classes,
        min_features=min_features,
        modality="video",
        template=prompt,
    ).build()

    return discover_features_from_videos(
        videos_or_folder=videos_or_folder,
        prompt=resolved_prompt,
        provider=provider,
        as_set=as_set,
        num_frames=num_frames,
        output_dir=output_dir,
        output_filename=output_filename,
        use_audio=use_audio,
        max_videos_to_sample=max_videos_to_sample,
        max_total_frames_payload=max_total_frames_payload,
        random_seed=random_seed,
        system_prompt=system_prompt,
    )

def discover_features_multiclass_tabular(
    file_or_folder: Union[str, Path],
    text_column: str,
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_tabular_features.json",
    as_set: bool = True,
    min_features: Optional[int] = None,
    max_rows: Optional[int] = None,
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Discover tabular features with optional task and system prompts.

    The task ``prompt`` may use the discovery placeholders; ``system_prompt``
    controls model behavior.
    """
    resolved_prompt = DiscoveryPromptBuilder(
        classes=classes,
        min_features=min_features,
        modality="tabular",
        template=prompt,
    ).build()

    return discover_features_from_tabular(
        file_or_folder=file_or_folder,
        text_column=text_column,
        prompt=resolved_prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename,
        max_rows=max_rows,
        system_prompt=system_prompt,
    )
