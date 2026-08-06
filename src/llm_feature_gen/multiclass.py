"""Multi-class text feature discovery and generation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .discover import discover_features_from_texts
from .generate import generate_features
from .prompts import multiclass_discovery_prompt
from .providers.openai_provider import OpenAIProvider


class MultiClassDiscoveryPromptBuilder:
    """Build a text discovery prompt tailored to an arbitrary class list.

    The builder fills the multiclass discovery prompt template with the class
    names and the requested feature count, so the provider is asked for
    features that separate exactly those classes.

    Args:
        classes: Class names the discovered features should distinguish.
            At least two are required.
        min_features: Minimum number of features to request. Defaults to
            ``max(10, 3 * len(classes))``.

    Raises:
        ValueError: If fewer than two classes are given, or ``min_features``
            is not a positive integer.

    Example:
        ```python
        prompt = MultiClassDiscoveryPromptBuilder(
            classes=["sadness", "fear", "joy"],
            min_features=12,
        ).build()
        ```
    """

    def __init__(self, classes: Sequence[str], min_features: Optional[int] = None) -> None:
        classes = list(classes)
        if len(classes) < 2:
            raise ValueError("At least 2 classes are required.")
        if min_features is not None and min_features < 1:
            raise ValueError("min_features must be a positive integer.")

        self.classes = classes
        self.min_features = min_features

    def build(self) -> str:
        n_classes = len(self.classes)
        class_list = "\n".join(f"  - {class_name}" for class_name in self.classes)
        min_features = self.min_features if self.min_features is not None else max(10, n_classes * 3)

        return multiclass_discovery_prompt.format(
            n_classes=n_classes,
            class_list=class_list,
            min_features=min_features,
        )


def discover_features_multiclass(
    texts_or_file: Union[str, Path, List[str]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_text_features.json",
    as_set: bool = True,
    min_features: Optional[int] = None,
) -> Dict[str, Any]:
    """Discover text features that distinguish all supplied classes.

    Builds a class-aware prompt with
    [MultiClassDiscoveryPromptBuilder][llm_feature_gen.multiclass.MultiClassDiscoveryPromptBuilder]
    and delegates to
    [discover_features_from_texts][llm_feature_gen.discover.discover_features_from_texts].

    Args:
        texts_or_file: Raw text, a list of texts, a document path, or a folder
            of supported text documents to discover from.
        classes: Class names the discovered features should distinguish.
        provider: Provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        output_dir: Directory where the schema JSON is written.
        output_filename: Filename of the schema JSON artifact.
        as_set: When ``True``, all texts are combined into a single request so
            the provider discovers one shared schema.
        min_features: Minimum number of features to request. Defaults to
            ``max(10, 3 * len(classes))``.

    Returns:
        The discovery payload, normally a dictionary with a
        ``proposed_features`` list.

    Raises:
        ValueError: If fewer than two classes are given.

    Example:
        ```python
        discovered = discover_features_multiclass(
            texts_or_file="discover_texts",
            classes=["sadness", "fear", "joy"],
        )
        ```
    """
    prompt = MultiClassDiscoveryPromptBuilder(
        classes=classes,
        min_features=min_features,
    ).build()

    return discover_features_from_texts(
        texts_or_file=texts_or_file,
        prompt=prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename,
    )


def generate_features_multiclass(
    root_folder: Union[str, Path],
    discovered_features: Union[str, Path, Dict[str, Any]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    merge_to_single_csv: bool = True,
    merged_csv_name: str = "all_feature_values.csv",
) -> Dict[str, str]:
    """Generate feature-value CSVs for two or more classes.

    Args:
        root_folder: Dataset root containing one subfolder per class.
        discovered_features: Schema to score against — either a path to a
            discovery JSON artifact or an already-loaded schema dictionary
            (which is written to a temporary JSON file first).
        classes: Class-folder names to process. Each must exist under
            ``root_folder``.
        provider: Provider instance. Defaults to
            [OpenAIProvider][llm_feature_gen.providers.OpenAIProvider].
        output_dir: Directory where CSV outputs are written.
        merge_to_single_csv: Also write one concatenated CSV across classes.
        merged_csv_name: Filename of the merged CSV artifact.

    Returns:
        Mapping from class name to generated CSV path; the merged CSV is
        stored under the ``"__merged__"`` key when enabled.

    Raises:
        ValueError: If fewer than two classes are given.
        FileNotFoundError: If a class subfolder is missing under
            ``root_folder``.

    Example:
        ```python
        csv_paths = generate_features_multiclass(
            root_folder="texts",
            discovered_features="outputs/discovered_text_features.json",
            classes=["sadness", "fear", "joy"],
        )
        ```
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
    )


def run_multiclass_pipeline(
    discover_folder: Union[str, Path],
    train_folder: Union[str, Path],
    test_folder: Union[str, Path],
    classes: Sequence[str],
    provider: OpenAIProvider,
    output_dir: Union[str, Path] = "outputs",
    min_features: Optional[int] = None,
) -> Dict[str, Any]:
    """Run discovery, train generation, and test generation for multiple classes.

    The three stages share one discovered schema: features are discovered from
    ``discover_folder``, then feature values are generated for the train and
    test splits so both use identical columns.

    Args:
        discover_folder: Folder of texts used only for feature discovery.
        train_folder: Class-organized folder scored for the training split.
        test_folder: Class-organized folder scored for the test split.
        classes: Class-folder names present in both splits.
        provider: Provider instance used for all three stages.
        output_dir: Base output directory. The schema JSON is written here;
            train CSVs go to ``<output_dir>/train_generated/`` and test CSVs
            to ``<output_dir>/test_generated/``.
        min_features: Minimum number of features to request during discovery.

    Returns:
        A dictionary with three keys — ``"discovered_features"`` (the schema
        payload), ``"train_csv_paths"`` and ``"test_csv_paths"`` (the mappings
        returned by
        [generate_features_multiclass][llm_feature_gen.multiclass.generate_features_multiclass],
        each including a ``"__merged__"`` entry).

    Raises:
        ValueError: If fewer than two classes are given.
        FileNotFoundError: If a class subfolder is missing in either split.

    Example:
        ```python
        results = run_multiclass_pipeline(
            discover_folder="discover_texts",
            train_folder="texts/train",
            test_folder="texts/test",
            classes=["sadness", "fear", "joy"],
            provider=OpenAIProvider(),
        )
        train_csv = results["train_csv_paths"]["__merged__"]
        ```
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
    )

    print("Generating train features...")
    train_csv_paths = generate_features_multiclass(
        root_folder=train_folder,
        discovered_features=features_path,
        classes=classes,
        provider=provider,
        output_dir=output_dir / "train_generated",
        merged_csv_name="train_feature_values.csv",
    )

    print("Generating test features...")
    test_csv_paths = generate_features_multiclass(
        root_folder=test_folder,
        discovered_features=features_path,
        classes=classes,
        provider=provider,
        output_dir=output_dir / "test_generated",
        merged_csv_name="test_feature_values.csv",
    )

    return {
        "discovered_features": discovered,
        "train_csv_paths": train_csv_paths,
        "test_csv_paths": test_csv_paths,
    }
