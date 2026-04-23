"""Multi-class aware feature generation for llm-feature-gen.

The original package assumed a **fixed two-class workflow** in its discovery
prompt and partially in its generation logic.  This module replaces those
assumptions with a fully generic implementation that works cleanly for 2, 3, 4,
or any number of classes.

Key changes vs. the original
-----------------------------
1. ``discover_features_multiclass`` – new discovery entry point that injects
   the actual class names into the prompt so the LLM knows it is looking for
   features that separate *N* categories, not just two.
2. ``generate_features_multiclass`` – orchestrates generation for an arbitrary
   class list, forwarding to the existing ``assign_feature_values_from_folder``
   machinery which already works for any number of classes.
3. ``MultiClassDiscoveryPromptBuilder`` – builds a prompt dynamically from the
   class names, replacing the hardcoded two-class description.

Usage example::

    from llm_feature_gen.multiclass import discover_features_multiclass, generate_features_multiclass
    from llm_feature_gen.providers.local_provider import LocalProvider

    provider = LocalProvider(
        base_url="https://litellm.vse.cz/",
        api_key="sk-...",
        default_text_model="qwen3.6-35b",
    )

    # 4-class CLINC150 task
    classes = ["travel_alert", "travel_suggestion", "flight_status", "book_flight"]

    discovered = discover_features_multiclass(
        texts_or_file="data/discover",
        classes=classes,
        provider=provider,
        output_dir="outputs",
    )

    csv_paths = generate_features_multiclass(
        root_folder="data/train",
        discovered_features=discovered,
        classes=classes,
        provider=provider,
        output_dir="outputs/train",
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from llm_feature_gen.discover import discover_features_from_texts
from llm_feature_gen.generate import generate_features, load_discovered_features
from llm_feature_gen.providers.openai_provider import OpenAIProvider


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class MultiClassDiscoveryPromptBuilder:
    """Build a discovery prompt that is aware of the actual class count.

    Parameters
    ----------
    classes : list of str
        The class names present in the dataset.  The prompt will explicitly
        name the classes so the LLM can tailor its feature proposals.

    Examples
    --------
    >>> builder = MultiClassDiscoveryPromptBuilder(
    ...     classes=["travel_alert", "travel_suggestion", "flight_status"]
    ... )
    >>> prompt = builder.build()
    """

    _BASE_PROMPT = (
        "You are an expert in natural language processing and explainable artificial intelligence.\n"
        "Your task is to analyze textual examples and discover interpretable, human-understandable\n"
        "features that can differentiate between {n_classes} text categories:\n"
        "{class_list}\n"
        "\n"
        "You must reason purely from the textual content itself, without assuming any predefined\n"
        "feature types, and describe the distinguishing properties as if explaining them to a\n"
        "human researcher.\n"
        "\n"
        "I am providing you with example texts. Each text belongs to one of the {n_classes} categories\n"
        "listed above, but you are NOT told which text belongs to which category.\n"
        "\n"
        "Your task is to:\n"
        "  1. Examine all texts carefully and propose key textual features that appear to\n"
        "     systematically vary across subsets of the texts — features that together would be\n"
        "     sufficient to distinguish all {n_classes} categories in a classification model.\n"
        "  2. Focus on characteristics meaningful to a human reader: differences in intent,\n"
        "     narrative structure, linguistic choices, communicative function, level of abstraction,\n"
        "     specificity, urgency, or discourse patterns.\n"
        "  3. Express each feature as a short snake_case attribute name.\n"
        "  4. For each feature, provide a brief explanation of why it could be discriminative\n"
        "     across the {n_classes} hidden groups.\n"
        "  5. For each feature, provide 3–6 possible values (short descriptive strings, 2–5 words).\n"
        "  6. Provide at least {min_features} distinct features (more is better).\n"
        "  7. Respond ONLY with valid JSON — no preamble, no markdown, no explanation outside JSON.\n"
        "\n"
        "Output format (strict JSON):\n"
        "{{\n"
        '  "proposed_features": [\n'
        "    {{\n"
        '      "feature": "feature_name_1",\n'
        '      "description": "why it separates the {n_classes} groups",\n'
        '      "possible_values": ["value1", "value2", "value3"]\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
    )

    def __init__(self, classes: Sequence[str]) -> None:
        if len(classes) < 2:
            raise ValueError("At least 2 classes are required.")
        self.classes = list(classes)

    def build(self) -> str:
        """Return the formatted discovery prompt."""
        n = len(self.classes)
        class_list = "\n".join(f"  - {c}" for c in self.classes)
        # More classes → request more features so each pair is distinguishable
        min_features = max(10, n * 3)
        return self._BASE_PROMPT.format(
            n_classes=n,
            class_list=class_list,
            min_features=min_features,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_features_multiclass(
    texts_or_file: Union[str, Path, List[str]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    output_filename: str = "discovered_text_features.json",
    as_set: bool = True,
) -> Dict[str, Any]:
    """Discover features for an arbitrary number of classes.

    Unlike the default ``discover_features_from_texts``, this function injects
    the class names into the prompt so the LLM knows it must find features
    that separate *all N categories*, not just two.

    Parameters
    ----------
    texts_or_file : str, Path, or list of str
        Raw texts, a single file path, or a folder of ``.txt`` files.
    classes : sequence of str
        Names of all classes in the dataset (2 or more).
    provider : OpenAIProvider, optional
        LLM provider instance.
    output_dir : str or Path
        Directory where the JSON artifact is written.
    output_filename : str
        Filename for the JSON artifact.
    as_set : bool, default True
        When True, all texts are sent together in one request.

    Returns
    -------
    dict
        The discovered feature schema (``proposed_features`` key).
    """
    builder = MultiClassDiscoveryPromptBuilder(classes=classes)
    prompt = builder.build()

    return discover_features_from_texts(
        texts_or_file=texts_or_file,
        prompt=prompt,
        provider=provider,
        as_set=as_set,
        output_dir=output_dir,
        output_filename=output_filename,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_features_multiclass(
    root_folder: Union[str, Path],
    discovered_features: Union[str, Path, Dict[str, Any]],
    classes: Sequence[str],
    provider: Optional[OpenAIProvider] = None,
    output_dir: Union[str, Path] = "outputs",
    merge_to_single_csv: bool = True,
    merged_csv_name: str = "all_feature_values.csv",
) -> Dict[str, str]:
    """Generate feature values for *N* classes (N ≥ 2).

    This is a thin wrapper around the existing ``generate_features`` that
    makes the multi-class intent explicit and validates the class list.

    Parameters
    ----------
    root_folder : str or Path
        Dataset root with one sub-folder per class.
    discovered_features : str, Path, or dict
        Path to the JSON artifact produced by ``discover_features_multiclass``
        or a dict already loaded with ``load_discovered_features``.
    classes : sequence of str
        All class names to process (must each have a matching sub-folder).
    provider : OpenAIProvider, optional
        LLM provider instance.
    output_dir : str or Path
        Directory for CSV outputs.
    merge_to_single_csv : bool, default True
        Whether to write a single merged CSV.
    merged_csv_name : str
        Filename for the merged CSV.

    Returns
    -------
    dict mapping class name → CSV path.  When merged, also contains
    ``"__merged__"`` key.

    Raises
    ------
    ValueError
        If fewer than 2 classes are provided.
    FileNotFoundError
        If a required class sub-folder is missing.
    """
    classes = list(classes)
    if len(classes) < 2:
        raise ValueError(
            f"At least 2 classes required for multi-class generation, got {len(classes)}."
        )

    root_folder = Path(root_folder)
    for cls in classes:
        cls_dir = root_folder / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Class sub-folder not found: {cls_dir}. "
                f"Expected one folder per class inside {root_folder}."
            )

    # Persist features dict to a temporary JSON if a dict was passed directly
    if isinstance(discovered_features, dict):
        tmp_json = Path(output_dir) / "_tmp_discovered_features.json"
        tmp_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_json.write_text(
            json.dumps(discovered_features, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        discovered_features_path = tmp_json
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


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full 4-class pipeline in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_multiclass_pipeline(
    discover_folder: Union[str, Path],
    train_folder: Union[str, Path],
    test_folder: Union[str, Path],
    classes: Sequence[str],
    provider: OpenAIProvider,
    output_dir: Union[str, Path] = "outputs",
) -> Dict[str, Any]:
    """Run the complete discover → generate pipeline for N classes.

    Parameters
    ----------
    discover_folder : str or Path
        Flat folder of discovery sample texts.
    train_folder : str or Path
        Training data root (one sub-folder per class).
    test_folder : str or Path
        Test data root (one sub-folder per class).
    classes : sequence of str
        All class names.
    provider : OpenAIProvider
        LLM provider instance.
    output_dir : str or Path
        Root directory for all outputs.

    Returns
    -------
    dict with keys:
        ``"discovered_features"``  – the feature schema dict
        ``"train_csv_paths"``      – per-class + merged train CSV paths
        ``"test_csv_paths"``       – per-class + merged test CSV paths
    """
    output_dir = Path(output_dir)
    features_path = output_dir / "discovered_text_features.json"

    print(f"Discovering features for {len(classes)} classes: {classes}")
    discovered = discover_features_multiclass(
        texts_or_file=discover_folder,
        classes=classes,
        provider=provider,
        output_dir=output_dir,
        output_filename="discovered_text_features.json",
    )

    print("Generating train features …")
    train_csv_paths = generate_features_multiclass(
        root_folder=train_folder,
        discovered_features=features_path,
        classes=classes,
        provider=provider,
        output_dir=output_dir / "train_generated",
        merged_csv_name="train_feature_values.csv",
    )

    print("Generating test features …")
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
