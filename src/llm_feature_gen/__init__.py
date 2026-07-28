"""Stable public API for the ``llm_feature_gen`` package."""

from importlib.metadata import PackageNotFoundError, version

from .discover import (
    discover_features_from_images,
    discover_features_from_tabular,
    discover_features_from_texts,
    discover_features_from_videos,
)
from .generate import (
    assign_feature_values_from_folder,
    generate_features,
    generate_features_from_images,
    generate_features_from_tabular,
    generate_features_from_texts,
    generate_features_from_videos,
    load_discovered_features,
    parse_json_from_markdown,
)
from .prompts import DiscoveryPromptBuilder
from .batch import BatchTextCache, generate_features_batch, generate_features_from_texts_cached
from .multiclass import (
    discover_features_multiclass_tabular,
    discover_features_multiclass,
    discover_features_multiclass_images,
    discover_features_multiclass_videos,
    generate_features_multiclass,
    run_multiclass_pipeline,
)
from .providers import LocalProvider, OpenAIProvider

try:
    __version__ = version("llm-feature-gen")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "LocalProvider",
    "DiscoveryPromptBuilder",
    "OpenAIProvider",
    "__version__",
    "assign_feature_values_from_folder",
    "BatchTextCache",
    "discover_features_from_images",
    "discover_features_multiclass",
    "discover_features_multiclass_images",
    "discover_features_multiclass_videos",
    "discover_features_multiclass_tabular",
    "discover_features_from_tabular",
    "discover_features_from_texts",
    "discover_features_from_videos",
    "generate_features",
    "generate_features_batch",
    "generate_features_from_images",
    "generate_features_from_tabular",
    "generate_features_from_texts_cached",
    "generate_features_from_texts",
    "generate_features_from_videos",
    "generate_features_multiclass",
    "load_discovered_features",
    "parse_json_from_markdown",
    "run_multiclass_pipeline",
]
