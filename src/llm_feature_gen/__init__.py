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
from .providers import LocalProvider, OpenAIProvider

try:
    __version__ = version("llm-feature-gen")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "LocalProvider",
    "OpenAIProvider",
    "__version__",
    "assign_feature_values_from_folder",
    "discover_features_from_images",
    "discover_features_from_tabular",
    "discover_features_from_texts",
    "discover_features_from_videos",
    "generate_features",
    "generate_features_from_images",
    "generate_features_from_tabular",
    "generate_features_from_texts",
    "generate_features_from_videos",
    "load_discovered_features",
    "parse_json_from_markdown",
]
