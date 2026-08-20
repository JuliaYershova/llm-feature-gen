"""Discovery prompt templates and the builder that fills them in."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

_PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt by filename without extension, e.g. load_prompt("text_generation_prompt")."""
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found in {path.parent}")
    return path.read_text(encoding="utf-8")


text_generation_prompt = load_prompt("text_generation_prompt")
image_generation_prompt = load_prompt("image_generation_prompt")
video_generation_prompt = load_prompt("video_generation_prompt")
tabular_generation_prompt = load_prompt("tabular_generation_prompt")

# One discovery template per modality. Loaded eagerly so a missing
# file fails at import, not halfway through a run.
discovery_templates: Dict[str, str] = {
    "text": load_prompt("text_discovery_prompt"),
    "image": load_prompt("image_discovery_prompt"),
    "video": load_prompt("video_discovery_prompt"),
    "tabular": load_prompt("tabular_discovery_prompt"),
}


class DiscoveryPromptBuilder:
    """Fill a discovery template with the class list and feature count.

    Pass either ``classes`` (names go into the prompt) or ``n_classes``
    (classes stay anonymous). ``min_features`` defaults to max(10, 3 * n).
    A custom ``template`` can be passed instead of a bundled one; it must
    contain the {n_classes}, {class_list} and {min_features} placeholders.
    """

    def __init__(
        self,
        classes: Optional[Sequence[str]] = None,
        min_features: Optional[int] = None,
        modality: str = "text",
        n_classes: Optional[int] = None,
        template: Optional[str] = None,
    ) -> None:
        if classes is not None:
            classes = list(classes)
            if len(classes) < 2:
                raise ValueError("At least 2 classes are required.")
            if n_classes is not None and n_classes != len(classes):
                raise ValueError(
                    f"n_classes={n_classes} does not match len(classes)={len(classes)}."
                )
            n_classes = len(classes)
        else:
            n_classes = 2 if n_classes is None else n_classes
            if n_classes < 2:
                raise ValueError("At least 2 classes are required.")

        if min_features is not None and min_features < 1:
            raise ValueError("min_features must be a positive integer.")

        if template is None:
            if modality not in discovery_templates:
                raise ValueError(
                    f"Unknown modality {modality!r}. Available: {sorted(discovery_templates)}"
                )
            template = discovery_templates[modality]

        self.classes = classes
        self.n_classes = n_classes
        self.min_features = min_features
        self.template = template

    def build(self) -> str:
        # no names given -> keep the classes anonymous in the prompt
        names = self.classes or [f"category_{i}" for i in range(1, self.n_classes + 1)]
        class_list = "\n".join(f"  - {name}" for name in names)
        min_features = self.min_features or max(10, self.n_classes * 3)

        return self.template.format(
            n_classes=self.n_classes,
            class_list=class_list,
            min_features=min_features,
        )