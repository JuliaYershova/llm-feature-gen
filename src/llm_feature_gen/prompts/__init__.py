from pathlib import Path

def load_prompt(name: str) -> str:
    """
    Loads a text prompt by filename (without extension) from this package.
    Example: load_prompt("image_discovery_prompt")
    """
    path = Path(__file__).parent / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' not found in {path.parent}")
    return path.read_text(encoding="utf-8")

multiclass_text_discovery_prompt = load_prompt("multiclass_text_discovery_prompt")
multiclass_image_discovery_prompt = load_prompt("multiclass_image_discovery_prompt")
multiclass_video_discovery_prompt = load_prompt("multiclass_video_discovery_prompt")
multiclass_tabular_discovery_prompt = load_prompt("multiclass_tabular_discovery_prompt")

# Old name for the text prompt, kept so existing imports keep working.
multiclass_discovery_prompt = multiclass_text_discovery_prompt

