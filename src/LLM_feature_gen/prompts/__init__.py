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


image_discovery_prompt = load_prompt("discovery_prompt")
text_discovery_prompt = load_prompt("generation_prompt")

