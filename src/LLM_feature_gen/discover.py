# src/LLM_feature_gen/discover.py
from typing import List, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime

from .utils.image import image_to_base64
from .providers.openai_provider import OpenAIProvider
from .prompts import image_discovery_prompt
from dotenv import load_dotenv

# Load environment variables automatically
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

def discover_features_from_images(
    image_paths_or_folder: str | List[str],
    prompt: str = image_discovery_prompt,
    provider: Optional[OpenAIProvider] = None,
    as_set: bool = True,
    output_dir: str | Path = "outputs",
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper: takes a list of image file paths OR a folder path,
    converts images to base64, calls the provider, and saves the JSON result.

    Parameters
    ----------
    image_paths_or_folder : str | list[str]
        Either a path to a folder containing images or a list of image file paths.
    prompt : str
        Prompt text to send to the LLM (defaults to image_discovery_prompt).
    provider : OpenAIProvider, optional
        If None, creates one using environment variables.
    as_set : bool, default=True
        If True, sends all images in one joint request (comparative reasoning).
        If False, sends them individually and returns a list of results.
    output_dir : str | Path, default="outputs"
        Folder where the resulting JSON will be saved.
    output_filename : str, optional
        Optional custom name for the JSON file (without extension).

    Returns
    -------
    dict
        JSON-like object returned by the LLM.
    """

    # Initialize provider
    provider = provider or OpenAIProvider(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Determine image list
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

    # Convert images to base64
    b64_list: List[str] = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            b64_list.append(image_to_base64(np.array(img)))
        except Exception as e:
            print(f"Could not load {path}: {e}")

    if not b64_list:
        raise RuntimeError("Failed to load any valid images from input.")

    # Call provider
    result = (
        provider.image_features(b64_list, prompt=prompt)
        if as_set
        else provider.image_features(b64_list, prompt=prompt)
    )

    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define filename (timestamped if none provided)
    if output_filename is None:
        output_filename = f"discovered_features.json"

    output_path = output_dir / output_filename

    # Save JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Features saved to {output_path}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

    return result