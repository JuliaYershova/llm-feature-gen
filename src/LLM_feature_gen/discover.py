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
    as_set: bool = True,                     # <- default TRUE for discovery
    output_dir: str | Path = "outputs",
    output_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper: takes a list of image file paths OR a folder path,
    converts images to base64, calls the provider, and saves the JSON result.
    """
    # 1) init provider
    provider = provider or OpenAIProvider(
        api_key=AZURE_OPENAI_API_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

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
        )
    else:
        # per-image behavior
        result_list = provider.image_features(
            b64_list,
            prompt=prompt,
            as_set=False,
        )

    # - joint mode: result_list is like: [ { "proposed_features": [...] } ]
    # - per-image mode: result_list is list of dicts

    # 5) save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        output_filename = "discovered_features.json"

    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)

    print(f"Features saved to {output_path}")

    # return the FIRST (and only) element in joint mode to keep downstream simple
    if as_set and isinstance(result_list, list) and len(result_list) == 1:
        return result_list[0]

    return result_list