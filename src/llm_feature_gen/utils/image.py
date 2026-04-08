# src/llm_feature_gen/utils/image.py
import base64, io
from PIL import Image
import numpy as np


def image_to_base64(img_arr: np.ndarray, max_size: int = 384, quality: int = 85) -> str:
    img = Image.fromarray(img_arr)
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
