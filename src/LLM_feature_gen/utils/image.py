# src/LLM_feature_gen/utils/image.py
import base64, io
from PIL import Image
import numpy as np

def image_to_base64(img_arr: np.ndarray) -> str:
    img = Image.fromarray(img_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")