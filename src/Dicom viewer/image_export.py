"""
image_export.py

Converts an 8-bit windowed NumPy array into a base64-encoded PNG/JPEG.
"""

import base64
import io
from typing import Literal

import numpy as np
from PIL import Image


def array_to_base64_image(
    image_array: np.ndarray,
    image_format: Literal["PNG", "JPEG"] = "PNG",
    jpeg_quality: int = 90,
) -> str:
    if image_array.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 array for image export, got {image_array.dtype}. "
            f"Did you forget to run apply_window_level() first?"
        )

    pil_image = Image.fromarray(image_array, mode="L")

    buffer = io.BytesIO()
    if image_format == "JPEG":
        pil_image.save(buffer, format="JPEG", quality=jpeg_quality)
    else:
        pil_image.save(buffer, format="PNG", optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def array_to_data_uri(
    image_array: np.ndarray,
    image_format: Literal["PNG", "JPEG"] = "PNG",
) -> str:
    encoded = array_to_base64_image(image_array, image_format)
    mime = "image/png" if image_format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"