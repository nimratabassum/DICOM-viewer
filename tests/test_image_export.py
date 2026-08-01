"""
test_image_export.py

Covers image_export.py: converting an 8-bit windowed array into a
base64 PNG/JPEG, and the dtype guard that catches the "forgot to
window first" mistake before it produces a broken image.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from dicom_viewer.image_export import array_to_base64_image, array_to_data_uri


class TestArrayToBase64Image:
    def test_png_roundtrip_preserves_pixel_values(self):
        arr = np.array([[0, 128], [200, 255]], dtype=np.uint8)

        encoded = array_to_base64_image(arr, image_format="PNG")
        decoded_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded_bytes))
        result = np.array(image)

        np.testing.assert_array_equal(result, arr)

    def test_jpeg_format_produces_valid_image(self):
        arr = np.full((16, 16), 100, dtype=np.uint8)

        encoded = array_to_base64_image(arr, image_format="JPEG", jpeg_quality=90)
        decoded_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded_bytes))

        assert image.format == "JPEG"
        assert image.size == (16, 16)

    def test_wrong_dtype_raises_actionable_error(self):
        """The classic mistake this guard exists for: passing raw HU
        floats straight in without windowing first."""
        arr = np.array([[-1000.0, 50.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="apply_window_level"):
            array_to_base64_image(arr)

    def test_int_dtype_also_rejected(self):
        arr = np.array([[1, 2]], dtype=np.int32)

        with pytest.raises(ValueError):
            array_to_base64_image(arr)


class TestArrayToDataUri:
    def test_png_data_uri_has_correct_mime_prefix(self):
        arr = np.zeros((4, 4), dtype=np.uint8)

        uri = array_to_data_uri(arr, image_format="PNG")

        assert uri.startswith("data:image/png;base64,")

    def test_jpeg_data_uri_has_correct_mime_prefix(self):
        arr = np.zeros((4, 4), dtype=np.uint8)

        uri = array_to_data_uri(arr, image_format="JPEG")

        assert uri.startswith("data:image/jpeg;base64,")
