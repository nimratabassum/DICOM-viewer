"""
processing_engine.py

Vectorized NumPy operations for:
  1. Raw pixel -> Hounsfield Unit (HU) conversion
  2. Windowing/Leveling -> 8-bit displayable range (with MONOCHROME1 inversion)
  3. ROI (Region of Interest) statistics
"""

import numpy as np


def apply_rescale(pixel_array: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """output = pixel_value * RescaleSlope + RescaleIntercept"""
    rescaled = pixel_array.astype(np.float32) * np.float32(slope) + np.float32(intercept)
    return rescaled


def apply_window_level(
    hu_array: np.ndarray,
    window_width: float,
    window_center: float,
    invert: bool = False,
    output_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Linear W/L transform -> 8-bit displayable range. Fully vectorized.

    Args:
        invert: Set True when PhotometricInterpretation == 'MONOCHROME1'.
                MONOCHROME1 DICOMs store data so that LOW pixel values
                should display as WHITE (opposite of the normal
                MONOCHROME2 convention). If this flag is ignored, those
                images render as a photo-negative of the correct image.
                Inversion is applied AFTER windowing so the HU values
                themselves stay physically correct — only the final
                grayscale mapping direction flips.
    """
    if window_width <= 0:
        raise ValueError(f"window_width must be > 0, got {window_width}")

    window_width = np.float32(window_width)
    window_center = np.float32(window_center)

    low = window_center - window_width / np.float32(2.0)
    high = window_center + window_width / np.float32(2.0)

    normalized = (hu_array - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)

    scaled = (normalized * np.float32(255.0)).astype(output_dtype)

    if invert:
        # 255 - value flips the grayscale mapping direction. Vectorized,
        # single op, negligible cost even on large arrays.
        scaled = (255 - scaled.astype(np.int16)).astype(output_dtype)

    return scaled


def compute_full_range_window(hu_array: np.ndarray) -> tuple[float, float]:
    """Auto-derive a window spanning the actual data range."""
    min_val = float(np.min(hu_array))
    max_val = float(np.max(hu_array))
    width = max(max_val - min_val, 1.0)
    center = (max_val + min_val) / 2.0
    return width, center


def compute_roi_statistics(
    hu_array: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> dict:
    """
    Compute HU statistics for a rectangular Region of Interest (ROI).

    This is the standard "measure this area" tool used in real radiology
    viewers — e.g. select a suspicious region and check its mean HU to
    help distinguish tissue types (fluid vs. soft tissue vs. calcification).

    Args:
        hu_array: 2D array of Hounsfield Unit values (post-rescale, NOT
                   post-windowing — statistics must be computed on real
                   HU values, not on the display-scaled 0-255 image).
        row_start, row_end, col_start, col_end: pixel-index bounding box
                   of the ROI (row_end/col_end exclusive, Python slice style).

    Returns:
        dict with mean, min, max, std, and pixel_count for the ROI.

    Raises:
        ValueError: if the bounding box is out of range or empty.
    """
    rows, cols = hu_array.shape

    if not (0 <= row_start < row_end <= rows):
        raise ValueError(
            f"Invalid row range [{row_start}:{row_end}] for array with {rows} rows."
        )
    if not (0 <= col_start < col_end <= cols):
        raise ValueError(
            f"Invalid col range [{col_start}:{col_end}] for array with {cols} cols."
        )

    roi = hu_array[row_start:row_end, col_start:col_end]

    return {
        "mean_hu": float(np.mean(roi)),
        "min_hu": float(np.min(roi)),
        "max_hu": float(np.max(roi)),
        "std_hu": float(np.std(roi)),
        "pixel_count": int(roi.size),
    }