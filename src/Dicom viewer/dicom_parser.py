"""
dicom_parser.py

Handles loading and safe metadata extraction from DICOM files.
Uses pydicom for parsing. Designed to be defensive against missing
or malformed tags, and against compressed pixel data that requires
optional codec packages.
"""

from dataclasses import dataclass
from typing import Union, Tuple
import pydicom
from pydicom.dataset import FileDataset
from pydicom.multival import MultiValue
import numpy as np


@dataclass
class DicomMetadata:
    """Structured, type-safe container for the metadata we care about."""
    rows: int
    columns: int
    rescale_intercept: float
    rescale_slope: float
    window_center: float
    window_width: float
    pixel_spacing: Tuple[float, float]  # (row_spacing, col_spacing) in mm
    bits_stored: int
    pixel_representation: int  # 0 = unsigned, 1 = signed
    photometric_interpretation: str
    instance_number: int
    image_position_z: float  # z-coordinate for slice ordering in a series


def _first_value(value: Union[float, int, MultiValue, list]) -> float:
    """
    Many DICOM tags (notably WindowCenter/WindowWidth) can legally be
    stored as a single scalar OR a multi-value. We standardize on the
    first value for a default render.
    """
    if isinstance(value, (MultiValue, list, tuple)):
        return float(value[0])
    return float(value)


def _safe_get_float(ds: FileDataset, tag_name: str, default: float) -> float:
    """Fetch a numeric tag, falling back to a safe default if absent."""
    if not hasattr(ds, tag_name):
        return default
    raw = getattr(ds, tag_name)
    if raw is None:
        return default
    try:
        return _first_value(raw)
    except (TypeError, ValueError, IndexError):
        return default


def load_dicom_file(filepath: str) -> FileDataset:
    """
    Load a DICOM file from disk.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        pydicom.errors.InvalidDicomError: if the file isn't valid DICOM.
        ValueError: if the file has no pixel data.
    """
    ds = pydicom.dcmread(filepath, force=False)

    if "PixelData" not in ds:
        raise ValueError(
            f"'{filepath}' is a valid DICOM file but contains no PixelData "
            f"(SOP Class: {getattr(ds, 'SOPClassUID', 'unknown')}). "
            f"This may be a non-image object (SR, presentation state, etc.)."
        )
    return ds


def extract_metadata(ds: FileDataset) -> DicomMetadata:
    """
    Extract and normalize the metadata required for HU conversion,
    windowing, inversion, and slice ordering, with safe fallbacks
    for every field.
    """
    rows = int(ds.Rows)
    columns = int(ds.Columns)

    rescale_intercept = _safe_get_float(ds, "RescaleIntercept", default=0.0)
    rescale_slope = _safe_get_float(ds, "RescaleSlope", default=1.0)

    window_center = _safe_get_float(ds, "WindowCenter", default=40.0)
    window_width = _safe_get_float(ds, "WindowWidth", default=400.0)

    pixel_spacing_raw = getattr(ds, "PixelSpacing", None) or getattr(
        ds, "ImagerPixelSpacing", None
    )
    if pixel_spacing_raw is not None and len(pixel_spacing_raw) == 2:
        pixel_spacing = (float(pixel_spacing_raw[0]), float(pixel_spacing_raw[1]))
    else:
        pixel_spacing = (1.0, 1.0)

    bits_stored = int(getattr(ds, "BitsStored", 16))
    pixel_representation = int(getattr(ds, "PixelRepresentation", 0))
    photometric_interpretation = str(
        getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    )

    instance_number = int(getattr(ds, "InstanceNumber", 0))

    # Used to sort slices into correct physical order within a series.
    # ImagePositionPatient = [x, y, z] in patient coordinate system (mm).
    position = getattr(ds, "ImagePositionPatient", None)
    image_position_z = float(position[2]) if position and len(position) == 3 else float(instance_number)

    return DicomMetadata(
        rows=rows,
        columns=columns,
        rescale_intercept=rescale_intercept,
        rescale_slope=rescale_slope,
        window_center=window_center,
        window_width=window_width,
        pixel_spacing=pixel_spacing,
        bits_stored=bits_stored,
        pixel_representation=pixel_representation,
        photometric_interpretation=photometric_interpretation,
        instance_number=instance_number,
        image_position_z=image_position_z,
    )


def needs_inversion(metadata: DicomMetadata) -> bool:
    """
    Returns True if this image's PhotometricInterpretation is MONOCHROME1,
    meaning low pixel values should display as WHITE (inverted from the
    normal MONOCHROME2 convention where low = black).

    Without correcting for this, MONOCHROME1 images render as a photo
    negative of the correct image — anatomy is still there, but bone
    looks dark and air looks bright, which is diagnostically confusing.
    """
    return metadata.photometric_interpretation == "MONOCHROME1"


def extract_pixel_array(ds: FileDataset) -> np.ndarray:
    """
    Extract the raw pixel array, with explicit, actionable error handling
    for compressed transfer syntaxes that are missing their codec.

    pydicom needs an optional plugin installed to decode compressed pixel
    data (JPEG Baseline/Lossless, JPEG2000, JPEG-LS, RLE). Without it,
    ds.pixel_array raises a fairly cryptic error — we catch that and
    surface a clear, fixable message instead.
    """
    try:
        arr = ds.pixel_array
    except Exception as e:
        transfer_syntax = getattr(
            getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown"
        )
        raise RuntimeError(
            f"Failed to decode pixel data for Transfer Syntax "
            f"'{transfer_syntax}'. This file is very likely compressed "
            f"and missing a required codec package.\n"
            f"Fix: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm\n"
            f"Original error: {e}"
        ) from e

    # Widen dtype up-front so rescaling (which can go negative, e.g.
    # air = -1000 HU) never overflows/wraps around.
    if arr.dtype != np.int32 and arr.dtype != np.float64:
        arr = arr.astype(np.int32)

    return arr