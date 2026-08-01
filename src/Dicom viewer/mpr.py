"""
mpr.py

Multi-planar reconstruction (MPR) on a loaded DicomSeries volume.

A raw HU volume only lets you scroll through axial slices, the plane
the scan was actually acquired in. Real viewers also let you look at
coronal (front) and sagittal (side) views, cut through the same
volume along the other two axes.

That only looks correct if the voxels are cubes first. CT slices are
almost never isotropic, e.g. 0.7mm between pixels within a slice but
2.5mm between slices, so a naive coronal/sagittal slice through the
raw array comes out squashed. This module resamples the volume to
isotropic spacing before cutting along the other planes.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import zoom

from .dicom_series import DicomSeries


@dataclass
class VoxelSpacing:
    """Physical spacing in mm between voxel centers, one value per axis."""
    slice_spacing: float   # distance between consecutive axial slices
    row_spacing: float     # PixelSpacing[0]
    col_spacing: float     # PixelSpacing[1]


def _slice_spacing(series: DicomSeries) -> float:
    """
    Distance between consecutive slices along the z-axis, derived from
    image_position_z of consecutive slices rather than trusting a
    SliceThickness tag, which can legally differ from the actual
    spacing between slice centers (e.g. with overlapping slices).
    """
    if series.num_slices < 2:
        return 1.0
    z_positions = [m.image_position_z for m in series.slice_metadata]
    diffs = np.diff(z_positions)
    spacing = float(np.mean(np.abs(diffs)))
    return spacing if spacing > 0 else 1.0


def get_voxel_spacing(series: DicomSeries) -> VoxelSpacing:
    """Read the physical voxel spacing for a loaded series."""
    row_spacing, col_spacing = series.slice_metadata[0].pixel_spacing
    return VoxelSpacing(
        slice_spacing=_slice_spacing(series),
        row_spacing=row_spacing,
        col_spacing=col_spacing,
    )


def resample_to_isotropic(
    series: DicomSeries, target_spacing_mm: float = 1.0
) -> Tuple[np.ndarray, VoxelSpacing]:
    """
    Resample a DicomSeries volume so all three axes have equal physical
    spacing, target_spacing_mm apart.

    This is what makes coronal/sagittal cuts through the volume look
    like a real body instead of a stretched or squashed one. Uses
    linear interpolation, order=1: fast and fine for viewing, though a
    diagnostic-grade pipeline would want to make the interpolation
    order configurable per use case.

    Returns:
        (resampled_volume, new_spacing) where new_spacing has all three
        fields equal to target_spacing_mm (up to floating point noise
        from the zoom factors landing on non-integer output sizes).
    """
    spacing = get_voxel_spacing(series)
    zoom_factors = (
        spacing.slice_spacing / target_spacing_mm,
        spacing.row_spacing / target_spacing_mm,
        spacing.col_spacing / target_spacing_mm,
    )

    resampled = zoom(series.volume_hu, zoom_factors, order=1)

    new_spacing = VoxelSpacing(
        slice_spacing=target_spacing_mm,
        row_spacing=target_spacing_mm,
        col_spacing=target_spacing_mm,
    )
    return resampled, new_spacing


def get_axial_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """Slice along the acquisition plane (the original scan direction)."""
    return volume[index, :, :]


def get_coronal_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """
    Slice front-to-back through the body at a fixed row index.

    Only meaningful on an isotropically resampled volume — call
    resample_to_isotropic() first, otherwise the output is stretched
    along whichever axis had coarser spacing.
    """
    return volume[:, index, :]


def get_sagittal_slice(volume: np.ndarray, index: int) -> np.ndarray:
    """
    Slice side-to-side through the body at a fixed column index.

    Same isotropic-resampling requirement as get_coronal_slice().
    """
    return volume[:, :, index]
