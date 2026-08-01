"""
test_mpr.py

Covers mpr.py: voxel spacing derivation, isotropic resampling, and
axial/coronal/sagittal slicing.
"""

import numpy as np
import pytest

from dicom_viewer.dicom_series import load_dicom_series
from dicom_viewer.mpr import (
    get_axial_slice,
    get_coronal_slice,
    get_sagittal_slice,
    get_voxel_spacing,
    resample_to_isotropic,
)


class TestGetVoxelSpacing:
    def test_reads_row_col_spacing_from_metadata(self, synthetic_series_folder):
        series = load_dicom_series(synthetic_series_folder)

        spacing = get_voxel_spacing(series)

        assert spacing.row_spacing == pytest.approx(0.7)
        assert spacing.col_spacing == pytest.approx(0.7)

    def test_slice_spacing_derived_from_consecutive_z_positions(
        self, synthetic_series_folder
    ):
        """
        Fixture slices sit 10mm apart (z = 0, 10, 20, 30, 40), so the
        average gap between consecutive sorted slices should be 10mm,
        not whatever SliceThickness would have said.
        """
        series = load_dicom_series(synthetic_series_folder)

        spacing = get_voxel_spacing(series)

        assert spacing.slice_spacing == pytest.approx(10.0)

    def test_single_slice_series_defaults_slice_spacing_to_one(
        self, synthetic_dataset_factory, tmp_path
    ):
        from dicom_viewer.dicom_series import load_dicom_series

        folder = tmp_path / "one_slice"
        folder.mkdir()
        ds = synthetic_dataset_factory(image_position=(0.0, 0.0, 0.0))
        ds.save_as(str(folder / "only.dcm"), enforce_file_format=True)

        series = load_dicom_series(str(folder))
        spacing = get_voxel_spacing(series)

        assert spacing.slice_spacing == 1.0


class TestResampleToIsotropic:
    def test_output_spacing_matches_target(self, synthetic_series_folder):
        series = load_dicom_series(synthetic_series_folder)

        _, new_spacing = resample_to_isotropic(series, target_spacing_mm=1.0)

        assert new_spacing.slice_spacing == 1.0
        assert new_spacing.row_spacing == 1.0
        assert new_spacing.col_spacing == 1.0

    def test_volume_grows_when_upsampling_to_finer_spacing(self, synthetic_series_folder):
        """
        Original spacing is 10mm between slices; resampling to 1mm
        should produce roughly 10x more slices along that axis.
        """
        series = load_dicom_series(synthetic_series_folder)

        resampled, _ = resample_to_isotropic(series, target_spacing_mm=1.0)

        assert resampled.shape[0] > series.volume_hu.shape[0]

    def test_resampled_volume_is_3d(self, synthetic_series_folder):
        series = load_dicom_series(synthetic_series_folder)

        resampled, _ = resample_to_isotropic(series, target_spacing_mm=2.0)

        assert resampled.ndim == 3


class TestPlaneSlicing:
    def test_axial_slice_matches_volume_indexing(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        result = get_axial_slice(volume, 1)

        np.testing.assert_array_equal(result, volume[1, :, :])
        assert result.shape == (3, 4)

    def test_coronal_slice_matches_volume_indexing(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        result = get_coronal_slice(volume, 1)

        np.testing.assert_array_equal(result, volume[:, 1, :])
        assert result.shape == (2, 4)

    def test_sagittal_slice_matches_volume_indexing(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        result = get_sagittal_slice(volume, 2)

        np.testing.assert_array_equal(result, volume[:, :, 2])
        assert result.shape == (2, 3)
