"""
test_dicom_series.py

Covers dicom_series.py: loading a folder of slices into one ordered
volume, physical-position sorting, shape-consistency checking, and the
grouped loader (load_dicom_series_safe) added to handle folders that
mix more than one series together.
"""

import numpy as np
import pytest

from dicom_viewer.dicom_series import DicomSeries, load_dicom_series, load_dicom_series_safe


class TestLoadDicomSeries:
    def test_loads_all_slices(self, synthetic_series_folder):
        series = load_dicom_series(synthetic_series_folder)

        assert series.num_slices == 5
        assert series.volume_hu.shape == (5, 8, 8)

    def test_slices_are_sorted_by_physical_z_position_not_save_order(
        self, synthetic_series_folder
    ):
        """
        The fixture saves slices in a deliberately scrambled order
        (z=40, 10, 20, 0, 30) and encodes each slice's z-value into its
        own pixel values. If sorting is correct, slice 0 in the volume
        should be the z=0 slice (all pixels == 0), and slice 4 should
        be the z=40 slice (all pixels == 40).
        """
        series = load_dicom_series(synthetic_series_folder)

        z_values_in_order = [m.image_position_z for m in series.slice_metadata]
        assert z_values_in_order == sorted(z_values_in_order)

        assert series.get_slice_hu(0).mean() == pytest.approx(0.0)
        assert series.get_slice_hu(4).mean() == pytest.approx(40.0)

    def test_missing_folder_raises(self):
        with pytest.raises(NotADirectoryError):
            load_dicom_series("/no/such/folder")

    def test_empty_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="No .dcm files"):
            load_dicom_series(str(empty))

    def test_mixed_series_folder_raises_on_shape_mismatch(self, mixed_series_folder):
        """
        This is exactly the scenario the professor's feedback flagged:
        a folder that looks like one series but actually contains two.
        load_dicom_series should fail loudly and clearly rather than
        silently building a corrupt volume.
        """
        with pytest.raises(ValueError, match="Inconsistent slice dimensions"):
            load_dicom_series(mixed_series_folder)


class TestDicomSeriesGetSliceHu:
    def test_valid_index_returns_2d_array(self, synthetic_series_folder):
        series = load_dicom_series(synthetic_series_folder)

        sl = series.get_slice_hu(2)

        assert sl.shape == (8, 8)

    @pytest.mark.parametrize("bad_index", [-1, 5, 100])
    def test_out_of_range_index_raises_index_error(self, synthetic_series_folder, bad_index):
        series = load_dicom_series(synthetic_series_folder)

        with pytest.raises(IndexError):
            series.get_slice_hu(bad_index)


class TestLoadDicomSeriesSafe:
    def test_single_series_folder_returns_one_series(self, synthetic_series_folder):
        result = load_dicom_series_safe(synthetic_series_folder)

        assert len(result) == 1
        assert result[0].num_slices == 5

    def test_mixed_folder_splits_into_two_series_instead_of_failing(
        self, mixed_series_folder
    ):
        """
        The whole point of load_dicom_series_safe: the same folder that
        makes load_dicom_series() raise should load cleanly here, split
        into its two real series.
        """
        result = load_dicom_series_safe(mixed_series_folder)

        assert len(result) == 2
        slice_counts = sorted(s.num_slices for s in result)
        assert slice_counts == [2, 5]

    def test_largest_series_returned_first(self, mixed_series_folder):
        result = load_dicom_series_safe(mixed_series_folder)

        assert result[0].num_slices == 5  # main acquisition
        assert result[1].num_slices == 2  # scout series
