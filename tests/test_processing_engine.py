"""
test_processing_engine.py

Covers processing_engine.py: HU rescaling, window/level to 8-bit,
MONOCHROME1 inversion, auto-windowing, and ROI statistics.

These are pure NumPy functions with no I/O, so tests use small
hand-built arrays with known, hand-checkable outputs rather than
synthetic DICOM datasets.
"""

import numpy as np
import pytest

from dicom_viewer.processing_engine import (
    apply_rescale,
    apply_window_level,
    compute_full_range_window,
    compute_roi_statistics,
)


class TestApplyRescale:
    def test_identity_rescale(self):
        raw = np.array([[0, 100], [200, 300]], dtype=np.int32)

        result = apply_rescale(raw, slope=1.0, intercept=0.0)

        np.testing.assert_array_equal(result, raw.astype(np.float32))

    def test_typical_ct_rescale_to_hounsfield_units(self):
        """slope=1, intercept=-1024 is the standard CT rescale pair;
        raw 0 should land at -1024 HU (roughly air)."""
        raw = np.array([[0, 1024, 2024]], dtype=np.int32)

        result = apply_rescale(raw, slope=1.0, intercept=-1024.0)

        np.testing.assert_allclose(result, [[-1024.0, 0.0, 1000.0]])

    def test_output_dtype_is_float32(self):
        raw = np.zeros((4, 4), dtype=np.int32)

        result = apply_rescale(raw, slope=2.0, intercept=1.0)

        assert result.dtype == np.float32


class TestApplyWindowLevel:
    def test_midpoint_maps_to_middle_gray(self):
        hu = np.array([[40.0]])  # equals window_center

        result = apply_window_level(hu, window_width=400, window_center=40)

        # (40 - (-160)) / 400 * 255 = 127.5 -> 127 after uint8 truncation
        assert 126 <= result[0, 0] <= 128

    def test_below_window_clips_to_zero(self):
        hu = np.array([[-1000.0]])  # air, far below any soft-tissue window

        result = apply_window_level(hu, window_width=400, window_center=40)

        assert result[0, 0] == 0

    def test_above_window_clips_to_255(self):
        hu = np.array([[3000.0]])  # dense bone/metal, far above the window

        result = apply_window_level(hu, window_width=400, window_center=40)

        assert result[0, 0] == 255

    def test_zero_window_width_raises(self):
        hu = np.array([[0.0]])

        with pytest.raises(ValueError, match="window_width must be > 0"):
            apply_window_level(hu, window_width=0, window_center=40)

    def test_negative_window_width_raises(self):
        hu = np.array([[0.0]])

        with pytest.raises(ValueError, match="window_width must be > 0"):
            apply_window_level(hu, window_width=-50, window_center=40)

    def test_inversion_flips_output(self):
        hu = np.array([[3000.0, -1000.0]])  # -> [255, 0] before inversion

        normal = apply_window_level(hu, 400, 40, invert=False)
        inverted = apply_window_level(hu, 400, 40, invert=True)

        assert normal[0, 0] == 255 and normal[0, 1] == 0
        assert inverted[0, 0] == 0 and inverted[0, 1] == 255

    def test_output_dtype_matches_requested_dtype(self):
        hu = np.array([[0.0, 100.0]])

        result = apply_window_level(hu, 400, 40, output_dtype=np.uint8)

        assert result.dtype == np.uint8


class TestComputeFullRangeWindow:
    def test_spans_min_to_max(self):
        hu = np.array([[-500.0, 0.0], [250.0, 750.0]])

        width, center = compute_full_range_window(hu)

        assert width == pytest.approx(1250.0)
        assert center == pytest.approx(125.0)

    def test_constant_array_gets_minimum_width_of_one(self):
        """A flat array (max == min) would otherwise divide by zero
        during windowing, so width must floor at 1.0."""
        hu = np.full((4, 4), 40.0)

        width, center = compute_full_range_window(hu)

        assert width == 1.0
        assert center == 40.0


class TestComputeRoiStatistics:
    def test_stats_on_known_array(self):
        hu = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, 50.0, 60.0],
                [70.0, 80.0, 90.0],
            ]
        )

        stats = compute_roi_statistics(hu, row_start=0, row_end=2, col_start=0, col_end=2)

        # ROI = [[10, 20], [40, 50]]
        assert stats["mean_hu"] == pytest.approx(30.0)
        assert stats["min_hu"] == 10.0
        assert stats["max_hu"] == 50.0
        assert stats["pixel_count"] == 4

    def test_full_array_roi(self):
        hu = np.array([[1.0, 2.0], [3.0, 4.0]])

        stats = compute_roi_statistics(hu, 0, 2, 0, 2)

        assert stats["mean_hu"] == pytest.approx(2.5)
        assert stats["pixel_count"] == 4

    @pytest.mark.parametrize(
        "row_start,row_end,col_start,col_end",
        [
            (-1, 2, 0, 2),   # negative start
            (2, 1, 0, 2),    # end before start
            (0, 100, 0, 2),  # end beyond array bounds
            (0, 2, 0, 0),    # empty column range
        ],
    )
    def test_invalid_bounds_raise_value_error(
        self, row_start, row_end, col_start, col_end
    ):
        hu = np.zeros((3, 3))

        with pytest.raises(ValueError):
            compute_roi_statistics(hu, row_start, row_end, col_start, col_end)
