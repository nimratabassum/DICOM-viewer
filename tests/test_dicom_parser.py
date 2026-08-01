"""
test_dicom_parser.py

Covers dicom_parser.py: metadata extraction, the defensive fallbacks
for missing/malformed tags, MONOCHROME1 inversion detection, and pixel
array extraction.

Each test targets one specific behavior described in the module's own
docstrings, so a failing test points directly at which guarantee broke.
"""

import numpy as np
import pytest

from dicom_viewer.dicom_parser import (
    extract_metadata,
    extract_pixel_array,
    load_dicom_file,
    needs_inversion,
)


class TestExtractMetadataHappyPath:
    """extract_metadata() on a fully-populated dataset should read every
    tag straight through with no fallback logic involved."""

    def test_reads_all_fields_correctly(self, synthetic_dataset):
        meta = extract_metadata(synthetic_dataset)

        assert meta.rows == 8
        assert meta.columns == 8
        assert meta.rescale_slope == 1.0
        assert meta.rescale_intercept == -1000.0
        assert meta.window_center == 40.0
        assert meta.window_width == 400.0
        assert meta.pixel_spacing == (0.7, 0.7)
        assert meta.bits_stored == 16
        assert meta.pixel_representation == 1
        assert meta.photometric_interpretation == "MONOCHROME2"
        assert meta.instance_number == 1


class TestExtractMetadataFallbacks:
    """
    dicom_parser.py's whole design premise is that real-world files are
    missing tags, so every fallback path gets its own explicit test
    instead of only testing the happy path.
    """

    def test_missing_rescale_tags_default_to_identity(self, synthetic_dataset_factory):
        ds = synthetic_dataset_factory()
        del ds.RescaleSlope
        del ds.RescaleIntercept

        meta = extract_metadata(ds)

        assert meta.rescale_slope == 1.0
        assert meta.rescale_intercept == 0.0

    def test_missing_window_tags_default_to_soft_tissue_window(
        self, synthetic_dataset_factory
    ):
        ds = synthetic_dataset_factory(include_window_tags=False)

        meta = extract_metadata(ds)

        assert meta.window_center == 40.0
        assert meta.window_width == 400.0

    def test_window_center_as_multivalue_takes_first_value(
        self, synthetic_dataset_factory
    ):
        ds = synthetic_dataset_factory()
        ds.WindowCenter = [50.0, 60.0]
        ds.WindowWidth = [350.0, 300.0]

        meta = extract_metadata(ds)

        assert meta.window_center == 50.0
        assert meta.window_width == 350.0

    def test_missing_pixel_spacing_defaults_to_1mm(self, synthetic_dataset_factory):
        ds = synthetic_dataset_factory(include_pixel_spacing=False)

        meta = extract_metadata(ds)

        assert meta.pixel_spacing == (1.0, 1.0)

    def test_missing_bits_stored_defaults_to_16(self, synthetic_dataset_factory):
        ds = synthetic_dataset_factory()
        del ds.BitsStored

        meta = extract_metadata(ds)

        assert meta.bits_stored == 16

    def test_missing_image_position_falls_back_to_instance_number(
        self, synthetic_dataset_factory
    ):
        ds = synthetic_dataset_factory(instance_number=7)
        # no image_position passed in -> ImagePositionPatient absent

        meta = extract_metadata(ds)

        assert meta.image_position_z == 7.0

    def test_empty_multivalue_rescale_slope_falls_back_to_default(
        self, synthetic_dataset_factory
    ):
        """
        A tag that's present but holds an empty multi-value (legal at
        the pydicom level, but has nothing at index [0]) must not crash
        extraction — _first_value's IndexError should be caught and the
        default used, same as a genuinely missing tag.
        """
        from pydicom.multival import MultiValue

        ds = synthetic_dataset_factory()
        ds.RescaleSlope = MultiValue(float, [])

        meta = extract_metadata(ds)

        assert meta.rescale_slope == 1.0


class TestNeedsInversion:
    def test_monochrome1_needs_inversion(self, synthetic_dataset_factory):
        ds = synthetic_dataset_factory(photometric_interpretation="MONOCHROME1")
        meta = extract_metadata(ds)

        assert needs_inversion(meta) is True

    def test_monochrome2_does_not_need_inversion(self, synthetic_dataset):
        meta = extract_metadata(synthetic_dataset)

        assert needs_inversion(meta) is False


class TestLoadDicomFile:
    def test_loads_valid_file(self, synthetic_dcm_file):
        ds = load_dicom_file(synthetic_dcm_file)

        assert ds.Rows == 8
        assert ds.Columns == 8

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dicom_file("this/path/does/not/exist.dcm")

    def test_file_without_pixel_data_raises_value_error(
        self, tmp_path, synthetic_dataset_factory
    ):
        ds = synthetic_dataset_factory(include_pixel_data=False)
        filepath = tmp_path / "no_pixels.dcm"
        ds.save_as(str(filepath), enforce_file_format=True)

        with pytest.raises(ValueError, match="no PixelData"):
            load_dicom_file(str(filepath))


class TestExtractPixelArray:
    def test_widens_dtype_to_int32_to_avoid_overflow(self, synthetic_dataset_factory):
        pixel_array = np.full((8, 8), fill_value=100, dtype=np.int16)
        ds = synthetic_dataset_factory(pixel_array=pixel_array)

        arr = extract_pixel_array(ds)

        assert arr.dtype == np.int32
        assert arr.shape == (8, 8)
        assert (arr == 100).all()

    def test_decode_failure_raises_actionable_runtime_error(
        self, synthetic_dataset_factory, monkeypatch
    ):
        """
        Simulates a missing-codec failure (e.g. compressed transfer
        syntax without pylibjpeg installed) and checks the error message
        actually tells the user how to fix it, not just that it failed.
        """
        ds = synthetic_dataset_factory()

        def _boom(self):
            raise RuntimeError("codec not available")

        monkeypatch.setattr(
            type(ds), "pixel_array", property(_boom), raising=False
        )

        with pytest.raises(RuntimeError, match="pylibjpeg"):
            extract_pixel_array(ds)
