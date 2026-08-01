"""
conftest.py

Shared pytest fixtures for the test suite.

Every fixture here builds a SYNTHETIC DICOM dataset in memory rather
than depending on the sample scan in ct-lung-screening-nlst-series/.
That's a deliberate choice, not an oversight:

  - Tests stay fast (no reading real 512x512 pixel data off disk).
  - Tests are fully controlled: we can construct exact edge cases
    (missing tags, MONOCHROME1, zero window width, mismatched slice
    shapes) that may not exist anywhere in the sample data at all.
  - Tests don't silently break if the sample folder is ever moved,
    renamed, or replaced with a different scan.

The real sample series is still used for the manual smoke test in
`local series test.py`, which is a demo script, not part of this
automated suite.
"""

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _make_dataset(
    rows=8,
    cols=8,
    pixel_array=None,
    rescale_slope=1.0,
    rescale_intercept=-1000.0,
    window_center=40.0,
    window_width=400.0,
    photometric_interpretation="MONOCHROME2",
    instance_number=1,
    image_position=None,
    series_uid=None,
    include_pixel_data=True,
    include_window_tags=True,
    include_pixel_spacing=True,
) -> FileDataset:
    """Build a minimal-but-valid synthetic CT-slice DICOM dataset."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        None, {}, file_meta=file_meta, preamble=b"\x00" * 128
    )
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.StudyInstanceUID = generate_uid()

    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric_interpretation
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.InstanceNumber = instance_number

    if include_window_tags:
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width

    if include_pixel_spacing:
        ds.PixelSpacing = [0.7, 0.7]

    if image_position is not None:
        ds.ImagePositionPatient = list(image_position)

    if include_pixel_data:
        if pixel_array is None:
            pixel_array = np.zeros((rows, cols), dtype=np.int16)
        ds.PixelData = pixel_array.astype(np.int16).tobytes()

    return ds


@pytest.fixture
def synthetic_dataset():
    """A single valid synthetic CT slice, all tags present."""
    return _make_dataset()


@pytest.fixture
def synthetic_dataset_factory():
    """Factory version, for tests that need to tweak specific tags."""
    return _make_dataset


@pytest.fixture
def synthetic_dcm_file(tmp_path, synthetic_dataset):
    """A synthetic dataset written to a real .dcm file on disk."""
    filepath = tmp_path / "slice_0001.dcm"
    synthetic_dataset.save_as(str(filepath), enforce_file_format=True)
    return str(filepath)


@pytest.fixture
def synthetic_series_folder(tmp_path):
    """
    A folder of 5 synthetic slices forming one consistent series,
    with distinct pixel values and z-positions so ordering/statistics
    tests have something meaningful to check.
    """
    series_uid = generate_uid()
    folder = tmp_path / "series"
    folder.mkdir()

    # Deliberately saved out of physical order (z=40 before z=10) to
    # verify load_dicom_series sorts by position, not by filename/save order.
    z_positions = [40.0, 10.0, 20.0, 0.0, 30.0]
    for i, z in enumerate(z_positions):
        pixel_array = np.full((8, 8), fill_value=int(z), dtype=np.int16)
        ds = _make_dataset(
            pixel_array=pixel_array,
            instance_number=i + 1,
            image_position=(0.0, 0.0, z),
            series_uid=series_uid,
            rescale_slope=1.0,
            rescale_intercept=0.0,  # keep HU == raw pixel value for easy assertions
        )
        ds.save_as(str(folder / f"file_{i}.dcm"), enforce_file_format=True)

    return str(folder)


@pytest.fixture
def mixed_series_folder(tmp_path):
    """
    A folder containing TWO different series mixed together: a 5-slice
    main series and a 2-slice "scout" series with a different shape.
    Simulates the messy multi-series downloads the professor's feedback
    flagged as an untested edge case.
    """
    folder = tmp_path / "mixed"
    folder.mkdir()

    main_uid = generate_uid()
    for i, z in enumerate([0.0, 10.0, 20.0, 30.0, 40.0]):
        pixel_array = np.full((8, 8), fill_value=int(z), dtype=np.int16)
        ds = _make_dataset(
            pixel_array=pixel_array,
            instance_number=i + 1,
            image_position=(0.0, 0.0, z),
            series_uid=main_uid,
            rows=8,
            cols=8,
            rescale_slope=1.0,
            rescale_intercept=0.0,
        )
        ds.save_as(str(folder / f"main_{i}.dcm"), enforce_file_format=True)

    scout_uid = generate_uid()
    for i, z in enumerate([0.0, 5.0]):
        pixel_array = np.zeros((4, 4), dtype=np.int16)
        ds = _make_dataset(
            pixel_array=pixel_array,
            instance_number=i + 1,
            image_position=(0.0, 0.0, z),
            series_uid=scout_uid,
            rows=4,
            cols=4,
        )
        ds.save_as(str(folder / f"scout_{i}.dcm"), enforce_file_format=True)

    return str(folder)
