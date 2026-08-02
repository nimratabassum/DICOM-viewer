# DICOM Viewer

## Introduction

This project implements a backend pipeline for loading, processing, and rendering DICOM medical images. A DICOM file stores raw scanner integers plus a large set of hardware-specific metadata, not a directly viewable image, so the pipeline converts that raw data into Hounsfield Units, applies windowing to compress it into a displayable 8-bit range, and exports the result as PNG or JPEG.

In addition to the core image processing, this project emphasizes:

- Defensive parsing against missing or malformed DICOM metadata
- Correct handling of multi-series folders and non-isotropic voxel spacing
- Automated testing against synthetic edge cases
- A proper installable package layout

The sample data used throughout is the NLST lung-screening series (`sample_data/ct-lung-screening-nlst-series/`).

---

## Theoretical Background

### Hounsfield Unit Conversion

CT scanners record raw radiodensity as sensor integers, not physical units. Each file includes a `RescaleSlope` (m) and `RescaleIntercept` (b) tag that convert the raw array into Hounsfield Units (HU):

H(x, y) = m · P_raw(x, y) + b

This is implemented in `processing_engine.apply_rescale`. The array is upcast to 32-bit float before the multiply, since HU values go negative and the source data is unsigned, which otherwise causes integer overflow.

### Windowing and Leveling

A CT scan can span more than 4000 HU, while a standard display has only 256 grey levels. Windowing maps a chosen sub-range to the full 0-255 range. Given window center c and width w:

L = c − w/2, U = c + w/2

N(x, y) = 0 if H(x, y) ≤ L, (H(x, y) − L)/w if L < H(x, y) < U, 1 if H(x, y) ≥ U

D(x, y) = floor(N(x, y) · 255)

This is implemented in `processing_engine.apply_window_level`, fully vectorized over the array.

### Photometric Inversion

Files using `MONOCHROME1` invert the standard density-to-brightness relationship. The correction is applied after windowing, so the underlying HU values stay unaffected:

D_inverted(x, y) = 255 − D(x, y)

### ROI Statistics

For a rectangular region R of size m × n (T = m × n pixels), mean and standard deviation are computed as:

μ_ROI = (1/T) Σ R(i, j)

σ_ROI = sqrt( (1/T) Σ (R(i, j) − μ_ROI)² )

### Isotropic Resampling for Multi-Planar Reconstruction

CT slices typically have unequal spacing between axes, for example 0.7mm within a slice and 2.5mm between slices. A coronal or sagittal cut through the raw volume is distorted unless the volume is first resampled to equal spacing on all three axes. `mpr.resample_to_isotropic` performs this resampling using `scipy.ndimage.zoom`, deriving slice spacing from the actual `ImagePositionPatient` z-coordinates rather than the `SliceThickness` tag, which can disagree with true spacing when slices overlap.

---

## Project Structure

The repository follows the `src/` layout for installable Python packages.

```
src/dicom viewer/
    dicom_parser.py
    dicom_series.py
    series_grouping.py
    processing_engine.py
    mpr.py
    image_export.py

tests/
    conftest.py
    test_dicom_parser.py
    test_dicom_series.py
    test_series_grouping.py
    test_processing_engine.py
    test_mpr.py
    test_image_export.py

scripts/
    run_local_series.py

sample_data/
    ct-lung-screening-nlst-series/

docs/
    output_previews/

pyproject.toml
requirements.txt
README.md
```

### Module Description

**dicom_parser.py**
Loads a single `.dcm` file and extracts metadata into a `DicomMetadata` dataclass. Every field has a fallback default, since real files frequently omit tags. Also verifies the file contains `PixelData` before further processing.

**series_grouping.py**
Splits a folder into separate scans by reading only the `SeriesInstanceUID` tag. Public datasets often mix a scout/localizer series with the main acquisition in the same folder, which otherwise causes a shape mismatch when loaded as one series.

**dicom_series.py**
Aggregates a folder of slices into a single 3D volume, sorted by the z-coordinate from `ImagePositionPatient`. `load_dicom_series_safe` builds on `series_grouping` to handle folders containing more than one series.

**processing_engine.py**
Implements the rescale, windowing, and ROI statistics described above. Fully vectorized with NumPy.

**mpr.py**
Performs isotropic resampling and coronal/sagittal slicing on a loaded volume.

**image_export.py**
Converts a windowed 8-bit array into a base64-encoded PNG or JPEG.

**scripts/run_local_series.py**
A manual CLI for validating the pipeline on real data, with single-file, series, and batch modes. Batch mode logs failures instead of stopping on the first error.

**tests/**
An automated pytest suite built on synthetic DICOM datasets generated in `conftest.py`, targeting specific edge cases: missing tags, `MONOCHROME1` inversion, zero window width, and mixed-series folders.

---

## Requirements

This project requires:

- Python 3.9 or higher
- pydicom
- NumPy
- Pillow
- SciPy
- pytest (for running tests)

---

## Installation

Clone the repository:

```
git clone https://github.com/nimratabassum/DICOM-viewer.git
cd dicom-viewer
```

Install the package:

```
pip install -e .
```

For development and testing:

```
pip install -e ".[dev]"
```

Compressed transfer syntaxes (JPEG2000, JPEG Lossless, RLE) require additional codec packages that pydicom does not install by default:

```
pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
```

---

## Running the Pipeline

Run against the included sample series:

```
python scripts/run_local_series.py --series sample_data/ct-lung-screening-nlst-series
```

---

## Results

The renders below are produced by the full pipeline (rescale, window, export) at different axial levels within the same NLST series.

<img width="256" height="256" alt="test_1" src="https://github.com/user-attachments/assets/1e9faef0-64bb-4275-95d6-24c49b75746b" />

Upper chest, at the level where the trachea bifurcates into the main bronchi. The scattered dots visible in the lung fields are small vessels seen end-on in cross section.

<img width="256" height="256" alt="test_2" src="https://github.com/user-attachments/assets/6e847993-5678-472f-9815-cfac0a5322ba" />

Aortic arch level. Soft tissue windowing renders the mediastinum as a single uniform region without losing lung detail on either side.

<img width="256" height="256" alt="test_3" src="https://github.com/user-attachments/assets/05fec9de-eebf-46e2-bf07-86f71fdf576d" />

Upper abdomen. Liver occupies the right side of the frame, with the stomach and spleen visible on the left.

<img width="256" height="256" alt="test_4" src="https://github.com/user-attachments/assets/187abee6-58c3-4bc1-a186-f8d6f9629546" />

Chest level, similar to the first slice but positioned slightly lower.

<img width="256" height="256" alt="test_5" src="https://github.com/user-attachments/assets/d3fce5f2-b660-4ea4-abf8-b1ebd8a05a84" />

Heart level. Internal structure is visible within the mediastinum, consistent with chamber boundaries.

<img width="256" height="256" alt="test_6" src="https://github.com/user-attachments/assets/025e4487-339e-400a-b49e-b972dbc38674" />

Shoulder/neck junction. Clavicles and humeral heads are visible on both sides, with contrast present in the adjacent vessels.

<img width="256" height="256" alt="test_7" src="https://github.com/user-attachments/assets/c2b03ed1-42c5-4af6-8eab-a4866f1477bd" />

Heart level, a different slice within the same region as the two prior cardiac-level renders.

<img width="256" height="256" alt="test_8" src="https://github.com/user-attachments/assets/51cf2ee8-b348-458c-9abc-d40016f59e83" />

Upper abdomen. Liver dominates the right side, with a gas pocket visible in the stomach and the spleen visible as an adjacent oval structure.

<img width="256" height="256" alt="test_9" src="https://github.com/user-attachments/assets/600e7e83-52e1-46ad-9a51-035e8574ac6f" />

Mid-abdomen. Both kidneys are visible posteriorly, along with bowel loops and liver tissue.

<img width="256" height="256" alt="test_10" src="https://github.com/user-attachments/assets/221f5e30-6392-47c1-aa1a-19314c1708c9" />

Neck level, below the shoulder junction shown earlier. Trachea and spine are centered, with contrast-filled vessels visible near the humeral heads.

<img width="256" height="256" alt="test_11" src="https://github.com/user-attachments/assets/e1643ff9-7887-4ad6-9096-09aae4b2f5a5" />

Heart level, from a further point in the series. Rendering remains consistent across all cardiac-level slices, confirming the windowing behaves correctly across the full volume rather than on isolated slices.

---

## Testing

Unit tests are implemented using pytest.

Run tests:

```
pytest --cov=dicom_viewer --cov-report=term-missing
```

The test suite verifies:

- Metadata extraction with missing or malformed tags
- MONOCHROME1 inversion handling
- Zero window width edge cases
- Multi-series folder splitting
- Rescale and windowing correctness against known values
- MPR resampling and slicing behavior

---

## Known Limitations

- ROI selection is a fixed rectangle defined by row/column bounds; no freehand or polygon regions.
- MPR resampling uses linear interpolation, sufficient for viewing but not for diagnostic use.
- Only single-channel CT-style data is supported; RGB `PhotometricInterpretation` is not handled.

---

## References

- NEMA (2024). *Digital Imaging and Communications in Medicine (DICOM) Standard*. https://www.dicomstandard.org/
- Bushberg, J. T., Seibert, J. A., Leidholdt, E. M., & Boone, J. M. (2011). *The Essential Physics of Medical Imaging* (3rd ed.). Lippincott Williams & Wilkins.
- Seeram, E. (2015). *Computed Tomography: Physical Principles, Clinical Applications, and Quality Control* (4th ed.). Saunders.
- Mason, D., et al. (2023). *pydicom* (v2.4.4) [Software]. https://github.com/pydicom/pydicom
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362.
- Clark, K., Vendt, B., Smith, K., et al. (2013). The Cancer Imaging Archive (TCIA). *Journal of Digital Imaging*, 26(6), 1045-1057.
