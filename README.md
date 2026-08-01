# DICOM Project

A small Python pipeline for loading CT DICOM files, converting raw pixel
data to Hounsfield Units, windowing them into a viewable image, and
exporting the result as PNG/JPEG. It also handles whole series (a
folder of slices from one scan), not just single files.

I built this using the NLST lung-screening sample series
(`ct-lung-screening-nlst-series/`) as my test data.

## Why this exists

Raw pixel values in a DICOM file aren't directly viewable. CT scanners
store data as raw integers that need to be rescaled into Hounsfield
Units (HU) first, then "windowed" into an 8-bit range you can actually
render as a grayscale image. Get either step wrong and the image looks
fine but is medically meaningless, or looks visibly broken (inverted,
blown out, etc). This project handles that pipeline end to end, plus
the annoying edge cases real DICOM files throw at you (missing tags,
MONOCHROME1 inversion, files that aren't images at all, folders with
more than one scan mixed together).

## Project structure

```
dicom_parser.py       Load a single .dcm file, extract + normalize metadata
processing_engine.py   Pure NumPy: rescale -> HU, window/level -> 8-bit, ROI stats
dicom_series.py         Load a whole folder of slices as one ordered 3D volume
series_grouping.py      Split a folder into separate series by SeriesInstanceUID
mpr.py                  Resample a volume to isotropic spacing + coronal/sagittal slicing
image_export.py         Convert a windowed array to base64 PNG/JPEG
local series test.py    Manual smoke-test script (single file / one series / batch mode)
tests/                  Automated pytest suite
```

## Pipeline, step by step

1. **Load the file** (`dicom_parser.load_dicom_file`) — reads the
   `.dcm` file and checks it actually has `PixelData`. Some DICOM
   files are reports or presentation states with no image in them at
   all, so this fails early and clearly instead of crashing later on a
   confusing `AttributeError`.

2. **Extract metadata** (`dicom_parser.extract_metadata`) — pulls out
   rows/columns, rescale slope/intercept, window center/width, pixel
   spacing, and photometric interpretation, with a safe fallback for
   every field. Real files are missing tags more often than you'd
   expect, so nothing here assumes a tag is present.

3. **Rescale to HU** (`processing_engine.apply_rescale`) —
   `pixel_value * slope + intercept`. This turns the sensor's raw
   integers into actual Hounsfield Units, where air is about -1000 and
   dense bone is around +1000 to +3000.

4. **Window/level** (`processing_engine.apply_window_level`) — maps a
   chosen HU range down to 0-255 so it can be displayed. Also handles
   `MONOCHROME1` inversion, where low pixel values are supposed to
   render as white instead of black.

5. **Export** (`image_export.array_to_base64_image` /
   `array_to_data_uri`) — turns the windowed 8-bit array into a PNG or
   JPEG, base64-encoded.

For a full scan instead of one file, `dicom_series.load_dicom_series`
loads every slice in a folder, sorts them by physical position
(`ImagePositionPatient`'s z-coordinate, not filename), and stacks them
into one 3D HU volume.

## Handling messy multi-series folders

Not every folder you download is actually one clean series.
Downloads from public archives sometimes mix a scout/localizer series
in with the main scan, or bundle two scans from the same visit
together. `dicom_series.load_dicom_series` used to just fail with a
shape-mismatch error in that case, which tells you something's wrong
but not what to do about it.

`series_grouping.group_by_series` fixes that by reading just the
`SeriesInstanceUID` tag (no pixel data, so it's cheap) and splitting
the folder into per-series file lists. `dicom_series.load_dicom_series_safe`
builds on that: it loads every series it finds and returns them as a
list, largest first, instead of forcing you to pre-sort the folder by
hand.

## Multi-planar reconstruction

`mpr.py` adds coronal and sagittal views on top of the axial volume
you get from loading a series. CT slices almost never have equal
spacing in all three directions (e.g. 0.7mm between pixels within a
slice but 10mm between slices), so cutting straight through the raw
volume along another axis comes out stretched. `resample_to_isotropic`
fixes that first by resampling the whole volume to equal spacing in
every direction, then `get_coronal_slice` / `get_sagittal_slice` cut
through the result.

## Testing

The `tests/` folder has a real pytest suite, not just the manual
script. It runs against small synthetic DICOM datasets built in
`tests/conftest.py`, not the sample scan, so it can hit specific edge
cases on purpose (missing tags, `MONOCHROME1`, zero window width, a
folder with two series mixed together) instead of hoping the sample
data happens to contain them.

Run it with:

```bash
pip install -r requirements.txt
pytest --cov=. --cov-report=term-missing
```

`local series test.py` is separate from this suite. It's a manual
script for quickly checking a real file/folder/batch by eye (it prints
metadata and saves preview PNGs to `output_previews/`), not something
meant to run in CI.

## Requirements

```bash
pip install -r requirements.txt
```

Compressed DICOM transfer syntaxes (JPEG, JPEG2000, JPEG-LS, RLE) need
an extra codec package pydicom doesn't install by default. If you hit
a decode error, `dicom_parser.extract_pixel_array` will tell you
exactly which packages to install (`pylibjpeg`, `pylibjpeg-libjpeg`,
`pylibjpeg-openjpeg`, or `python-gdcm`).

## Known limitations

- ROI selection is a fixed rectangle passed in as row/col bounds, no
  freehand or polygon regions.
- `mpr.py`'s resampling uses linear interpolation, which is fine for
  viewing but not what you'd want for anything diagnostic.
- Everything here assumes CT-style single-channel data.
  `PhotometricInterpretation` values like `RGB` aren't handled.
