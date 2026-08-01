"""
test_dicom.py

ONE script, THREE modes:
  1. SINGLE FILE  — test one .dcm file
  2. ONE SERIES   — test one folder containing all slices of one scan
  3. BATCH        — recursively test every .dcm file under a root folder
                    (handles messy nested folder structures like TCIA downloads)

You can either:
  (a) Edit TARGET_PATH below and just run: python test_dicom.py
      The script auto-detects which mode fits based on what TARGET_PATH is.
  (b) Force a specific mode from the command line:
      python test_dicom.py --file "C:\\path\\to\\one_file.dcm"
      python test_dicom.py --series "C:\\path\\to\\one_series_folder"
      python test_dicom.py --batch "C:\\path\\to\\root_download_folder"
"""

import os
import sys
import base64
import argparse

from dicom_viewer.dicom_parser import load_dicom_file, extract_metadata, extract_pixel_array, needs_inversion
from dicom_viewer.dicom_series import load_dicom_series
from dicom_viewer.processing_engine import apply_rescale, apply_window_level, compute_roi_statistics, compute_full_range_window
from dicom_viewer.image_export import array_to_base64_image


# ============================================================
# EDIT THIS if you want to just run "python test_dicom.py" with no flags.
# Point it at EITHER a single .dcm file, a series folder, or a root
# download folder — the script figures out which mode to use.
# Leave as-is if you're using command-line flags instead.
# ============================================================
TARGET_PATH = r"C:\Users\Adeel\Desktop\Nimra tabassum project\DICOM project\ct-lung-screening-nlst-series\instance-0120.dcm"


# ------------------------------------------------------------
# MODE 1: single file
# ------------------------------------------------------------
def test_single_file(filepath: str, output_dir: str = "output_previews"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"MODE: SINGLE FILE")
    print(f"File: {filepath}")
    print("=" * 60)

    ds = load_dicom_file(filepath)
    meta = extract_metadata(ds)
    print(meta)

    invert = needs_inversion(meta)
    print(f"Needs inversion (MONOCHROME1): {invert}")

    raw = extract_pixel_array(ds)
    print(f"Raw pixel array shape: {raw.shape}, dtype: {raw.dtype}")

    hu = apply_rescale(raw, meta.rescale_slope, meta.rescale_intercept)
    print(f"HU range: [{hu.min():.1f}, {hu.max():.1f}]")

    ww, wc = meta.window_width, meta.window_center
    if ww <= 0:
        ww, wc = compute_full_range_window(hu)
        print(f"Invalid window in file tags, auto-computed instead: WW={ww:.1f}, WC={wc:.1f}")

    img = apply_window_level(hu, ww, wc, invert=invert)

    b64 = array_to_base64_image(img)
    out_name = os.path.splitext(os.path.basename(filepath))[0] + ".png"
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"[OK] Saved preview: {out_path}")

    # ROI stats on center region as a bonus check
    r, c = hu.shape
    r0, r1 = max(0, r // 2 - 15), min(r, r // 2 + 15)
    c0, c1 = max(0, c // 2 - 15), min(c, c // 2 + 15)
    stats = compute_roi_statistics(hu, r0, r1, c0, c1)
    print(f"Center ROI stats: {stats}")

    print("\n[SUCCESS] Single file test complete.\n")


# ------------------------------------------------------------
# MODE 2: one series folder
# ------------------------------------------------------------
def test_one_series(folder_path: str, output_dir: str = "output_previews"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"MODE: ONE SERIES")
    print(f"Folder: {folder_path}")
    print("=" * 60)

    try:
        series = load_dicom_series(folder_path)
    except Exception as e:
        print(f"[FAILED] Could not load as one series: {e}")
        print("\nIf this says 'Inconsistent slice dimensions', this folder")
        print("contains MORE THAN ONE series mixed together. Either point")
        print("this at a deeper single-series subfolder, or use --batch")
        print("mode instead to test every file individually regardless")
        print("of folder structure.")
        return

    print(f"[OK] Loaded {series.num_slices} slices.")
    print(f"Volume shape: {series.volume_hu.shape}")
    print(f"Inversion needed: {series.invert}")

    indices = sorted(set([0, series.num_slices // 2, series.num_slices - 1]))
    for idx in indices:
        hu_slice = series.get_slice_hu(idx)
        meta = series.slice_metadata[idx]
        img = apply_window_level(hu_slice, meta.window_width, meta.window_center, invert=series.invert)
        b64 = array_to_base64_image(img)
        out_path = os.path.join(output_dir, f"series_slice_{idx}.png")
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  Slice {idx}: mean HU={hu_slice.mean():.1f} -> saved {out_path}")

    mid_idx = series.num_slices // 2
    hu_mid = series.get_slice_hu(mid_idx)
    r, c = hu_mid.shape
    r0, r1 = max(0, r // 2 - 15), min(r, r // 2 + 15)
    c0, c1 = max(0, c // 2 - 15), min(c, c // 2 + 15)
    stats = compute_roi_statistics(hu_mid, r0, r1, c0, c1)
    print(f"\nCenter ROI stats (middle slice): {stats}")

    print("\n[SUCCESS] Series test complete.\n")


# ------------------------------------------------------------
# MODE 3: batch — recursively test every file, no assumptions about structure
# ------------------------------------------------------------
def test_batch(root_folder: str, output_dir: str = "output_previews"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(f"MODE: BATCH (recursive)")
    print(f"Root folder: {root_folder}")
    print("=" * 60)

    dcm_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(dirpath, fname))

    print(f"Found {len(dcm_files)} .dcm files.\n")
    if not dcm_files:
        print("No .dcm files found under this path.")
        return

    results = {"SUCCESS": 0, "FAILED": 0}
    failures = []

    for i, fp in enumerate(dcm_files, 1):
        try:
            ds = load_dicom_file(fp)
            meta = extract_metadata(ds)
            invert = needs_inversion(meta)
            raw = extract_pixel_array(ds)
            hu = apply_rescale(raw, meta.rescale_slope, meta.rescale_intercept)
            ww, wc = meta.window_width, meta.window_center
            if ww <= 0:
                ww, wc = compute_full_range_window(hu)
            img = apply_window_level(hu, ww, wc, invert=invert)
            b64 = array_to_base64_image(img)
            safe_name = f"{i:04d}_" + os.path.splitext(os.path.basename(fp))[0] + ".png"
            with open(os.path.join(output_dir, safe_name), "wb") as f:
                f.write(base64.b64decode(b64))
            results["SUCCESS"] += 1
            print(f"[{i}/{len(dcm_files)}] OK: {fp}")
        except Exception as e:
            results["FAILED"] += 1
            failures.append((fp, str(e)))
            print(f"[{i}/{len(dcm_files)}] FAILED: {fp}")
            print(f"     Reason: {e}")

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Success: {results['SUCCESS']}")
    print(f"Failed:  {results['FAILED']}")
    print(f"Total:   {len(dcm_files)}")
    print(f"\nPreviews saved to: {os.path.abspath(output_dir)}")

    if failures:
        print("\nFailure details:")
        for fp, err in failures:
            print(f"  - {fp}\n      {err}")


# ------------------------------------------------------------
# Mode auto-detection + entry point
# ------------------------------------------------------------
def looks_like_single_series_folder(folder_path: str) -> bool:
    """True if this folder directly contains .dcm files (no subfolders needed)."""
    try:
        entries = os.listdir(folder_path)
    except Exception:
        return False
    return any(f.lower().endswith(".dcm") for f in entries)


def main():
    parser = argparse.ArgumentParser(description="Unified DICOM pipeline tester")
    parser.add_argument("--file", help="Path to a single .dcm file")
    parser.add_argument("--series", help="Path to one series folder (all slices of one scan)")
    parser.add_argument("--batch", help="Path to a root folder to recursively test every .dcm file")
    args = parser.parse_args()

    if args.file:
        test_single_file(args.file)
        return
    if args.series:
        test_one_series(args.series)
        return
    if args.batch:
        test_batch(args.batch)
        return

    # No flags given — fall back to TARGET_PATH with auto-detection
    if TARGET_PATH == "PASTE_YOUR_PATH_HERE":
        print("No path provided.")
        print("Either edit TARGET_PATH at the top of this file, or use a flag:")
        print('  python test_dicom.py --file "C:\\path\\to\\file.dcm"')
        print('  python test_dicom.py --series "C:\\path\\to\\series_folder"')
        print('  python test_dicom.py --batch "C:\\path\\to\\root_folder"')
        sys.exit(1)

    if os.path.isfile(TARGET_PATH):
        test_single_file(TARGET_PATH)
    elif os.path.isdir(TARGET_PATH):
        if looks_like_single_series_folder(TARGET_PATH):
            print("(Auto-detected: this folder has .dcm files directly in it -> treating as ONE SERIES)")
            print("(If this is wrong and it's actually a mixed multi-series folder, re-run with --batch instead)\n")
            test_one_series(TARGET_PATH)
        else:
            print("(Auto-detected: no .dcm files directly in this folder, only subfolders -> using BATCH mode)\n")
            test_batch(TARGET_PATH)
    else:
        print(f"'{TARGET_PATH}' is not a valid file or folder path.")
        sys.exit(1)


if __name__ == "__main__":
    main()