"""
series_grouping.py

Splits a folder of .dcm files into separate series based on
SeriesInstanceUID.

dicom_series.py assumes every file in a folder belongs to one scan,
and raises if the resulting slice shapes don't match. That check
catches the problem but doesn't fix it. In practice, downloads from
public archives (e.g. TCIA) often mix several series into one folder,
sometimes a scout/localizer series next to the real acquisition, or
two scans from the same visit dumped together.

This module reads only the tags needed to group files (no pixel data),
so it stays fast even on a few hundred slices, and hands back one file
list per SeriesInstanceUID so the caller can run load_dicom_series()
on each group independently.
"""

import os
from collections import defaultdict
from typing import Dict, List

import pydicom


def _list_dcm_files(folder_path: str) -> List[str]:
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"'{folder_path}' is not a valid directory.")

    filepaths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    if not filepaths:
        raise ValueError(f"No .dcm files found in '{folder_path}'.")
    return filepaths


def group_by_series(folder_path: str) -> Dict[str, List[str]]:
    """
    Group every .dcm file in a folder by SeriesInstanceUID.

    Only the file meta and a handful of top-level tags are read for
    each file (stop_before_pixels=True), so this is cheap to run even
    before deciding how to load the data.

    Returns:
        A dict mapping SeriesInstanceUID -> sorted list of filepaths.
        Files with no SeriesInstanceUID tag are grouped under the key
        "UNKNOWN_SERIES" instead of being dropped, so nothing silently
        disappears.

    Raises:
        NotADirectoryError: if folder_path doesn't exist.
        ValueError: if the folder has no .dcm files.
    """
    filepaths = _list_dcm_files(folder_path)

    groups: Dict[str, List[str]] = defaultdict(list)
    for fp in filepaths:
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=False)
        series_uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES"))
        groups[series_uid].append(fp)

    for uid in groups:
        groups[uid].sort()

    return dict(groups)


def describe_groups(groups: Dict[str, List[str]]) -> str:
    """
    Human-readable summary of a group_by_series() result, useful for
    quickly checking whether a "batch" download is actually one series,
    or several mixed together, before committing to loading all of it.
    """
    lines = [f"Found {len(groups)} series:"]
    for uid, files in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  - {uid}: {len(files)} file(s)")
    return "\n".join(lines)
