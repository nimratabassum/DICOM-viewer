"""
test_series_grouping.py

Covers series_grouping.py: splitting a folder of .dcm files into
groups by SeriesInstanceUID.
"""

import pytest

from dicom_viewer.series_grouping import describe_groups, group_by_series


class TestGroupBySeries:
    def test_single_series_folder_returns_one_group(self, synthetic_series_folder):
        groups = group_by_series(synthetic_series_folder)

        assert len(groups) == 1
        (files,) = groups.values()
        assert len(files) == 5

    def test_mixed_folder_returns_two_groups(self, mixed_series_folder):
        groups = group_by_series(mixed_series_folder)

        assert len(groups) == 2
        sizes = sorted(len(files) for files in groups.values())
        assert sizes == [2, 5]

    def test_missing_folder_raises(self):
        with pytest.raises(NotADirectoryError):
            group_by_series("/no/such/folder")

    def test_empty_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="No .dcm files"):
            group_by_series(str(empty))

    def test_grouping_does_not_read_pixel_data(self, mixed_series_folder, monkeypatch):
        """
        group_by_series() is supposed to be a cheap metadata-only pass
        (stop_before_pixels=True) so it can run before deciding how to
        load a folder. This checks that guarantee directly instead of
        just trusting the docstring.
        """
        import pydicom

        original = pydicom.dcmread

        def _spy(fp, *args, **kwargs):
            assert kwargs.get("stop_before_pixels") is True
            return original(fp, *args, **kwargs)

        monkeypatch.setattr(pydicom, "dcmread", _spy)

        group_by_series(mixed_series_folder)


class TestDescribeGroups:
    def test_output_mentions_series_count_and_file_counts(self, mixed_series_folder):
        groups = group_by_series(mixed_series_folder)

        summary = describe_groups(groups)

        assert "Found 2 series" in summary
        assert "5 file(s)" in summary
        assert "2 file(s)" in summary
