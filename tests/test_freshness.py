"""Tests for read-before-edit freshness tracking."""

from supercoder.context.freshness import FileFreshnessTracker


def test_freshness_tracker_allows_unchanged_read_file(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('ok')\n")
    tracker = FileFreshnessTracker(tmp_path)

    tracker.mark_read(target)
    result = tracker.check_edit(target)

    assert result.allowed is True
    assert result.reason == "file is unchanged since last read"


def test_freshness_tracker_denies_file_without_prior_read(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("print('ok')\n")
    tracker = FileFreshnessTracker(tmp_path)

    result = tracker.check_edit(target)

    assert result.allowed is False
    assert "not read before edit" in result.reason


def test_freshness_tracker_denies_file_changed_after_read(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    tracker = FileFreshnessTracker(tmp_path)

    tracker.mark_read(target)
    target.write_text("VALUE = 2\n")
    result = tracker.check_edit(target)

    assert result.allowed is False
    assert "changed since last read" in result.reason


def test_freshness_tracker_marks_written_file_fresh(tmp_path):
    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n")
    tracker = FileFreshnessTracker(tmp_path)

    tracker.mark_read(target)
    target.write_text("VALUE = 2\n")
    tracker.mark_written(target)
    result = tracker.check_edit(target)

    assert result.allowed is True
