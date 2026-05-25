"""Tests for RepoMap persistence behavior."""

from unittest.mock import MagicMock

from supercoder.repomap.map import RepoMap


def test_repomap_persistence_errors_are_logged(tmp_path, monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr("supercoder.repomap.map.get_logger", lambda: fake_logger)

    def fail_write(_path, _content, encoding="utf-8"):
        raise OSError("disk full")

    monkeypatch.setattr("supercoder.repomap.map.AtomicFileWriter.write", fail_write)
    repo_map = RepoMap(tmp_path)
    repo_map._get_files = MagicMock(return_value=[])

    assert repo_map.get_repo_map() == ""
    fake_logger.log_error.assert_called_once()
