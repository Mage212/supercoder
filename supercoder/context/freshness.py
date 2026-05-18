"""Host-side freshness tracking for read-before-edit enforcement."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..logging import get_logger


@dataclass(frozen=True)
class FileSnapshot:
    """File metadata captured when content was shown to the model."""

    mtime_ns: int
    size: int
    sha256: str | None = None

    @property
    def hash_present(self) -> bool:
        return self.sha256 is not None


@dataclass(frozen=True)
class FreshnessResult:
    """Result of checking whether a file is safe to edit."""

    allowed: bool
    reason: str
    snapshot: FileSnapshot | None = None
    current: FileSnapshot | None = None


class FileFreshnessTracker:
    """Track files that were exposed to the model before editing."""

    DEFAULT_MAX_HASH_BYTES = 1_000_000

    def __init__(self, repo_root: Path | None = None, max_hash_bytes: int = DEFAULT_MAX_HASH_BYTES):
        self.repo_root = repo_root.resolve() if repo_root else None
        self.max_hash_bytes = max_hash_bytes
        self._reads: dict[Path, FileSnapshot] = {}

    def mark_read(self, path: Path, *, source: str = "file-read") -> FileSnapshot:
        """Record that a file's current content was shown to the model."""
        resolved = path.resolve()
        snapshot = self._snapshot(resolved)
        self._reads[resolved] = snapshot
        self._log(
            path=resolved,
            source=source,
            action="mark_read",
            status="recorded",
            reason="file content exposed to model",
            snapshot=snapshot,
        )
        return snapshot

    def check_edit(self, path: Path, *, source: str = "code-edit") -> FreshnessResult:
        """Return whether an existing file can be edited using known-fresh context."""
        resolved = path.resolve()
        snapshot = self._reads.get(resolved)
        if snapshot is None:
            result = FreshnessResult(
                allowed=False,
                reason="File was not read before edit. Use file-read or @file first.",
            )
            self._log(
                path=resolved,
                source=source,
                action="check_edit",
                status="denied",
                reason=result.reason,
            )
            return result

        current = self._snapshot(resolved)
        if snapshot != current:
            result = FreshnessResult(
                allowed=False,
                reason="File changed since last read. Read it again before editing.",
                snapshot=snapshot,
                current=current,
            )
            self._log(
                path=resolved,
                source=source,
                action="check_edit",
                status="denied",
                reason=result.reason,
                snapshot=current,
            )
            return result

        result = FreshnessResult(
            allowed=True,
            reason="file is unchanged since last read",
            snapshot=snapshot,
            current=current,
        )
        self._log(
            path=resolved,
            source=source,
            action="check_edit",
            status="allowed",
            reason=result.reason,
            snapshot=current,
        )
        return result

    def mark_written(self, path: Path, *, source: str = "code-edit") -> FileSnapshot:
        """Record the post-write file state as fresh for follow-up edits."""
        resolved = path.resolve()
        snapshot = self._snapshot(resolved)
        self._reads[resolved] = snapshot
        self._log(
            path=resolved,
            source=source,
            action="mark_written",
            status="recorded",
            reason="file changed by SuperCoder",
            snapshot=snapshot,
        )
        return snapshot

    def _snapshot(self, path: Path) -> FileSnapshot:
        stat = path.stat()
        sha256: str | None = None
        if path.is_file() and stat.st_size <= self.max_hash_bytes:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
        return FileSnapshot(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            sha256=sha256,
        )

    def _display_path(self, path: Path) -> str:
        if self.repo_root:
            try:
                return path.relative_to(self.repo_root).as_posix()
            except ValueError:
                pass
        return str(path)

    def _log(
        self,
        *,
        path: Path,
        source: str,
        action: str,
        status: str,
        reason: str,
        snapshot: FileSnapshot | None = None,
    ) -> None:
        get_logger().log_freshness_check(
            path=self._display_path(path),
            source=source,
            action=action,
            status=status,
            reason=reason,
            size=snapshot.size if snapshot else None,
            hash_present=snapshot.hash_present if snapshot else False,
        )
