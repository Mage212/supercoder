"""Shared helpers for read-only tools."""

from __future__ import annotations

import fnmatch
from difflib import get_close_matches
from pathlib import Path

IGNORE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "target",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".pytest_cache",
    "egg-info",
    ".eggs",
    ".mypy_cache",
    ".ruff_cache",
}

IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".class"}
IGNORE_NAMES = {".DS_Store"}


def resolve_within_root(
    path_input: str, allowed_root: Path | None
) -> tuple[Path | None, str | None]:
    """Resolve a path and ensure it stays inside allowed_root when configured."""
    if allowed_root is None:
        return Path(path_input or ".").resolve(), None

    root = allowed_root.resolve()
    raw_path = Path(path_input or ".")
    path = (root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, f"Error: Path '{path_input}' is outside the project directory"
    return path, None


def relative_display_path(path: Path, root: Path | None) -> str:
    """Return a stable relative path for tool output when possible."""
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def is_ignored_path(path: Path, root: Path | None = None) -> bool:
    """Return True for junk/cache paths that should not be searched by tools."""
    try:
        rel = path.resolve().relative_to(root.resolve()) if root else path
    except ValueError:
        rel = path

    for part in rel.parts:
        if part in IGNORE_DIRS or part in IGNORE_NAMES:
            return True
        if part.startswith(".") and part != ".env.example":
            return True

    return path.suffix in IGNORE_SUFFIXES


def matches_file_pattern(path: Path, pattern: str, root: Path | None = None) -> bool:
    """Match filePattern against both file name and relative path."""
    if not pattern:
        return True
    rel = relative_display_path(path, root)
    return fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern)


def is_binary_file(path: Path, sample_size: int = 4096) -> bool:
    """Detect obvious binary files without reading the full file."""
    try:
        with path.open("rb") as f:
            sample = f.read(sample_size)
    except OSError:
        return False

    if not sample:
        return False
    if b"\0" in sample:
        return True

    allowed_controls = {8, 9, 10, 12, 13, 27}
    control_bytes = sum(1 for byte in sample if byte < 32 and byte not in allowed_controls)
    return control_bytes / len(sample) > 0.30


def format_size(size: int) -> str:
    """Format byte size in a compact human-readable form."""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size // 1024}KB"
    return f"{size // (1024 * 1024)}MB"


def find_similar_files(
    missing_path: Path,
    allowed_root: Path | None,
    limit: int = 5,
    candidate_limit: int = 5000,
) -> list[str]:
    """Return close file path suggestions for a missing file."""
    search_root = allowed_root.resolve() if allowed_root else missing_path.parent.resolve()
    if missing_path.parent.exists():
        search_root = missing_path.parent.resolve()
        if allowed_root:
            try:
                search_root.relative_to(allowed_root.resolve())
            except ValueError:
                search_root = allowed_root.resolve()

    if not search_root.exists() or not search_root.is_dir():
        return []

    candidates: list[Path] = []
    try:
        iterator = search_root.iterdir() if missing_path.parent.exists() else search_root.rglob("*")
        for candidate in iterator:
            if len(candidates) >= candidate_limit:
                break
            if candidate.is_file() and not is_ignored_path(candidate, allowed_root):
                candidates.append(candidate)
    except OSError:
        return []

    if not candidates:
        return []

    by_name = {candidate.name: candidate for candidate in candidates}
    close_names = get_close_matches(missing_path.name, list(by_name), n=limit * 2, cutoff=0.45)

    suggestions: list[Path] = []
    for name in close_names:
        candidate = by_name[name]
        if candidate not in suggestions:
            suggestions.append(candidate)

    if len(suggestions) < limit:
        rel_map = {
            relative_display_path(candidate, allowed_root): candidate for candidate in candidates
        }
        close_paths = get_close_matches(
            missing_path.as_posix(), list(rel_map), n=limit * 2, cutoff=0.35
        )
        for rel in close_paths:
            candidate = rel_map[rel]
            if candidate not in suggestions:
                suggestions.append(candidate)

    return [relative_display_path(candidate, allowed_root) for candidate in suggestions[:limit]]
