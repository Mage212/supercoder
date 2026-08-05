"""Repository Map generator."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from ..logging import get_logger
from ..tools.tool_utils import is_ignored_path
from ..utils.atomic_writer import AtomicFileWriter
from .tag_extractor import TagExtractor


class RepoMap:
    """Generates a compact, cache-friendly map of the repository structure.

    The map is designed to be byte-stable across turns when files are
    unchanged, so that a local LLM backend (llama-server / vLLM / SGLang) can
    reuse its prefix KV cache. Two invariants support that:

    1. Determinism: file selection and rendering are sorted, so identical file
       sets produce identical bytes regardless of filesystem inode order.
    2. Content-hash cache: a digest over the selected files' (path, mtime,
       size) short-circuits both extraction and rendering on cache hits, and
       skips the on-disk rewrite of repo_map.txt.
    """

    def __init__(self, root: str | Path = "."):
        self.root = Path(root).resolve()
        self.extractor = TagExtractor()
        self.storage_dir = self.root / ".supercoder" / "repomap"
        # Content-hash cache for the rendered map.
        self._cache_key: str | None = None
        self._cached_map: str | None = None

    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_repo_map(self, max_tokens: int = 1024) -> str:
        """Generate a token-limited repository map and persist it.

        Returns a cached render unchanged when the selected files' contents
        (as summarized by mtime + size) have not changed since the last call.
        """
        files = self._get_files()
        cache_key = self._compute_cache_key(files, max_tokens)

        # Fast path: identical inputs → return cached render without touching
        # tree-sitter or rewriting repo_map.txt.
        if cache_key == self._cache_key and self._cached_map is not None:
            return self._cached_map

        all_tags = []
        for file in files:
            tags = self.extractor.extract(str(file))
            all_tags.extend(tags)

        repo_map = self._render_tree(all_tags, max_tokens)

        # Persist to file and sidecar metadata only on a cache miss.
        try:
            self._ensure_storage_dir()
            map_file = self.storage_dir / "repo_map.txt"
            AtomicFileWriter.write(map_file, repo_map)
            self._write_meta_sidecar(cache_key, len(files))
        except Exception as e:
            get_logger().log_error(e)

        self._cache_key = cache_key
        self._cached_map = repo_map
        return repo_map

    def _compute_cache_key(self, files: list[Path], max_tokens: int) -> str:
        """Build a SHA256 digest over the selected files' identity + freshness.

        Keyed on (relative path, mtime_ns, size) so that content edits, path
        additions/removals, and reorderings all invalidate the cache, while
        repeated calls on unchanged files keep the key stable. ``max_tokens``
        is part of the key so that a different render budget is honored rather
        than served from a stale cache entry.
        """
        if not files:
            return "empty"
        parts: list[str] = [f"max_tokens={max_tokens}"]
        for f in files:
            try:
                rel = f.relative_to(self.root).as_posix()
            except ValueError:
                rel = str(f)
            try:
                st = f.stat()
                parts.append(f"{rel}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{rel}:missing")
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return digest[:16]

    def _write_meta_sidecar(self, cache_key: str, file_count: int) -> None:
        """Write a human-readable freshness marker next to repo_map.txt."""
        meta = {
            "cache_key": cache_key,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "file_count": file_count,
        }
        meta_path = self.storage_dir / "repo_map.meta.json"
        AtomicFileWriter.write(meta_path, json.dumps(meta, indent=2))

    def _get_files(self) -> list[Path]:
        """Get relevant source files in deterministic (sorted) order.

        Sorting is required for two reasons:
        - rglob order depends on directory inode order and is not stable across
          runs, which would bust the LLM prefix cache on every call.
        - The 50-file cap below otherwise selects an arbitrary subset; a sorted
          selection is reproducible.
        """
        source_extensions = {".py", ".scala", ".java", ".js", ".ts", ".go", ".rs"}
        files = []

        for path in self.root.rglob("*"):
            if is_ignored_path(path, self.root):
                continue
            if path.is_file() and path.suffix in source_extensions:
                files.append(path)

        files.sort()
        # Limit total files for performance
        return files[:50]

    def _render_tree(self, tags, max_tokens) -> str:
        """Render tags into a tree structure with deterministic file order."""
        if not tags:
            return ""

        # Group by file
        by_file: dict[str, list] = {}
        for tag in tags:
            if tag.file not in by_file:
                by_file[tag.file] = []
            by_file[tag.file].append(tag)

        output = []
        total_tokens = 0

        # Sort by file path for stable rendering order.
        for file in sorted(by_file.keys()):
            file_tags = by_file[file]
            rel_path = Path(file).relative_to(self.root)
            output.append(f"{rel_path}:")

            for tag in file_tags:
                line = f"  {tag.name} {tag.kind}"
                output.append(line)

                # Simple token check
                total_tokens += len(line) // 4
                if total_tokens > max_tokens:
                    output.append("  ...")
                    return "\n".join(output)

        return "\n".join(output)
