"""Glob tool for finding files without reading their contents."""

from __future__ import annotations

from pathlib import Path

from .base import BaseTool, ToolDefinition
from .tool_utils import is_ignored_path, relative_display_path, resolve_within_root


class GlobTool(BaseTool):
    """Find files by glob pattern."""

    HARD_MAX_RESULTS = 1000

    def __init__(self, allowed_root: Path | None = None):
        self.allowed_root = allowed_root

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="glob",
            description="Find files by glob pattern without reading file contents.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search from",
                        "default": ".",
                    },
                    "maxResults": {
                        "type": "integer",
                        "description": "Maximum number of paths to return",
                        "default": 100,
                    },
                    "includeDirs": {
                        "type": "boolean",
                        "description": "Include directories in results",
                        "default": False,
                    },
                },
                "required": ["pattern"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"

        pattern = str(args.get("pattern", "")).strip()
        search_path = str(args.get("path", "."))
        max_results = min(max(int(args.get("maxResults", 100)), 1), self.HARD_MAX_RESULTS)
        include_dirs = bool(args.get("includeDirs", False))

        if not pattern:
            return "Error: pattern is required"

        root, error = resolve_within_root(search_path, self.allowed_root)
        if error:
            return error
        if root is None:
            return "Error: Invalid search path"
        if not root.exists():
            return f"Error: Path '{search_path}' not found"
        if not root.is_dir():
            return f"Error: Path '{search_path}' is not a directory"

        matches: list[Path] = []
        truncated = False

        try:
            for candidate in root.rglob(pattern):
                if is_ignored_path(candidate, self.allowed_root):
                    continue
                if candidate.is_dir() and not include_dirs:
                    continue
                matches.append(candidate)
                if len(matches) >= max_results:
                    truncated = True
                    break
        except ValueError as e:
            return f"Error: Invalid glob pattern '{pattern}': {e}"
        except OSError as e:
            return f"Error running glob: {e}"

        matches = sorted(matches, key=lambda p: relative_display_path(p, self.allowed_root))
        shown = [relative_display_path(path, self.allowed_root) for path in matches[:max_results]]

        header = (
            f"Glob: {pattern}\n"
            f"Path: {relative_display_path(root, self.allowed_root)}\n"
            f"Found: {len(matches)}"
        )
        if truncated:
            header += f" (showing first {max_results}, truncated)"

        if not shown:
            return f"{header}\n\nNo matches found."

        return f"{header}\n{'-' * 50}\n" + "\n".join(shown)
