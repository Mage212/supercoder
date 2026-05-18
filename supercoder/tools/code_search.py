"""Code search tool using ripgrep or Python fallback."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .base import BaseTool, ToolDefinition
from .tool_utils import (
    is_binary_file,
    is_ignored_path,
    matches_file_pattern,
    relative_display_path,
    resolve_within_root,
)


class SearchMatch:
    """A normalized code search match."""

    def __init__(self, path: Path, line: int, column: int, text: str):
        self.path = path
        self.line = line
        self.column = column
        self.text = text


class CodeSearchTool(BaseTool):
    """Search for code patterns in the project."""

    HARD_MAX_RESULTS = 100
    MAX_CONTEXT_LINES = 3
    MAX_FILE_BYTES = 1024 * 1024

    def __init__(self, allowed_root: Path | None = None):
        self.allowed_root = allowed_root

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code-search",
            description="Search for code patterns in the project using ripgrep or fallback search.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search pattern (text or regex)"},
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search",
                        "default": ".",
                    },
                    "maxResults": {
                        "type": "integer",
                        "description": "Maximum number of matches to return",
                        "default": 10,
                    },
                    "filePattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files, e.g. '*.py'",
                    },
                    "contextLines": {
                        "type": "integer",
                        "description": "Context lines around each match (0-3)",
                        "default": 0,
                    },
                },
                "required": ["query"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        query = args.get("query", "")
        search_path = str(args.get("path", "."))
        max_results = min(max(int(args.get("maxResults", 10)), 1), self.HARD_MAX_RESULTS)
        file_pattern = args.get("filePattern", "")
        context_lines = min(max(int(args.get("contextLines", 0)), 0), self.MAX_CONTEXT_LINES)

        if not query:
            return "Error: query is required"

        root, error = resolve_within_root(search_path, self.allowed_root)
        if error:
            return error
        if root is None:
            return "Error: Invalid search path"
        if not root.exists():
            return f"Error: Path '{search_path}' not found"

        if shutil.which("rg"):
            matches, total, error = self._ripgrep(query, root, max_results, file_pattern)
            if error:
                return error
            return self._format_results(query, root, matches, total, context_lines, "rg")

        matches, total, error = self._python_search(query, root, max_results, file_pattern)
        if error:
            return error
        return self._format_results(query, root, matches, total, context_lines, "python")

    def _ripgrep(
        self, query: str, root: Path, max_results: int, file_pattern: str
    ) -> tuple[list[SearchMatch], int, str | None]:
        """Search using rg when available."""
        cmd = [
            "rg",
            "--line-number",
            "--column",
            "--no-heading",
            "--with-filename",
            "--color=never",
            "--max-filesize",
            "1M",
        ]
        if file_pattern:
            cmd.extend(["--glob", file_pattern])
        target = "." if root.is_dir() else root.name
        cmd.extend([query, target])

        try:
            result = subprocess.run(
                cmd,
                cwd=root if root.is_dir() else root.parent,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return [], 0, "Error: code-search timed out after 30s"
        except Exception as e:
            return [], 0, f"Error running rg: {e}"

        if result.returncode == 1:
            return [], 0, None
        if result.returncode not in (0, 1):
            stderr = result.stderr.strip()
            return [], 0, f"Error running rg: {stderr or 'search failed'}"

        matches: list[SearchMatch] = []
        total = 0
        cwd = root if root.is_dir() else root.parent
        for raw_line in result.stdout.splitlines():
            parsed = self._parse_rg_line(raw_line, cwd)
            if parsed is None:
                continue
            total += 1
            if len(matches) < max_results:
                matches.append(parsed)

        return matches, total, None

    def _python_search(
        self, query: str, root: Path, max_results: int, file_pattern: str
    ) -> tuple[list[SearchMatch], int, str | None]:
        """Fallback search using Python regex scanning."""
        try:
            pattern = re.compile(query)
        except re.error as e:
            return [], 0, f"Error: invalid regex query: {e}"

        files = [root] if root.is_file() else root.rglob("*")
        matches: list[SearchMatch] = []
        total = 0

        for path in files:
            if not path.is_file():
                continue
            if is_ignored_path(path, self.allowed_root):
                continue
            if not matches_file_pattern(path, file_pattern, self.allowed_root):
                continue
            try:
                if path.stat().st_size > self.MAX_FILE_BYTES or is_binary_file(path):
                    continue
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_number, line in enumerate(f, start=1):
                        match = pattern.search(line)
                        if not match:
                            continue
                        total += 1
                        if len(matches) < max_results:
                            matches.append(
                                SearchMatch(
                                    path=path,
                                    line=line_number,
                                    column=match.start() + 1,
                                    text=line.rstrip("\n"),
                                )
                            )
            except OSError:
                continue

        return matches, total, None

    def _parse_rg_line(self, raw_line: str, cwd: Path) -> SearchMatch | None:
        parts = raw_line.split(":", 3)
        if len(parts) != 4:
            return None
        file_part, line_part, column_part, text = parts
        try:
            return SearchMatch(
                path=(cwd / file_part).resolve(),
                line=int(line_part),
                column=int(column_part),
                text=text,
            )
        except ValueError:
            return None

    def _format_results(
        self,
        query: str,
        root: Path,
        matches: list[SearchMatch],
        total: int,
        context_lines: int,
        engine: str,
    ) -> str:
        location = relative_display_path(root, self.allowed_root)
        shown = len(matches)
        header = f"Search: {query!r}\nEngine: {engine}\nPath: {location}\nFound: {total} matches"
        if total > shown:
            header += f" (showing first {shown}, truncated)"

        if not matches:
            return f"{header}\n\nNo matches found."

        blocks = [self._format_match(match, context_lines) for match in matches]
        return f"{header}\n{'-' * 50}\n" + "\n\n".join(blocks)

    def _format_match(self, match: SearchMatch, context_lines: int) -> str:
        rel = relative_display_path(match.path, self.allowed_root)
        if context_lines <= 0:
            return f"{rel}:{match.line}:{match.column}: {match.text}"

        start = max(match.line - context_lines, 1)
        end = match.line + context_lines
        context: list[str] = []
        try:
            with match.path.open("r", encoding="utf-8", errors="replace") as f:
                for line_number, line in enumerate(f, start=1):
                    if line_number < start:
                        continue
                    if line_number > end:
                        break
                    marker = ">" if line_number == match.line else " "
                    context.append(f"{marker} {line_number:4d}: {line.rstrip()}")
        except OSError:
            context.append(f"> {match.line:4d}: {match.text}")

        return f"{rel}:{match.line}:{match.column}\n" + "\n".join(context)
