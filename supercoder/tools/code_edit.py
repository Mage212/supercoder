"""Diff-based code editing tool."""

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..utils.atomic_writer import AtomicFileWriter
from .base import BaseTool, ToolDefinition

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager


class CodeEditTool(BaseTool):
    """Edit code using diff-based operations with atomic writes."""

    def __init__(
        self,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        allowed_root: Path | None = None,
    ):
        """Initialize with optional checkpoint manager and allowed root directory.

        Args:
            checkpoint_manager: Optional CheckpointManager for backup/rollback support
            allowed_root: If set, file paths must be within this directory (path traversal guard)
        """
        self.checkpoint = checkpoint_manager
        self.allowed_root = allowed_root

    def _safe_write(self, path: Path, content: str) -> None:
        """Write file with backup and atomic write.

        Args:
            path: Target file path
            content: Content to write
        """
        # Backup before modifying (if checkpoint is active)
        if self.checkpoint:
            self.checkpoint.backup_file(path)

        # Atomic write
        AtomicFileWriter.write(path, content)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code-edit",
            description=(
                "Edit code files. Operations: "
                "search_replace (find and replace text), "
                "insert_after (insert content after a matching line), "
                "insert_before (insert content before a matching line), "
                "replace_lines (replace a line range), "
                "append (append to end of file), "
                "create (create a new file)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to edit or create"},
                    "operation": {
                        "type": "string",
                        "description": "Edit operation to perform",
                        "enum": ["search_replace", "insert_after", "insert_before", "replace_lines", "append", "create"],
                    },
                    "search": {"type": "string", "description": "Text to find (for search_replace)"},
                    "replace": {"type": "string", "description": "Replacement text (for search_replace)"},
                    "after": {"type": "string", "description": "Line to insert after (for insert_after)"},
                    "before": {"type": "string", "description": "Line to insert before (for insert_before)"},
                    "content": {"type": "string", "description": "New content (for create, insert_after, insert_before, replace_lines, append)"},
                    "startLine": {"type": "integer", "description": "Start line number (for replace_lines)"},
                    "endLine": {"type": "integer", "description": "End line number (for replace_lines)"},
                },
                "required": ["filepath", "operation"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        filepath = args.get("filepath", args.get("fileName", ""))
        operation = args.get("operation", "search_replace")

        if not filepath:
            return "Error: filepath is required"

        path = Path(filepath)

        # Validate path stays within the allowed root (when configured)
        if self.allowed_root is not None:
            try:
                path.resolve().relative_to(self.allowed_root)
            except ValueError:
                return f"Error: Path '{filepath}' is outside the project directory"

        # Handle create operation separately
        if operation == "create":
            return self._create_file(path, args.get("content", ""))

        # For other operations, file must exist
        if not path.exists():
            return f"Error: File '{filepath}' not found"

        try:
            if operation == "search_replace":
                return self._search_replace(path, args.get("search", ""), args.get("replace", ""))
            elif operation == "insert_after":
                return self._insert_after(path, args.get("after", ""), args.get("content", ""))
            elif operation == "insert_before":
                return self._insert_before(path, args.get("before", ""), args.get("content", ""))
            elif operation == "replace_lines":
                return self._replace_lines(
                    path, args.get("startLine", 1), args.get("endLine", 1), args.get("content", "")
                )
            elif operation == "append":
                return self._append(path, args.get("content", ""))
            else:
                return f"Error: Unknown operation '{operation}'"

        except Exception as e:
            return f"Error: {e}"

    def _generate_diff(self, before: str, after: str, filepath: Path) -> str:
        """Generate unified diff between before and after content."""
        diff_lines = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(filepath),
            tofile=str(filepath),
            lineterm="",
        )
        # Join with newlines to create proper diff output
        return "\n".join(diff_lines)

    def _create_file(self, path: Path, content: str) -> str:
        """Create a new file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Track created file for rollback
            if self.checkpoint:
                self.checkpoint.track_created_file(path)

            AtomicFileWriter.write(path, content)
            # For new files, show the content as all additions
            diff = self._generate_diff("", content, path)
            return f"✅ Created file: {path}\n\n{diff}" if diff else f"✅ Created file: {path}"
        except Exception as e:
            return f"Error creating file: {e}"

    def _search_replace(self, path: Path, search: str, replace: str) -> str:
        """Search and replace text in file."""
        if not search:
            return "Error: search string is required"

        content_before = path.read_text()

        if search not in content_before:
            # Try to find similar matches
            lines = content_before.splitlines()
            similar = [line.strip()[:60] for line in lines if search[:20] in line][:3]
            hint = "\nSimilar lines found:\n" + "\n".join(similar) if similar else ""
            return f"Error: Search string not found in {path}{hint}"

        # Count occurrences
        count = content_before.count(search)

        if count > 1:
            return (
                f"Error: Search string found {count} times in {path}. "
                f"Provide a more specific search string with additional context lines to uniquely identify the target."
            )

        content_after = content_before.replace(search, replace)
        self._safe_write(path, content_after)

        # Generate diff
        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Replaced 1 occurrence in {path}\n\n{diff}"

    def _insert_after(self, path: Path, after: str, content: str) -> str:
        """Insert content after a matching line."""
        if not after:
            return "Error: 'after' string is required"

        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()

        matching = [i for i, line in enumerate(lines) if after in line]
        if not matching:
            return f"Error: Line containing '{after[:50]}' not found"
        if len(matching) > 1:
            return (
                f"Error: '{after[:50]}' found on {len(matching)} lines. "
                f"Provide a more specific string to uniquely identify the target line."
            )

        i = matching[0]
        new_lines = content.splitlines()
        lines = lines[: i + 1] + new_lines + lines[i + 1 :]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Inserted {len(new_lines)} line(s) after line {i + 1} in {path}\n\n{diff}"

    def _insert_before(self, path: Path, before: str, content: str) -> str:
        """Insert content before a matching line."""
        if not before:
            return "Error: 'before' string is required"

        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()

        matching = [i for i, line in enumerate(lines) if before in line]
        if not matching:
            return f"Error: Line containing '{before[:50]}' not found"
        if len(matching) > 1:
            return (
                f"Error: '{before[:50]}' found on {len(matching)} lines. "
                f"Provide a more specific string to uniquely identify the target line."
            )

        i = matching[0]
        new_lines = content.splitlines()
        lines = lines[:i] + new_lines + lines[i:]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Inserted {len(new_lines)} line(s) before line {i + 1} in {path}\n\n{diff}"

    def _replace_lines(self, path: Path, start: int, end: int, content: str) -> str:
        """Replace a range of lines."""
        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()
        total = len(lines)

        if start < 1 or start > total:
            return f"Error: startLine {start} out of range (1-{total})"
        if end < start or end > total:
            return f"Error: endLine {end} invalid (must be {start}-{total})"

        new_lines = content.splitlines() if content else []
        lines = lines[: start - 1] + new_lines + lines[end:]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Replaced lines {start}-{end} with {len(new_lines)} line(s) in {path}\n\n{diff}"

    def _append(self, path: Path, content: str) -> str:
        """Append content to end of file."""
        content_before = path.read_text()
        if not content_before.endswith("\n"):
            content_before_normalized = content_before + "\n"
        else:
            content_before_normalized = content_before
        content_after = content_before_normalized + content + "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Appended to {path}\n\n{diff}"
