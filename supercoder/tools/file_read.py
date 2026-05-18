"""Smart file reading tool with token and byte limits."""

from __future__ import annotations

from pathlib import Path

from ..context.freshness import FileFreshnessTracker
from ..logging import get_logger
from ..permissions import PermissionPolicy
from .base import BaseTool, ToolDefinition
from .tool_utils import (
    find_similar_files,
    format_size,
    is_binary_file,
    relative_display_path,
    resolve_within_root,
)


class FileReadTool(BaseTool):
    """Read files with optional line range limits."""

    DEFAULT_MAX_BYTES = 64_000
    HARD_MAX_BYTES = 256_000
    HARD_MAX_LINES = 5000

    def __init__(
        self,
        allowed_root: Path | None = None,
        permission_policy: PermissionPolicy | None = None,
        freshness_tracker: FileFreshnessTracker | None = None,
    ):
        self.allowed_root = allowed_root
        self.permission_policy = permission_policy
        self.freshness_tracker = freshness_tracker

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file-read",
            description="Read a file with optional line range.",
            parameters={
                "type": "object",
                "properties": {
                    "fileName": {"type": "string", "description": "Path to the file to read"},
                    "startLine": {
                        "type": "integer",
                        "description": "Start line (1-indexed)",
                        "default": 1,
                    },
                    "endLine": {
                        "type": "integer",
                        "description": "End line (1-indexed, inclusive)",
                    },
                    "maxLines": {
                        "type": "integer",
                        "description": "Maximum lines to return",
                        "default": 200,
                    },
                    "maxBytes": {
                        "type": "integer",
                        "description": "Maximum bytes of text to return",
                        "default": self.DEFAULT_MAX_BYTES,
                    },
                },
                "required": ["fileName"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        file_name = args.get("fileName", args.get("filename", ""))
        start_line = max(int(args.get("startLine", 1)), 1)
        end_line_arg = args.get("endLine")
        end_line = max(int(end_line_arg), start_line) if end_line_arg else None
        max_lines = min(max(int(args.get("maxLines", 200)), 1), self.HARD_MAX_LINES)
        max_bytes = min(
            max(int(args.get("maxBytes", self.DEFAULT_MAX_BYTES)), 1), self.HARD_MAX_BYTES
        )

        if not file_name:
            return "Error: fileName is required"

        path, error = resolve_within_root(file_name, self.allowed_root)
        if error:
            return error
        if path is None:
            return "Error: Invalid file path"

        if self.permission_policy:
            decision = self.permission_policy.check_path(path, "read")
            if decision.denied:
                get_logger().log_permission_decision(
                    tool_name=self.definition.name,
                    subject=self.permission_policy.relative_path(path),
                    action=decision.action.value,
                    reason=decision.reason,
                    source=decision.source,
                    matched_rule=decision.matched_rule,
                )
                return self.permission_policy.format_denial(file_name, decision)

        if not path.exists():
            suggestions = find_similar_files(path, self.allowed_root)
            message = f"Error: File '{file_name}' not found"
            if suggestions:
                message += "\n\nDid you mean:\n" + "\n".join(f"- {s}" for s in suggestions)
            return message

        if path.is_dir():
            return f"Error: '{file_name}' is a directory, not a file"

        if is_binary_file(path):
            size = format_size(path.stat().st_size)
            return f"Error: File '{file_name}' appears to be binary ({size}); refusing to read as text."

        try:
            selected: list[tuple[int, str]] = []
            used_bytes = 0
            stop_reason = ""
            last_seen = 0

            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line_number, line in enumerate(f, start=1):
                    last_seen = line_number
                    if line_number < start_line:
                        continue
                    if end_line is not None and line_number > end_line:
                        break
                    if len(selected) >= max_lines:
                        stop_reason = f"maxLines={max_lines}"
                        break

                    encoded_len = len(line.encode("utf-8", errors="replace"))
                    if selected and used_bytes + encoded_len > max_bytes:
                        stop_reason = f"maxBytes={max_bytes}"
                        break
                    if not selected and encoded_len > max_bytes:
                        line = line.encode("utf-8", errors="replace")[:max_bytes].decode(
                            "utf-8", errors="replace"
                        )
                        encoded_len = len(line.encode("utf-8", errors="replace"))
                        stop_reason = f"maxBytes={max_bytes}"

                    selected.append((line_number, line.rstrip("\n")))
                    used_bytes += encoded_len

                    if stop_reason:
                        break

            display_name = relative_display_path(path, self.allowed_root)
            size_bytes = path.stat().st_size
            file_size = format_size(size_bytes)

            if not selected:
                if size_bytes == 0 and self.freshness_tracker:
                    self.freshness_tracker.mark_read(path, source=self.definition.name)
                return (
                    f"File: {display_name}\n"
                    f"Size: {file_size}\n"
                    f"Lines: no content selected from startLine={start_line}"
                )

            first_line = selected[0][0]
            last_line = selected[-1][0]
            formatted = "\n".join(f"{number:4d}: {line}" for number, line in selected)

            info = f"Lines {first_line}-{last_line}; bytes returned {used_bytes}; file size {file_size}"
            if stop_reason:
                info += f" (truncated by {stop_reason}; use startLine/endLine to see more)"
            elif end_line is None and last_seen >= last_line:
                info += " (use startLine/endLine to see more if needed)"

            if self.freshness_tracker:
                self.freshness_tracker.mark_read(path, source=self.definition.name)

            return f"File: {display_name}\n{info}\n{'-' * 50}\n{formatted}"

        except Exception as e:
            return f"Error reading file: {e}"
