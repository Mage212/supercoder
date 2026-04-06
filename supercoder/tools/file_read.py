"""Smart file reading tool with token limits."""

from pathlib import Path

from .base import BaseTool, ToolDefinition


class FileReadTool(BaseTool):
    """Read files with optional line range limits."""

    def __init__(self, allowed_root: Path | None = None):
        self.allowed_root = allowed_root

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file-read",
            description="Read a file with optional line range.",
            parameters={
                "type": "object",
                "properties": {
                    "fileName": {"type": "string", "description": "Path to the file to read"},
                    "startLine": {"type": "integer", "description": "Start line (1-indexed)", "default": 1},
                    "endLine": {"type": "integer", "description": "End line (1-indexed, inclusive)"},
                    "maxLines": {"type": "integer", "description": "Maximum lines to return", "default": 200},
                },
                "required": ["fileName"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        file_name = args.get("fileName", args.get("filename", ""))
        start_line = args.get("startLine", 1)
        end_line = args.get("endLine")
        max_lines = args.get("maxLines", 200)

        if not file_name:
            return "Error: fileName is required"

        path = Path(file_name)

        if self.allowed_root is not None:
            try:
                path.resolve().relative_to(self.allowed_root)
            except ValueError:
                return f"Error: Path '{file_name}' is outside the project directory"

        if not path.exists():
            return f"Error: File '{file_name}' not found"

        if path.is_dir():
            return f"Error: '{file_name}' is a directory, not a file"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total = len(lines)

            # Apply line range
            start = max(1, start_line) - 1
            end = min(end_line, total) if end_line else min(start + max_lines, total)

            selected = lines[start:end]

            # Format with line numbers
            formatted = "\n".join(f"{start + i + 1:4d}: {line}" for i, line in enumerate(selected))

            # Build result
            header = f"📂 File: {file_name}"
            info = f"📊 Lines {start + 1}-{end} of {total}"

            if end < total:
                info += " (use startLine/endLine to see more)"

            return f"{header}\n{info}\n{'─' * 50}\n{formatted}"

        except Exception as e:
            return f"Error reading file: {e}"
