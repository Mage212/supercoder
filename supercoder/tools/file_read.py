"""Smart file reading tool with token limits."""

from pathlib import Path
from .base import BaseTool, ToolDefinition


class FileReadTool(BaseTool):
    """Read files with optional line range limits."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file-read",
            description='Read a file. Args: {"fileName": "path", "startLine": N, "endLine": M, "maxLines": 100}'
        )
    
    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        file_name = args.get("fileName", args.get("filename", ""))
        start_line = args.get("startLine", 1)
        end_line = args.get("endLine")
        max_lines = args.get("maxLines", 200)
        
        if not file_name:
            return "Error: fileName is required"
        
        path = Path(file_name)
        if not path.exists():
            return f"Error: File '{file_name}' not found"
        
        if path.is_dir():
            return f"Error: '{file_name}' is a directory, not a file"
        
        try:
            content = path.read_text(encoding='utf-8', errors='replace')
            lines = content.splitlines()
            total = len(lines)
            
            # Apply line range
            start = max(1, start_line) - 1
            if end_line:
                end = min(end_line, total)
            else:
                end = min(start + max_lines, total)
            
            selected = lines[start:end]
            
            # Format with line numbers
            formatted = "\n".join(
                f"{start + i + 1:4d}: {line}" 
                for i, line in enumerate(selected)
            )
            
            # Build result
            header = f"📂 File: {file_name}"
            info = f"📊 Lines {start + 1}-{end} of {total}"
            
            if end < total:
                info += f" (use startLine/endLine to see more)"
            
            return f"{header}\n{info}\n{'─' * 50}\n{formatted}"
            
        except Exception as e:
            return f"Error reading file: {e}"
