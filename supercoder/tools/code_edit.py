"""Diff-based code editing tool."""

from pathlib import Path
from .base import BaseTool, ToolDefinition


class CodeEditTool(BaseTool):
    """Edit code using diff-based operations."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code-edit",
            description='''Edit code. Operations:
- search_replace: {"filepath": "...", "operation": "search_replace", "search": "old", "replace": "new"}
- insert_after: {"filepath": "...", "operation": "insert_after", "after": "line", "content": "new"}
- replace_lines: {"filepath": "...", "operation": "replace_lines", "startLine": N, "endLine": M, "content": "new"}
- create: {"filepath": "...", "operation": "create", "content": "file content"}'''
        )
    
    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        filepath = args.get("filepath", args.get("fileName", ""))
        operation = args.get("operation", "search_replace")
        
        if not filepath:
            return "Error: filepath is required"
        
        path = Path(filepath)
        
        # Handle create operation separately
        if operation == "create":
            return self._create_file(path, args.get("content", ""))
        
        # For other operations, file must exist
        if not path.exists():
            return f"Error: File '{filepath}' not found"
        
        try:
            if operation == "search_replace":
                return self._search_replace(
                    path, 
                    args.get("search", ""), 
                    args.get("replace", "")
                )
            elif operation == "insert_after":
                return self._insert_after(
                    path,
                    args.get("after", ""),
                    args.get("content", "")
                )
            elif operation == "insert_before":
                return self._insert_before(
                    path,
                    args.get("before", ""),
                    args.get("content", "")
                )
            elif operation == "replace_lines":
                return self._replace_lines(
                    path,
                    args.get("startLine", 1),
                    args.get("endLine", 1),
                    args.get("content", "")
                )
            elif operation == "append":
                return self._append(path, args.get("content", ""))
            else:
                return f"Error: Unknown operation '{operation}'"
                
        except Exception as e:
            return f"Error: {e}"
    
    def _create_file(self, path: Path, content: str) -> str:
        """Create a new file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return f"✅ Created file: {path}"
        except Exception as e:
            return f"Error creating file: {e}"
    
    def _search_replace(self, path: Path, search: str, replace: str) -> str:
        """Search and replace text in file."""
        if not search:
            return "Error: search string is required"
        
        content = path.read_text()
        
        if search not in content:
            # Try to find similar matches
            lines = content.splitlines()
            similar = [l.strip()[:60] for l in lines if search[:20] in l][:3]
            hint = f"\nSimilar lines found:\n" + "\n".join(similar) if similar else ""
            return f"Error: Search string not found in {path}{hint}"
        
        # Count occurrences
        count = content.count(search)
        
        new_content = content.replace(search, replace)
        path.write_text(new_content)
        
        return f"✅ Replaced {count} occurrence(s) in {path}"
    
    def _insert_after(self, path: Path, after: str, content: str) -> str:
        """Insert content after a matching line."""
        if not after:
            return "Error: 'after' string is required"
        
        lines = path.read_text().splitlines()
        
        for i, line in enumerate(lines):
            if after in line:
                # Insert new content after this line
                new_lines = content.splitlines()
                lines = lines[:i + 1] + new_lines + lines[i + 1:]
                path.write_text("\n".join(lines) + "\n")
                return f"✅ Inserted {len(new_lines)} line(s) after line {i + 1} in {path}"
        
        return f"Error: Line containing '{after[:50]}' not found"
    
    def _insert_before(self, path: Path, before: str, content: str) -> str:
        """Insert content before a matching line."""
        if not before:
            return "Error: 'before' string is required"
        
        lines = path.read_text().splitlines()
        
        for i, line in enumerate(lines):
            if before in line:
                new_lines = content.splitlines()
                lines = lines[:i] + new_lines + lines[i:]
                path.write_text("\n".join(lines) + "\n")
                return f"✅ Inserted {len(new_lines)} line(s) before line {i + 1} in {path}"
        
        return f"Error: Line containing '{before[:50]}' not found"
    
    def _replace_lines(self, path: Path, start: int, end: int, content: str) -> str:
        """Replace a range of lines."""
        lines = path.read_text().splitlines()
        total = len(lines)
        
        if start < 1 or start > total:
            return f"Error: startLine {start} out of range (1-{total})"
        if end < start or end > total:
            return f"Error: endLine {end} invalid (must be {start}-{total})"
        
        new_lines = content.splitlines() if content else []
        lines = lines[:start - 1] + new_lines + lines[end:]
        path.write_text("\n".join(lines) + "\n")
        
        return f"✅ Replaced lines {start}-{end} with {len(new_lines)} line(s) in {path}"
    
    def _append(self, path: Path, content: str) -> str:
        """Append content to end of file."""
        current = path.read_text()
        if not current.endswith("\n"):
            current += "\n"
        path.write_text(current + content + "\n")
        return f"✅ Appended to {path}"
