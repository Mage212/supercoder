"""Diff-based code editing tool."""

import difflib
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
    
    def _generate_diff(self, before: str, after: str, filepath: Path) -> str:
        """Generate unified diff between before and after content."""
        diff_lines = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(filepath),
            tofile=str(filepath),
            lineterm=""
        )
        # Join with newlines to create proper diff output
        return "\n".join(diff_lines)
    
    def _create_file(self, path: Path, content: str) -> str:
        """Create a new file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
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
            similar = [l.strip()[:60] for l in lines if search[:20] in l][:3]
            hint = f"\nSimilar lines found:\n" + "\n".join(similar) if similar else ""
            return f"Error: Search string not found in {path}{hint}"
        
        # Count occurrences
        count = content_before.count(search)
        
        content_after = content_before.replace(search, replace)
        path.write_text(content_after)
        
        # Generate diff
        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Replaced {count} occurrence(s) in {path}\n\n{diff}"
    
    def _insert_after(self, path: Path, after: str, content: str) -> str:
        """Insert content after a matching line."""
        if not after:
            return "Error: 'after' string is required"
        
        content_before = path.read_text()
        lines = content_before.splitlines()
        
        for i, line in enumerate(lines):
            if after in line:
                # Insert new content after this line
                new_lines = content.splitlines()
                lines = lines[:i + 1] + new_lines + lines[i + 1:]
                content_after = "\n".join(lines) + "\n"
                path.write_text(content_after)
                
                diff = self._generate_diff(content_before, content_after, path)
                return f"✅ Inserted {len(new_lines)} line(s) after line {i + 1} in {path}\n\n{diff}"
        
        return f"Error: Line containing '{after[:50]}' not found"
    
    def _insert_before(self, path: Path, before: str, content: str) -> str:
        """Insert content before a matching line."""
        if not before:
            return "Error: 'before' string is required"
        
        content_before = path.read_text()
        lines = content_before.splitlines()
        
        for i, line in enumerate(lines):
            if before in line:
                new_lines = content.splitlines()
                lines = lines[:i] + new_lines + lines[i:]
                content_after = "\n".join(lines) + "\n"
                path.write_text(content_after)
                
                diff = self._generate_diff(content_before, content_after, path)
                return f"✅ Inserted {len(new_lines)} line(s) before line {i + 1} in {path}\n\n{diff}"
        
        return f"Error: Line containing '{before[:50]}' not found"
    
    def _replace_lines(self, path: Path, start: int, end: int, content: str) -> str:
        """Replace a range of lines."""
        content_before = path.read_text()
        lines = content_before.splitlines()
        total = len(lines)
        
        if start < 1 or start > total:
            return f"Error: startLine {start} out of range (1-{total})"
        if end < start or end > total:
            return f"Error: endLine {end} invalid (must be {start}-{total})"
        
        new_lines = content.splitlines() if content else []
        lines = lines[:start - 1] + new_lines + lines[end:]
        content_after = "\n".join(lines) + "\n"
        path.write_text(content_after)
        
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
        path.write_text(content_after)
        
        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Appended to {path}\n\n{diff}"

