"""Code search tool using git grep or fallback."""

import subprocess
from pathlib import Path
from .base import BaseTool, ToolDefinition


class CodeSearchTool(BaseTool):
    """Search for code patterns in the project."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code-search",
            description='Search code. Args: {"query": "pattern", "maxResults": 10, "filePattern": "*.py"}'
        )
    
    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        query = args.get("query", "")
        max_results = args.get("maxResults", 10)
        file_pattern = args.get("filePattern", "")
        
        if not query:
            return "Error: query is required"
        
        # Try git grep first (faster), fall back to grep
        try:
            result = self._git_grep(query, max_results, file_pattern)
            if result:
                return result
        except Exception:
            pass
        
        # Fallback to regular grep
        return self._fallback_grep(query, max_results)
    
    def _git_grep(self, query: str, max_results: int, file_pattern: str) -> str:
        """Search using git grep."""
        cmd = ["git", "grep", "-n", "-I", "--color=never"]
        
        # Add context lines
        cmd.extend(["-C", "2"])
        
        cmd.append(query)
        cmd.append(".")
        
        if file_pattern:
            cmd.extend(["--", file_pattern])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode not in (0, 1):  # 1 = no matches
            raise Exception(result.stderr)
        
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        
        # Limit results
        limited = lines[:max_results * 5]  # ~5 lines per result with context
        
        total_matches = len([l for l in lines if l and ":" in l and not l.startswith("--")])
        shown = min(total_matches, max_results)
        
        header = f"🔍 Search: '{query}'\n📊 Found {total_matches} matches (showing ~{shown})"
        
        if not lines or not lines[0]:
            return f"{header}\n\nNo matches found."
        
        return f"{header}\n{'─' * 50}\n" + "\n".join(limited)
    
    def _fallback_grep(self, query: str, max_results: int) -> str:
        """Fallback search using grep."""
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", "--include=*.scala", 
                 "--include=*.js", "--include=*.ts", query, "."],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            lines = result.stdout.strip().split("\n")[:max_results * 3]
            
            if not lines or not lines[0]:
                return f"🔍 No matches found for '{query}'"
            
            return f"🔍 Search: '{query}'\n{'─' * 50}\n" + "\n".join(lines)
            
        except Exception as e:
            return f"Error searching: {e}"
