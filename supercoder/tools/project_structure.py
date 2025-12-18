"""Project structure tool."""

from pathlib import Path
from .base import BaseTool, ToolDefinition


# Directories to ignore
IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", 
    "target", "build", "dist", ".idea", ".vscode", ".pytest_cache",
    "egg-info", ".eggs", ".mypy_cache", ".ruff_cache"
}

# File patterns to ignore
IGNORE_PATTERNS = {".pyc", ".pyo", ".so", ".dylib", ".class", ".DS_Store"}


class ProjectStructureTool(BaseTool):
    """Show project directory structure."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="project-structure",
            description='Show project structure. Args: {"maxDepth": 3, "maxFiles": 50, "path": "."}'
        )
    
    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        max_depth = args.get("maxDepth", 3)
        max_files = args.get("maxFiles", 50)
        root_path = args.get("path", ".")
        
        root = Path(root_path)
        if not root.exists():
            return f"Error: Path '{root_path}' not found"
        
        output = ["📁 Project Structure:"]
        counter = {"files": 0, "dirs": 0}
        
        self._build_tree(root, output, 0, max_depth, max_files, counter)
        
        output.append(f"\n📊 Total: {counter['dirs']} directories, {counter['files']} files shown")
        
        return "\n".join(output)
    
    def _build_tree(
        self, 
        path: Path, 
        output: list, 
        depth: int, 
        max_depth: int,
        max_files: int,
        counter: dict
    ) -> None:
        """Recursively build directory tree."""
        if depth >= max_depth or counter["files"] >= max_files:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        
        for item in items:
            # Skip ignored items
            if item.name in IGNORE_DIRS:
                continue
            if item.suffix in IGNORE_PATTERNS:
                continue
            if item.name.startswith(".") and item.name != ".env.example":
                continue
            
            prefix = "  " * depth
            
            if item.is_dir():
                output.append(f"{prefix}📁 {item.name}/")
                counter["dirs"] += 1
                self._build_tree(item, output, depth + 1, max_depth, max_files, counter)
            else:
                if counter["files"] < max_files:
                    size = self._format_size(item.stat().st_size)
                    output.append(f"{prefix}📄 {item.name} ({size})")
                    counter["files"] += 1
    
    def _format_size(self, size: int) -> str:
        """Format file size in human readable format."""
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size // 1024}KB"
        else:
            return f"{size // (1024 * 1024)}MB"
