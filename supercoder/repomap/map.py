"""Repository Map generator."""

import networkx as nx
from pathlib import Path
from .tag_extractor import TagExtractor

class RepoMap:
    """Generates a compact map of the repository structure."""
    
    def __init__(self, root: str = "."):
        self.root = Path(root).resolve()
        self.extractor = TagExtractor()
        self.graph = nx.MultiDiGraph()
        self.storage_dir = self.root / ".supercoder" / "repomap"
    
    def _ensure_storage_dir(self):
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def get_repo_map(self, max_tokens: int = 1024) -> str:
        """Generate a token-limited repository map and persist it."""
        # 1. Scan files and extract tags
        files = self._get_files()
        all_tags = []
        
        for file in files:
            tags = self.extractor.extract(str(file))
            all_tags.extend(tags)
            
            # Add to graph (simplified: just nodes for now)
            # Full implementation would add edges based on usage
            self.graph.add_node(str(file), tags=tags)
        
        # 2. Render tree
        repo_map = self._render_tree(all_tags, max_tokens)
        
        # 3. Persist to file
        try:
            self._ensure_storage_dir()
            map_file = self.storage_dir / "repo_map.txt"
            map_file.write_text(repo_map, encoding="utf-8")
        except Exception:
            pass  # Fail silently - persistence is a nice-to-have
            
        return repo_map

    
    def _get_files(self) -> list[Path]:
        """Get relevant source files."""
        # Simplified file finding
        source_extensions = {".py", ".scala", ".java", ".js", ".ts", ".go", ".rs"}
        files = []
        
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix in source_extensions:
                if any(p in path.parts for p in [".git", "venv", "__pycache__", "node_modules"]):
                    continue
                files.append(path)
        
        # Limit total files for performance
        return files[:50]
    
    def _render_tree(self, tags, max_tokens) -> str:
        """Render tags into a tree structure."""
        if not tags:
            return ""
            
        # Group by file
        by_file = {}
        for tag in tags:
            if tag.file not in by_file:
                by_file[tag.file] = []
            by_file[tag.file].append(tag)
            
        output = []
        total_tokens = 0
        
        for file, file_tags in by_file.items():
            rel_path = Path(file).relative_to(self.root)
            output.append(f"{rel_path}:")
            
            for tag in file_tags:
                line = f"  {tag.name} {tag.kind}"
                output.append(line)
                
                # Simple token check
                total_tokens += len(line) // 4
                if total_tokens > max_tokens:
                    output.append("  ...")
                    return "\n".join(output)
                    
        return "\n".join(output)
