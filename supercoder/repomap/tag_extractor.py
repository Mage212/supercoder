"""Extract code structure using tree-sitter."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    from tree_sitter_languages import get_parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False


@dataclass
class Tag:
    """A code definition tag."""
    name: str
    kind: str  # "class", "function", "def", "class_def"
    file: str
    line: int


class TagExtractor:
    """Extract tags from code files."""
    
    def __init__(self):
        self._cache = {}
    
    def extract(self, file_path: str) -> list[Tag]:
        """Extract tags from a file."""
        if not HAS_TREE_SITTER:
            return self._fallback_extract(file_path)
        
        lang = self._detect_language(file_path)
        if not lang:
            return []
        
        try:
            parser = get_parser(lang)
            content = Path(file_path).read_bytes()
            tree = parser.parse(content)
            
            tags = []
            self._visit_node(tree.root_node, file_path, tags)
            return tags
            
        except Exception:
            return self._fallback_extract(file_path)
    
    def _detect_language(self, path: str) -> str | None:
        """Detect tree-sitter language from extension."""
        ext = Path(path).suffix.lower()
        return {
            ".py": "python", 
            ".scala": "scala", 
            ".java": "java",
            ".ts": "typescript", 
            ".js": "javascript", 
            ".go": "go", 
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".rb": "ruby",
            ".php": "php",
        }.get(ext)
    
    def _visit_node(self, node, file_path: str, tags: list[Tag]) -> None:
        """Recursively visit nodes to find definitions."""
        # This is a simplified traversal. 
        # A full implementation would use proper queries for each language.
        # Here we rely on common node type names.
        
        node_type = node.type
        
        if node_type in ("function_definition", "class_definition", "method_definition"):
            name = self._get_node_name(node)
            if name:
                kind = "class" if "class" in node_type else "function"
                tags.append(Tag(name, kind, file_path, node.start_point[0] + 1))
        
        # Recurse
        for child in node.children:
            self._visit_node(child, file_path, tags)

    def _get_node_name(self, node) -> str | None:
        """Extract name from a definition node."""
        # Most languages have a 'name' field or similar child
        for child in node.children:
            if child.type == "identifier" or child.type == "name":
                return child.text.decode("utf-8")
            if child.type == "function_declarator": # C++
                return self._get_node_name(child)
        return None

    def _fallback_extract(self, file_path: str) -> list[Tag]:
        """Simple regex-based extraction as fallback."""
        import re
        tags = []
        try:
            content = Path(file_path).read_text(errors="ignore")
            lines = content.splitlines()
            
            patterns = [
                (r'^\s*(?:class|object|trait|struct)\s+(\w+)', 'class'),
                (r'^\s*(?:def|func|fn|function)\s+(\w+)', 'function'),
            ]
            
            for i, line in enumerate(lines):
                for pattern, kind in patterns:
                    match = re.search(pattern, line)
                    if match:
                        tags.append(Tag(match.group(1), kind, file_path, i + 1))
        except Exception:
            pass
        return tags
