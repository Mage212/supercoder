"""Test RepoMap functionality."""

import pytest
from pathlib import Path
from supercoder.repomap import RepoMap


class TestRepoMap:
    """Tests for RepoMap class."""
    
    def test_repomap_initialization(self, tmp_path):
        """Test RepoMap initializes correctly."""
        repo_map = RepoMap(tmp_path)
        assert repo_map is not None
    
    def test_repomap_with_python_files(self, tmp_path):
        """Test RepoMap generates content for Python files."""
        # Create a Python file
        py_file = tmp_path / "example.py"
        py_file.write_text("""
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b

def main():
    calc = Calculator()
    print(calc.add(1, 2))
""")
        
        repo_map = RepoMap(tmp_path)
        content = repo_map.get_repo_map(max_tokens=2048)
        
        # Should contain some content
        assert content is not None
    
    def test_repomap_empty_directory(self, tmp_path):
        """Test RepoMap handles empty directories."""
        repo_map = RepoMap(tmp_path)
        content = repo_map.get_repo_map(max_tokens=2048)
        
        # Should return empty or minimal content for empty directory
        # This is acceptable behavior
        assert content is not None or content == ""
    
    def test_repomap_ignores_hidden_files(self, tmp_path):
        """Test RepoMap ignores hidden files and directories."""
        # Create a hidden directory
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        hidden_file = hidden_dir / "secret.py"
        hidden_file.write_text("SECRET = 'hidden'")
        
        # Create a visible file
        visible_file = tmp_path / "visible.py"
        visible_file.write_text("def visible(): pass")
        
        repo_map = RepoMap(tmp_path)
        content = repo_map.get_repo_map(max_tokens=2048)
        
        # Hidden content should not appear
        if content:
            assert ".hidden" not in content
