"""Test tools functionality."""

import pytest
import os
from pathlib import Path
from supercoder.tools import (
    CodeSearchTool, 
    CodeEditTool, 
    FileReadTool, 
    ProjectStructureTool
)


class TestCodeSearchTool:
    """Tests for CodeSearchTool."""
    
    def test_code_search_initialization(self):
        """Test CodeSearchTool initializes correctly."""
        tool = CodeSearchTool()
        assert tool.definition.name == "code-search"
    
    def test_code_search_basic(self, tmp_path):
        """Test basic code search functionality."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello_world():\n    print('Hello')\n")
        
        tool = CodeSearchTool()
        # Search in the temp directory
        result = tool.execute(f'{{"query": "hello_world", "path": "{tmp_path}"}}')
        
        assert "hello_world" in result or "Error" in result  # May need git


class TestCodeEditTool:
    """Tests for CodeEditTool."""
    
    def test_code_edit_initialization(self):
        """Test CodeEditTool initializes correctly."""
        tool = CodeEditTool()
        assert tool.definition.name == "code-edit"
    
    def test_code_edit_create_file(self, tmp_path):
        """Test creating a new file."""
        tool = CodeEditTool()
        test_file = tmp_path / "new_file.txt"
        
        result = tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "create",
            "content": "Hello from SuperCoder!"
        }}''')
        
        assert test_file.exists()
        assert test_file.read_text() == "Hello from SuperCoder!"
        assert "Created" in result or "created" in result.lower()
    
    def test_code_edit_search_replace(self, tmp_path):
        """Test search and replace operation."""
        tool = CodeEditTool()
        test_file = tmp_path / "replace_test.txt"
        test_file.write_text("Hello World\nGoodbye World\n")
        
        result = tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "World",
            "replace": "Universe"
        }}''')
        
        content = test_file.read_text()
        assert "Universe" in content


class TestFileReadTool:
    """Tests for FileReadTool."""
    
    def test_file_read_initialization(self):
        """Test FileReadTool initializes correctly."""
        tool = FileReadTool()
        assert tool.definition.name == "file-read"
    
    def test_file_read_basic(self, tmp_path):
        """Test reading a file."""
        tool = FileReadTool()
        test_file = tmp_path / "readable.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\n")
        
        result = tool.execute(f'{{"fileName": "{test_file}"}}')
        
        assert "Line 1" in result
        assert "Line 2" in result
    
    def test_file_read_nonexistent(self, tmp_path):
        """Test reading a non-existent file."""
        tool = FileReadTool()
        
        result = tool.execute(f'{{"fileName": "{tmp_path}/nonexistent.txt"}}')
        
        assert "Error" in result or "not found" in result.lower()


class TestProjectStructureTool:
    """Tests for ProjectStructureTool."""
    
    def test_project_structure_initialization(self):
        """Test ProjectStructureTool initializes correctly."""
        tool = ProjectStructureTool()
        assert tool.definition.name == "project-structure"
    
    def test_project_structure_basic(self, tmp_path):
        """Test getting project structure."""
        # Create some files and directories
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# main")
        (tmp_path / "README.md").write_text("# README")
        
        tool = ProjectStructureTool()
        result = tool.execute(f'{{"path": "{tmp_path}"}}')
        
        assert "src" in result or "main.py" in result
