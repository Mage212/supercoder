"""Test tools functionality."""

import stat

from supercoder.tools import (
    CodeEditTool,
    CodeSearchTool,
    FileReadTool,
    GlobTool,
    ProjectStructureTool,
)
from supercoder.utils.atomic_writer import AtomicFileWriter


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

    def test_code_search_path_pattern_and_context(self, tmp_path):
        """Search respects path, file pattern, result cap, and context lines."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "app.py").write_text("def alpha():\n    return 'needle'\n")
        (src / "app.txt").write_text("needle in text\n")

        tool = CodeSearchTool(allowed_root=tmp_path)
        result = tool.execute(
            f'{{"query": "needle", "path": "{src}", "filePattern": "*.py", '
            '"maxResults": 5, "contextLines": 1}'
        )

        assert "src/app.py" in result
        assert "src/app.txt" not in result
        assert "return 'needle'" in result
        assert "Engine:" in result

    def test_code_search_python_fallback(self, tmp_path, monkeypatch):
        """Python fallback works when ripgrep is unavailable."""
        monkeypatch.setattr("supercoder.tools.code_search.shutil.which", lambda _name: None)
        test_file = tmp_path / "fallback.py"
        test_file.write_text("class Fallback:\n    pass\n")

        tool = CodeSearchTool(allowed_root=tmp_path)
        result = tool.execute(f'{{"query": "Fallback", "path": "{tmp_path}"}}')

        assert "fallback.py" in result
        assert "Engine: python" in result


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
        test_file.write_text("Hello World\nGoodbye Earth\n")

        tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "Hello World",
            "replace": "Hello Universe"
        }}''')

        content = test_file.read_text()
        assert "Hello Universe" in content

    def test_atomic_writer_preserves_existing_file_permissions(self, tmp_path):
        """Atomic replacement preserves mode bits such as executable files."""
        test_file = tmp_path / "script.sh"
        test_file.write_text("#!/bin/sh\necho old\n")
        test_file.chmod(0o755)

        AtomicFileWriter.write(test_file, "#!/bin/sh\necho new\n")

        mode = stat.S_IMODE(test_file.stat().st_mode)
        assert mode == 0o755
        assert test_file.read_text() == "#!/bin/sh\necho new\n"


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

    def test_file_read_binary_guard(self, tmp_path):
        """Binary files are not read as replacement-character text."""
        tool = FileReadTool(allowed_root=tmp_path)
        test_file = tmp_path / "image.bin"
        test_file.write_bytes(b"abc\x00def")

        result = tool.execute(f'{{"fileName": "{test_file}"}}')

        assert "binary" in result
        assert "refusing" in result

    def test_file_read_max_bytes(self, tmp_path):
        """Large text output is capped by maxBytes."""
        tool = FileReadTool(allowed_root=tmp_path)
        test_file = tmp_path / "large.txt"
        test_file.write_text(("a" * 100) + "\nsecond line\n")

        result = tool.execute(f'{{"fileName": "{test_file}", "maxBytes": 20}}')

        assert "maxBytes=20" in result
        assert "second line" not in result

    def test_file_read_missing_file_suggestions(self, tmp_path):
        """Missing files include nearby path suggestions."""
        tool = FileReadTool(allowed_root=tmp_path)
        (tmp_path / "readable.txt").write_text("ok\n")

        result = tool.execute(f'{{"fileName": "{tmp_path}/readble.txt"}}')

        assert "Did you mean" in result
        assert "readable.txt" in result


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


class TestGlobTool:
    """Tests for GlobTool."""

    def test_glob_initialization(self):
        """Test GlobTool initializes correctly."""
        tool = GlobTool()
        assert tool.definition.name == "glob"

    def test_glob_finds_files_without_hidden_junk(self, tmp_path):
        """Glob returns matching paths and skips hidden/cache directories."""
        src = tmp_path / "src"
        src.mkdir()
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (src / "app.py").write_text("print('ok')\n")
        (src / "note.txt").write_text("content mentions app.py but should not matter\n")
        (hidden / "secret.py").write_text("print('hidden')\n")

        tool = GlobTool(allowed_root=tmp_path)
        result = tool.execute('{"pattern": "**/*.py"}')

        assert "src/app.py" in result
        assert "secret.py" not in result
        assert "content mentions" not in result

    def test_glob_rejects_outside_root(self, tmp_path):
        """Glob path validation uses allowed_root."""
        tool = GlobTool(allowed_root=tmp_path)

        result = tool.execute('{"pattern": "*.py", "path": "/"}')

        assert "outside the project directory" in result
