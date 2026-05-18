"""Test RepoMap functionality."""

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

    def test_repomap_ignores_runtime_and_environment_dirs(self, tmp_path):
        """RepoMap should not inject runtime artifacts or virtualenv files."""
        source_file = tmp_path / "app.py"
        source_file.write_text("def visible_app(): pass")

        checkpoint_dir = tmp_path / ".supercoder" / "checkpoints" / "old"
        checkpoint_dir.mkdir(parents=True)
        checkpoint_file = checkpoint_dir / "snapshot.py"
        checkpoint_file.write_text("def checkpoint_noise(): pass")

        venv_dir = tmp_path / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        venv_file = venv_dir / "dependency.py"
        venv_file.write_text("def dependency_noise(): pass")

        repo_map = RepoMap(tmp_path)
        files = repo_map._get_files()

        assert source_file in files
        assert checkpoint_file not in files
        assert venv_file not in files
