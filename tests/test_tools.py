"""Test tools functionality."""

import stat

import pytest

from supercoder.context.freshness import FileFreshnessTracker
from supercoder.permissions import PermissionPolicy
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

    def test_code_search_filters_sensitive_paths_with_python_fallback(self, tmp_path, monkeypatch):
        """Python fallback never returns matches from sensitive files."""
        monkeypatch.setattr("supercoder.tools.code_search.shutil.which", lambda _name: None)
        (tmp_path / "visible.py").write_text("TOKEN = 'needle'\n")
        (tmp_path / "credentials.json").write_text('{"token": "needle"}\n')

        tool = CodeSearchTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )
        result = tool.execute('{"query": "needle", "path": "."}')

        assert "visible.py" in result
        assert "credentials.json" not in result

    def test_code_search_filters_sensitive_paths_with_rg(self, tmp_path):
        """rg results are filtered before they reach the model."""
        (tmp_path / "visible.py").write_text("TOKEN = 'needle'\n")
        (tmp_path / "credentials.json").write_text('{"token": "needle"}\n')

        tool = CodeSearchTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )
        result = tool.execute('{"query": "needle", "path": "."}')

        assert "visible.py" in result
        assert "credentials.json" not in result


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

    def test_code_edit_search_replace_preserves_crlf(self, tmp_path):
        """M4: whitespace-normalized/fuzzy match must not corrupt CRLF files.

        Regression: _find_best_match computed char offsets with
        ``sum(len(ln)+1 ...)`` on ``splitlines()``, which drops the ``\\r`` of a
        ``\\r\\n`` separator. Offsets were then 1 byte short per CRLF line, so the
        applied replacement stole a ``\\n`` from the previous line and left a
        dangling ``\\r``, corrupting adjacent lines.
        """
        tool = CodeEditTool()
        test_file = tmp_path / "crlf_test.py"
        # CRLF line endings; search uses different whitespace than the file so
        # the match goes through the whitespace_normalized path (not exact).
        test_file.write_bytes(b"line one\r\n  hello   world  \r\nline three\r\n")

        tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "hello world",
            "replace": "REPLACED"
        }}''')

        raw = test_file.read_bytes()
        # The replacement must apply...
        assert b"REPLACED" in raw
        # ...without corrupting line endings: line one keeps its \r\n, and no
        # dangling \r is left next to REPLACED (the M4 corruption signature).
        assert b"line one\r\n" in raw, f"line one CRLF corrupted: {raw!r}"
        assert b"\rREPLACED" not in raw, f"dangling \\r before REPLACED: {raw!r}"
        assert b"REPLACED\r" not in raw, f"dangling \\r after REPLACED: {raw!r}"
        # CRLF structure must stay balanced (no orphaned \r or \n).
        assert raw.count(b"\r") == raw.count(b"\n"), f"unbalanced CRLF after edit: {raw!r}"

    def test_code_edit_search_replace_crlf_no_dangling_cr(self, tmp_path):
        """A whitespace-normalized match on a CRLF line leaves no dangling \\r.

        Companion to test_code_edit_search_replace_preserves_crlf: focuses on the
        specific M4 corruption signature (a \\r stranded next to the replacement)
        when the search text differs from the file content only in whitespace.
        """
        tool = CodeEditTool()
        test_file = tmp_path / "crlf_dangling.py"
        test_file.write_bytes(b"alpha\r\n  x   y   z  \r\nomega\r\n")

        tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "x y z",
            "replace": "DONE"
        }}''')

        raw = test_file.read_bytes()
        assert b"DONE" in raw
        # No dangling \r adjacent to the replacement (the M4 signature).
        assert b"\rDONE" not in raw
        assert b"DONE\r" not in raw
        # alpha keeps its \r\n; CRLF stays balanced overall.
        assert b"alpha\r\n" in raw
        assert raw.count(b"\r") == raw.count(b"\n")

    @pytest.mark.parametrize(
        "operation,extra_args,verify",
        [
            (
                "insert_after",
                '"after": "line two", "content": "INSERTED"',
                lambda raw: b"INSERTED" in raw,
            ),
            (
                "insert_before",
                '"before": "line two", "content": "INSERTED"',
                lambda raw: b"INSERTED" in raw,
            ),
            (
                "replace_lines",
                '"startLine": 1, "endLine": 1, "content": "REPLACED"',
                lambda raw: b"REPLACED" in raw,
            ),
            (
                "append",
                '"content": "APPENDED"',
                lambda raw: b"APPENDED" in raw,
            ),
        ],
    )
    def test_code_edit_preserves_crlf_across_operations(
        self, tmp_path, operation, extra_args, verify
    ):
        """R2-3 (M4 scope): ALL edit operations must preserve CRLF line endings.

        Previously only search_replace was fixed; insert_after/insert_before/
        replace_lines/append still used read_text() with universal-newline
        translation, silently rewriting CRLF files to LF on every edit.
        """
        tool = CodeEditTool()
        test_file = tmp_path / "crlf_all.py"
        test_file.write_bytes(b"line one\r\nline two\r\nline three\r\n")

        tool.execute(
            f'''{{
            "filepath": "{test_file}",
            "operation": "{operation}",
            {extra_args}
        }}'''
        )

        raw = test_file.read_bytes()
        # The operation took effect.
        assert verify(raw), f"{operation} did not apply: {raw!r}"
        # Untouched original lines must keep their CRLF endings. For replace_lines
        # line one is replaced, so check the two lines that must survive intact.
        assert b"line two\r\n" in raw, f"{operation} flattened original CRLF: {raw!r}"
        assert b"line three\r\n" in raw, f"{operation} flattened original CRLF: {raw!r}"
        # The original CRLF separators must survive on the untouched lines. The
        # old read_text() path flattened ALL of them to LF (0 remaining).
        assert raw.count(b"\r\n") >= 2, (
            f"{operation} lost original CRLF separators on untouched lines: {raw!r}"
        )

    def test_code_edit_aborts_when_checkpoint_backup_fails(self, tmp_path):
        """Existing files are not written when checkpoint backup cannot be created."""

        class FailingCheckpoint:
            current = object()

            def backup_file(self, _path):
                return False

        tool = CodeEditTool(checkpoint_manager=FailingCheckpoint())
        test_file = tmp_path / "backup_fail.txt"
        test_file.write_text("Hello World\n")

        result = tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "Hello World",
            "replace": "Hello Universe"
        }}''')

        assert "Could not create checkpoint backup" in result
        assert test_file.read_text() == "Hello World\n"

    def test_code_edit_preview_search_replace_does_not_write(self, tmp_path):
        """Preview builds a diff without changing the file."""
        tool = CodeEditTool()
        test_file = tmp_path / "preview_replace.txt"
        test_file.write_text("Hello World\n")

        preview = tool.preview_edit(
            {
                "filepath": str(test_file),
                "operation": "search_replace",
                "search": "Hello World",
                "replace": "Hello Universe",
            }
        )

        assert preview.ok
        assert "---" in preview.diff
        assert "+++" in preview.diff
        assert "-Hello World" in preview.diff
        assert "+Hello Universe" in preview.diff
        assert test_file.read_text() == "Hello World\n"

    def test_code_edit_preview_create_does_not_write(self, tmp_path):
        """Create preview shows all additions without creating the file."""
        tool = CodeEditTool()
        test_file = tmp_path / "preview_create.txt"

        preview = tool.preview_edit(
            {
                "filepath": str(test_file),
                "operation": "create",
                "content": "created\n",
            }
        )

        assert preview.ok
        assert "+created" in preview.diff
        assert not test_file.exists()

    def test_code_edit_preview_append_does_not_write(self, tmp_path):
        """Append preview builds a diff without appending content."""
        tool = CodeEditTool()
        test_file = tmp_path / "preview_append.txt"
        test_file.write_text("first\n")

        preview = tool.preview_edit(
            {
                "filepath": str(test_file),
                "operation": "append",
                "content": "second",
            }
        )

        assert preview.ok
        assert "+second" in preview.diff
        assert test_file.read_text() == "first\n"

    def test_code_edit_preview_replace_lines_does_not_write(self, tmp_path):
        """Replace-lines preview builds a diff without replacing lines."""
        tool = CodeEditTool()
        test_file = tmp_path / "preview_lines.txt"
        test_file.write_text("one\ntwo\nthree\n")

        preview = tool.preview_edit(
            {
                "filepath": str(test_file),
                "operation": "replace_lines",
                "startLine": 2,
                "endLine": 2,
                "content": "updated",
            }
        )

        assert preview.ok
        assert "-two" in preview.diff
        assert "+updated" in preview.diff
        assert test_file.read_text() == "one\ntwo\nthree\n"

    def test_code_edit_requires_prior_read_with_freshness_tracker(self, tmp_path):
        """Freshness tracker blocks edits to files the model has not seen."""
        test_file = tmp_path / "guarded.txt"
        test_file.write_text("Hello World\n")
        tracker = FileFreshnessTracker(tmp_path)
        tool = CodeEditTool(freshness_tracker=tracker)

        result = tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "Hello World",
            "replace": "Hello Universe"
        }}''')

        assert "not read before edit" in result
        assert test_file.read_text() == "Hello World\n"

    def test_file_read_marks_file_fresh_for_code_edit(self, tmp_path):
        """A successful file-read allows a subsequent edit."""
        test_file = tmp_path / "fresh.txt"
        test_file.write_text("Hello World\n")
        tracker = FileFreshnessTracker(tmp_path)
        read_tool = FileReadTool(freshness_tracker=tracker)
        edit_tool = CodeEditTool(freshness_tracker=tracker)

        read_tool.execute(f'{{"fileName": "{test_file}"}}')
        result = edit_tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "Hello World",
            "replace": "Hello Universe"
        }}''')

        assert "Replaced" in result
        assert test_file.read_text() == "Hello Universe\n"

    def test_code_edit_blocks_file_changed_after_read(self, tmp_path):
        """External file changes invalidate a prior read snapshot."""
        test_file = tmp_path / "stale.txt"
        test_file.write_text("VALUE = 1\n")
        tracker = FileFreshnessTracker(tmp_path)
        read_tool = FileReadTool(freshness_tracker=tracker)
        edit_tool = CodeEditTool(freshness_tracker=tracker)

        read_tool.execute(f'{{"fileName": "{test_file}"}}')
        test_file.write_text("VALUE = 2\n")
        result = edit_tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "VALUE = 2",
            "replace": "VALUE = 3"
        }}''')

        assert "changed since last read" in result
        assert test_file.read_text() == "VALUE = 2\n"

    def test_successful_code_edit_keeps_file_fresh_for_next_edit(self, tmp_path):
        """SuperCoder's own writes refresh the snapshot for follow-up edits."""
        test_file = tmp_path / "sequence.txt"
        test_file.write_text("first\nsecond\n")
        tracker = FileFreshnessTracker(tmp_path)
        read_tool = FileReadTool(freshness_tracker=tracker)
        edit_tool = CodeEditTool(freshness_tracker=tracker)

        read_tool.execute(f'{{"fileName": "{test_file}"}}')
        first = edit_tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "first",
            "replace": "updated"
        }}''')
        second = edit_tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "search_replace",
            "search": "second",
            "replace": "done"
        }}''')

        assert "Replaced" in first
        assert "Replaced" in second
        assert test_file.read_text() == "updated\ndone\n"

    def test_code_edit_create_existing_file_is_blocked(self, tmp_path):
        """Create does not overwrite existing files."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("original\n")
        tool = CodeEditTool()

        result = tool.execute(f'''{{
            "filepath": "{test_file}",
            "operation": "create",
            "content": "replacement"
        }}''')

        assert "already exists" in result
        assert test_file.read_text() == "original\n"

    def test_code_edit_denies_sensitive_path(self, tmp_path):
        """Sensitive files cannot be created or edited."""
        tool = CodeEditTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )
        target = tmp_path / ".env"

        result = tool.execute(f'''{{
            "filepath": "{target}",
            "operation": "create",
            "content": "TOKEN=secret"
        }}''')

        assert "Permission denied" in result
        assert not target.exists()

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

    def test_file_read_denies_sensitive_path(self, tmp_path):
        """Sensitive files are blocked before content is read."""
        secret = tmp_path / ".env"
        secret.write_text("TOKEN=secret\n")
        tool = FileReadTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )

        result = tool.execute('{"fileName": ".env"}')

        assert "Permission denied" in result
        assert "TOKEN=secret" not in result

    def test_file_read_allows_env_example(self, tmp_path):
        """.env.example is safe documentation and remains readable."""
        sample = tmp_path / ".env.example"
        sample.write_text("TOKEN=\n")
        tool = FileReadTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )

        result = tool.execute('{"fileName": ".env.example"}')

        assert "TOKEN=" in result


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

    def test_glob_filters_sensitive_paths(self, tmp_path):
        """Glob returns paths only, but still hides sensitive path names."""
        (tmp_path / "visible.txt").write_text("ok\n")
        (tmp_path / "credentials.json").write_text("{}\n")

        tool = GlobTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )
        result = tool.execute('{"pattern": "*"}')

        assert "visible.txt" in result
        assert "credentials.json" not in result

    def test_project_structure_filters_sensitive_paths(self, tmp_path):
        """Project tree output should not reveal sensitive path names."""
        (tmp_path / "README.md").write_text("# ok\n")
        (tmp_path / "credentials.json").write_text("{}\n")

        tool = ProjectStructureTool(
            allowed_root=tmp_path,
            permission_policy=PermissionPolicy(tmp_path),
        )
        result = tool.execute('{"path": ".", "maxDepth": 2, "maxFiles": 10}')

        assert "README.md" in result
        assert "credentials.json" not in result


class TestGlobPatternSafety:
    """L2 (code-review-2026-06-23): glob must return a clear error for absolute
    or '..'-containing patterns instead of crashing with an unhandled
    NotImplementedError from Path.rglob. (code_search via rg does not crash —
    it returns an empty result — so this guard is glob-only.)"""

    def test_glob_absolute_pattern_returns_error_not_crash(self, tmp_path):
        tool = GlobTool(allowed_root=tmp_path)
        result = tool.execute('{"pattern": "/etc/hosts"}')
        assert result.startswith("Error")
        assert "absolute" in result.lower() or "relative" in result.lower()

    def test_glob_dotdot_pattern_returns_error_not_crash(self, tmp_path):
        tool = GlobTool(allowed_root=tmp_path)
        result = tool.execute('{"pattern": "../secret.txt"}')
        assert result.startswith("Error")


class TestGlobNormalUseStillWorks:
    """Guard: the safety check must not break valid relative globs."""

    def test_glob_relative_pattern_finds_file(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "mod.py").write_text("y = 2\n")
        tool = GlobTool(allowed_root=tmp_path)
        result = tool.execute('{"pattern": "**/*.py"}')
        assert "mod.py" in result
        assert result.startswith("Glob:")
