"""Tests for Phase 3D fixes (backlog B-036): ReDoS, shell operators, streaming parser."""

from supercoder.agent.streaming_tool_parser import StreamingToolCallParser, _ToolCallBuffer
from supercoder.permissions import PermissionPolicy
from supercoder.tools.code_search import CodeSearchTool

# ── D1: ReDoS timeout in Python-fallback search ──


class TestReDoSTimeout:
    def test_redos_regex_times_out(self, tmp_path):
        """A pathological regex must be bounded by SIGALRM, not hang forever."""
        (tmp_path / "f.py").write_text("a" * 30 + "!")
        tool = CodeSearchTool(allowed_root=tmp_path)
        matches, _total, error = tool._python_search("(a+)+$", tmp_path, 10, "")
        assert error is not None
        assert "timed out" in error
        assert matches == []

    def test_invalid_regex_returns_error(self, tmp_path):
        (tmp_path / "f.py").write_text("content")
        tool = CodeSearchTool(allowed_root=tmp_path)
        _, _, error = tool._python_search("[", tmp_path, 10, "")
        assert error is not None
        assert "invalid regex" in error

    def test_normal_regex_finds_matches(self, tmp_path):
        (tmp_path / "f.py").write_text("hello world\nhello again\n")
        tool = CodeSearchTool(allowed_root=tmp_path)
        matches, total, _ = tool._python_search("hello", tmp_path, 10, "")
        assert total == 2
        assert len(matches) == 2


# ── D2: redirection operators (> <) detected as control operators ──


class TestRedirectionOperators:
    def _policy(self, tmp_path):
        return PermissionPolicy(tmp_path)

    def test_output_redirection_detected(self, tmp_path):
        assert self._policy(tmp_path)._has_shell_control_operator("cat f > /etc/passwd")

    def test_append_redirection_detected(self, tmp_path):
        assert self._policy(tmp_path)._has_shell_control_operator("echo x >> ~/.bashrc")

    def test_input_redirection_detected(self, tmp_path):
        assert self._policy(tmp_path)._has_shell_control_operator("grep s < .env")

    def test_stderr_redirection_detected(self, tmp_path):
        assert self._policy(tmp_path)._has_shell_control_operator("cmd 2>&1")

    def test_redirection_in_single_quotes_not_detected(self, tmp_path):
        assert not self._policy(tmp_path)._has_shell_control_operator("echo 'a > b'")

    def test_redirection_in_double_quotes_not_detected(self, tmp_path):
        assert not self._policy(tmp_path)._has_shell_control_operator('echo "a > b"')

    def test_plain_command_not_detected(self, tmp_path):
        assert not self._policy(tmp_path)._has_shell_control_operator("ls -la")


# ── D3: streaming parser unbalanced flag ──


class TestUnbalancedFlag:
    def test_extra_closing_brace_sets_unbalanced(self):
        buf = _ToolCallBuffer()
        for ch in '{"a":1}}':
            buf.feed_char(ch)
        assert buf.unbalanced is True

    def test_unbalanced_buffer_not_complete(self):
        buf = _ToolCallBuffer()
        for ch in '{"a":1}}':
            buf.feed_char(ch)
        assert buf.is_complete() is False

    def test_balanced_json_not_flagged(self):
        buf = _ToolCallBuffer()
        for ch in '{"a": [1, 2]}':
            buf.feed_char(ch)
        assert buf.unbalanced is False
        assert buf.is_complete() is True


# ── D4: tool_id routing over index ──


class _Fn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _Delta:
    def __init__(self, index, id="", name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _Fn(name=name, arguments=arguments)


class TestToolIdRouting:
    def test_tool_id_routes_over_index(self):
        """Deltas with a stable tool_id but changing index accumulate into one call."""
        parser = StreamingToolCallParser()
        parser.feed_chunk([_Delta(index=0, id="A", name="t", arguments='{"x":')])
        parser.feed_chunk([_Delta(index=1, id="A", arguments="1}")])
        native, _raw = parser.finalize()
        assert len(native) == 1
        assert native[0]["arguments"] == {"x": 1}

    def test_missing_id_routes_by_index(self):
        """Deltas without an id still accumulate by index (existing behavior)."""
        parser = StreamingToolCallParser()
        parser.feed_chunk([_Delta(index=0, name="t", arguments='{"x":')])
        parser.feed_chunk([_Delta(index=0, arguments="1}")])
        native, _ = parser.finalize()
        assert len(native) == 1
        assert native[0]["arguments"] == {"x": 1}
