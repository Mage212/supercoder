"""Tests for the unified renderers module (Task 1.2).

Smoke tests: construct each renderer with sample data, print to a recording
console, and assert the output contains expected substrings. No pixel-level
assertions (brittle); we verify shape, not exact spacing.
"""

from rich.console import Console

from supercoder.ui import render


def _render_to_text(renderable, width: int = 100) -> str:
    """Render a rich renderable to plain text via a recording console."""
    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


class TestUserMessage:
    def test_renders_without_error(self):
        result = render.render_user_message("hello world")
        text = _render_to_text(result)
        assert "hello world" in text
        assert "You" in text

    def test_live_and_history_styles_are_identical(self):
        """Fixes the inconsistency between repl.py:237 (live) and :795 (restored).

        Both `live=True` and `live=False` must produce the same rendering so
        the scrollback and the restored session history match.
        """
        live_text = _render_to_text(render.render_user_message("hi", live=True))
        hist_text = _render_to_text(render.render_user_message("hi", live=False))
        assert live_text == hist_text


class TestAssistantMessage:
    def test_renders_content(self):
        result = render.render_assistant_message("Here is some **bold** text.")
        text = _render_to_text(result)
        assert "bold" in text  # markdown word appears
        assert "SuperCoder" in text  # header brand

    def test_model_tag_in_header(self):
        result = render.render_assistant_message("ok", model="openai/gpt-4o")
        text = _render_to_text(result)
        assert "gpt-4o" in text

    def test_elapsed_in_header(self):
        result = render.render_assistant_message("ok", elapsed_s=12.3)
        text = _render_to_text(result)
        assert "12s" in text

    def test_interrupted_marker(self):
        result = render.render_assistant_message("ok", interrupted=True)
        text = _render_to_text(result)
        assert "interrupted" in text

    def test_empty_content_does_not_crash(self):
        result = render.render_assistant_message("")
        text = _render_to_text(result)
        assert "SuperCoder" in text


class TestReasoning:
    def test_renders_content(self):
        result = render.render_reasoning("Let me think about this.")
        text = _render_to_text(result)
        assert "Let me think about this." in text
        assert "Reasoning" in text

    def test_streaming_suffix(self):
        result = render.render_reasoning("thinking...", streaming=True)
        text = _render_to_text(result)
        assert "thinking" in text.lower()


class TestToolCallCompact:
    def test_uses_tool_icon(self):
        result = render.render_tool_call_compact("file-read", {"fileName": "src/main.py"})
        text = _render_to_text(result)
        assert "📖" in text
        assert "file-read" in text
        assert "src/main.py" in text

    def test_unknown_tool_uses_default_icon(self):
        result = render.render_tool_call_compact("mcp__unknown__tool", {})
        text = _render_to_text(result)
        assert "⚙" in text

    def test_no_args_still_renders(self):
        result = render.render_tool_call_compact("project-structure", {})
        text = _render_to_text(result)
        assert "project-structure" in text


class TestToolCallExpanded:
    def test_renders_json(self):
        result = render.render_tool_call_expanded(
            "code-edit", {"path": "a.py", "operation": "create"}
        )
        text = _render_to_text(result)
        assert "code-edit" in text
        assert "a.py" in text
        assert "create" in text


class TestToolResult:
    def test_compact_one_line(self):
        result = render.render_tool_result("read 3 files", policy="compact")
        text = _render_to_text(result)
        assert "read 3 files" in text
        assert "✔" in text

    def test_hidden_policy(self):
        result = render.render_tool_result("hidden result", policy="hidden")
        text = _render_to_text(result)
        assert "hidden result" in text

    def test_error_policy(self):
        result = render.render_tool_result("File not found", policy="error", name="file-read")
        text = _render_to_text(result)
        assert "File not found" in text
        assert "file-read" in text

    def test_expanded_with_diff(self):
        diff = "--- a.py\n+++ b.py\n@@ -1 +1 @@\n-old\n+new\n"
        result = render.render_tool_result(
            "edited", policy="expanded", name="code-edit", diff_text=diff
        )
        text = _render_to_text(result)
        assert "code-edit" in text

    def test_expanded_with_code(self):
        result = render.render_tool_result(
            "ran command", policy="expanded", name="command-exec", code_text="ls -la", lexer="bash"
        )
        text = _render_to_text(result)
        assert "command-exec" in text


class TestContextBar:
    def test_zero_usage(self):
        result = render.render_context_bar(0, 10000, width=10)
        text = _render_to_text(result)
        assert "0/10,000 tokens" in text

    def test_full_usage(self):
        result = render.render_context_bar(10000, 10000, width=10)
        text = _render_to_text(result)
        assert "10,000/10,000 tokens" in text

    def test_partial_usage(self):
        result = render.render_context_bar(5000, 10000, width=10)
        text = _render_to_text(result)
        assert "5,000/10,000 tokens" in text

    def test_width_argument(self):
        # Footer width vs /stats width produce different bar lengths.
        footer = render.render_context_bar(50, 100, width=10)
        stats = render.render_context_bar(50, 100, width=20)
        assert _render_to_text(footer) != _render_to_text(stats)


class TestModePrompt:
    def test_known_mode(self):
        result = render.render_mode_prompt("CODE", "gpt-4o")
        text = _render_to_text(result)
        assert "gpt-4o" in text

    def test_unknown_mode_falls_back(self):
        # Should not crash on an unknown mode key.
        result = render.render_mode_prompt("UNKNOWN", "gpt-4o")
        text = _render_to_text(result)
        assert "gpt-4o" in text
