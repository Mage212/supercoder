"""Tests for streaming display logic in _handle_chat.

These tests simulate the paragraph-tracking and offset logic used inside
_handle_chat() to prevent regressions in the streaming display pipeline.
"""
import re

# ──────────────────────────────────────────────────────────────────────────────
# Helpers that replicate the logic in repl.py without instantiating SuperCoderREPL
# ──────────────────────────────────────────────────────────────────────────────

def simulate_streaming(tokens: list[str], tool_calling_type: str = "supercoder"):
    """Simulate the full streaming display pipeline.

    Runs tokens through StreamingDisplayBuffer → print_new_paragraphs, then
    calls stop_streaming, and returns the concatenated text that would have
    been passed to console.print(Markdown(...)).
    """
    from supercoder.streaming_buffer import StreamingDisplayBuffer

    buf = StreamingDisplayBuffer(tool_calling_type)
    accumulated = ""
    printed_up_to = 0
    printed_parts = []

    def print_new_paragraphs():
        nonlocal printed_up_to
        unprinted = accumulated[printed_up_to:]
        if not unprinted:
            return
        boundary = unprinted.rfind("\n\n")
        if boundary > 0:
            to_print = unprinted[:boundary]
            if to_print.strip():
                printed_parts.append(to_print)
            printed_up_to += boundary + 2
        elif len(unprinted) >= 300:
            last_nl = unprinted.rfind("\n")
            if last_nl > 0:
                to_print = unprinted[:last_nl]
                if to_print.strip():
                    printed_parts.append(to_print)
                printed_up_to += last_nl + 1

    for tok in tokens:
        chunk = buf.add(tok)
        if chunk:
            accumulated += chunk
            print_new_paragraphs()

    # stop_streaming
    remaining = buf.flush()
    accumulated += remaining
    unprinted = accumulated[printed_up_to:]
    if unprinted.strip():
        printed_parts.append(unprinted)

    return printed_parts


def simulate_filter(text: str) -> str:
    """Replicate _filter_special_tokens from repl.py."""
    text = re.sub(r"```tool_code\s*\n?.*?\n?```", "", text, flags=re.DOTALL)
    text = re.sub(r"<@TOOL>.*?</@TOOL>", "", text, flags=re.DOTALL)
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"<@TOOL_RESULT>.*?</@TOOL_RESULT>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|start\|>.*?<\|call\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^|]+\|>", "", text)
    # NEW behaviour: 3+ newlines → 2, preserve \n\n for Markdown paragraphs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Bug #1 + #2: Text not displayed due to paragraph offset vs count mismatch
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamingDisplay:

    def test_leading_newlines_simple_response(self):
        """Qwen3.5 always starts with \\n\\n — full response must still display."""
        tokens = ["\n\n", "Привет!", " 👋", " Как", " дела", "?"]
        parts = simulate_streaming(tokens)
        full = "".join(parts)
        assert "Привет!" in full
        assert "👋" in full
        assert "Как дела?" in full

    def test_single_paragraph_no_trailing_newline(self):
        """Single dense paragraph without closing \\n\\n must be printed in stop_streaming."""
        tokens = list("Hello! How can I help you today?")
        parts = simulate_streaming(tokens)
        full = "".join(parts)
        assert "Hello!" in full
        assert "How can I help you today?" in full

    def test_multi_paragraph_all_printed(self):
        """Every paragraph in a multi-paragraph response must be printed exactly once."""
        text = "Para one.\n\nPara two.\n\nPara three."
        tokens = list(text)
        parts = simulate_streaming(tokens)
        full = " ".join(parts)
        assert "Para one." in full
        assert "Para two." in full
        assert "Para three." in full

    def test_no_duplication(self):
        """Text must not appear more than once across all printed parts."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        tokens = list(text)
        parts = simulate_streaming(tokens)
        full = " ".join(parts)
        # Each distinct sentence appears exactly once
        assert full.count("First paragraph.") == 1
        assert full.count("Second paragraph.") == 1
        assert full.count("Third paragraph.") == 1

    def test_text_before_tool_call_is_printed(self):
        """Text that precedes a tool call tag must be printed."""
        text_before = "Похоже, нет файлов.\n\n"
        tool_tag = '<@TOOL>{"name":"command-exec","arguments":{"command":"ls -la"}}</@TOOL>'
        tokens = list(text_before + tool_tag)
        parts = simulate_streaming(tokens)
        full = "".join(parts)
        # User-visible text should appear
        assert "Похоже, нет файлов." in full
        # Tool tag must NOT appear
        assert "<@TOOL>" not in full

    def test_leading_nn_with_tool_call_text_visible(self):
        """\\n\\n prefix + text + tool call: text must be visible, tag must not."""
        tokens = (
            list("\n\nЯ проверю файлы.\n\n") +
            list('<@TOOL>{"name":"project-structure","arguments":{}}</@TOOL>')
        )
        parts = simulate_streaming(tokens)
        full = "".join(parts)
        assert "Я проверю файлы." in full
        assert "<@TOOL>" not in full


# ──────────────────────────────────────────────────────────────────────────────
# Bug #2: _filter_special_tokens must preserve \n\n
# ──────────────────────────────────────────────────────────────────────────────

class TestFilterSpecialTokens:

    def test_double_newlines_preserved(self):
        """The filter must NOT collapse \\n\\n into \\n."""
        result = simulate_filter("Paragraph A.\n\nParagraph B.")
        assert "\n\n" in result, "\\n\\n must be preserved for Markdown paragraphs"

    def test_triple_newlines_collapsed_to_double(self):
        """Three or more newlines should be collapsed to two."""
        result = simulate_filter("A.\n\n\nB.")
        assert "\n\n\n" not in result
        assert "\n\n" in result

    def test_single_newline_unchanged(self):
        """Single \\n must not be affected."""
        result = simulate_filter("Line one.\nLine two.")
        assert "\n" in result
        assert result.count("\n") == 1

    def test_tool_tags_removed(self):
        """Tool call tags must be stripped by the filter."""
        text = 'Some text <@TOOL>{"name":"x"}</@TOOL> more text'
        result = simulate_filter(text)
        assert "<@TOOL>" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_strip_called(self):
        """Leading/trailing whitespace is stripped."""
        result = simulate_filter("  \n\nHello\n\n  ")
        assert result == "Hello"
