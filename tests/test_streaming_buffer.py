"""Tests for StreamingDisplayBuffer.

Verifies that tool call tags are hidden from display while normal text
passes through, including edge cases like partial tags, multiple tags,
tags at stream start, and format-specific behaviors.
"""


from supercoder.streaming_buffer import StreamingDisplayBuffer


# ---------------------------------------------------------------------------
# Helper: feed tokens one-by-one and collect displayed text
# ---------------------------------------------------------------------------
def feed_tokens(buf: StreamingDisplayBuffer, tokens: list[str]) -> str:
    """Feed a list of tokens and return all displayed text concatenated."""
    parts = []
    for token in tokens:
        chunk = buf.add(token)
        if chunk:
            parts.append(chunk)
    remaining = buf.flush()
    if remaining:
        parts.append(remaining)
    return "".join(parts)


def feed_text(buf: StreamingDisplayBuffer, text: str, chunk_size: int = 5) -> str:
    """Feed text in fixed-size chunks and return displayed text."""
    tokens = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return feed_tokens(buf, tokens)


# ===========================================================================
# Supercoder format tests
# ===========================================================================
class TestSupercoderFormat:
    """Tests for the default <@TOOL>...</@TOOL> format."""

    def test_plain_text_passes_through(self):
        """Normal text without any tags is fully displayed."""
        buf = StreamingDisplayBuffer("supercoder")
        result = feed_text(buf, "Hello, this is a normal response without any tools.")
        assert "Hello" in result
        assert "normal response" in result

    def test_tag_completely_hidden(self):
        """A complete <@TOOL>...</@TOOL> block is never shown."""
        buf = StreamingDisplayBuffer("supercoder")
        text = 'Let me read the file.\n\n<@TOOL>{"name": "file-read", "arguments": {"fileName": "main.py"}}</@TOOL>'
        result = feed_text(buf, text)
        assert "Let me read the file." in result
        assert "<@TOOL>" not in result
        assert "file-read" not in result
        assert "</@TOOL>" not in result

    def test_text_before_tag_shown(self):
        """Text before a tag is displayed, the tag itself is hidden."""
        buf = StreamingDisplayBuffer("supercoder")
        result = feed_text(buf, 'Here is some text.\n\n<@TOOL>{"name": "test"}</@TOOL>')
        assert "Here is some text." in result
        assert "<@TOOL>" not in result

    def test_text_after_tag_shown(self):
        """Text after a closing tag is displayed."""
        buf = StreamingDisplayBuffer("supercoder")
        result = feed_text(buf, '<@TOOL>{"name": "test"}</@TOOL>\n\nAfter the tool call.')
        assert "After the tool call." in result
        assert "<@TOOL>" not in result

    def test_partial_opener_held_back(self):
        """A partial opener like '<@' is held back until resolved."""
        buf = StreamingDisplayBuffer("supercoder")

        # Feed text ending with '<@'
        chunk1 = buf.add("Some text <@")
        # '<@' matches start of '<@TOOL>' — should be held back
        # 'Some text ' should be released
        assert chunk1 is None or "<@" not in (chunk1 or "")

        # Now complete it as NOT a tag
        chunk2 = buf.add("mention>")
        # '<@mention>' is not '<@TOOL>' — should be released
        remaining = buf.flush()
        full = (chunk1 or "") + (chunk2 or "") + remaining
        assert "<@mention>" in full

    def test_lone_angle_bracket_released(self):
        """A '<' not followed by '@' is eventually released."""
        buf = StreamingDisplayBuffer("supercoder")
        result = feed_text(buf, "a < b and c > d")
        assert "a < b" in result or ("a" in result and "< b" in result)

    def test_multiple_tags_all_hidden(self):
        """Multiple consecutive tags are all hidden."""
        buf = StreamingDisplayBuffer("supercoder")
        text = (
            'First.\n\n<@TOOL>{"name": "a"}</@TOOL>\n\n'
            'Second.\n\n<@TOOL>{"name": "b"}</@TOOL>\n\n'
            "Done."
        )
        result = feed_text(buf, text)
        assert "First." in result
        assert "Second." in result
        assert "Done." in result
        assert "<@TOOL>" not in result
        assert "</@TOOL>" not in result

    def test_tag_at_start_of_response(self):
        """Response starting directly with a tag — nothing displayed until after."""
        buf = StreamingDisplayBuffer("supercoder")
        result = feed_text(buf, '<@TOOL>{"name": "test"}</@TOOL>')
        # The tag content should not appear
        assert "<@TOOL>" not in result
        assert "test" not in result or result.strip() == ""

    def test_token_by_token_streaming(self):
        """Feeding character-by-character still works correctly."""
        buf = StreamingDisplayBuffer("supercoder")
        text = 'Hello!\n\n<@TOOL>{"name": "x"}</@TOOL>'
        # Feed one character at a time
        result = feed_tokens(buf, list(text))
        assert "Hello!" in result
        assert "<@TOOL>" not in result


# ===========================================================================
# Qwen format tests
# ===========================================================================
class TestQwenFormat:
    """Tests for the to=tool:name {json} format (no closing tag)."""

    def test_qwen_tag_hidden(self):
        """Qwen-style tool call is hidden."""
        buf = StreamingDisplayBuffer("qwen_like")
        text = 'Let me search.\n\nto=tool:code-search {"query": "def main"}\n\nDone.'
        result = feed_text(buf, text)
        assert "Let me search." in result
        assert "to=tool:" not in result
        assert "code-search" not in result
        assert "Done." in result

    def test_qwen_consumes_to_newline(self):
        """Qwen format consumes everything up to the newline."""
        buf = StreamingDisplayBuffer("qwen_like")
        text = 'to=tool:file-read {"fileName": "test.py"}\nAfter.'
        result = feed_text(buf, text)
        assert "to=tool:" not in result
        assert "After." in result


# ===========================================================================
# GLM format tests
# ===========================================================================
class TestGlmFormat:
    """Tests for <tool_call>...</tool_call> format."""

    def test_glm_tag_hidden(self):
        """GLM-style tool call is hidden."""
        buf = StreamingDisplayBuffer("glm_tool_call")
        text = "Reading file.\n\n<tool_call>file-read<arg_key>fileName</arg_key><arg_value>main.py</arg_value></tool_call>"
        result = feed_text(buf, text)
        assert "Reading file." in result
        assert "<tool_call>" not in result
        assert "</tool_call>" not in result


# ===========================================================================
# XML function format tests
# ===========================================================================
class TestXmlFormat:
    """Tests for <function_call name="...">...</function_call> format."""

    def test_xml_tag_hidden(self):
        """XML function call is hidden."""
        buf = StreamingDisplayBuffer("xml_function")
        text = 'Checking.\n\n<function_call name="file-read">{"fileName": "x.py"}</function_call>'
        result = feed_text(buf, text)
        assert "Checking." in result
        assert "<function_call" not in result
        assert "</function_call>" not in result


# ===========================================================================
# JSON block format tests
# ===========================================================================
class TestJsonBlockFormat:
    """Tests for ```json ... ``` format."""

    def test_json_block_hidden(self):
        """JSON code block tool call is hidden."""
        buf = StreamingDisplayBuffer("json_block")
        text = 'Let me try.\n\n```json\n{"tool": "file-read", "arguments": {"fileName": "a.py"}}\n```\n\nDone.'
        result = feed_text(buf, text)
        assert "Let me try." in result
        assert "```json" not in result
        assert "Done." in result


# ===========================================================================
# Edge cases
# ===========================================================================
class TestEdgeCases:
    def test_empty_input(self):
        """Empty tokens produce no output."""
        buf = StreamingDisplayBuffer("supercoder")
        assert buf.add("") is None
        assert buf.flush() == ""

    def test_flush_returns_remaining_buffer(self):
        """flush() returns everything still held (partial opener suffix)."""
        buf = StreamingDisplayBuffer("supercoder")
        # Text ending with '<@' is held because it could be start of '<@TOOL>'
        chunk = buf.add("held text <@")
        result = buf.flush()
        full = (chunk or "") + result
        assert "held text" in full
        assert "<@" in full

    def test_reset_clears_state(self):
        """reset() clears buffer and tag state."""
        buf = StreamingDisplayBuffer("supercoder")
        buf.add("some <@TOOL>partial")
        buf.reset()
        assert buf.flush() == ""
        assert buf._in_tag is False

    def test_unknown_format_defaults_to_supercoder(self):
        """Unknown tool_calling_type falls back to supercoder."""
        buf = StreamingDisplayBuffer("unknown_format")
        assert buf._opener == "<@TOOL>"
