#!/usr/bin/env python3
"""Tests for multi-format tool call parser."""

import pytest

from supercoder.agent.tool_parser import (
    GlmToolCallParser,
    JsonBlockParser,
    QwenStyleParser,
    SupercoderTagParser,
    SupercoderTagFallbackParser,
    ToolCallParser,
    XmlFunctionParser,
)


class TestSupercoderTagParser:
    """Test our native format."""

    def test_basic_tool_call(self):
        parser = SupercoderTagParser()
        text = 'Some text <@TOOL>{"name": "file-read", "arguments": {"path": "main.py"}}</@TOOL> more text'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-read"
        assert result.format_name == "supercoder_tag"

    def test_no_match(self):
        parser = SupercoderTagParser()
        text = "No tool call here"
        assert parser.try_parse(text) is None

    def test_multiple_tool_calls(self):
        parser = SupercoderTagParser()
        text = """
<@TOOL>{"name": "file-read", "arguments": {"path": "a.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"path": "b.py"}}</@TOOL>
"""
        results = parser.try_parse_all(text)
        assert len(results) == 2
        assert results[0].arguments["path"] == "a.py"
        assert results[1].arguments["path"] == "b.py"


class TestQwenStyleParser:
    """Test Qwen-style format: to=tool:name {...}"""

    def test_simple_format(self):
        parser = QwenStyleParser()
        text = 'I will read the file. to=tool:file-read {"path": "main.py"}'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-read"
        assert result.arguments == {"path": "main.py"}
        assert result.format_name == "qwen_style"

    def test_dot_separator(self):
        parser = QwenStyleParser()
        text = 'to=tool.code-edit {"file": "test.py", "operation": "create", "content": "print(1)"}'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "code-edit"
        assert result.arguments["file"] == "test.py"

    def test_no_match(self):
        parser = QwenStyleParser()
        text = "Regular text without Qwen markers"
        assert parser.try_parse(text) is None


class TestJsonBlockParser:
    """Test JSON code block format."""

    def test_json_block_with_tool_key(self):
        parser = JsonBlockParser()
        text = """Here's the command:
```json
{"tool": "file-read", "arguments": {"path": "main.py"}}
```
"""
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-read"
        assert result.arguments == {"path": "main.py"}

    def test_json_block_with_name_key(self):
        parser = JsonBlockParser()
        text = '```json\n{"name": "code-edit", "arguments": {"file": "test.py"}}\n```'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "code-edit"

    def test_no_tool_key(self):
        parser = JsonBlockParser()
        text = '```json\n{"key": "value"}\n```'
        assert parser.try_parse(text) is None


class TestXmlFunctionParser:
    """Test XML-style function calls."""

    def test_function_call_tag(self):
        parser = XmlFunctionParser()
        text = '<function_call name="file-read">{"path": "main.py"}</function_call>'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-read"
        assert result.arguments == {"path": "main.py"}

    def test_with_newlines(self):
        parser = XmlFunctionParser()
        text = """<function_call name="code-edit">
{"file": "app.py", "content": "print('hello')"}
</function_call>"""
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "code-edit"


class TestToolCallParser:
    """Test waterfall parser integration."""

    def test_supercoder_format_priority(self):
        parser = ToolCallParser()
        # Both formats present - supercoder should win
        text = '<@TOOL>{"name": "file-read", "arguments": {}}</@TOOL> also to=tool:file-read {"path": "other.py"}'
        result = parser.parse(text)

        assert result is not None
        assert result.format_name == "supercoder_tag"

    def test_fallback_to_qwen(self):
        parser = ToolCallParser()
        text = 'Let me search. to=tool:code-search {"query": "test"}'
        result = parser.parse(text)

        assert result is not None
        assert result.format_name == "qwen_style"

    def test_no_tool_call(self):
        parser = ToolCallParser()
        text = "Just regular text, no tool calls here."
        assert parser.parse(text) is None

    def test_supported_formats(self):
        parser = ToolCallParser()
        formats = parser.supported_formats

        assert "supercoder_tag" in formats
        assert "supercoder_tag_fallback" in formats
        assert "qwen_style" in formats
        assert "json_block" in formats
        assert "xml_function" in formats
        assert "glm_tool_call" in formats
        assert len(formats) == 6  # 6 parsers including fallback and GLM


class TestGlmToolCallParser:
    """Test GLM-style <tool_call> format."""

    def test_single_argument(self):
        parser = GlmToolCallParser()
        text = "<tool_call>file-read<arg_key>fileName</arg_key><arg_value>main.py</arg_value></tool_call>"
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-read"
        assert result.arguments == {"fileName": "main.py"}
        assert result.format_name == "glm_tool_call"

    def test_multiple_arguments(self):
        parser = GlmToolCallParser()
        text = "<tool_call>project-structure<arg_key>maxDepth</arg_key><arg_value>3<arg_key>maxFiles</arg_key><arg_value>50<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>"
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "project-structure"
        assert result.arguments["maxDepth"] == 3
        assert result.arguments["maxFiles"] == 50
        assert result.arguments["path"] == "."

    def test_with_surrounding_text(self):
        parser = GlmToolCallParser()
        text = "Давайте посмотрим структуру проекта.<tool_call>project-structure<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>"
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "project-structure"
        assert result.arguments == {"path": "."}

    def test_no_match(self):
        parser = GlmToolCallParser()
        text = "Regular text without GLM markers"
        assert parser.try_parse(text) is None

    def test_multiple_tool_calls(self):
        """Test parsing multiple tool calls in one response - real scenario from GLM-4.7-Flash."""
        parser = GlmToolCallParser()
        text = "<tool_call>file-read<arg_key>fileName</arg_key><arg_value>main.py</arg_value></tool_call><tool_call>file-read<arg_key>fileName</arg_key><arg_value>hello.py</arg_value></tool_call><tool_call>file-read<arg_key>fileName</arg_key><arg_value>dungeon_crawler.py</arg_value></tool_call>"
        results = parser.try_parse_all(text)

        assert len(results) == 3
        assert results[0].arguments["fileName"] == "main.py"
        assert results[1].arguments["fileName"] == "hello.py"
        assert results[2].arguments["fileName"] == "dungeon_crawler.py"

    def test_parse_all_supercoder(self):
        parser = ToolCallParser()
        text = """
<@TOOL>{"name": "file-read", "arguments": {"path": "a.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"path": "b.py"}}</@TOOL>
"""
        results = parser.parse_all(text)
        assert len(results) == 2


class TestSupercoderTagFallbackParser:
    """Regression tests for qwen3.5-4b failure modes."""

    def test_missing_closing_tag(self):
        """qwen3.5-4b sometimes omits </@TOOL> at end of response."""
        parser = SupercoderTagFallbackParser()
        text = '<@TOOL>{"name": "code-search", "arguments": {"query": "def", "maxResults": 50}}'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "code-search"
        assert result.arguments["query"] == "def"
        assert result.format_name == "supercoder_tag_fallback"

    def test_no_activation_when_closing_tag_present(self):
        """Fallback must not activate when main parser would match."""
        parser = SupercoderTagFallbackParser()
        text = '<@TOOL>{"name": "file-read", "arguments": {"path": "main.py"}}</@TOOL>'
        result = parser.try_parse(text)
        assert result is None  # let SupercoderTagParser handle it

    def test_waterfall_picks_up_missing_closing_tag(self):
        """ToolCallParser should use fallback when closing tag absent."""
        parser = ToolCallParser()
        text = '<@TOOL>{"name": "code-search", "arguments": {"query": "def"}}'
        result = parser.parse(text)

        assert result is not None
        assert result.name == "code-search"

    def test_extra_gt_before_closing_tag(self):
        """qwen3.5-4b sometimes emits '}></@TOOL>' instead of '}</@TOOL>'."""
        parser = SupercoderTagParser()
        text = '<@TOOL>{"name": "file-create", "arguments": {"fileName": "config.py"}}></@TOOL>'
        result = parser.try_parse(text)

        assert result is not None
        assert result.name == "file-create"

    def test_single_quote_strings_repaired(self):
        """qwen3.5-4b uses single-quoted JSON strings instead of double."""
        parser = ToolCallParser()
        text = "<@TOOL>{\"name\": \"file-create\", \"arguments\": {\"content\": 'print(\"Hello\")'}}</@TOOL>"
        result = parser.parse(text)

        assert result is not None
        assert result.name == "file-create"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
