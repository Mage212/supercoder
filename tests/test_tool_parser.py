#!/usr/bin/env python3
"""Tests for multi-format tool call parser."""

import pytest
from supercoder.agent.tool_parser import (
    ToolCallParser,
    SupercoderTagParser,
    QwenStyleParser,
    JsonBlockParser,
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
        text = '''
<@TOOL>{"name": "file-read", "arguments": {"path": "a.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"path": "b.py"}}</@TOOL>
'''
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
        text = '''Here's the command:
```json
{"tool": "file-read", "arguments": {"path": "main.py"}}
```
'''
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
        text = '''<function_call name="code-edit">
{"file": "app.py", "content": "print('hello')"}
</function_call>'''
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
        assert "qwen_style" in formats
        assert "json_block" in formats
        assert "xml_function" in formats
        assert len(formats) == 4  # Only 4 parsers now
    
    def test_parse_all_supercoder(self):
        parser = ToolCallParser()
        text = '''
<@TOOL>{"name": "file-read", "arguments": {"path": "a.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"path": "b.py"}}</@TOOL>
'''
        results = parser.parse_all(text)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
