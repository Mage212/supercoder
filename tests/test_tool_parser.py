#!/usr/bin/env python3
"""Tests for multi-format tool call parser."""

import pytest
from supercoder.agent.tool_parser import (
    ToolCallParser,
    SupercoderTagParser,
    QwenStyleParser,
    JsonBlockParser,
    XmlFunctionParser,
    PythonicParser,
    GenericJsonParser,
)


class TestSupercoderTagParser:
    """Test our native format."""
    
    def test_basic_tool_call(self):
        parser = SupercoderTagParser()
        text = 'Some text <@TOOL>{"name": "file_read", "arguments": "{\\"path\\": \\"main.py\\"}"}</@TOOL> more text'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "file_read"
        assert result.format_name == "supercoder_tag"
    
    def test_no_match(self):
        parser = SupercoderTagParser()
        text = "No tool call here"
        assert parser.try_parse(text) is None


class TestQwenStyleParser:
    """Test Qwen/OSS model format."""
    
    def test_project_structure_call(self):
        parser = QwenStyleParser()
        text = '<|start|>assistant<|channel|>commentary to=tool:project-structure <|constrain|>json<|message|>{"maxDepth": 3, "path": "."}<|call|>'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "project-structure"  # kebab-case
        assert result.arguments == {"maxDepth": 3, "path": "."}
        assert result.format_name == "qwen_style"
    
    def test_create_call(self):
        parser = QwenStyleParser()
        text = '<|start|>assistant<|channel|>commentary to=tool:create <|constrain|>json<|message|>{"filepath":"test.py","content":"print(1)"}<|call|>'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "code-edit"  # mapped from 'create'
        assert "filepath" in result.arguments
    
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
{"tool": "file_read", "args": {"path": "main.py"}}
```
'''
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "file_read"
        assert result.arguments == {"path": "main.py"}
    
    def test_json_block_with_function_key(self):
        parser = JsonBlockParser()
        text = '```json\n{"function": "code_edit", "arguments": {"filepath": "test.py"}}\n```'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "code_edit"
    
    def test_no_tool_key(self):
        parser = JsonBlockParser()
        text = '```json\n{"key": "value"}\n```'
        assert parser.try_parse(text) is None


class TestXmlFunctionParser:
    """Test XML-style function calls."""
    
    def test_function_call_tag(self):
        parser = XmlFunctionParser()
        text = '<function_call name="file_read">{"path": "main.py"}</function_call>'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "file_read"
    
    def test_tool_tag(self):
        parser = XmlFunctionParser()
        text = '<tool name="search">query text</tool>'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "search"
        assert result.arguments == "query text"


class TestPythonicParser:
    """Test Python function call syntax."""
    
    def test_basic_function_call(self):
        parser = PythonicParser()
        text = 'file-read(path="main.py")'  # kebab-case
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "file-read"  # kebab-case
        assert result.arguments == {"path": "main.py"}
    
    def test_multiple_args(self):
        parser = PythonicParser()
        text = 'code-edit(filepath="test.py", operation="create", content="hello")'  # kebab-case
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "code-edit"
        assert result.arguments["filepath"] == "test.py"
    
    def test_unknown_function_ignored(self):
        parser = PythonicParser()
        text = 'unknown_func(arg="value")'
        assert parser.try_parse(text) is None


class TestGenericJsonParser:
    """Test generic JSON fallback."""
    
    def test_json_with_tool_key(self):
        parser = GenericJsonParser()
        text = 'The tool call is: {"tool": "file_read", "path": "main.py"}'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "file_read"
    
    def test_json_with_action_key(self):
        parser = GenericJsonParser()
        text = '{"action": "code_edit", "params": {"filepath": "test.py"}}'
        result = parser.try_parse(text)
        
        assert result is not None
        assert result.name == "code_edit"


class TestToolCallParser:
    """Test waterfall parser integration."""
    
    def test_supercoder_format_priority(self):
        parser = ToolCallParser()
        # Both formats present - supercoder should win
        text = '<@TOOL>{"name": "file_read", "arguments": "{}"}</@TOOL> also file_read(path="other.py")'
        result = parser.parse(text)
        
        assert result is not None
        assert result.format_name == "supercoder_tag"
    
    def test_fallback_to_qwen(self):
        parser = ToolCallParser()
        text = '<|start|>assistant<|channel|>commentary to=tool:search <|message|>{"query": "test"}<|call|>'
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
        assert len(formats) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
