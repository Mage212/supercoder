"""Multi-format tool call parser.

Supports tool calling formats based on model's tool_calling_type configuration:
1. Our native format: <@TOOL>{"name": "...", "arguments": "..."}</@TOOL>
2. Qwen-style: to=tool:name {...}
3. JSON code blocks: ```json {"tool": "...", "arguments": {...}} ```
4. XML: <function_call name="...">...</function_call>
"""

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract a complete JSON object starting at position `start` (must be '{').

    Handles nested braces correctly, unlike a simple non-greedy regex.
    Returns the matched JSON string, or None if braces are unbalanced.
    """
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


@dataclass
class ToolCall:
    """Parsed tool call result."""
    name: str
    arguments: dict[str, Any] | str
    raw_match: str = ""  # The original matched text
    format_name: str = ""  # Which parser matched
    
    def to_dict(self) -> dict:
        """Convert to dict format expected by CoderAgent."""
        return {
            "name": self.name,
            "arguments": self.arguments if isinstance(self.arguments, str) else json.dumps(self.arguments)
        }


class BaseToolParser(ABC):
    """Base class for tool call parsers."""
    
    name: str = "base"
    
    @abstractmethod
    def try_parse(self, text: str) -> ToolCall | None:
        """Try to parse a tool call from text. Return None if no match."""
        pass


class SupercoderTagParser(BaseToolParser):
    """Parse our native <@TOOL>...</@TOOL> format.
    
    Expected format:
    <@TOOL>{"name": "tool-name", "arguments": {"arg1": "value1"}}</@TOOL>
    """
    
    name = "supercoder_tag"
    pattern = re.compile(r'<@TOOL>(.*?)</@TOOL>', re.DOTALL)
    
    def try_parse(self, text: str) -> ToolCall | None:
        match = self.pattern.search(text)
        if not match:
            return None
        
        try:
            content = match.group(1).strip()
            data = json.loads(content)
            return ToolCall(
                name=data.get("name", ""),
                arguments=data.get("arguments", ""),
                raw_match=match.group(0),
                format_name=self.name
            )
        except (json.JSONDecodeError, KeyError):
            return None
    
    def try_parse_all(self, text: str) -> list[ToolCall]:
        """Parse ALL tool calls from text, not just the first one."""
        results = []
        
        for match in self.pattern.finditer(text):
            try:
                content = match.group(1).strip()
                data = json.loads(content)
                results.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", ""),
                    raw_match=match.group(0),
                    format_name=self.name
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        
        return results


class QwenStyleParser(BaseToolParser):
    """Parse Qwen-style formats.
    
    Supports:
    1. Simple format (as instructed): to=tool:file-read {"path": "main.py"}
    2. Full Qwen format (model default): <|start|>...<|channel|>...to=tool:name...<|message|>{...}<|call|>
    """
    
    name = "qwen_style"

    # Pattern to find tool name followed by the start of a JSON object.
    # We capture only the name; the JSON object is extracted via balanced-brace parsing.
    _name_pattern_simple = re.compile(
        r'to=(?:tool[:\s.])?([a-zA-Z0-9_-]+)\s*(\{)',
        re.IGNORECASE
    )
    _name_pattern_full = re.compile(
        r'<\|start\|>.*?to=(?:tool[:\s.])?([a-zA-Z0-9_-]+).*?<\|message\|>(\{)',
        re.DOTALL | re.IGNORECASE
    )
    _name_pattern_channel = re.compile(
        r'<\|channel\|>.*?to=(?:tool[:\s.])?([a-zA-Z0-9_-]+).*?<\|message\|>(\{)',
        re.DOTALL | re.IGNORECASE
    )

    def try_parse(self, text: str) -> ToolCall | None:
        for pattern in (self._name_pattern_simple, self._name_pattern_full, self._name_pattern_channel):
            match = pattern.search(text)
            if match:
                result = self._parse_with_balanced_braces(text, match)
                if result:
                    return result
        return None

    def _parse_with_balanced_braces(self, text: str, name_match) -> ToolCall | None:
        tool_name = name_match.group(1).strip()
        json_start = name_match.start(2)  # position of the opening '{'
        json_text = _extract_balanced_json(text, json_start)
        if json_text is None:
            return None
        try:
            args = json.loads(json_text)
            raw = text[name_match.start():json_start + len(json_text)]
            return ToolCall(
                name=tool_name,
                arguments=args,
                raw_match=raw,
                format_name=self.name
            )
        except json.JSONDecodeError:
            return None


class JsonBlockParser(BaseToolParser):
    """Parse JSON code blocks with tool info.
    
    Expected format (as instructed in system prompt):
    ```json
    {"tool": "file-read", "arguments": {"path": "main.py"}}
    ```
    """
    
    name = "json_block"
    # Require explicit 'json' marker to avoid matching arbitrary code examples
    pattern = re.compile(r'```json\s*\n(.*?)\n?```', re.DOTALL)

    def try_parse(self, text: str) -> ToolCall | None:
        match = self.pattern.search(text)
        if not match:
            return None

        try:
            content = match.group(1).strip()
            data = json.loads(content)

            # Support key names as instructed: "tool" and "arguments"
            name = data.get("tool") or data.get("name")
            if not name:
                return None

            args = data.get("arguments") or data.get("args")
            # Only treat as tool call if 'arguments'/'args' key is explicitly present
            if args is None:
                return None
            
            return ToolCall(
                name=name,
                arguments=args,
                raw_match=match.group(0),
                format_name=self.name
            )
        except (json.JSONDecodeError, KeyError):
            return None


class XmlFunctionParser(BaseToolParser):
    """Parse XML-style function calls.
    
    Expected format (as instructed in system prompt):
    <function_call name="file-read">{"path": "main.py"}</function_call>
    """
    
    name = "xml_function"
    pattern = re.compile(
        r'<function_call\s+name=["\']([^"\']+)["\']>(.*?)</function_call>',
        re.DOTALL
    )
    
    def try_parse(self, text: str) -> ToolCall | None:
        match = self.pattern.search(text)
        if not match:
            return None
        
        name = match.group(1)
        args_str = match.group(2).strip()
        
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            args = args_str
        
        return ToolCall(
            name=name,
            arguments=args,
            raw_match=match.group(0),
            format_name=self.name
        )


class GlmToolCallParser(BaseToolParser):
    """Parse GLM-style <tool_call> format.
    
    GLM-4.7-Flash uses a unique format:
    <tool_call>tool-name<arg_key>param1</arg_key><arg_value>value1<arg_key>param2</arg_key><arg_value>value2</arg_value></tool_call>
    
    Note: The closing </arg_value> tag only appears at the end, not after each value.
    """
    
    name = "glm_tool_call"
    # Match <tool_call>...</tool_call> with tool name and args inside
    pattern = re.compile(r'<tool_call>([^<]+)(.*?)</tool_call>', re.DOTALL)
    # Match arg_key/arg_value pairs - value ends at next <arg_key> or </arg_value>
    arg_pattern = re.compile(r'<arg_key>([^<]+)</arg_key><arg_value>([^<]*)', re.DOTALL)
    
    def try_parse(self, text: str) -> ToolCall | None:
        match = self.pattern.search(text)
        if not match:
            return None
        
        tool_name = match.group(1).strip()
        args_section = match.group(2)
        
        # Parse arg_key/arg_value pairs
        args = {}
        for arg_match in self.arg_pattern.finditer(args_section):
            key = arg_match.group(1).strip()
            value = arg_match.group(2).strip()
            # Try to convert to appropriate type
            if value.isdigit():
                args[key] = int(value)
            elif value.lower() in ('true', 'false'):
                args[key] = value.lower() == 'true'
            else:
                args[key] = value
        
        return ToolCall(
            name=tool_name,
            arguments=args,
            raw_match=match.group(0),
            format_name=self.name
        )
    
    def _parse_match(self, match) -> ToolCall:
        """Parse a single regex match into a ToolCall."""
        tool_name = match.group(1).strip()
        args_section = match.group(2)
        
        args = {}
        for arg_match in self.arg_pattern.finditer(args_section):
            key = arg_match.group(1).strip()
            value = arg_match.group(2).strip()
            if value.isdigit():
                args[key] = int(value)
            elif value.lower() in ('true', 'false'):
                args[key] = value.lower() == 'true'
            else:
                args[key] = value
        
        return ToolCall(
            name=tool_name,
            arguments=args,
            raw_match=match.group(0),
            format_name=self.name
        )
    
    def try_parse_all(self, text: str) -> list[ToolCall]:
        """Parse ALL tool calls from text, not just the first one."""
        results = []
        for match in self.pattern.finditer(text):
            try:
                results.append(self._parse_match(match))
            except Exception:
                continue
        return results


class ToolCallParser:
    """Waterfall parser that tries multiple formats in order.
    
    Usage:
        parser = ToolCallParser()
        result = parser.parse(llm_response_text)
        if result:
            print(f"Found {result.name} call via {result.format_name}")
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        # Parsers in priority order - matching our supported tool_calling_types
        self.parsers: list[BaseToolParser] = [
            SupercoderTagParser(),  # tool_calling_type: supercoder
            QwenStyleParser(),      # tool_calling_type: qwen_like
            JsonBlockParser(),      # tool_calling_type: json_block
            XmlFunctionParser(),    # tool_calling_type: xml_function
            GlmToolCallParser(),    # tool_calling_type: glm_tool_call
        ]
    
    def parse(self, text: str) -> ToolCall | None:
        """Try each parser in order, return first match."""
        for parser in self.parsers:
            try:
                result = parser.try_parse(text)
                if result:
                    if self.debug:
                        print(f"[DEBUG] Tool call parsed by {parser.name}: {result.name}")
                    return result
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Parser {parser.name} error: {e}")
                continue
        return None
    
    def parse_all(self, text: str) -> list[ToolCall]:
        """Parse ALL tool calls from text using first matching parser with multi-call support.
        
        Returns a list of all tool calls found (may be empty).
        """
        # Try each parser that supports try_parse_all
        for parser in self.parsers:
            if hasattr(parser, 'try_parse_all'):
                results = parser.try_parse_all(text)
                if results:
                    if self.debug:
                        print(f"[DEBUG] Found {len(results)} tool calls via {parser.name}")
                    return results
        
        # Fall back to single parse for other formats
        result = self.parse(text)
        if result:
            return [result]
        return []
    
    def add_parser(self, parser: BaseToolParser, priority: int = -1) -> None:
        """Add a custom parser. Use priority=-1 to add at end."""
        if priority < 0:
            self.parsers.append(parser)
        else:
            self.parsers.insert(priority, parser)
    
    @property
    def supported_formats(self) -> list[str]:
        """Return list of supported format names."""
        return [p.name for p in self.parsers]
