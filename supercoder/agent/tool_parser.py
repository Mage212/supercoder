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
    
    # Pattern 1: Simple format (what we instruct in prompt)
    pattern_simple = re.compile(
        r'to=(?:tool[:\s.])?([a-zA-Z0-9_-]+)\s+(\{.*?\})',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 2: Full Qwen format with markers (what model actually uses)
    pattern_full = re.compile(
        r'<\|start\|>.*?to=(?:tool[:\s.])?([a-zA-Z0-9_-]+).*?<\|message\|>(\{.*?\})<\|call\|>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 3: channel format without start (some models)
    pattern_channel = re.compile(
        r'<\|channel\|>.*?to=(?:tool[:\s.])?([a-zA-Z0-9_-]+).*?<\|message\|>(\{.*?\})',
        re.DOTALL | re.IGNORECASE
    )
    
    def try_parse(self, text: str) -> ToolCall | None:
        # Try simple format first (what we instructed)
        match = self.pattern_simple.search(text)
        if match:
            return self._parse_match(match)
        
        # Try full Qwen format (what model may use)
        match = self.pattern_full.search(text)
        if match:
            return self._parse_match(match)
        
        # Try channel format
        match = self.pattern_channel.search(text)
        if match:
            return self._parse_match(match)
        
        return None
    
    def _parse_match(self, match) -> ToolCall | None:
        tool_name = match.group(1).strip()
        json_text = match.group(2)
        
        try:
            args = json.loads(json_text)
            return ToolCall(
                name=tool_name,
                arguments=args,
                raw_match=match.group(0),
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
    pattern = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
    
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
            
            args = data.get("arguments") or data.get("args") or {}
            
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
        """Parse ALL tool calls from text using first matching parser.
        
        Returns a list of all tool calls found (may be empty).
        """
        # Try SupercoderTagParser first as it supports multiple calls
        supercoder_parser = self.parsers[0]  # SupercoderTagParser
        if hasattr(supercoder_parser, 'try_parse_all'):
            results = supercoder_parser.try_parse_all(text)
            if results:
                if self.debug:
                    print(f"[DEBUG] Found {len(results)} tool calls via {supercoder_parser.name}")
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
