"""Multi-format tool call parser.

Supports various LLM output formats for tool calling:
1. Our native format: <@TOOL>{"name": "...", "arguments": "..."}</@TOOL>
2. Qwen-style: <|start|>...<|channel|>...to=tool:name...<|message|>...<|call|>
3. JSON code blocks: ```json {"tool": "...", "args": {...}} ```
4. XML: <function_call name="...">...</function_call>
5. Pythonic: tool_name(arg1="value1")
6. Generic JSON with tool/function keys
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
    """Parse our native <@TOOL>...</@TOOL> format, including inside ```tool_code blocks."""
    
    name = "supercoder_tag"
    pattern = re.compile(r'<@TOOL>(.*?)</@TOOL>', re.DOTALL)
    # Pattern to extract content from ```tool_code blocks
    code_block_pattern = re.compile(r'```tool_code\s*\n?(.*?)\n?```', re.DOTALL)
    
    def try_parse(self, text: str) -> ToolCall | None:
        # First, try to extract from tool_code block if present
        code_match = self.code_block_pattern.search(text)
        if code_match:
            text = code_match.group(1)
        
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
        
        # First, try to extract from tool_code block if present
        code_match = self.code_block_pattern.search(text)
        if code_match:
            text = code_match.group(1)
        
        # Find all matches
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
    """Parse Qwen/OSS model format: various Qwen-style patterns
    
    Examples:
    <|start|>assistant<|channel|>commentary to=tool:project-structure <|message|>...<|call|>
    <|start|>assistant<|channel|>commentary to=TOOL project-structure <|message|>...<|call|>
    <|channel|>commentary to=code-search <|constrain|>...<|message|>{...}
    to=tool.file-read {"fileName":"hello.py"}
    """
    
    name = "qwen_style"
    
    # Pattern 1: Full Qwen format with <|start|>...<|call|>
    pattern_full = re.compile(
        r'<\|start\|>.*?to=(?:tool[:\s.]|TOOL\s+)(\S+).*?<\|message\|>(.*?)<\|call\|>',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 2: Simple format: to=tool.name {...} or to=tool:name {...}
    pattern_simple = re.compile(
        r'to=tool[:\.](\S+)\s+(\{.*?\})',
        re.DOTALL | re.IGNORECASE
    )
    
    # Pattern 3: gpt-oss format: <|channel|>...to=name OR to=TOOL name...<|message|>{...}
    pattern_gpt_oss = re.compile(
        r'<\|channel\|>.*?to=(?:TOOL\s+)?(\S+).*?<\|message\|>(\{[^}]+\})',
        re.DOTALL | re.IGNORECASE
    )
    
    def try_parse(self, text: str) -> ToolCall | None:
        # Try full Qwen format first
        match = self.pattern_full.search(text)
        if not match:
            # Try gpt-oss format: <|channel|>...to=name...<|message|>{...}
            match = self.pattern_gpt_oss.search(text)
        if not match:
            # Try simple format: to=tool.name {...}
            match = self.pattern_simple.search(text)
        
        if not match:
            return None
        
        tool_name = match.group(1).strip()
        message_content = match.group(2).strip()
        
        # Map tool names from Qwen format to our tool names
        # Note: Our tools use kebab-case (e.g., "project-structure", "code-edit")
        tool_name_map = {
            "create": "code-edit",
            "read": "file-read", 
            "search": "code-search",
            "exec": "command-exec",
        }
        
        # Use mapped name if exists, otherwise keep original (don't replace dashes)
        mapped_name = tool_name_map.get(tool_name, tool_name)
        
        # Try to parse message as JSON arguments
        try:
            args = json.loads(message_content)
            # Transform args to match our tool format if needed
            if mapped_name == "code-edit" and "filepath" in args:
                args["operation"] = args.get("operation", "create")
        except json.JSONDecodeError:
            args = message_content
        
        return ToolCall(
            name=mapped_name,
            arguments=args,
            raw_match=match.group(0),
            format_name=self.name
        )


class JsonBlockParser(BaseToolParser):
    """Parse JSON code blocks with tool/function info.
    
    Examples:
    ```json
    {"tool": "file_read", "args": {"path": "main.py"}}
    ```
    
    ```json
    {"function": "code_edit", "arguments": {"filepath": "test.py"}}
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
            
            # Support various key names
            name = data.get("tool") or data.get("function") or data.get("name") or data.get("tool_name")
            if not name:
                return None
            
            args = data.get("args") or data.get("arguments") or data.get("parameters") or data.get("params") or {}
            
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
    
    Examples:
    <function_call name="file_read">{"path": "main.py"}</function_call>
    <tool_call name="code_edit" args='{"filepath": "test.py"}' />
    <tool name="search">query text</tool>
    """
    
    name = "xml_function"
    patterns = [
        # <function_call name="...">...</function_call>
        re.compile(r'<function_call\s+name=["\']([^"\']+)["\']>(.*?)</function_call>', re.DOTALL),
        # <tool_call name="..." args="..." />
        re.compile(r'<tool_call\s+name=["\']([^"\']+)["\']\s+args=["\']([^"\']+)["\']', re.DOTALL),
        # <tool name="...">...</tool>
        re.compile(r'<tool\s+name=["\']([^"\']+)["\']>(.*?)</tool>', re.DOTALL),
        # <call_function name="...">...</call_function>
        re.compile(r'<call_function\s+name=["\']([^"\']+)["\']>(.*?)</call_function>', re.DOTALL),
    ]
    
    def try_parse(self, text: str) -> ToolCall | None:
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1)
                args_str = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                # Try to parse as JSON
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
        return None


class PythonicParser(BaseToolParser):
    """Parse function-call style syntax.
    
    Examples:
    file_read(path="main.py")
    code_edit(filepath="test.py", operation="create", content="hello")
    
    Note: Only matches known tool names to avoid false positives.
    """
    
    name = "pythonic"
    known_tools = {
        "file-read", "code-edit", "code-search", "project-structure", 
        "command-exec", "read_file", "write_file", "search", "execute"
    }
    
    def try_parse(self, text: str) -> ToolCall | None:
        # Match tool_name(args) pattern
        for tool_name in self.known_tools:
            pattern = re.compile(rf'{tool_name}\s*\((.*?)\)', re.DOTALL)
            match = pattern.search(text)
            if match:
                args_str = match.group(1).strip()
                args = self._parse_kwargs(args_str)
                if args is not None:
                    return ToolCall(
                        name=tool_name,
                        arguments=args,
                        raw_match=match.group(0),
                        format_name=self.name
                    )
        return None
    
    def _parse_kwargs(self, args_str: str) -> dict | None:
        """Parse Python kwargs string to dict."""
        if not args_str:
            return {}
        
        # Try simple key=value parsing
        result = {}
        # Match key="value" or key='value' or key=value patterns
        pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
        
        for match in pattern.finditer(args_str):
            key = match.group(1)
            value = match.group(2) or match.group(3) or match.group(4)
            result[key] = value
        
        return result if result else None


class GenericJsonParser(BaseToolParser):
    """Parse any JSON object that looks like a tool call.
    
    This is a fallback parser that tries to find JSON with tool-related keys.
    """
    
    name = "generic_json"
    # Match JSON objects
    pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    
    tool_keys = {"tool", "function", "name", "tool_name", "action"}
    args_keys = {"args", "arguments", "parameters", "params", "input"}
    
    def try_parse(self, text: str) -> ToolCall | None:
        for match in self.pattern.finditer(text):
            try:
                data = json.loads(match.group(0))
                if not isinstance(data, dict):
                    continue
                
                # Check for tool-like structure
                name = None
                for key in self.tool_keys:
                    if key in data:
                        name = data[key]
                        break
                
                if not name:
                    continue
                
                # Get arguments
                args = {}
                for key in self.args_keys:
                    if key in data:
                        args = data[key]
                        break
                
                # If no explicit args key, use remaining keys as args
                if not args:
                    args = {k: v for k, v in data.items() if k not in self.tool_keys}
                
                return ToolCall(
                    name=name,
                    arguments=args,
                    raw_match=match.group(0),
                    format_name=self.name
                )
            except json.JSONDecodeError:
                continue
        
        return None


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
        # Parsers in priority order
        self.parsers: list[BaseToolParser] = [
            SupercoderTagParser(),
            QwenStyleParser(),
            JsonBlockParser(),
            XmlFunctionParser(),
            PythonicParser(),
            GenericJsonParser(),
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

