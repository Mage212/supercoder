"""Tool calling prompt templates for different model types.

Each model may expect tools to be called in a specific format.
This module provides format-specific instructions that are injected
into the system prompt based on the model's `tool_calling_type`.
"""

# Tool calling instruction templates for each supported type
TOOL_CALLING_PROMPTS = {
    # Our native format - most explicit and clear
    "supercoder": """# Tool Calling
Call tools with <@TOOL>{"name": "<tool-name>", "arguments": <json-args>}</@TOOL>

**Example:**
<@TOOL>{"name": "file-read", "arguments": {"path": "main.py"}}</@TOOL>

**Multiple tool calls:**
You can call multiple tools in one response:
<@TOOL>{"name": "file-read", "arguments": {"path": "file1.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"path": "file2.py"}}</@TOOL>
""",

    # Qwen-style format used by gpt-oss, deepresearch, and similar models
    "qwen_like": """# Tool Calling
Call tools using this format:
to=tool:<tool-name> <json-arguments>

**Example:**
to=tool:file-read {"path": "main.py"}

**Available argument formats:**
to=tool:code-edit {"file": "app.py", "operation": "create", "content": "print('hello')"}
to=tool:command-exec {"command": "ls -la", "timeout": 30}
to=tool:code-search {"query": "def main", "path": "."}
to=tool:project-structure {"path": "."}

**Important:** Always use valid JSON for arguments. Use double quotes for strings.
""",

    # JSON code block format - common with many instruction-tuned models
    "json_block": """# Tool Calling
Call tools using JSON code blocks:

```json
{"tool": "<tool-name>", "arguments": {"arg1": "value1", "arg2": "value2"}}
```

**Example:**
```json
{"tool": "file-read", "arguments": {"path": "main.py"}}
```

**For code-edit:**
```json
{"tool": "code-edit", "arguments": {"file": "app.py", "operation": "create", "content": "print('hello')"}}
```

**Important:** Use proper JSON formatting with double quotes.
""",

    # XML function call format
    "xml_function": """# Tool Calling
Call tools using XML syntax:

<function_call name="<tool-name>">
{"arg1": "value1", "arg2": "value2"}
</function_call>

**Example:**
<function_call name="file-read">
{"path": "main.py"}
</function_call>

**For code-edit:**
<function_call name="code-edit">
{"file": "app.py", "operation": "create", "content": "print('hello')"}
</function_call>
""",

    # OpenAI-compatible function calling (for reference, though most use native)
    "openai_native": """# Tool Calling
You have access to tools. When you need to use a tool, respond with a function call.
The system will execute the tool and provide results.

Call tools by specifying the tool name and arguments in JSON format.
""",
}

# Valid tool calling types for validation
VALID_TOOL_CALLING_TYPES = set(TOOL_CALLING_PROMPTS.keys())


def get_tool_calling_prompt(tool_calling_type: str) -> str:
    """Get tool calling instructions for the specified type.
    
    Args:
        tool_calling_type: The type of tool calling format to use.
                          Valid values: supercoder, qwen_like, json_block, xml_function
    
    Returns:
        String containing tool calling instructions for the system prompt.
        Falls back to 'supercoder' format if type is unknown.
    """
    return TOOL_CALLING_PROMPTS.get(tool_calling_type, TOOL_CALLING_PROMPTS["supercoder"])


def get_available_types() -> list[str]:
    """Get list of available tool calling types."""
    return list(TOOL_CALLING_PROMPTS.keys())
