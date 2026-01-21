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
<@TOOL>{"name": "file-read", "arguments": {"fileName": "main.py"}}</@TOOL>

**Multiple tool calls:**
You can call multiple tools in one response:
<@TOOL>{"name": "file-read", "arguments": {"fileName": "file1.py"}}</@TOOL>
<@TOOL>{"name": "file-read", "arguments": {"fileName": "file2.py"}}</@TOOL>
""",

    # Qwen-style format used by gpt-oss, deepresearch, and similar models
    "qwen_like": """# Tool Calling
Call tools using this format:
to=tool:<tool-name> <json-arguments>

**Example:**
to=tool:file-read {"fileName": "main.py"}

**Available argument formats:**
to=tool:code-edit {"filepath": "app.py", "operation": "create", "content": "print('hello')"}
to=tool:command-exec {"command": "ls -la", "timeout": 30}
to=tool:code-search {"query": "def main"}
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
{"tool": "file-read", "arguments": {"fileName": "main.py"}}
```

**For code-edit:**
```json
{"tool": "code-edit", "arguments": {"filepath": "app.py", "operation": "create", "content": "print('hello')"}}
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
{"fileName": "main.py"}
</function_call>

**For code-edit:**
<function_call name="code-edit">
{"filepath": "app.py", "operation": "create", "content": "print('hello')"}
</function_call>
""",

    # GLM-4.7-Flash style format with <tool_call> tags and arg_key/arg_value pairs
    "glm_tool_call": """# Tool Calling
Call tools using XML-style tags with argument key-value pairs:

<tool_call>tool-name<arg_key>arg1</arg_key><arg_value>value1<arg_key>arg2</arg_key><arg_value>value2</arg_value></tool_call>

**Example - read a file:**
<tool_call>file-read<arg_key>fileName</arg_key><arg_value>main.py</arg_value></tool_call>

**Example - search code:**
<tool_call>code-search<arg_key>query</arg_key><arg_value>def main<arg_key>maxResults</arg_key><arg_value>10</arg_value></tool_call>

**Example - create file:**
<tool_call>code-edit<arg_key>filepath</arg_key><arg_value>app.py<arg_key>operation</arg_key><arg_value>create<arg_key>content</arg_key><arg_value>print('hello')</arg_value></tool_call>

**Important:** Use <arg_key> and <arg_value> tags for each argument.
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
