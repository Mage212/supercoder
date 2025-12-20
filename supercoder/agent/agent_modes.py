"""Agent modes for SuperCoder.

Defines different operating modes that change agent behavior and available tools.
"""

from enum import Enum
from dataclasses import dataclass


class AgentMode(Enum):
    """Operating modes for the coding agent."""
    
    CODE = "code"  # Default mode: can edit files, run commands
    ASK = "ask"    # Q&A mode: can only read and analyze code


@dataclass
class ModeConfig:
    """Configuration for an agent mode."""
    
    name: str
    prompt_suffix: str
    allowed_tools: list[str] | None  # None means all tools allowed
    
    
# Read-only tools that are always safe in ask mode
ASK_MODE_TOOLS = [
    "file-read",          # Read file contents
    "code-search",        # Search code patterns
    "project-structure",  # Show directory structure
]


MODE_CONFIGS = {
    AgentMode.CODE: ModeConfig(
        name="code",
        prompt_suffix="""
MODE: CODE (Full editing enabled)

You are in CODE mode with FULL capabilities. You MUST use the available tools to accomplish tasks:

✅ AVAILABLE TOOLS:
- file-read: Read files → {"fileName": "path/to/file.py"}
- code-edit: Edit files → {"file": "path", "search": "old", "replace": "new"}
- code-search: Search code → {"query": "pattern"}
- command-exec: RUN SCRIPTS AND COMMANDS → {"command": "python script.py"}
- project-structure: Show files → {"path": "."}

⚡ TO RUN A SCRIPT, use command-exec:
{"command": "python dungeon_crawler.py"}

You are AUTHORIZED and EXPECTED to:
- Edit and modify files
- Execute shell commands and scripts
- Create new files
- Use ALL available tools

Do NOT refuse to execute commands. Use the tools provided.
""",
        allowed_tools=None,  # All tools allowed
    ),
    AgentMode.ASK: ModeConfig(
        name="ask",
        prompt_suffix="""
IMPORTANT: You are in ASK mode.

You CAN:
- Answer questions about the code
- Explain how code works
- Suggest approaches and solutions
- Read files to understand the codebase

You CANNOT:
- Edit or modify any files
- Create new files  
- Execute commands that modify the system

If asked to make changes, explain what would be needed and suggest using /code command.
""",
        allowed_tools=ASK_MODE_TOOLS,
    ),
}

