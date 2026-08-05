"""Tools module."""

from .base import BaseTool, ToolDefinition
from .code_edit import CodeEditTool
from .code_search import CodeSearchTool
from .command_exec import CommandExecutionTool
from .file_read import FileReadTool
from .glob_tool import GlobTool
from .project_structure import ProjectStructureTool
from .recall import RecallTool

# All available tools
ALL_TOOLS = [
    FileReadTool(),
    CodeSearchTool(),
    GlobTool(),
    CodeEditTool(),
    ProjectStructureTool(),
    CommandExecutionTool(),
    RecallTool(),
]

# Aliases for tool names that models commonly hallucinate instead of the
# canonical kebab-case names. Applied uniformly in both native and streaming
# paths (coder_agent.py) before the ``name not in self.tools`` check, so a
# model that calls ``read``/``view``/``cat`` still reaches ``file-read``.
TOOL_ALIASES: dict[str, str] = {
    # file-read synonyms (models trained on other agents' toolsets)
    "read": "file-read",
    "view": "file-read",
    "cat": "file-read",
    "open": "file-read",
    "show": "file-read",
    "get_file": "file-read",
    "readfile": "file-read",
    # underscore variants
    "file_read": "file-read",
    # code-edit synonyms (qwen3.5 / small models invent these)
    "file-create": "code-edit",
    "file-write": "code-edit",
    "create-file": "code-edit",
    "write-file": "code-edit",
    "file_edit": "code-edit",
    "code_edit": "code-edit",
    "edit": "code-edit",
    "write": "code-edit",
    "replace": "code-edit",
    # code-search synonyms
    "code_search": "code-search",
    "search": "code-search",
    "grep": "code-search",
    "find": "code-search",
    # command-exec synonyms
    "run-command": "command-exec",
    "run_command": "command-exec",
    "execute": "command-exec",
    "run": "command-exec",
    "shell": "command-exec",
    "bash": "command-exec",
    # project-structure synonyms
    "ls": "project-structure",
    "list": "project-structure",
    "tree": "project-structure",
    "structure": "project-structure",
    # recall synonyms (retrieval over past events)
    "search_log": "recall",
    "history": "recall",
    "recall_events": "recall",
}

__all__ = [
    "ALL_TOOLS",
    "TOOL_ALIASES",
    "BaseTool",
    "CodeEditTool",
    "CodeSearchTool",
    "CommandExecutionTool",
    "FileReadTool",
    "GlobTool",
    "ProjectStructureTool",
    "RecallTool",
    "ToolDefinition",
]
