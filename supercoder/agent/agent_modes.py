"""Agent modes for SuperCoder.

Defines operating modes and short runtime instructions for each mode.
Mode enforcement is handled by the host before tool execution; these
instructions are sent in-band only when the mode changes.
"""

from dataclasses import dataclass
from enum import Enum

from ..ui import theme


class AgentMode(Enum):
    """Operating modes for the coding agent."""

    ASK = "ask"  # Q&A mode: can only read and analyze code
    PLAN = "plan"  # Planning mode: read-only plus plans under .supercoder/plans
    CODE = "code"  # Default work mode: file edits require host approval
    ACCEPT_EDITS = "accept-edits"  # Editing mode: file edits allowed


@dataclass
class ModeConfig:
    """Configuration for an agent mode."""

    name: str
    instruction: str
    allowed_tools: list[str] | None  # None means all tools allowed
    toolbar: str
    prompt_label: str
    # Presentation tokens sourced from ui.theme.MODE_STYLE (single source of truth).
    color: str = ""
    icon: str = ""


READ_ONLY_TOOLS = [
    "file-read",  # Read file contents
    "code-search",  # Search code patterns
    "glob",  # Find files by pattern
    "project-structure",  # Show directory structure
]

ASK_MODE_TOOLS = READ_ONLY_TOOLS
PLAN_MODE_TOOLS = [*READ_ONLY_TOOLS, "code-edit"]

MODE_CYCLE = [
    AgentMode.ASK,
    AgentMode.PLAN,
    AgentMode.CODE,
    AgentMode.ACCEPT_EDITS,
]


MODE_CONFIGS = {
    AgentMode.ASK: ModeConfig(
        name="ask",
        instruction=(
            "Current mode: ASK. You may answer questions and inspect code with read-only "
            "tools only. Do not edit files and do not run shell commands."
        ),
        allowed_tools=ASK_MODE_TOOLS,
        toolbar="read/search only; no edits or shell",
        prompt_label="ask",
        color=theme.MODE_STYLE["ASK"]["color"],
        icon=theme.MODE_STYLE["ASK"]["icon"],
    ),
    AgentMode.PLAN: ModeConfig(
        name="plan",
        instruction=(
            "Current mode: PLAN. You may inspect the project and prepare plans. "
            "Project file edits and shell commands are blocked. You may save plans only "
            "under .supercoder/plans/; the host will enforce date-prefixed plan filenames."
        ),
        allowed_tools=PLAN_MODE_TOOLS,
        toolbar="read/search; plans only in .supercoder/plans",
        prompt_label="plan",
        color=theme.MODE_STYLE["PLAN"]["color"],
        icon=theme.MODE_STYLE["PLAN"]["icon"],
    ),
    AgentMode.CODE: ModeConfig(
        name="code",
        instruction=(
            "Current mode: CODE. You may inspect code and run useful shell commands through "
            "the host permission flow. File edits require host approval before execution; "
            "switch to /accept-edits when the user wants edits applied without per-edit prompts."
        ),
        allowed_tools=None,
        toolbar="read/search; shell/edit ask",
        prompt_label="code",
        color=theme.MODE_STYLE["CODE"]["color"],
        icon=theme.MODE_STYLE["CODE"]["icon"],
    ),
    AgentMode.ACCEPT_EDITS: ModeConfig(
        name="accept-edits",
        instruction=(
            "Current mode: ACCEPT-EDITS. You may inspect code and edit files when needed. "
            "Shell commands still go through the host permission flow."
        ),
        allowed_tools=None,
        toolbar="edits allowed; shell still asks",
        prompt_label="accept",
        color=theme.MODE_STYLE["ACCEPT_EDITS"]["color"],
        icon=theme.MODE_STYLE["ACCEPT_EDITS"]["icon"],
    ),
}
