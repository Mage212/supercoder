"""System prompts for the agent."""

from .tool_calling_prompts import get_tool_calling_prompt

# Compact prompt optimized for local/smaller models
# Tool calling instructions are injected dynamically based on model's tool_calling_type
SYSTEM_PROMPT = """You are a coding assistant.

{tool_calling_instructions}

Available tools:
{tools}

# Rules
1. Read files before editing
2. Use diff-based edits when possible
3. Ask before destructive commands
4. Be concise in responses
"""

SYSTEM_PROMPT_LEAN = """You are a coding assistant.
{tool_calling_instructions}
Tools: {tools}
"""


# Prompt for context summarization (/compact command)
CONTEXT_SUMMARY_PROMPT = """Analyze the following conversation history and create a concise but informative summary.

Important guidelines:
- Highlight key decisions and results
- Emphasize the most recent messages (they are most relevant)
- Preserve important technical context (files, functions, errors, code changes)
- The summary should allow continuing work without losing context

Conversation history:
{conversation_history}

Create a summary in this format:

## Working Context
[Brief description of the task and current state]

## Key Results
[What was accomplished]

## Current Focus
[What was being worked on in recent messages]

## Important Details
[Technical details to remember: file paths, function names, decisions made]
"""


# Cache-aware compact request appended as the next user message in the current chat.
# This preserves the existing prompt prefix so llama.cpp-like backends can reuse KV cache.
CACHE_AWARE_COMPACT_REQUEST = """[Context maintenance request]

Create a concise continuation summary of the conversation so far.

Rules:
- Return only the summary, no preamble.
- Do not call tools.
- Do not continue the user's task.
- Preserve current goal, key decisions, files touched/read, commands/results, errors, and next step.

Format:

## Working Context
[Task goal and current state]

## Key Decisions
[Important choices and constraints]

## Files and Commands
[Relevant paths, tools, commands, and results]

## Current Focus
[The exact next step to continue from]
"""


def build_system_prompt(
    tools: list,
    rules: str = "",
    tool_calling_type: str = "supercoder",
    mode_suffix: str = "",
    native_tools: bool = False,
    lean: bool = False,
) -> str:
    """Build system prompt with available tools and project rules.

    Args:
        tools: List of available tools.
        rules: Optional project-specific rules to include.
        tool_calling_type: Type of tool calling format (only used when native_tools=False).
        mode_suffix: Additional prompt suffix for specific modes (e.g., ask mode).
        native_tools: If True, tools are passed via API — skip verbose format instructions.
        lean: If True, use shorter prompts for weak/local models.
    """
    if not tools:
        tool_list = "(no tools available yet)"
    else:
        tool_list = "\n".join(f"- {t.definition.name}: {t.definition.description}" for t in tools)

    # Get tool calling instructions
    if native_tools:
        # Tools are passed via the API `tools` parameter — minimal prompt
        tool_calling_instructions = (
            "You have access to tools. Call them when needed to accomplish the task. "
            "The system handles tool execution and returns results automatically."
        )
    else:
        tool_calling_instructions = get_tool_calling_prompt(tool_calling_type)

    template = SYSTEM_PROMPT_LEAN if lean else SYSTEM_PROMPT
    prompt = template.format(tools=tool_list, tool_calling_instructions=tool_calling_instructions)

    # Skip project rules in lean mode to save tokens
    if rules:
        if lean:
            compact_rules = _compact_project_rules(rules)
            prompt += (
                "\n# Project Rules (mandatory)\n"
                "Follow these project rules. They override shorter lean-mode instructions:\n"
                f"{compact_rules}\n"
            )
        else:
            prompt += f"\n{rules}"

    # Add mode-specific suffix (e.g., ask mode restrictions)
    if mode_suffix:
        prompt += f"\n{mode_suffix}"

    return prompt


def _compact_project_rules(rules: str, max_chars: int = 4000) -> str:
    """Compact project rules for lean mode without dropping them entirely."""
    compact = "\n".join(line.strip() for line in rules.splitlines() if line.strip())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 18].rstrip() + "\n...[truncated]"
