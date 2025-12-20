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


def build_system_prompt(
    tools: list, 
    rules: str = "", 
    tool_calling_type: str = "supercoder",
    mode_suffix: str = ""
) -> str:
    """Build system prompt with available tools and project rules.
    
    Args:
        tools: List of available tools.
        rules: Optional project-specific rules to include.
        tool_calling_type: Type of tool calling format to use in instructions.
                          Valid: supercoder, qwen_like, json_block, xml_function
        mode_suffix: Additional prompt suffix for specific modes (e.g., ask mode).
    """
    if not tools:
        tool_list = "(no tools available yet)"
    else:
        tool_list = "\n".join(
            f"- {t.definition.name}: {t.definition.description}" 
            for t in tools
        )
    
    # Get tool calling instructions for this model type
    tool_calling_instructions = get_tool_calling_prompt(tool_calling_type)
    
    prompt = SYSTEM_PROMPT.format(
        tools=tool_list,
        tool_calling_instructions=tool_calling_instructions
    )
    
    # Add project rules if provided
    if rules:
        prompt += f"\n{rules}"
    
    # Add mode-specific suffix (e.g., ask mode restrictions)
    if mode_suffix:
        prompt += f"\n{mode_suffix}"
    
    return prompt

