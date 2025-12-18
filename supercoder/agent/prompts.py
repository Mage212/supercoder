"""System prompts for the agent."""

# Compact prompt optimized for local/smaller models
SYSTEM_PROMPT = """You are a coding assistant.

# Tool Calling
Call tools with <@TOOL>{{"name": "<tool-name>", "arguments": "<json-args>"}}</@TOOL>

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


def build_system_prompt(tools: list, rules: str = "") -> str:
    """Build system prompt with available tools and project rules.
    
    Args:
        tools: List of available tools.
        rules: Optional project-specific rules to include.
    """
    if not tools:
        tool_list = "(no tools available yet)"
    else:
        tool_list = "\n".join(
            f"- {t.definition.name}: {t.definition.description}" 
            for t in tools
        )
    
    prompt = SYSTEM_PROMPT.format(tools=tool_list)
    
    # Add project rules if provided
    if rules:
        prompt += f"\n{rules}"
    
    return prompt

