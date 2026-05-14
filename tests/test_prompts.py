"""Tests for prompt construction and mode policy."""

from supercoder.agent.agent_modes import MODE_CONFIGS, AgentMode
from supercoder.agent.prompts import build_system_prompt


def test_lean_prompt_includes_compact_project_rules():
    rules = """
# Project Rules
The following are project-specific coding rules you MUST follow:

## safety
Use AtomicWriter for file writes.
"""

    prompt = build_system_prompt([], rules=rules, native_tools=True, lean=True)

    assert "Project Rules (mandatory)" in prompt
    assert "Use AtomicWriter for file writes." in prompt


def test_code_mode_prompt_uses_cautious_command_policy():
    suffix = MODE_CONFIGS[AgentMode.CODE].prompt_suffix

    assert "Do NOT refuse to execute commands" not in suffix
    assert "Refuse clearly dangerous commands" in suffix
    assert "Ask for confirmation" in suffix


def test_code_edit_prompt_example_matches_tool_schema():
    suffix = MODE_CONFIGS[AgentMode.CODE].prompt_suffix

    assert '"filepath"' in suffix
    assert '"operation"' in suffix
    assert '{"file": "path"' not in suffix
