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


def test_code_mode_instruction_uses_cautious_command_policy():
    instruction = MODE_CONFIGS[AgentMode.CODE].instruction

    assert "Do NOT refuse to execute commands" not in instruction
    assert "host permission flow" in instruction
    assert "File edits are blocked" in instruction


def test_accept_edits_instruction_enables_file_edits():
    instruction = MODE_CONFIGS[AgentMode.ACCEPT_EDITS].instruction

    assert "edit files" in instruction
    assert "Shell commands still go through" in instruction
