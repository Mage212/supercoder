"""Tests for Phase 2 rollback semantics (backlog B-035).

Covers the rollback fixes from docs/code-review-2026-06-15.md:
- B4: RollbackResult contract — ``_restore_files`` tracks failed paths so a
  partial rollback is never reported as fully successful.
- B1: rollback only when the current turn actually performed file edits — a
  read-only/tool exception must not roll back unrelated state.
- B3: a successful code-edit survives the turn (mid-turn commit then a fresh
  checkpoint keeps subsequent edits protected).
"""

from unittest.mock import MagicMock

from supercoder.agent.agent_modes import AgentMode
from supercoder.agent.coder_agent import CoderAgent
from supercoder.checkpoint import CheckpointManager, RollbackResult
from supercoder.context import ContextConfig
from supercoder.llm.base import CompletionResult, NativeToolCall
from supercoder.tools.base import BaseTool, ToolDefinition
from supercoder.tools.code_edit import CodeEditTool


class FailingTool(BaseTool):
    """Tool that always raises during execute()."""

    @property
    def definition(self):
        return ToolDefinition(name="failing-tool", description="Always fails")

    def execute(self, arguments):
        raise RuntimeError("boom")


def _make_agent(tmp_path, tools):
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.config = MagicMock()
    mock_llm.config.model = "test-model"
    agent = CoderAgent(
        llm=mock_llm,
        tools=tools,
        context_config=ContextConfig(max_tokens=32000),
        streaming=False,
        use_repo_map=False,
        repo_root=str(tmp_path),
    )
    return agent, mock_llm


# ── B4: RollbackResult contract ──


class TestRollbackResultContract:
    def test_truthy_when_anything_attempted(self):
        assert bool(RollbackResult(restored=["a.py"]))
        assert bool(RollbackResult(failed=["b.py"]))
        assert not bool(RollbackResult())

    def test_empty_rollback_when_no_current_checkpoint(self, tmp_path):
        cm = CheckpointManager(tmp_path)
        result = cm.rollback()
        assert isinstance(result, RollbackResult)
        assert result.restored == []
        assert result.failed == []

    def test_restore_files_reports_missing_backup_as_failed(self, tmp_path):
        """A vanished backup must surface in ``failed`` instead of being silently skipped."""
        cm = CheckpointManager(tmp_path)
        cp = cm.create("test")
        src = tmp_path / "src.py"
        src.write_text("original\n")
        assert cm.backup_file(src) is True

        # Remove the backup copy so restore cannot find it.
        checkpoint_dir = tmp_path / ".supercoder" / "checkpoints" / cp.id
        for entry in checkpoint_dir.iterdir():
            if entry.name != "metadata.json":
                entry.unlink()

        restored, failed = cm._restore_files(cp)
        assert restored == []
        assert any("src.py" in p for p in failed)


# ── B1: rollback only on actual file edits ──


class TestRollbackOnlyOnEdits:
    def test_tool_exception_without_edits_skips_rollback(self, tmp_path):
        """A non-edit tool raising must not roll back (nothing was changed)."""
        agent, mock_llm = _make_agent(tmp_path, [FailingTool()])
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[NativeToolCall(id="c1", name="failing-tool", arguments={})],
                raw_tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "failing-tool", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Recovered.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("go"))

        assert [e for e in events if e["type"] == "rollback"] == []
        assert any(e["type"] == "error" for e in events)

    def test_edit_then_failing_tool_rolls_back_the_edit(self, tmp_path):
        """B1 core: an applied code-edit followed by a failing tool rolls back the edit."""
        (tmp_path / "target.py").write_text("a = 1\n")
        agent, mock_llm = _make_agent(tmp_path, [CodeEditTool(), FailingTool()])
        agent.set_mode(AgentMode.ACCEPT_EDITS)
        agent.freshness_tracker.mark_read(tmp_path / "target.py", source="test")
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="c1",
                        name="code-edit",
                        arguments={
                            "filepath": "target.py",
                            "operation": "search_replace",
                            "search": "a = 1",
                            "replace": "a = 2",
                        },
                    ),
                    NativeToolCall(id="c2", name="failing-tool", arguments={}),
                ],
                raw_tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "failing-tool", "arguments": "{}"},
                    },
                ],
            ),
            CompletionResult(content="Recovered.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("go"))
        rollback_events = [e for e in events if e["type"] == "rollback"]
        assert rollback_events  # B1: rollback triggered because has_file_edits was True
        assert any("target.py" in p for p in rollback_events[0]["content"]["restored"])
        assert (tmp_path / "target.py").read_text() == "a = 1\n"  # edit reverted


# ── B3: edits survive mid-turn commit; fresh checkpoint protects later edits ──


class TestCheckpointProtectionAcrossTurn:
    def test_successful_edit_is_committed_not_rolled_back(self, tmp_path):
        """After edits + a final no-tool-call response, the edit stays applied."""
        (tmp_path / "target.py").write_text("a = 1\n")
        agent, mock_llm = _make_agent(tmp_path, [CodeEditTool()])
        agent.set_mode(AgentMode.ACCEPT_EDITS)
        # Satisfy read-before-edit freshness so the edit applies in the test.
        agent.freshness_tracker.mark_read(tmp_path / "target.py", source="test")
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="c1",
                        name="code-edit",
                        arguments={
                            "filepath": "target.py",
                            "operation": "search_replace",
                            "search": "a = 1",
                            "replace": "a = 2",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("edit"))

        assert not any(e["type"] == "rollback" for e in events)
        assert (tmp_path / "target.py").read_text() == "a = 2\n"

    def test_two_sequential_edits_both_persisted(self, tmp_path):
        """B3 core: a second edit in the same turn is protected by the fresh checkpoint."""
        (tmp_path / "target.py").write_text("a = 1\nb = 2\n")
        agent, mock_llm = _make_agent(tmp_path, [CodeEditTool()])
        agent.set_mode(AgentMode.ACCEPT_EDITS)
        agent.freshness_tracker.mark_read(tmp_path / "target.py", source="test")
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="c1",
                        name="code-edit",
                        arguments={
                            "filepath": "target.py",
                            "operation": "search_replace",
                            "search": "a = 1",
                            "replace": "a = 99",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="c2",
                        name="code-edit",
                        arguments={
                            "filepath": "target.py",
                            "operation": "search_replace",
                            "search": "b = 2",
                            "replace": "b = 98",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("edit twice"))
        assert not any(e["type"] == "rollback" for e in events)
        content = (tmp_path / "target.py").read_text()
        assert "a = 99" in content
        assert "b = 98" in content
