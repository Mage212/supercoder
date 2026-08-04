"""Tests for Phase 2 rollback semantics (backlog B-035).

Covers the rollback fixes from docs/code-review-2026-06-15.md:
- B4: RollbackResult contract — ``_restore_files`` tracks failed paths so a
  partial rollback is never reported as fully successful.
- B1: rollback only when the current turn actually performed file edits — a
  read-only/tool exception must not roll back unrelated state.
- B3: a successful code-edit survives the turn (mid-turn commit then a fresh
  checkpoint keeps subsequent edits protected).
"""

import json
from pathlib import Path
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


class TestBackupFileDedup:
    def test_backup_file_dedup_relative_and_absolute(self, tmp_path, monkeypatch):
        """backup_file must dedup on a normalized key, not mix relative/absolute.

        Regression: the membership check used ``str(file_path)`` while storage
        used ``str(file_path.absolute())``. Calling backup_file twice on the
        same file — once relative, once absolute — missed the existing entry and
        re-copied the (possibly already-edited) file over the pristine backup,
        silently destroying rollback. Normalize the key in both places.
        """
        cm = CheckpointManager(tmp_path)
        cm.create("dedup test")
        src = tmp_path / "src.py"
        src.write_text("pristine\n")

        # Run from tmp_path so a relative path resolves to the same file.
        monkeypatch.chdir(tmp_path)

        # First backup with an absolute path.
        assert cm.backup_file(src) is True
        backup_entries_before = dict(cm.current.files)

        # Simulate an edit happening to the file between the two backup calls.
        src.write_text("modified\n")

        # Second backup with a relative path (different str(), same file on disk).
        assert cm.backup_file(Path("src.py")) is True

        # No new backup entry should have been created, and the stored backup
        # must still hold the pristine content (not the modified version).
        assert cm.current.files == backup_entries_before, (
            "second backup_file call created a duplicate or clobbered the key"
        )
        backup_path = Path(next(iter(cm.current.files.values())))
        assert backup_path.read_text() == "pristine\n", (
            "pristine backup was overwritten by the second (post-edit) backup_file call"
        )


class TestUndoPathContainment:
    """M5: /undo must never restore or delete paths outside the repo root.

    Checkpoint metadata (``.supercoder/checkpoints/<id>/metadata.json``) lives
    inside the repo and is therefore untrusted input — a cloned malicious repo
    can plant a checkpoint whose ``files``/``created_files`` point anywhere on
    the host. Before this fix, ``_restore_files`` copied backup→original and
    unlinked created_files with no containment check, so a planted checkpoint
    could overwrite ``~/.zshrc`` or delete arbitrary files on the first ``/undo``.
    """

    def test_undo_rejects_files_outside_repo(self, tmp_path):
        # repo is a subdir; victim lives OUTSIDE it (as a sibling).
        repo = tmp_path / "repo"
        repo.mkdir()
        cm = CheckpointManager(repo)
        cp_id = "20240101_000000_000000_planted"
        cp_dir = repo / ".supercoder" / "checkpoints" / cp_id
        cp_dir.mkdir(parents=True)

        # Victim file OUTSIDE the repo.
        victim = tmp_path / "victim.txt"
        victim.write_text("ORIGINAL INNOCENT CONTENT")

        # Planted backup with malicious content.
        bak = cp_dir / "deadbeef.bak"
        bak.write_text("PWNED BY PLANTED CHECKPOINT")

        meta = {
            "id": cp_id,
            "timestamp": "2024-01-01T00:00:00",
            "description": "planted",
            "files": {str(victim): str(bak)},
            "created_files": [],
        }
        (cp_dir / "metadata.json").write_text(json.dumps(meta))

        result = cm.undo_by_id(cp_id)

        # The victim file must be untouched and reported as failed (containment).
        assert victim.read_text() == "ORIGINAL INNOCENT CONTENT"
        assert any(str(victim) in p for p in result.failed), (
            "out-of-repo restore target should be reported as failed, not applied"
        )

    def test_undo_rejects_created_files_outside_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        cm = CheckpointManager(repo)
        cp_id = "20240101_000000_000000_planted2"
        cp_dir = repo / ".supercoder" / "checkpoints" / cp_id
        cp_dir.mkdir(parents=True)

        # A file OUTSIDE the repo that the planted checkpoint claims to have "created".
        outside = tmp_path / "outside_created.txt"
        outside.write_text("should not be deleted")

        meta = {
            "id": cp_id,
            "timestamp": "2024-01-01T00:00:00",
            "description": "planted",
            "files": {},
            "created_files": [str(outside)],
        }
        (cp_dir / "metadata.json").write_text(json.dumps(meta))

        result = cm.undo_by_id(cp_id)

        assert outside.exists(), "out-of-repo created_file must not be deleted"
        assert any(str(outside) in p for p in result.failed)

    def test_undo_still_restores_in_repo_files(self, tmp_path):
        """Containment check must not break the legitimate in-repo undo path."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "target.py").write_text("original\n")
        cm = CheckpointManager(repo)
        cm.create("edit")
        assert cm.backup_file(repo / "target.py") is True
        (repo / "target.py").write_text("edited\n")
        cm.commit()

        checkpoints = cm.list_checkpoints()
        result = cm.undo_by_id(checkpoints[0].id)

        assert any("target.py" in p for p in result.restored)
        assert (repo / "target.py").read_text() == "original\n"


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
