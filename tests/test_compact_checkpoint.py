"""Tests for compact goal re-injection and structured provenance checkpoint.

These cover points 2.1-2.4 of the epoch-based memory architecture:
- the original task goal is captured from the first user_input and survives
  compaction (set_initial_summary re-injects it);
- compact_context produces a host-generated provenance block (git HEAD,
  changed files, last test result) without an extra LLM call;
- the new task_goal and provenance fields round-trip through ChatSession
  serialization and remain covered by the session trust gate.
"""

import subprocess

from supercoder.agent.coder_agent import CoderAgent
from supercoder.context import ContextConfig, ContextWindowManager
from supercoder.context.session_manager import ChatSession, SessionManager
from supercoder.llm.base import CompletionResult, Message
from supercoder.tools import ALL_TOOLS


class StubLLM:
    """Minimal LLM stub returning a fixed summary for compact."""

    model = "stub-model"

    def __init__(self, summary: str = "## Working Context\nstub summary"):
        self._summary = summary

    def chat_with_tools_interruptible(self, messages, tools, abort_controller, **kwargs):
        return CompletionResult(
            content=self._summary, tool_calls=[], reasoning="", usage=None, truncated=False
        )


class TestGoalCapture:
    """ContextWindowManager.set_task_goal captures the first goal only."""

    def test_first_goal_is_kept(self):
        cm = ContextWindowManager(ContextConfig())
        assert cm.get_task_goal() is None

        cm.set_task_goal("Refactor the auth module")
        assert cm.get_task_goal() == "Refactor the auth module"

    def test_second_goal_does_not_overwrite(self):
        cm = ContextWindowManager(ContextConfig())
        cm.set_task_goal("first goal")

        cm.set_task_goal("second, different goal")
        # The original task focus must persist across refocusing within a task.
        assert cm.get_task_goal() == "first goal"

    def test_empty_goal_ignored(self):
        cm = ContextWindowManager(ContextConfig())
        cm.set_task_goal("   ")
        assert cm.get_task_goal() is None


class TestGoalReinjection:
    """set_initial_summary re-injects the goal as a user_input after compact."""

    def test_goal_reinjected_as_user_input(self):
        cm = ContextWindowManager(ContextConfig())
        cm.set_task_goal("Implement feature X")

        cm.set_initial_summary("summary of progress", recent_messages=None)

        messages = cm.get_messages_for_api()
        # Layout: [system?][repo_map?][compact_summary][goal user_input]
        goal_msgs = [m for m in messages if "Implement feature X" in m.content]
        assert len(goal_msgs) == 1
        goal_msg = goal_msgs[0]
        assert goal_msg.role == "user"
        # Marked as user_input so the model treats it as task framing.
        assert goal_msg.display_type == "user_input"

    def test_goal_survives_second_compact(self):
        cm = ContextWindowManager(ContextConfig())
        cm.set_task_goal("Long-running task goal")

        cm.set_initial_summary("first summary", recent_messages=None)
        cm.set_initial_summary("second summary", recent_messages=None)

        messages = cm.get_messages_for_api()
        assert any("Long-running task goal" in m.content for m in messages)

    def test_no_goal_no_reinjection(self):
        cm = ContextWindowManager(ContextConfig())
        # No goal captured.
        cm.set_initial_summary("summary", recent_messages=None)

        messages = cm.get_messages_for_api()
        # Only the compact_summary message (plus optional system/repo_map).
        types = [m.display_type for m in messages]
        assert "compact_summary" in types
        # No synthetic user_input goal block.
        assert not any("[Original Task Goal" in m.content for m in messages)

    def test_provenance_included_in_summary_block(self):
        cm = ContextWindowManager(ContextConfig())
        cm.set_task_goal("some goal")

        provenance = (
            "## Session Provenance\n"
            "- git HEAD: abc1234\n"
            "- files changed this session: 2 (a.py, b.py)\n"
            "- last test result: PASS — pytest"
        )
        cm.set_initial_summary("summary text", recent_messages=None, provenance=provenance)

        messages = cm.get_messages_for_api()
        summary_msg = next(m for m in messages if m.display_type == "compact_summary")
        assert "git HEAD: abc1234" in summary_msg.content
        assert "summary text" in summary_msg.content


class TestSessionCheckpointPersistence:
    """ChatSession round-trips task_goal and provenance through to_dict/from_dict."""

    def test_round_trip_with_checkpoint_fields(self):
        session = ChatSession(
            id="abc12345",
            title="test",
            created_at="2026-08-05T08:00:00",
            last_modified="2026-08-05T08:00:00",
            messages=[],
            task_goal="The original goal",
            provenance="## Session Provenance\n- git HEAD: abc1234",
        )

        data = session.to_dict()
        assert data["task_goal"] == "The original goal"
        assert "git HEAD: abc1234" in data["provenance"]

        restored = ChatSession.from_dict(data)
        assert restored.task_goal == "The original goal"
        assert "git HEAD: abc1234" in (restored.provenance or "")

    def test_round_trip_without_checkpoint_fields_backwards_compatible(self):
        # Old sessions saved before this feature have no task_goal/provenance.
        data = {
            "id": "abc12345",
            "title": "old",
            "created_at": "2026-08-01T00:00:00",
            "last_modified": "2026-08-01T00:00:00",
            "is_compacted": False,
            "messages": [],
        }
        session = ChatSession.from_dict(data)
        assert session.task_goal is None
        assert session.provenance is None


class TestProvenanceBlock:
    """compact_context builds a host-generated provenance block (no extra LLM call)."""

    def test_build_provenance_includes_git_head_and_goal(self, tmp_path, monkeypatch):
        agent = CoderAgent(StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path))
        agent.context.set_task_goal("ship the feature")

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="abc1234\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=" M a.py\n?? b.py\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        provenance = agent._build_provenance()

        assert provenance is not None
        assert "git HEAD: abc1234" in provenance
        assert "ship the feature" in provenance
        assert "a.py" in provenance and "b.py" in provenance

    def test_build_provenance_none_when_nothing_available(self, tmp_path, monkeypatch):
        agent = CoderAgent(StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path))

        # No goal, and git commands fail (not a repo).
        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repo")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert agent._build_provenance() is None

    def test_compact_injects_provenance_into_summary(self, tmp_path, monkeypatch):
        agent = CoderAgent(
            StubLLM(summary="working context summary"),
            tools=ALL_TOOLS,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.context.set_task_goal("the goal that must survive")
        agent.context.add_message(
            Message("user", "the goal that must survive", display_type="user_input")
        )
        agent.context.add_message(Message("assistant", "work in progress", display_type="response"))
        # Make sure there's enough to compact.
        agent.context.add_message(Message("user", "second turn", display_type="user_input"))
        agent.context.add_message(Message("assistant", "more work", display_type="response"))

        def fake_run(cmd, **kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")
            if cmd[:2] == ["git", "status"]:
                return subprocess.CompletedProcess(cmd, 0, stdout=" M src/x.py\n", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        _summary, _before, _after = agent.compact_context()

        messages = agent.context.get_messages_for_api()
        summary_msg = next(m for m in messages if m.display_type == "compact_summary")
        # Provenance was folded into the summary block.
        assert "git HEAD: deadbeef" in summary_msg.content
        assert "src/x.py" in summary_msg.content
        # Goal was re-injected as its own user_input.
        goal_msgs = [m for m in messages if "the goal that must survive" in m.content]
        assert any(m.display_type == "user_input" for m in goal_msgs)


class TestLastTestResult:
    """_last_test_result classifies test-runner output by parsed counts.

    The old substring heuristic misclassified real pytest output: pytest
    prints ``3 passed, 0 failed in 1.2s`` and the presence of the literal
    "failed" flipped it to FAIL. The fix parses numeric counts instead.
    """

    def _agent_with_test_result(self, tmp_path, tool_output, *, command=None):
        agent = CoderAgent(StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path))
        meta = {"command": command} if command else None
        agent.context.add_message(
            Message(
                "tool",
                tool_output,
                tool_call_id="call_test",
                name="command-exec",
                display_type="tool_result",
                display_meta=meta,
            )
        )
        return agent

    def test_pass_classification(self, tmp_path):
        agent = self._agent_with_test_result(tmp_path, "3 passed in 1.2s", command="pytest -x")
        result = agent._last_test_result()
        assert result is not None
        assert result.startswith("PASS")

    def test_pass_with_zero_failed(self, tmp_path):
        # Real pytest output: "3 passed, 0 failed" — the regression case.
        agent = self._agent_with_test_result(
            tmp_path, "3 passed, 0 failed in 1.2s", command="pytest"
        )
        result = agent._last_test_result()
        assert result is not None
        assert result.startswith("PASS"), f"expected PASS, got {result!r}"

    def test_fail_classification(self, tmp_path):
        agent = self._agent_with_test_result(
            tmp_path, "1 failed, 2 passed in 1.0s", command="pytest"
        )
        result = agent._last_test_result()
        assert result is not None
        assert result.startswith("FAIL")

    def test_error_classification(self, tmp_path):
        agent = self._agent_with_test_result(tmp_path, "2 errors in 0.5s", command="pytest")
        result = agent._last_test_result()
        assert result is not None
        assert result.startswith("FAIL")

    def test_run_fallback_no_counts(self, tmp_path):
        # Test output with no parseable pass/fail counts (e.g. still running
        # or a runner without standard summary lines).
        agent = self._agent_with_test_result(
            tmp_path, "running test suite...\ncompiling", command="cargo test"
        )
        result = agent._last_test_result()
        # Recognized as a test run but with no clear PASS/FAIL -> RUN.
        assert result is not None
        assert result.startswith("RUN")

    def test_non_test_command_ignored(self, tmp_path):
        # An unrelated command whose output happens to contain the substring
        # "test" must not be misclassified as a test run.
        agent = self._agent_with_test_result(
            tmp_path, "testing_notes.md\nsrc/main.py", command="ls"
        )
        assert agent._last_test_result() is None

    def test_no_test_in_history(self, tmp_path):
        agent = CoderAgent(StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path))
        # No messages at all.
        assert agent._last_test_result() is None


class TestSessionCompactPersistence:
    """task_goal and provenance persist through update_session_after_compact
    and survive a save/load cycle. This covers the kwargs branches that no
    other test exercises."""

    def test_update_session_after_compact_persists_new_fields(self, tmp_path):
        manager = SessionManager(tmp_path, allow_loading=True)
        session = ChatSession(
            id="sess1",
            title="t",
            created_at="2026-08-05T08:00:00",
            last_modified="2026-08-05T08:00:00",
            messages=[Message("user", "hi", display_type="user_input")],
        )
        recent = [Message("user", "latest", display_type="user_input")]

        manager.update_session_after_compact(
            session,
            "summary text",
            recent,
            provenance="## Session Provenance\n- git HEAD: abc1234",
            task_goal="the original goal",
        )

        loaded = manager.load_session("sess1")
        assert loaded is not None
        assert loaded.task_goal == "the original goal"
        assert loaded.provenance is not None
        assert "git HEAD: abc1234" in loaded.provenance

    def test_compact_save_load_goal_survives(self, tmp_path, monkeypatch):
        """End-to-end: compact_context persists the goal, and load_session
        restores it into the live context so future compactions keep it."""
        agent = CoderAgent(
            StubLLM(),
            tools=ALL_TOOLS,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.context.set_task_goal("goal that must survive reload")
        agent.context.add_message(
            Message("user", "goal that must survive reload", display_type="user_input")
        )
        agent.context.add_message(Message("assistant", "wip", display_type="response"))
        agent.context.add_message(Message("user", "second turn", display_type="user_input"))
        agent.context.add_message(Message("assistant", "more wip", display_type="response"))

        # Create a session so compact_context has somewhere to persist.
        agent.current_session = agent.session_manager.create_new_session()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a repo")

        monkeypatch.setattr(subprocess, "run", fake_run)

        agent.compact_context()

        # Reload the session into a fresh agent and confirm the goal survived.
        session_id = agent.current_session.id
        reloaded = CoderAgent(
            StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path)
        )
        assert reloaded.load_session(session_id) is True
        assert reloaded.context.get_task_goal() == "goal that must survive reload"

    def test_provenance_field_round_trips_but_not_restored_to_context(self, tmp_path):
        """The provenance field persists on the session object (round-trips
        through save/load), but by design the agent's ContextWindowManager does
        not carry a provenance attribute — provenance is rebuilt fresh by
        _build_provenance on the next compact, not read back from disk. The
        string also survives via the compact_summary message it was folded
        into."""
        manager = SessionManager(tmp_path, allow_loading=True)
        session = ChatSession(
            id="sess2",
            title="t",
            created_at="2026-08-05T08:00:00",
            last_modified="2026-08-05T08:00:00",
            messages=[],
        )
        manager.update_session_after_compact(
            session, "summary", None, provenance="## Session Provenance\n- x: 1"
        )

        loaded = manager.load_session("sess2")
        assert loaded is not None
        # Field round-trips on the session object.
        assert loaded.provenance == "## Session Provenance\n- x: 1"

        # And the agent's ContextWindowManager has no provenance attribute to
        # restore into (it is rebuilt by _build_provenance, not read on load).
        agent = CoderAgent(StubLLM(), tools=ALL_TOOLS, use_repo_map=False, repo_root=str(tmp_path))
        assert not hasattr(agent.context, "provenance")
