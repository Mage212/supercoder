from unittest.mock import MagicMock

import yaml

from supercoder.agent.agent_modes import AgentMode
from supercoder.agent.coder_agent import CoderAgent
from supercoder.agent.loop_guard import AgentLoopGuard, LoopGuardConfig
from supercoder.config import Config
from supercoder.context import ContextConfig
from supercoder.llm.base import CompletionResult, NativeToolCall
from supercoder.tools.base import BaseTool, ToolDefinition


class CountingTool(BaseTool):
    def __init__(self):
        self.calls = 0

    @property
    def definition(self):
        return ToolDefinition(name="loop-tool", description="Count calls")

    def execute(self, arguments):
        self.calls += 1
        return "ok"


class FailingCodeEditTool(BaseTool):
    def __init__(self):
        self.calls = 0

    @property
    def definition(self):
        return ToolDefinition(name="code-edit", description="Fail edits")

    def execute(self, arguments):
        self.calls += 1
        return "Error: Search string not found in target.py"


def tool_result(name: str = "loop-tool") -> CompletionResult:
    return CompletionResult(
        content="",
        tool_calls=[NativeToolCall(id="call_loop", name=name, arguments={"x": 1})],
        raw_tool_calls=[
            {
                "id": "call_loop",
                "type": "function",
                "function": {"name": name, "arguments": '{"x":1}'},
            }
        ],
    )


def code_edit_result(call_id: str, search: str) -> CompletionResult:
    args = {
        "filepath": "target.py",
        "operation": "search_replace",
        "search": search,
        "replace": "new",
    }
    return CompletionResult(
        content="",
        tool_calls=[NativeToolCall(id=call_id, name="code-edit", arguments=args)],
        raw_tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "code-edit", "arguments": "{}"},
            }
        ],
    )


def make_agent(mock_llm, tool, tmp_path, loop_detection=None):
    return CoderAgent(
        llm=mock_llm,
        tools=[tool],
        context_config=ContextConfig(max_tokens=32000),
        streaming=False,
        use_repo_map=False,
        repo_root=str(tmp_path),
        loop_detection=loop_detection,
    )


def test_loop_guard_fingerprint_is_stable_and_non_reversible():
    guard = AgentLoopGuard(LoopGuardConfig())

    left = guard.tool_call_fingerprint("tool", {"b": 2, "a": 1})
    right = guard.tool_call_fingerprint("tool", {"a": 1, "b": 2})

    assert left == right
    assert "tool" not in left
    assert left != "tool:{'a': 1, 'b': 2}"
    assert len(left) == 64


def test_identical_tool_call_gets_corrective_result_before_execution(tmp_path):
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.chat_with_tools_interruptible.side_effect = [
        tool_result(),
        tool_result(),
        tool_result(),
        CompletionResult(content="Changed strategy.", tool_calls=[]),
    ]
    tool = CountingTool()
    agent = make_agent(mock_llm, tool, tmp_path)

    events = list(agent.chat_turn("loop"))
    loop_results = [
        e
        for e in events
        if e["type"] == "tool_result" and "Loop detected" in e["content"]["result"]
    ]

    assert tool.calls == 2
    assert len(loop_results) == 1
    assert not any(e["type"] == "error" for e in events)


def test_identical_tool_call_stops_when_repeated_after_corrective_result(tmp_path):
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.chat_with_tools_interruptible.side_effect = [
        tool_result(),
        tool_result(),
        tool_result(),
        tool_result(),
    ]
    tool = CountingTool()
    agent = make_agent(mock_llm, tool, tmp_path)

    events = list(agent.chat_turn("loop"))
    error = next(e for e in events if e["type"] == "error")

    assert tool.calls == 2
    assert "Loop detected" in error["content"]


def test_loop_detection_can_be_disabled(tmp_path):
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.chat_with_tools_interruptible.side_effect = [
        tool_result(),
        tool_result(),
        tool_result(),
        CompletionResult(content="Done.", tool_calls=[]),
    ]
    tool = CountingTool()
    agent = make_agent(mock_llm, tool, tmp_path, loop_detection={"enabled": False})

    events = list(agent.chat_turn("loop"))

    assert tool.calls == 3
    assert not any("Loop detected" in str(e.get("content")) for e in events)


def test_no_progress_code_edit_stops_after_corrective_attempt(tmp_path):
    mock_llm = MagicMock()
    mock_llm.model = "test-model"
    mock_llm.chat_with_tools_interruptible.side_effect = [
        code_edit_result("call_1", "old-a"),
        code_edit_result("call_2", "old-b"),
        code_edit_result("call_3", "old-c"),
    ]
    tool = FailingCodeEditTool()
    agent = make_agent(
        mock_llm,
        tool,
        tmp_path,
        loop_detection={
            "identical_tool_call_threshold": 99,
            "no_progress_edit_threshold": 2,
        },
    )
    agent.set_mode(AgentMode.ACCEPT_EDITS)

    events = list(agent.chat_turn("fix target"))
    error = next(e for e in events if e["type"] == "error")

    assert tool.calls == 2
    assert "Loop detected" in error["content"]


def test_loop_detection_config_loads_from_yaml(tmp_path, monkeypatch):
    config_file = tmp_path / ".supercoder.yaml"
    config_file.write_text(
        yaml.dump(
            {
                "api_key": "test-key",
                "loop_detection": {
                    "enabled": False,
                    "identical_tool_call_threshold": 4,
                },
            }
        )
    )
    monkeypatch.chdir(tmp_path)

    config = Config.load()

    assert config.loop_detection["enabled"] is False
    assert config.loop_detection["identical_tool_call_threshold"] == 4


class TestResultClassificationScope:
    """Loop-guard result classification must be scoped to the first line.

    Regression: _result_class matched ``"denied" in lower`` / ``"not allowed"
    in lower`` / etc. against the WHOLE result text, so reading a documentation
    file whose body mentions "denied" classified the read as a denied result —
    and reading the same file 3x tripped identical_tool_error. Classification of
    status words must look only at the first line (which carries the actual
    status header), not the body.
    """

    def _guard(self):
        return AgentLoopGuard(LoopGuardConfig())

    def test_denied_in_body_not_classified_as_denied(self):
        guard = self._guard()
        result = (
            "File: docs/access-control.md\n"
            "    1: # Access Control\n"
            "    2: Requests can be denied when permissions are missing.\n"
        )
        cls = guard._result_class(result)
        assert cls is None, f"body-only 'denied' must not classify as denied (got {cls!r})"

    def test_denied_on_first_line_still_classified(self):
        guard = self._guard()
        # A denial that does not start with "error" (which would classify as
        # error: first) must still be recognized as denied.
        cls = guard._result_class("Permission denied for write")
        assert cls is not None
        assert cls.startswith("denied:")

    def test_not_allowed_in_body_not_classified(self):
        guard = self._guard()
        result = "File: policy.md\n    1: Some operations are not allowed in production.\n"
        assert guard._result_class(result) is None

    def test_cancelled_in_body_not_classified(self):
        guard = self._guard()
        result = "File: changelog.md\n    1: The cancelled feature was removed in v2.\n"
        assert guard._result_class(result) is None

    def test_error_on_first_line_still_classified(self):
        guard = self._guard()
        cls = guard._result_class("Error executing tool: boom")
        assert cls is not None and cls.startswith("error:")
