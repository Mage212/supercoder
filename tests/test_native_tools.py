"""Tests for native API tool calling (non-streaming mode).

Covers:
- ToolDefinition.to_openai_schema() for all tools
- CompletionResult / NativeToolCall dataclasses
- Message.to_api_dict() with role=tool and tool_calls
- CoderAgent.chat_turn() event flow
- chat_with_tools() response parsing
"""

from unittest.mock import MagicMock

import pytest

from supercoder.llm.base import CompletionResult, Message, NativeToolCall
from supercoder.tools import ALL_TOOLS
from supercoder.tools.base import ToolDefinition

# ──────────────────────────────────────────────
# ToolDefinition → OpenAI schema
# ──────────────────────────────────────────────


class TestToolDefinitionSchema:
    """Verify each tool produces a valid OpenAI-compatible tool schema."""

    def test_basic_schema_structure(self):
        td = ToolDefinition(
            name="test-tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )
        schema = td.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test-tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"]["type"] == "object"
        assert "x" in schema["function"]["parameters"]["properties"]

    def test_schema_without_parameters(self):
        td = ToolDefinition(name="simple", description="No params")
        schema = td.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "simple"
        assert "parameters" not in schema["function"]

    @pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.definition.name)
    def test_all_tools_produce_valid_schema(self, tool):
        """Every tool must produce a valid schema with required fields."""
        schema = tool.definition.to_openai_schema()

        assert schema["type"] == "function"
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert isinstance(func["name"], str)
        assert len(func["name"]) > 0
        assert isinstance(func["description"], str)
        assert len(func["description"]) > 0

        # If parameters present, it must be a valid JSON Schema object
        if "parameters" in func:
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_file_read_schema_has_required_filename(self):
        from supercoder.tools.file_read import FileReadTool

        schema = FileReadTool().definition.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "fileName" in params["properties"]
        assert "fileName" in params["required"]

    def test_code_edit_schema_has_operation_enum(self):
        from supercoder.tools.code_edit import CodeEditTool

        schema = CodeEditTool().definition.to_openai_schema()
        op = schema["function"]["parameters"]["properties"]["operation"]
        assert "enum" in op
        assert "search_replace" in op["enum"]
        assert "create" in op["enum"]

    def test_command_exec_schema_has_required_command(self):
        from supercoder.tools.command_exec import CommandExecutionTool

        schema = CommandExecutionTool().definition.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "command" in params["properties"]
        assert "command" in params["required"]


# ──────────────────────────────────────────────
# Message serialization
# ──────────────────────────────────────────────


class TestMessageSerialization:
    """Verify Message.to_api_dict() handles all roles correctly."""

    def test_simple_user_message(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_api_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_assistant_message_with_tool_calls(self):
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "file-read", "arguments": '{"fileName": "x.py"}'},
            }
        ]
        msg = Message(role="assistant", content="Let me read that.", tool_calls=tool_calls)
        d = msg.to_api_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Let me read that."
        assert d["tool_calls"] == tool_calls

    def test_tool_result_message(self):
        msg = Message(
            role="tool",
            content="File contents here...",
            tool_call_id="call_1",
            name="file-read",
        )
        d = msg.to_api_dict()
        assert d["role"] == "tool"
        assert d["content"] == "File contents here..."
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "file-read"

    def test_tool_call_id_not_included_for_user(self):
        msg = Message(role="user", content="Hi")
        d = msg.to_api_dict()
        assert "tool_call_id" not in d
        assert "tool_calls" not in d
        assert "name" not in d

    def test_name_only_included_for_tool_role(self):
        """name field should only appear for role=tool messages."""
        msg = Message(role="assistant", content="x", name="file-read")
        d = msg.to_api_dict()
        assert "name" not in d


# ──────────────────────────────────────────────
# CompletionResult / NativeToolCall
# ──────────────────────────────────────────────


class TestCompletionResult:
    def test_no_tool_calls(self):
        result = CompletionResult(content="Hello!", tool_calls=[], reasoning="")
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.reasoning == ""

    def test_with_tool_calls(self):
        tc = NativeToolCall(id="call_1", name="file-read", arguments={"fileName": "x.py"})
        result = CompletionResult(content="", tool_calls=[tc], reasoning="thinking...")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "file-read"
        assert result.tool_calls[0].arguments == {"fileName": "x.py"}
        assert result.reasoning == "thinking..."

    def test_raw_tool_calls_preserved(self):
        raw = [{"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]
        result = CompletionResult(content="", tool_calls=[], raw_tool_calls=raw)
        assert result.raw_tool_calls == raw


# ──────────────────────────────────────────────
# CoderAgent.chat_turn() event flow
# ──────────────────────────────────────────────


class TestChatTurnEventFlow:
    """Test the native agent loop using mocked LLM responses."""

    def _make_agent(self):
        """Create a CoderAgent with mocked LLM for testing."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"

        tool = FileReadTool()
        agent = CoderAgent(
            llm=mock_llm,
            tools=[tool],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
        )
        return agent, mock_llm

    def test_simple_text_response(self):
        """LLM returns text only, no tool calls → response + done."""
        agent, mock_llm = self._make_agent()
        mock_llm.chat_with_tools_interruptible.return_value = CompletionResult(
            content="Hello, how can I help?",
            tool_calls=[],
            reasoning="",
        )

        events = list(agent.chat_turn("Hi"))
        types = [e["type"] for e in events]

        assert "response" in types
        assert "done" in types
        assert events[-1]["type"] == "done"

        # Check the text
        response_event = next(e for e in events if e["type"] == "response")
        assert response_event["content"] == "Hello, how can I help?"

    def test_context_reference_is_attached_before_user_prompt(self, tmp_path):
        """@path references are expanded as separate user context before the prompt."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        (tmp_path / "main.py").write_text("def main():\n    return 1\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.return_value = CompletionResult(
            content="Done.",
            tool_calls=[],
        )
        agent = CoderAgent(
            llm=mock_llm,
            tools=[FileReadTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Review @main.py"))
        messages = agent.context.get_messages()
        api_messages = mock_llm.chat_with_tools_interruptible.call_args[0][0]

        assert events[0]["type"] == "context_attachment"
        assert [m.display_type for m in messages[:2]] == ["context_attachment", "user_input"]
        assert '<attached_file path="main.py"' in messages[0].content
        assert api_messages[1].display_type == "context_attachment"
        assert api_messages[2].content == "Review @main.py"

    def test_response_with_reasoning(self):
        """LLM returns reasoning + text → thinking + response + done."""
        agent, mock_llm = self._make_agent()
        mock_llm.chat_with_tools_interruptible.return_value = CompletionResult(
            content="The answer is 42.",
            tool_calls=[],
            reasoning="Let me think about this...",
        )

        events = list(agent.chat_turn("What is the meaning of life?"))
        types = [e["type"] for e in events]

        assert types[0] == "thinking"
        assert "response" in types
        assert "done" in types

    def test_tool_call_then_response(self):
        """LLM calls a tool, then responds with text."""
        agent, mock_llm = self._make_agent()

        # First call: LLM wants to call file-read
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_1", name="file-read", arguments={"fileName": "test.txt"}
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "file-read", "arguments": '{"fileName": "test.txt"}'},
                    }
                ],
            ),
            # Second call: LLM responds with text
            CompletionResult(
                content="The file doesn't exist.",
                tool_calls=[],
            ),
        ]

        events = list(agent.chat_turn("Read test.txt"))
        types = [e["type"] for e in events]

        assert "tool_call" in types
        assert "tool_result" in types
        assert "response" in types
        assert "done" in types

    def test_unknown_tool_yields_error(self):
        """LLM calls a non-existent tool → error event."""
        agent, mock_llm = self._make_agent()
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[NativeToolCall(id="call_1", name="unknown-tool", arguments={})],
                raw_tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "unknown-tool", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Sorry, that tool doesn't exist.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("Do something"))
        types = [e["type"] for e in events]
        assert "error" in types

    def test_llm_error_yields_error_event(self):
        """LLM raises exception → error event."""
        agent, mock_llm = self._make_agent()
        mock_llm.chat_with_tools_interruptible.side_effect = Exception("Connection refused")

        events = list(agent.chat_turn("Hello"))
        types = [e["type"] for e in events]
        assert "error" in types
        error_event = next(e for e in events if e["type"] == "error")
        assert "Connection refused" in error_event["content"]

    def test_max_iterations_limit(self):
        """Agent should stop after MAX_TOOL_ITERATIONS to prevent infinite loops."""
        agent, mock_llm = self._make_agent()

        # LLM always returns a tool call → infinite loop
        def always_tool_call(*args, **kwargs):
            return CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(id="call_x", name="file-read", arguments={"fileName": "x"})
                ],
                raw_tool_calls=[
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "file-read", "arguments": '{"fileName": "x"}'},
                    }
                ],
            )

        mock_llm.chat_with_tools_interruptible.side_effect = always_tool_call

        events = list(agent.chat_turn("Loop forever"))

        # Must eventually stop with an error
        types = [e["type"] for e in events]
        assert "error" in types
        error_event = next(e for e in events if e["type"] == "error")
        assert "limit" in error_event["content"].lower()

    def test_tool_results_use_role_tool(self):
        """Verify that tool results are added to context as role='tool' with tool_call_id."""
        agent, mock_llm = self._make_agent()

        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(id="call_42", name="file-read", arguments={"fileName": "x.py"})
                ],
                raw_tool_calls=[
                    {
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "file-read", "arguments": '{"fileName": "x.py"}'},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        list(agent.chat_turn("Read x.py"))

        # Check context messages
        messages = agent.context.get_messages()
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) >= 1

        tool_msg = tool_msgs[0]
        assert tool_msg.tool_call_id == "call_42"
        assert tool_msg.name == "file-read"

    def test_command_deny_skips_confirmation_and_execution(self, tmp_path):
        """Denied commands return a tool result without asking the user."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_cmd",
                        name="command-exec",
                        arguments={"command": "sudo echo blocked"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {
                            "name": "command-exec",
                            "arguments": '{"command": "sudo echo blocked"}',
                        },
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[CommandExecutionTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Run risky command"))
        types = [event["type"] for event in events]
        tool_result = next(event for event in events if event["type"] == "tool_result")

        assert "command_confirm" not in types
        assert "Permission denied" in tool_result["content"]["result"]

    def test_command_allow_skips_confirmation(self, tmp_path):
        """Allowed commands execute without a confirmation event."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_cmd",
                        name="command-exec",
                        arguments={"command": "printf allowed"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {
                            "name": "command-exec",
                            "arguments": '{"command": "printf allowed"}',
                        },
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[CommandExecutionTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
            permissions={"command-exec": {"allow": ["printf allowed"]}},
        )

        events = list(agent.chat_turn("Run allowed command"))
        types = [event["type"] for event in events]
        tool_result = next(event for event in events if event["type"] == "tool_result")

        assert "command_confirm" not in types
        assert "allowed" in tool_result["content"]["result"]

    def test_command_default_ask_yields_confirmation(self, tmp_path):
        """Unknown commands still ask for approval by default."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_cmd",
                        name="command-exec",
                        arguments={"command": "echo ask"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {
                            "name": "command-exec",
                            "arguments": '{"command": "echo ask"}',
                        },
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[CommandExecutionTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Run unknown command"))
        types = [event["type"] for event in events]

        assert "command_confirm" in types

    def test_large_tool_result_is_masked_in_context(self, tmp_path):
        """Large tool outputs are offloaded and only compact text reaches context."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.base import BaseTool, ToolDefinition

        class BigOutputTool(BaseTool):
            @property
            def definition(self):
                return ToolDefinition(name="big-output", description="Return large output")

            def execute(self, arguments):
                return ("H" * 3500) + "MIDDLE_ONLY_SECRET" + ("T" * 5000)

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[NativeToolCall(id="call_big", name="big-output", arguments={})],
                raw_tool_calls=[
                    {
                        "id": "call_big",
                        "type": "function",
                        "function": {"name": "big-output", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        agent = CoderAgent(
            llm=mock_llm,
            tools=[BigOutputTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Run big output"))
        tool_event = next(e for e in events if e["type"] == "tool_result")
        tool_text = tool_event["content"]["result"]

        assert "[Tool output compacted]" in tool_text
        assert "MIDDLE_ONLY_SECRET" not in tool_text

        tool_msgs = [m for m in agent.context.get_messages() if m.role == "tool"]
        assert "[Tool output compacted]" in tool_msgs[0].content
        assert "MIDDLE_ONLY_SECRET" not in tool_msgs[0].content

        offloaded = list((tmp_path / ".supercoder" / "tool-outputs").glob("*.txt"))
        assert len(offloaded) == 1
        assert "MIDDLE_ONLY_SECRET" in offloaded[0].read_text()

    def test_glob_is_available_in_ask_mode(self, tmp_path):
        """glob is a read-only discovery tool, so ASK mode includes it."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"

        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        agent.set_mode(AgentMode.ASK)
        ask_tools = [tool.definition.name for tool in agent._get_tools_for_mode()]
        assert "glob" in ask_tools
        assert "code-edit" not in ask_tools

    def test_compact_context_uses_cache_aware_chat_path(self):
        """Manual compact should append a maintenance request to the current chat."""
        agent, mock_llm = self._make_agent()
        for i in range(8):
            agent.context.add_message(Message("user", f"Message {i}", display_type="user_input"))
        mock_llm.chat_with_tools_interruptible.return_value = CompletionResult(
            content="Compact summary",
            tool_calls=[],
        )

        summary, _before, _after = agent.compact_context()

        assert summary == "Compact summary"
        mock_llm.chat_with_tools_interruptible.assert_called_once()
        args, kwargs = mock_llm.chat_with_tools_interruptible.call_args
        assert args[0][-1].role == "user"
        assert "Context maintenance request" in args[0][-1].content
        assert kwargs["tool_choice"] == "none"

        messages = agent.context.get_messages()
        assert messages[0].display_type == "compact_summary"
        assert len(messages) == 7  # summary + protected_recent_steps default

    def test_auto_compact_runs_before_large_turn(self):
        """Auto-compact should run at a safe boundary before the next user turn."""
        agent, mock_llm = self._make_agent()
        agent.context.config.max_tokens = 2000
        agent.context.config.reserved_for_response = 0
        agent.context.config.auto_compact_threshold = 0.5
        agent.context.config.compression_threshold = 10.0
        agent.context.config.protected_recent_steps = 6

        agent.context.add_message(Message("user", "x" * 1200, display_type="user_input"))
        for i in range(6):
            agent.context.add_message(Message("assistant", f"short {i}", display_type="response"))
        agent.context.update_actual_usage(1200)

        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(content="Compact summary", tool_calls=[]),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("Continue"))
        types = [e["type"] for e in events]

        assert "auto_compact" in types
        assert "response" in types
        assert mock_llm.chat_with_tools_interruptible.call_count == 2


# ──────────────────────────────────────────────
# Session serialization with new fields
# ──────────────────────────────────────────────


class TestSessionSerialization:
    """Verify sessions can round-trip messages with tool_calls and tool_call_id."""

    def test_round_trip_native_messages(self):
        from supercoder.context.session_manager import ChatSession

        session = ChatSession(
            id="test-1",
            title="Test",
            created_at="2026-01-01T00:00:00",
            last_modified="2026-01-01T00:00:00",
            messages=[
                Message(role="user", content="Read a file"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "file-read", "arguments": '{"fileName": "x.py"}'},
                        }
                    ],
                ),
                Message(
                    role="tool",
                    content="file contents...",
                    tool_call_id="call_1",
                    name="file-read",
                ),
                Message(role="assistant", content="Here are the contents."),
            ],
        )

        # Serialize → deserialize
        data = session.to_dict()
        restored = ChatSession.from_dict(data)

        assert len(restored.messages) == 4

        # Check assistant message preserved tool_calls
        assistant_msg = restored.messages[1]
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) == 1
        assert assistant_msg.tool_calls[0]["id"] == "call_1"

        # Check tool message preserved tool_call_id and name
        tool_msg = restored.messages[2]
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.name == "file-read"

    def test_backward_compat_old_sessions(self):
        """Old sessions without tool_call_id should still load fine."""
        from supercoder.context.session_manager import ChatSession

        old_data = {
            "id": "old-1",
            "title": "Old Session",
            "created_at": "2025-01-01T00:00:00",
            "last_modified": "2025-01-01T00:00:00",
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        }

        session = ChatSession.from_dict(old_data)
        assert len(session.messages) == 2
        assert session.messages[0].tool_call_id is None
        assert session.messages[0].tool_calls is None
        assert session.messages[0].name is None
