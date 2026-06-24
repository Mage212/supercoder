"""Tests for native API tool calling (non-streaming mode).

Covers:
- ToolDefinition.to_openai_schema() for all tools
- CompletionResult / NativeToolCall dataclasses
- Message.to_api_dict() with role=tool and tool_calls
- CoderAgent.chat_turn() event flow
- chat_with_tools() response parsing
"""

from datetime import date
from unittest.mock import MagicMock

import pytest

from supercoder.llm.base import CompletionResult, Message, NativeToolCall, UsageStats
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
        assert [m.display_type for m in messages[:3]] == [
            "mode_policy",
            "context_attachment",
            "user_input",
        ]
        assert '<attached_file path="main.py"' in messages[1].content
        assert api_messages[1].display_type == "mode_policy"
        assert api_messages[2].display_type == "context_attachment"
        assert api_messages[3].content == "Review @main.py"

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
        agent.loop_detection = {"enabled": False}

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

    def test_text_tool_call_fallback_executes_supercoder_tag(self, tmp_path):
        """Native mode should recover textual <@TOOL> calls when providers omit tool_calls."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        (tmp_path / "test.txt").write_text("hello\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content=(
                    'I will read it.\n<@TOOL>{"name": "file-read", '
                    '"arguments": {"fileName": "test.txt"}}</@TOOL>'
                ),
                tool_calls=[],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[FileReadTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Read test.txt"))
        types = [event["type"] for event in events]
        response = next(event for event in events if event["type"] == "response")
        assistant_with_tools = next(
            msg
            for msg in agent.context.get_messages()
            if msg.role == "assistant" and msg.tool_calls
        )
        tool_msg = next(msg for msg in agent.context.get_messages() if msg.role == "tool")

        assert "tool_call" in types
        assert "tool_result" in types
        assert response["content"] == "I will read it."
        assert "<@TOOL>" not in response["content"]
        assert assistant_with_tools.tool_calls[0]["id"].startswith("fallback_call_")
        assert assistant_with_tools.tool_calls[0]["function"]["name"] == "file-read"
        assert tool_msg.tool_call_id == assistant_with_tools.tool_calls[0]["id"]

    def test_text_tool_call_fallback_executes_qwen_style(self, tmp_path):
        """Native mode should recover qwen-style textual tool calls."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        (tmp_path / "test.txt").write_text("hello\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content='to=tool:file-read {"fileName": "test.txt"}',
                tool_calls=[],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[FileReadTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Read test.txt"))
        types = [event["type"] for event in events]

        assert "tool_call" in types
        assert "tool_result" in types
        assert types.index("tool_call") < types.index("response")

    def test_json_example_without_arguments_does_not_trigger_retry(self):
        """Ordinary JSON examples should not be treated as malformed tool calls."""
        agent, mock_llm = self._make_agent()
        mock_llm.chat_with_tools_interruptible.return_value = CompletionResult(
            content='```json\n{"name": "demo"}\n```',
            tool_calls=[],
        )

        events = list(agent.chat_turn("Show JSON"))
        types = [event["type"] for event in events]

        assert "tool_retry" not in types
        assert "tool_call" not in types
        assert events[-1]["type"] == "done"

    def test_malformed_text_tool_call_retries_then_executes_native_call(self, tmp_path):
        """Malformed textual tool attempts should retry before giving up."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        (tmp_path / "test.txt").write_text("hello\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content='<@TOOL>{"name": "file-read", "arguments": {"fileName": "test.txt"',
                tool_calls=[],
            ),
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_retry",
                        name="file-read",
                        arguments={"fileName": "test.txt"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_retry",
                        "type": "function",
                        "function": {
                            "name": "file-read",
                            "arguments": '{"fileName": "test.txt"}',
                        },
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[FileReadTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Read test.txt"))
        retry_events = [event for event in events if event["type"] == "tool_retry"]
        types = [event["type"] for event in events]
        second_api_messages = mock_llm.chat_with_tools_interruptible.call_args_list[1][0][0]

        assert len(retry_events) == 1
        assert retry_events[0]["content"]["attempt"] == 1
        assert "tool_call" in types
        assert "tool_result" in types
        assert "native tool call interface" in second_api_messages[-1].content

    def test_malformed_text_tool_call_stops_after_two_retries(self):
        """Malformed textual tool attempts should stop after the retry budget."""
        agent, mock_llm = self._make_agent()
        malformed = CompletionResult(
            content='<@TOOL>{"name": "file-read", "arguments": {"fileName": "test.txt"',
            tool_calls=[],
        )
        mock_llm.chat_with_tools_interruptible.side_effect = [malformed, malformed, malformed]

        events = list(agent.chat_turn("Read test.txt"))
        retry_events = [event for event in events if event["type"] == "tool_retry"]
        error_event = next(event for event in events if event["type"] == "error")

        assert [event["content"]["attempt"] for event in retry_events] == [1, 2]
        assert "after 2 retries" in error_event["content"]

    def test_truncated_tool_call_is_not_executed(self, tmp_path):
        """Truncated tool calls are returned to the model as retryable tool errors."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.file_read import FileReadTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        tool = FileReadTool()
        tool.execute = MagicMock(return_value="SHOULD NOT RUN")
        raw_tool_call = {
            "id": "call_truncated",
            "type": "function",
            "function": {"name": "file-read", "arguments": '{"fileName": "missing.txt"}'},
        }
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_truncated",
                        name="file-read",
                        arguments={"fileName": "missing.txt"},
                    )
                ],
                raw_tool_calls=[raw_tool_call],
                truncated=True,
            ),
            CompletionResult(content="Retried safely.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=[tool],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Read missing.txt"))
        tool_results = [event for event in events if event["type"] == "tool_result"]

        tool.execute.assert_not_called()
        assert "not executed" in tool_results[0]["content"]["result"]
        assert mock_llm.chat_with_tools_interruptible.call_count == 2

    def test_command_deny_skips_confirmation_and_execution(self, tmp_path, monkeypatch):
        """Denied commands return a tool result without asking the user."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        fake_logger = MagicMock()
        monkeypatch.setattr("supercoder.agent.coder_agent.get_logger", lambda: fake_logger)
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
        fake_logger.log_tool_call.assert_any_call(
            "command-exec", '{"command": "sudo echo blocked"}'
        )
        fake_logger.log_tool_result.assert_any_call(
            "command-exec", tool_result["content"]["result"]
        )

    def test_command_allow_skips_confirmation(self, tmp_path, monkeypatch):
        """Allowed commands execute without a confirmation event."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        fake_logger = MagicMock()
        monkeypatch.setattr("supercoder.agent.coder_agent.get_logger", lambda: fake_logger)
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
        assert fake_logger.log_tool_call.call_count == 1
        assert fake_logger.log_tool_result.call_count == 1

    def test_command_default_ask_yields_confirmation(self, tmp_path, monkeypatch):
        """Unknown commands still ask for approval by default."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        fake_logger = MagicMock()
        monkeypatch.setattr("supercoder.agent.coder_agent.get_logger", lambda: fake_logger)
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
        fake_logger.log_tool_call.assert_any_call("command-exec", '{"command": "echo ask"}')
        fake_logger.log_tool_result.assert_any_call(
            "command-exec", "Command execution cancelled by user."
        )

    def test_command_session_approval_skips_next_confirmation(self, tmp_path, monkeypatch):
        """Session approvals are stored in memory and reused during the process."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        fake_logger = MagicMock()
        monkeypatch.setattr("supercoder.agent.coder_agent.get_logger", lambda: fake_logger)
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        raw_tool_call = {
            "id": "call_cmd",
            "type": "function",
            "function": {
                "name": "command-exec",
                "arguments": '{"command": "printf session"}',
            },
        }
        tool_call = NativeToolCall(
            id="call_cmd",
            name="command-exec",
            arguments={"command": "printf session"},
        )
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(content="", tool_calls=[tool_call], raw_tool_calls=[raw_tool_call]),
            CompletionResult(content="Done.", tool_calls=[]),
            CompletionResult(content="", tool_calls=[tool_call], raw_tool_calls=[raw_tool_call]),
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

        first_events = []
        for event in agent.chat_turn("Run command"):
            if event["type"] == "command_confirm":
                event["result"].update({"approved": True, "decision": "allow_session"})
            first_events.append(event)
        second_events = list(agent.chat_turn("Run command again"))

        assert "command_confirm" in [event["type"] for event in first_events]
        assert "command_confirm" not in [event["type"] for event in second_events]
        assert agent.permission_policy.check_command("printf session").source == "session"

    def test_command_persistent_approval_writes_project_rule(self, tmp_path, monkeypatch):
        """Persistent approvals are written to .supercoder/permissions.yaml."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.tools.command_exec import CommandExecutionTool

        fake_logger = MagicMock()
        monkeypatch.setattr("supercoder.agent.coder_agent.get_logger", lambda: fake_logger)
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
                        arguments={"command": "printf persistent"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {
                            "name": "command-exec",
                            "arguments": '{"command": "printf persistent"}',
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

        events = []
        for event in agent.chat_turn("Run persistent command"):
            if event["type"] == "command_confirm":
                event["result"].update({"approved": True, "decision": "allow_persistent"})
            events.append(event)

        assert "command_confirm" in [event["type"] for event in events]
        assert (tmp_path / ".supercoder" / "permissions.yaml").exists()
        assert agent.permission_policy.check_command("printf persistent").source == "persistent"
        fake_logger.log_permission_rule_change.assert_called()

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
        display_text = tool_event["content"]["display_result"]

        assert "[Tool output compacted]" in tool_text
        assert "MIDDLE_ONLY_SECRET" not in tool_text
        assert "[Tool output compacted]" not in display_text
        assert "MIDDLE_ONLY_SECRET" not in display_text
        assert "Preview head:" in display_text
        assert "Preview tail:" in display_text
        assert tool_event["content"]["masked"] is True
        assert tool_event["content"]["original_size"] == len(
            ("H" * 3500) + "MIDDLE_ONLY_SECRET" + ("T" * 5000)
        )
        assert tool_event["content"]["omitted_chars"] > 0
        assert tool_event["content"]["offload_path"].startswith(".supercoder/tool-outputs/")

        tool_msgs = [m for m in agent.context.get_messages() if m.role == "tool"]
        assert "[Tool output compacted]" in tool_msgs[0].content
        assert "MIDDLE_ONLY_SECRET" not in tool_msgs[0].content
        assert tool_msgs[0].display_result == display_text
        assert tool_msgs[0].display_policy == "compact"
        assert tool_msgs[0].display_meta["masked"] is True
        assert tool_msgs[0].display_meta["original_size"] == tool_event["content"]["original_size"]
        assert tool_msgs[0].display_meta["offload_path"].startswith(".supercoder/tool-outputs/")

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

    def test_code_mode_keeps_code_edit_available_for_host_approval(self, tmp_path):
        """CODE mode keeps edit tools available so the host can ask before execution."""
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

        code_tools = [tool.definition.name for tool in agent._get_tools_for_mode()]
        assert "code-edit" in code_tools

    def test_set_mode_does_not_change_system_prompt(self, tmp_path):
        """Mode switches must not invalidate the stable system prompt prefix."""
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

        before = agent.context.get_messages_for_api()[0].content
        agent.set_mode(AgentMode.ASK)
        after = agent.context.get_messages_for_api()[0].content

        assert before == after

    def test_mode_policy_is_announced_once_after_mode_change(self, tmp_path):
        """The in-band mode instruction should not be repeated on every turn."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(content="One.", tool_calls=[]),
            CompletionResult(content="Two.", tool_calls=[]),
            CompletionResult(content="Three.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        agent.set_mode(AgentMode.ASK)
        list(agent.chat_turn("First"))
        list(agent.chat_turn("Second"))
        agent.set_mode(AgentMode.PLAN)
        list(agent.chat_turn("Third"))

        policies = [m for m in agent.context.get_messages() if m.display_type == "mode_policy"]
        assert [p.content.splitlines()[1].split(".")[0] for p in policies] == [
            "Current mode: ASK",
            "Current mode: PLAN",
        ]

    def test_mode_policy_is_announced_again_after_compact(self, tmp_path):
        """Compacted history should get the current mode policy on the next turn."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(content="Before compact.", tool_calls=[]),
            CompletionResult(content="Compact summary", tool_calls=[]),
            CompletionResult(content="After compact.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        agent.set_mode(AgentMode.ASK)
        list(agent.chat_turn("Before"))
        agent.compact_context()
        list(agent.chat_turn("After"))

        policies = [m for m in agent.context.get_messages() if m.display_type == "mode_policy"]
        assert len(policies) == 1
        assert "Current mode: ASK" in policies[0].content

    def test_ask_mode_blocks_code_edit(self, tmp_path):
        """ASK mode is enforced by the host even if the model calls code-edit."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "blocked.txt",
                            "operation": "create",
                            "content": "nope",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.ASK)

        events = list(agent.chat_turn("Create a file"))
        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "ASK mode is read-only" in result
        assert not (tmp_path / "blocked.txt").exists()

    def test_plan_mode_blocks_command_exec(self, tmp_path):
        """PLAN mode can inspect and save plans, but cannot run shell commands."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

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
                        arguments={"command": "echo blocked"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_cmd",
                        "type": "function",
                        "function": {"name": "command-exec", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.PLAN)

        events = list(agent.chat_turn("Run command"))
        types = [event["type"] for event in events]
        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "command_confirm" not in types
        assert "PLAN mode blocks shell commands" in result

    def test_plan_mode_blocks_project_file_edit(self, tmp_path):
        """PLAN mode must not rewrite normal project files."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('old')\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "src/app.py",
                            "operation": "create",
                            "content": "print('new')\n",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.PLAN)

        events = list(agent.chat_turn("Edit project file"))
        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "PLAN mode cannot edit project files" in result
        assert (tmp_path / "src" / "app.py").read_text() == "print('old')\n"

    def test_plan_mode_allows_dated_plan_file_create(self, tmp_path):
        """PLAN mode rewrites simple filenames into dated .supercoder/plans files."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        today = date.today().isoformat()
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_plan",
                        name="code-edit",
                        arguments={
                            "filepath": "implementation.md",
                            "operation": "create",
                            "content": "# Plan\n",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_plan",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.PLAN)

        events = list(agent.chat_turn("Save a plan"))
        tool_call = next(e for e in events if e["type"] == "tool_call")["content"]
        expected = tmp_path / ".supercoder" / "plans" / f"{today}-implementation.md"

        assert tool_call["arguments"]["filepath"] == f".supercoder/plans/{today}-implementation.md"
        assert expected.read_text() == "# Plan\n"

    def test_plan_mode_dedupes_created_plan_filenames(self, tmp_path):
        """PLAN mode should avoid overwriting an existing dated plan on create."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        today = date.today().isoformat()
        plan_dir = tmp_path / ".supercoder" / "plans"
        plan_dir.mkdir(parents=True)
        (plan_dir / f"{today}-plan.md").write_text("old")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_plan",
                        name="code-edit",
                        arguments={"operation": "create", "content": "new"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_plan",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.PLAN)

        events = list(agent.chat_turn("Save a plan"))
        tool_call = next(e for e in events if e["type"] == "tool_call")["content"]

        assert tool_call["arguments"]["filepath"] == f".supercoder/plans/{today}-plan-2.md"
        assert (plan_dir / f"{today}-plan.md").read_text() == "old"
        assert (plan_dir / f"{today}-plan-2.md").read_text() == "new"

    def test_code_mode_cancelled_code_edit_does_not_write(self, tmp_path):
        """CODE mode asks before file edits and respects cancellation."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "blocked.txt",
                            "operation": "create",
                            "content": "nope",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = []
        for event in agent.chat_turn("Create a file"):
            if event["type"] == "edit_confirm":
                event["result"].update({"approved": False})
            events.append(event)

        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "edit_confirm" in [event["type"] for event in events]
        assert "File edit cancelled by user" in result
        assert not (tmp_path / "blocked.txt").exists()

    def test_code_mode_approved_code_edit_writes_file(self, tmp_path):
        """CODE mode applies file edits only after host approval."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "approved.txt",
                            "operation": "create",
                            "content": "ok",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = []
        for event in agent.chat_turn("Create a file"):
            if event["type"] == "edit_confirm":
                event["result"].update({"approved": True})
            events.append(event)

        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]
        assert "edit_confirm" in [event["type"] for event in events]
        assert "Created file" in result
        assert (tmp_path / "approved.txt").read_text() == "ok"

    def test_code_mode_edit_without_fresh_read_returns_preflight_error(self, tmp_path):
        """CODE mode does not ask approval for an edit that cannot pass preflight."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        target = tmp_path / "main.py"
        target.write_text("old\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "main.py",
                            "operation": "search_replace",
                            "search": "old",
                            "replace": "new",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("Edit file"))
        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "edit_confirm" not in [event["type"] for event in events]
        assert "File was not read before edit" in result
        assert target.read_text() == "old\n"

    def test_code_mode_edit_after_file_read_sends_diff_preview(self, tmp_path):
        """CODE mode sends a prepared diff preview before asking edit approval."""
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        target = tmp_path / "main.py"
        target.write_text("old\n")
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_read",
                        name="file-read",
                        arguments={"fileName": "main.py"},
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {"name": "file-read", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "main.py",
                            "operation": "search_replace",
                            "search": "old",
                            "replace": "new",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = []
        for event in agent.chat_turn("Edit file"):
            if event["type"] == "edit_confirm":
                preview = event["content"].get("preview")
                assert preview is not None
                assert "-old" in preview.diff
                assert "+new" in preview.diff
                event["result"].update({"approved": True})
            events.append(event)

        edit_results = [
            event["content"]["result"]
            for event in events
            if event["type"] == "tool_result" and event["content"]["name"] == "code-edit"
        ]
        assert [event["type"] for event in events].count("edit_confirm") == 1
        assert "Replaced 1 occurrence" in edit_results[0]
        assert target.read_text() == "new\n"

    def test_code_mode_apply_and_accept_edits_skips_followup_confirm(self, tmp_path):
        """Edit approval can switch the rest of the active loop to accept-edits."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        first_call = NativeToolCall(
            id="call_first",
            name="code-edit",
            arguments={
                "filepath": "first.txt",
                "operation": "create",
                "content": "first",
            },
        )
        second_call = NativeToolCall(
            id="call_second",
            name="code-edit",
            arguments={
                "filepath": "second.txt",
                "operation": "create",
                "content": "second",
            },
        )
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[first_call, second_call],
                raw_tool_calls=[
                    {
                        "id": "call_first",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    },
                    {
                        "id": "call_second",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    },
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = []
        for event in agent.chat_turn("Create files"):
            if event["type"] == "edit_confirm":
                event["result"].update({"approved": True, "decision": "apply_and_accept_edits"})
            events.append(event)

        assert [event["type"] for event in events].count("edit_confirm") == 1
        assert agent.mode == AgentMode.ACCEPT_EDITS
        assert (tmp_path / "first.txt").read_text() == "first"
        assert (tmp_path / "second.txt").read_text() == "second"
        second_messages = mock_llm.chat_with_tools_interruptible.call_args_list[1].args[0]
        assert any(
            getattr(message, "display_type", None) == "mode_policy"
            and "Current mode: ACCEPT-EDITS" in message.content
            for message in second_messages
        )

    def test_accept_edits_mode_allows_code_edit_create(self, tmp_path):
        """ACCEPT_EDITS preserves the current low-friction edit behavior."""
        from supercoder.agent.agent_modes import AgentMode
        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[
                    NativeToolCall(
                        id="call_edit",
                        name="code-edit",
                        arguments={
                            "filepath": "created.txt",
                            "operation": "create",
                            "content": "ok",
                        },
                    )
                ],
                raw_tool_calls=[
                    {
                        "id": "call_edit",
                        "type": "function",
                        "function": {"name": "code-edit", "arguments": "{}"},
                    }
                ],
            ),
            CompletionResult(content="Done.", tool_calls=[]),
        ]
        agent = CoderAgent(
            llm=mock_llm,
            tools=ALL_TOOLS,
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )
        agent.set_mode(AgentMode.ACCEPT_EDITS)

        events = list(agent.chat_turn("Create a file"))
        result = next(e for e in events if e["type"] == "tool_result")["content"]["result"]

        assert "edit_confirm" not in [event["type"] for event in events]
        assert "Created file" in result
        assert (tmp_path / "created.txt").read_text() == "ok"

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

    def test_auto_compact_runs_after_model_response_usage(self):
        """Auto-compact should use usage from the latest model response."""
        agent, mock_llm = self._make_agent()
        agent.context.config.max_tokens = 2000
        agent.context.config.reserved_for_response = 0
        agent.context.config.auto_compact_threshold = 0.5
        agent.context.config.compression_threshold = 10.0
        agent.context.config.protected_recent_steps = 1

        assert agent.context.should_auto_compact() is False

        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="Done.",
                tool_calls=[],
                usage=UsageStats(prompt_tokens=600, completion_tokens=600, total_tokens=1200),
            ),
            CompletionResult(content="Compact summary", tool_calls=[]),
        ]

        events = list(agent.chat_turn("Continue"))
        types = [e["type"] for e in events]

        assert "auto_compact" in types
        assert "response" in types
        assert types.index("response") < types.index("auto_compact")
        assert mock_llm.chat_with_tools_interruptible.call_count == 2

    def test_auto_compact_after_tool_results_before_next_model_call(self):
        """Tool responses stay API-valid before compacting a large tool-call turn."""
        agent, mock_llm = self._make_agent()
        agent.context.config.max_tokens = 2000
        agent.context.config.reserved_for_response = 0
        agent.context.config.auto_compact_threshold = 0.5
        agent.context.config.compression_threshold = 10.0
        agent.context.config.protected_recent_steps = 1
        tool_call = NativeToolCall(id="call_unknown", name="missing-tool", arguments={})
        raw_tool_call = {
            "id": "call_unknown",
            "type": "function",
            "function": {"name": "missing-tool", "arguments": "{}"},
        }

        mock_llm.chat_with_tools_interruptible.side_effect = [
            CompletionResult(
                content="",
                tool_calls=[tool_call],
                raw_tool_calls=[raw_tool_call],
                usage=UsageStats(prompt_tokens=1000, completion_tokens=250, total_tokens=1250),
            ),
            CompletionResult(content="Compact summary", tool_calls=[]),
            CompletionResult(content="Done.", tool_calls=[]),
        ]

        events = list(agent.chat_turn("Call a tool"))
        types = [e["type"] for e in events]

        assert "auto_compact" in types
        assert types.index("error") < types.index("auto_compact")
        assert types.index("auto_compact") < types.index("response")
        assert mock_llm.chat_with_tools_interruptible.call_count == 3


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


class TestNativeAbortBetweenTools:
    """M5 (code-review-2026-06-23): double-ESC must be able to abort a native
    chat_turn between tool iterations, not only during LLM streaming. A long-
    running command-exec (timeout up to 120s) would otherwise block the abort."""

    def test_abort_surfaces_as_aborted_event(self, tmp_path):
        from unittest.mock import MagicMock

        from supercoder.agent.coder_agent import CoderAgent
        from supercoder.context import ContextConfig
        from supercoder.llm.base import CompletionResult, NativeToolCall
        from supercoder.tools.file_read import FileReadTool

        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.config = MagicMock()
        mock_llm.config.model = "test-model"
        # First call returns a tool call so the loop iterates; the second call
        # (which must NOT be reached if abort works) returns plain text.
        call_count = {"n": 0}

        def fake_chat(messages, tools, abort_controller, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Arm abort during the first LLM call (simulating a double-ESC
                # that arrives while the first tool result is processed). The
                # check at the top of the next loop iteration must fire.
                abort_controller.abort()
                return CompletionResult(
                    content="",
                    tool_calls=[NativeToolCall(id="c1", name="file-read", arguments={})],
                    usage=None,
                )
            return CompletionResult(content="done", tool_calls=[], usage=None)

        mock_llm.chat_with_tools_interruptible.side_effect = fake_chat

        agent = CoderAgent(
            llm=mock_llm,
            tools=[FileReadTool()],
            context_config=ContextConfig(max_tokens=32000),
            streaming=False,
            use_repo_map=False,
            repo_root=str(tmp_path),
        )

        events = list(agent.chat_turn("hi"))
        types = [e.get("type") for e in events]
        assert "aborted" in types, (
            f"expected an 'aborted' event between tool iterations, got {types}"
        )
        # The second LLM call must not have happened — abort stopped the turn.
        assert call_count["n"] == 1, "abort should have stopped the turn before the 2nd LLM call"
