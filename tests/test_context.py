"""Test context management functionality."""

from supercoder.context import ContextConfig, ContextWindowManager, TokenCounter
from supercoder.llm.base import Message, UsageStats


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_token_counter_estimation(self):
        """Test that token counter provides reasonable estimates."""
        tc = TokenCounter(use_tiktoken=False)

        text = "Hello, this is a test message for token counting."
        tokens = tc.count(text)

        # Rough estimate: ~4 chars per token
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count

    def test_token_counter_with_code(self):
        """Test token counting for code."""
        tc = TokenCounter(use_tiktoken=False)

        code = """def hello_world():
    print("Hello, World!")
    return 42"""

        tokens = tc.count(code)
        assert tokens > 0

    def test_tiktoken_availability(self):
        """Test that tiktoken-based counter reports accurate counting."""
        tc = TokenCounter(use_tiktoken=True)
        # Should have accurate counting if tiktoken is available
        assert tc.has_accurate_counting is True


class TestContextWindowManager:
    """Tests for ContextWindowManager class."""

    def test_context_manager_initialization(self):
        """Test ContextWindowManager initializes correctly."""
        config = ContextConfig(
            max_tokens=1000, reserved_for_response=200, compression_threshold=0.5
        )
        cm = ContextWindowManager(config)

        assert cm is not None

    def test_add_messages(self):
        """Test adding messages to context."""
        config = ContextConfig(max_tokens=1000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("You are a helpful assistant.")

        cm.add_message(Message("user", "Hello"))
        cm.add_message(Message("assistant", "Hi there!"))

        stats = cm.get_stats()
        assert stats.message_count == 2

    def test_context_stats(self):
        """Test context statistics tracking."""
        config = ContextConfig(max_tokens=1000, reserved_for_response=200)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("You are a helpful assistant.")

        for i in range(5):
            cm.add_message(Message("user", f"Message {i}: test content"))
            cm.add_message(Message("assistant", f"Response {i}"))

        stats = cm.get_stats()
        assert stats.message_count == 10
        assert stats.used_tokens > 0
        assert stats.utilization_percent >= 0

    def test_context_clear(self):
        """Test clearing context."""
        config = ContextConfig(max_tokens=1000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("System prompt")

        cm.add_message(Message("user", "Hello"))
        cm.add_message(Message("assistant", "Hi"))

        cm.clear()
        stats = cm.get_stats()
        assert stats.message_count == 0

    def test_get_messages_for_api_filters_thinking(self):
        """Test that get_messages_for_api() excludes thinking messages."""
        config = ContextConfig(max_tokens=10000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("System")

        cm.add_message(Message("user", "Hello", display_type="user_input"))
        cm.add_message(Message("assistant", "Reasoning...", display_type="thinking"))
        cm.add_message(Message("assistant", "Hi!", display_type="response"))

        api_msgs = cm.get_messages_for_api()
        roles = [m.role for m in api_msgs]
        assert roles == ["system", "user", "assistant"]  # thinking filtered out

    def test_get_messages_for_api_keeps_other_display_types(self):
        """Test that non-thinking display types pass through to API."""
        config = ContextConfig(max_tokens=10000)
        cm = ContextWindowManager(config)

        cm.add_message(Message("user", "Hello", display_type="user_input"))
        cm.add_message(Message("assistant", "Result", display_type="response"))
        cm.add_message(
            Message("tool", "output", tool_call_id="tc1", name="f", display_type="tool_result")
        )

        api_msgs = cm.get_messages_for_api()
        assert len(api_msgs) == 3  # all pass through

    def test_actual_usage_prefers_total_tokens(self):
        """API total_tokens is the primary context usage metric."""
        cm = ContextWindowManager(ContextConfig(max_tokens=100000))

        cm.update_actual_usage(
            UsageStats(prompt_tokens=1000, completion_tokens=2000, total_tokens=37162)
        )

        assert cm.get_stats().used_tokens == 37162

    def test_actual_usage_falls_back_to_prompt_plus_completion(self):
        """Compatible APIs that omit total_tokens still provide a usable total."""
        cm = ContextWindowManager(ContextConfig(max_tokens=100000))

        cm.update_actual_usage(
            UsageStats(prompt_tokens=16000, completion_tokens=1265, total_tokens=0)
        )

        assert cm.get_stats().used_tokens == 17265

    def test_add_message_does_not_reset_latest_api_usage(self):
        """Local history changes should not erase the latest API-reported total."""
        cm = ContextWindowManager(ContextConfig(max_tokens=100000))
        cm.update_actual_usage(UsageStats(prompt_tokens=10, completion_tokens=5, total_tokens=1200))

        cm.add_message(Message("tool", "large local result", tool_call_id="tc1", name="file-read"))

        assert cm.get_stats().used_tokens == 1200

    def test_fallback_estimate_counts_structured_api_payload(self):
        """Fallback counting includes tool calls, tool results, and tools schema."""
        config = ContextConfig(max_tokens=100000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("System")
        cm.add_message(Message("user", "Read file", display_type="user_input"))
        plain_tokens = cm.get_stats().used_tokens

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "file-read",
                    "arguments": '{"fileName":"supercoder/context/window_manager.py"}',
                },
            }
        ]
        cm.add_message(Message("assistant", "", tool_calls=tool_calls, display_type="tool_call"))
        cm.add_message(
            Message(
                "tool",
                "file contents",
                tool_call_id="call_1",
                name="file-read",
                display_type="tool_result",
            )
        )
        cm.set_tools_schema(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "file-read",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"fileName": {"type": "string"}},
                        },
                    },
                }
            ]
        )

        assert cm.get_stats().used_tokens > plain_tokens

    def test_should_auto_compact_uses_protected_step_floor(self):
        """Auto-compact should wait until there is older history to summarize."""
        config = ContextConfig(
            max_tokens=200,
            reserved_for_response=0,
            auto_compact_threshold=0.1,
            protected_recent_steps=2,
            compression_threshold=1.0,
        )
        cm = ContextWindowManager(config)
        cm.set_system_prompt("System")

        cm.add_message(Message("user", "x" * 80))
        cm.add_message(Message("assistant", "y" * 80))
        assert cm.should_auto_compact() is False

        cm.add_message(Message("user", "z" * 80))
        assert cm.should_auto_compact() is True

    def test_protected_recent_messages_keep_tool_call_pairs(self):
        """Protected tail keeps native tool-call exchanges API-valid."""
        config = ContextConfig(protected_recent_steps=1)
        cm = ContextWindowManager(config)
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "file-read", "arguments": "{}"},
            }
        ]

        cm.add_message(Message("user", "old"))
        cm.add_message(Message("assistant", "", tool_calls=tool_calls, display_type="tool_call"))
        cm.add_message(
            Message(
                "tool",
                "result",
                tool_call_id="call_1",
                name="file-read",
                display_type="tool_result",
            )
        )

        protected = cm.get_protected_recent_messages()

        assert [m.role for m in protected] == ["assistant", "tool"]
        assert protected[0].tool_calls == tool_calls

    def test_protected_recent_messages_keep_context_attachment_with_user_input(self):
        """Protected tail keeps @path context together with its user prompt."""
        config = ContextConfig(protected_recent_steps=1)
        cm = ContextWindowManager(config)

        cm.add_message(Message("user", "old"))
        cm.add_message(Message("user", "attached file body", display_type="context_attachment"))
        cm.add_message(Message("user", "review @main.py", display_type="user_input"))

        protected = cm.get_protected_recent_messages()

        assert [m.display_type for m in protected] == ["context_attachment", "user_input"]

    def test_set_initial_summary_keeps_recent_messages(self):
        """Compact summary is followed by exact recent working context."""
        cm = ContextWindowManager(ContextConfig())
        recent = [Message("user", "latest", display_type="user_input")]

        cm.set_initial_summary("summary", recent)

        messages = cm.get_messages()
        assert len(messages) == 2
        assert messages[0].display_type == "compact_summary"
        assert messages[1].content == "latest"
