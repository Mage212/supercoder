"""Tests for StreamingToolCallParser — state machine for streaming tool call accumulation."""

import pytest

from supercoder.agent.streaming_tool_parser import StreamingToolCallParser, _ToolCallBuffer

# --- Helpers ---


class FakeFunction:
    def __init__(self, name: str | None = None, arguments: str | None = None):
        self.name = name
        self.arguments = arguments


class FakeDelta:
    def __init__(
        self,
        index: int,
        id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
    ):
        self.index = index
        self.id = id
        self.function = FakeFunction(name, arguments)


# --- Unit tests: _ToolCallBuffer state machine ---


class TestToolCallBuffer:
    def test_empty_buffer_not_complete(self):
        buf = _ToolCallBuffer()
        assert not buf.is_complete()

    def test_single_object_becomes_complete(self):
        buf = _ToolCallBuffer()
        for ch in "{}":
            buf.feed_char(ch)
        assert buf.is_complete()
        assert buf.buffer == "{}"

    def test_nested_objects_depth_tracking(self):
        buf = _ToolCallBuffer()
        for ch in '{"a": [1, 2]}':
            buf.feed_char(ch)
        assert buf.is_complete()
        assert buf.depth == 0

    def test_string_with_braces_no_depth_change(self):
        buf = _ToolCallBuffer()
        for ch in '{"code": "if (x) { return; }"}':
            buf.feed_char(ch)
        assert buf.is_complete()
        assert buf.depth == 0

    def test_escaped_quote_stays_in_string(self):
        buf = _ToolCallBuffer()
        for ch in '{"msg": "say \\"hello\\""}':
            buf.feed_char(ch)
        assert buf.is_complete()
        assert buf.in_string is False

    def test_escape_sequences_tracked(self):
        buf = _ToolCallBuffer()
        for ch in '{"path": "C:\\\\Users\\\\file"}':
            buf.feed_char(ch)
        assert buf.is_complete()


# --- Integration tests: StreamingToolCallParser ---


class TestStreamingToolCallParser:
    def test_single_tool_call_accumulation(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="call_1", name="read_file", arguments='{"pat')])
        parser.feed_chunk([FakeDelta(0, arguments='h": "main.py"}')])

        assert not parser.has_incomplete_tool_calls()
        native, _raw = parser.finalize()
        assert len(native) == 1
        assert native[0]["name"] == "read_file"
        assert native[0]["arguments"] == {"path": "main.py"}

    def test_multiple_tool_calls_different_indices(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="call_1", name="read", arguments='{"a":')])
        parser.feed_chunk([FakeDelta(1, id="call_2", name="write", arguments='{"b":')])
        parser.feed_chunk([FakeDelta(0, arguments=' "x"}')])
        parser.feed_chunk([FakeDelta(1, arguments=' "y"}')])

        native, _raw = parser.finalize()
        assert len(native) == 2
        assert native[0]["name"] == "read"
        assert native[1]["name"] == "write"

    def test_collision_detection_remapping(self):
        parser = StreamingToolCallParser()
        # First call at index 0, completes
        parser.feed_chunk([FakeDelta(0, id="call_1", name="func1", arguments='{"x": 1}')])

        # Second call at same index 0 but different ID → remapped
        parser.feed_chunk([FakeDelta(0, id="call_2", name="func2", arguments='{"y": 2}')])

        native, _raw = parser.finalize()
        assert len(native) == 2
        assert native[0]["name"] == "func1"
        assert native[1]["name"] == "func2"

    def test_has_incomplete_on_truncated_json(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk(
            [FakeDelta(0, id="call_1", name="edit", arguments='{"file": "a.py", "content": "hel')]
        )

        assert parser.has_incomplete_tool_calls()

    def test_has_not_incomplete_on_complete_json(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="call_1", name="read", arguments='{"path": "x"}')])

        assert not parser.has_incomplete_tool_calls()

    def test_finalize_returns_empty_for_no_buffers(self):
        parser = StreamingToolCallParser()
        native, _raw = parser.finalize()
        assert native == []
        assert _raw is None

    def test_reset_clears_state(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="call_1", name="f", arguments='{"a": 1}')])
        parser.reset()
        native, _raw = parser.finalize()
        assert native == []

    def test_continuation_without_id_routes_correctly(self):
        parser = StreamingToolCallParser()
        # Start with ID
        parser.feed_chunk([FakeDelta(0, id="call_1", name="func", arguments='{"k":')])
        # Continue without ID
        parser.feed_chunk([FakeDelta(0, arguments=' "v"}')])

        native, _ = parser.finalize()
        assert len(native) == 1
        assert native[0]["arguments"] == {"k": "v"}


# --- 3-level JSON recovery ---


class TestThreeLevelRecovery:
    def test_valid_json_level1(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="f", arguments='{"key": "val"}')])
        native, _ = parser.finalize()
        assert native[0]["arguments"] == {"key": "val"}

    def test_unclosed_string_auto_closed(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="f", arguments='{"key": "val')])

        native, _ = parser.finalize()
        assert native[0]["arguments"]["key"] == "val"

    def test_unclosed_braces_auto_closed(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="f", arguments='{"key": "val"')])

        native, _ = parser.finalize()
        assert native[0]["arguments"] == {"key": "val"}

    def test_broken_json_level3_jsonrepair(self):
        parser = StreamingToolCallParser()
        # Trailing comma — level 1 & 2 fail, level 3 (json-repair) fixes it
        parser.feed_chunk([FakeDelta(0, id="c1", name="f", arguments='{"key": "val",}')])

        native, _ = parser.finalize()
        assert native[0]["arguments"] == {"key": "val"}

    def test_empty_arguments_returns_empty_dict(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="f", arguments=None)])

        native, _ = parser.finalize()
        assert len(native) == 1
        assert native[0]["arguments"] == {}

    def test_missing_provider_id_gets_fallback_id(self):
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id=None, name="f", arguments="{}")])

        native, raw = parser.finalize()
        assert native[0]["id"] == "call_0"
        assert raw is not None
        assert raw[0]["id"] == "call_0"


# --- E2E streaming simulation ---


class TestStreamingSimulation:
    def test_character_by_character_streaming(self):
        """Simulate real token-by-token streaming."""
        parser = StreamingToolCallParser()
        chunks = ['{"', "query", '": "', "What is", " the weather", " in Paris", '?"}']

        parser.feed_chunk([FakeDelta(0, id="c1", name="search", arguments=chunks[0])])
        for chunk in chunks[1:]:
            parser.feed_chunk([FakeDelta(0, arguments=chunk)])

        native, _raw = parser.finalize()
        assert len(native) == 1
        assert native[0]["arguments"] == {"query": "What is the weather in Paris?"}
        assert _raw is not None

    def test_interleaved_tool_calls(self):
        """Multiple tool calls arriving interleaved."""
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="read", arguments='{"lo')])
        parser.feed_chunk([FakeDelta(1, id="c2", name="search", arguments='{"qu')])
        parser.feed_chunk([FakeDelta(0, arguments='cation":')])
        parser.feed_chunk([FakeDelta(1, arguments='ery":')])
        parser.feed_chunk([FakeDelta(0, arguments=' "NY"}')])
        parser.feed_chunk([FakeDelta(1, arguments=' "AI"}')])

        native, _ = parser.finalize()
        assert len(native) == 2
        assert native[0]["name"] == "read"
        assert native[0]["arguments"] == {"location": "NY"}
        assert native[1]["name"] == "search"
        assert native[1]["arguments"] == {"query": "AI"}

    def test_truncated_stream_detected(self):
        """Simulate max_tokens cutoff mid-JSON."""
        parser = StreamingToolCallParser()
        parser.feed_chunk([FakeDelta(0, id="c1", name="edit", arguments='{"file_path": "/path/to')])

        assert parser.has_incomplete_tool_calls()

        native, _ = parser.finalize()
        # Should still produce a result via recovery
        assert len(native) == 1
        assert native[0]["name"] == "edit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
