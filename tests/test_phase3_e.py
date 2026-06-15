"""Tests for Phase 3E fixes (backlog B-036): sliding pairs, token margin, backup hash."""

from supercoder.checkpoint import CheckpointManager
from supercoder.context.token_counter import TokenCounter
from supercoder.context.window_manager import ContextConfig, ContextWindowManager
from supercoder.llm.base import Message

# ── E1: sliding compression keeps tool-call pairs ──


class TestSlidingCompressionPairs:
    def _history_with_tool_call(self):
        return [
            Message("user", "first question"),
            Message(
                "assistant",
                "",
                tool_calls=[
                    {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
                ],
            ),
            Message("tool", "result", tool_call_id="c1", name="t"),
            Message("user", "second question"),
            Message("assistant", "done"),
        ]

    def test_assistant_and_result_removed_together(self):
        config = ContextConfig(
            max_tokens=100, compression_strategy="sliding", min_messages_to_keep=2
        )
        cm = ContextWindowManager(config)
        cm.history = self._history_with_tool_call()
        cm._last_response_total_tokens = 999  # force past the compression target

        cm._sliding_window_compress()

        has_call = any(m.role == "assistant" and m.tool_calls for m in cm.history)
        has_result = any(m.role == "tool" for m in cm.history)
        # Both removed together — no orphaned assistant(tool_calls) or tool(result).
        assert has_call == has_result


# ── E2: cl100k_base fallback margin ──


class TestTokenFallbackMargin:
    def test_fallback_flag_for_unknown_model(self):
        assert TokenCounter(model="some-unknown-local-llama").is_fallback_encoding is True

    def test_no_fallback_for_known_model(self):
        assert TokenCounter(model="gpt-4").is_fallback_encoding is False

    def test_margin_applied_on_fallback(self):
        tc = TokenCounter(model="some-unknown-local-llama")
        text = "hello world " * 20
        raw = len(tc.encoder.encode(text))
        assert tc.count(text) == int(raw * tc.FALLBACK_MARGIN)

    def test_has_accurate_counting_reflects_fallback(self):
        assert TokenCounter(model="gpt-4").has_accurate_counting is True
        assert TokenCounter(model="some-unknown-local-llama").has_accurate_counting is False


# ── E3: checkpoint backup name uses hash (no collisions) ──


class TestCheckpointBackupHash:
    def test_colliding_paths_get_distinct_backups(self, tmp_path):
        # Under the old "__"-join scheme both mapped to the same backup filename.
        f1 = tmp_path / "a" / "b.py"
        f1.parent.mkdir()
        f1.write_text("one")
        f2 = tmp_path / "a__b.py"
        f2.write_text("two")

        cm = CheckpointManager(tmp_path)
        cp = cm.create("test")
        assert cm.backup_file(f1) is True
        assert cm.backup_file(f2) is True

        assert cp.files[str(f1.absolute())] != cp.files[str(f2.absolute())]

    def test_backup_then_restore_round_trip(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("original\n")

        cm = CheckpointManager(tmp_path)
        cm.create("test")
        assert cm.backup_file(f) is True

        f.write_text("modified\n")
        result = cm.rollback()
        assert f.read_text() == "original\n"
        assert any("src.py" in p for p in result.restored)
