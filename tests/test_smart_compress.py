"""Tests for B-037: _smart_compress must preserve tool-call parity.

Findings from code review:
- #1 (Critical): _smart_compress produced orphaned assistant(tool_calls) without
  its tool(result), or vice versa, on the emergency-fallback path. OpenAI-
  compatible APIs reject such messages with HTTP 400.
- #2 (Important): _smart_compress pairing ignored the token budget, so
  should_emergency_compress() stayed True and the fallback never reached its
  goal.
- #4 (Minor): scoring used the streaming-only <@TOOL_RESULT> marker and an
  'error' substring, which were ineffective/noisy in native mode.

These tests are written FIRST (TDD) and are expected to fail against the
current implementation.
"""

from supercoder.context.token_counter import TokenCounter
from supercoder.context.window_manager import ContextConfig, ContextWindowManager
from supercoder.llm.base import Message


def _call(owner_id: str = "c1") -> dict:
    return {"id": owner_id, "type": "function", "function": {"name": "t", "arguments": "{}"}}


def _assistant_with_call(call_id: str = "c1") -> Message:
    return Message("assistant", "", tool_calls=[_call(call_id)])


def _tool_result(call_id: str = "c1", content: str = "result") -> Message:
    return Message("tool", content, tool_call_id=call_id, name="t")


def find_orphans(history: list[Message]) -> tuple[list[str], list[str]]:
    """Return (orphaned_call_ids, orphaned_result_ids).

    orphaned_call_ids: assistant(tool_calls) whose id has no matching tool(result)
    anywhere after them.
    orphaned_result_ids: tool(result) whose tool_call_id has no preceding
    assistant(tool_calls).
    """
    call_ids_emitted: set[str] = set()
    call_ids_with_result: set[str] = set()
    orphan_results: list[str] = []
    for m in history:
        if m.role == "assistant" and m.tool_calls:
            for c in m.tool_calls:
                cid = c.get("id")
                if cid:
                    call_ids_emitted.add(cid)
        elif m.role == "tool" and m.tool_call_id:
            if m.tool_call_id in call_ids_emitted:
                call_ids_with_result.add(m.tool_call_id)
            else:
                orphan_results.append(m.tool_call_id)
    orphan_calls = [c for c in call_ids_emitted if c not in call_ids_with_result]
    return orphan_calls, orphan_results


def _make_cm(
    history: list[Message], max_tokens: int, min_keep: int = 2, protected_steps: int = 6
) -> ContextWindowManager:
    config = ContextConfig(
        max_tokens=max_tokens,
        compression_strategy="smart",
        min_messages_to_keep=min_keep,
        protected_recent_steps=protected_steps,
    )
    cm = ContextWindowManager(config)
    cm.history = [m for m in history]
    # Force the emergency threshold so _smart_compress is the exercised path.
    cm._last_response_total_tokens = max_tokens - 1
    return cm


# ── Stage 1 (#1): tool-call parity ──


class TestSmartCompressionParity:
    """After _smart_compress, no orphaned tool_calls or tool_results may remain."""

    def test_no_orphaned_call_when_result_is_large(self):
        """Large tool result may be dropped by budget; its call must follow it out."""
        history = [
            Message("user", "q0"),
            _assistant_with_call("c1"),
            _tool_result("c1", "result " * 30),
            Message("user", "q1"),
            Message("assistant", "ok"),
            Message("user", "q2"),
            Message("assistant", "done final"),
        ]
        cm = _make_cm(history, max_tokens=40, min_keep=4)
        cm._smart_compress()
        orphan_calls, orphan_results = find_orphans(cm.history)
        assert orphan_calls == [], f"orphaned tool_calls: {orphan_calls}"
        assert orphan_results == [], f"orphaned tool_results: {orphan_results}"

    def test_no_orphaned_result_when_owner_is_droppable(self):
        """High-score tool result may survive scoring while its empty owner does not."""
        history = [
            Message("user", "q0 early"),
            _assistant_with_call("c1"),
            _tool_result("c1", "ERROR stacktrace: " + "z" * 40),
            Message("user", "q1"),
            Message("assistant", "fix1"),
            Message("user", "q2"),
            Message("assistant", "fix2"),
            Message("user", "q3"),
            Message("assistant", "fix3 final"),
        ]
        cm = _make_cm(history, max_tokens=60, min_keep=4)
        cm._smart_compress()
        orphan_calls, orphan_results = find_orphans(cm.history)
        assert orphan_calls == [], f"orphaned tool_calls: {orphan_calls}"
        assert orphan_results == [], f"orphaned tool_results: {orphan_results}"

    def test_no_orphans_with_parallel_tool_calls(self):
        """assistant(tool_calls=[c1,c2]) with two results must keep all or drop all."""
        history = [
            Message("user", "q0"),
            Message(
                "assistant",
                "",
                tool_calls=[_call("c1"), _call("c2")],
            ),
            _tool_result("c1", "r1 " * 20),
            _tool_result("c2", "r2 " * 20),
            Message("user", "q1"),
            Message("assistant", "ok1"),
            Message("user", "q2"),
            Message("assistant", "ok2 final"),
        ]
        cm = _make_cm(history, max_tokens=50, min_keep=4)
        cm._smart_compress()
        orphan_calls, orphan_results = find_orphans(cm.history)
        assert orphan_calls == [], f"orphaned tool_calls: {orphan_calls}"
        assert orphan_results == [], f"orphaned tool_results: {orphan_results}"

    def test_parity_invariant_matches_protected_recent(self):
        """The same parity guarantee that get_protected_recent_messages provides
        must hold after _smart_compress."""
        history = [
            Message("user", "q0"),
            _assistant_with_call("c1"),
            _tool_result("c1", "result " * 15),
            Message("user", "q1"),
            _assistant_with_call("c2"),
            _tool_result("c2", "other " * 15),
            Message("user", "q2"),
            Message("assistant", "answer final"),
        ]
        cm = _make_cm(history, max_tokens=45, min_keep=3)
        cm._smart_compress()
        orphan_calls, orphan_results = find_orphans(cm.history)
        assert orphan_calls == [] and orphan_results == [], (
            f"parity broken: calls={orphan_calls} results={orphan_results}"
        )


# ── Stage 2 (#2): token budget respect ──


class TestSmartCompressionBudget:
    """_smart_compress must not leave the history above max_tokens when it can
    avoid it by dropping non-protected messages. Parity is never broken to do
    so (D-036)."""

    def test_does_not_exceed_max_tokens(self):
        tc = TokenCounter(model="some-unknown-local")
        # The recent (protected) messages are small; the older ones are big
        # and droppable. The compressor should drop the big old ones to get
        # back under max_tokens.
        history = [
            Message("user", "big question " * 20),
            Message("assistant", "big answer with code ```x``` " * 20),
            Message("user", "another big " * 20),
            Message("assistant", "another big answer " * 20),
            Message("user", "small q"),
            Message("assistant", "small a final"),
        ]
        cm = _make_cm(history, max_tokens=100, min_keep=2)
        cm._smart_compress()
        actual = sum(tc.count(m.content) for m in cm.history)
        assert actual <= cm.config.max_tokens, (
            f"history {actual} tokens exceeds max_tokens {cm.config.max_tokens}"
        )

    def test_parity_preferred_over_budget_when_only_protected_remains(self):
        """If the only thing left is protected messages that form a tool-call
        pair exceeding the budget, parity wins: exceed budget rather than
        break parity (D-036)."""
        history = [
            Message("user", "q0"),
            _assistant_with_call("c1"),
            _tool_result("c1", "huge " * 50),  # alone exceeds budget
            Message("assistant", "final"),
        ]
        cm = _make_cm(history, max_tokens=30, min_keep=4)  # protect everything
        cm._smart_compress()
        orphan_calls, orphan_results = find_orphans(cm.history)
        # Parity must hold even though budget cannot be satisfied.
        assert orphan_calls == [] and orphan_results == [], (
            f"parity broken to save tokens (violates D-036): {orphan_calls}/{orphan_results}"
        )


# ── Stage 3 (#3): dead 'summarize' strategy removed ──


class TestNoSummarizeStrategy:
    """The 'summarize' compression strategy was unreachable dead code (production
    always uses 'smart'; not user-configurable). It must be removed."""

    def test_summarize_not_in_literal(self):
        import typing

        from supercoder.context.window_manager import ContextConfig

        # Extract the Literal args from the dataclass field type annotation.
        strategy_field = next(
            f for f in typing.get_type_hints(ContextConfig) if f == "compression_strategy"
        )
        annotation = typing.get_type_hints(ContextConfig)[strategy_field]
        args = typing.get_args(annotation)
        assert "summarize" not in args, (
            f"'summarize' must be removed from compression_strategy Literal, got {args}"
        )
        assert "smart" in args and "sliding" in args

    def test_summarize_compress_method_removed(self):
        from supercoder.context import window_manager as wm

        assert not hasattr(wm.ContextWindowManager, "_summarize_compress"), (
            "_summarize_compress dead-code method must be removed"
        )
        # _smart_compress and _sliding_window_compress remain.
        assert hasattr(wm.ContextWindowManager, "_smart_compress")
        assert hasattr(wm.ContextWindowManager, "_sliding_window_compress")

    def test_compress_dispatcher_has_no_summarize_branch(self):
        """The dispatcher body must not mention 'summarize'."""
        import inspect

        from supercoder.context import window_manager as wm

        source = inspect.getsource(wm.ContextWindowManager._compress)
        assert "summarize" not in source, f"_compress() still references summarize:\n{source}"


# ── Stage 4 (#4): display_type-based scoring ──


class TestSmartCompressionScoring:
    """Scoring must reflect native-mode semantics (D-017), not the streaming-only
    <@TOOL_RESULT> marker or a substring 'error' match that fires on reasoning."""

    def test_native_tool_result_scored_high_without_marker(self):
        """A role='tool' message with plain content (native mode) must rank as
        important even though it has no <@TOOL_RESULT> marker."""
        history = [
            # Old, plain low-value messages
            Message("user", "filler " * 5, display_type="user_input"),
            Message("assistant", "filler reply", display_type="response"),
            Message("user", "filler2 " * 5, display_type="user_input"),
            Message("assistant", "filler reply2", display_type="response"),
            # Native tool exchange (no <@TOOL_RESULT> marker anywhere)
            _assistant_with_call("c1"),
            _tool_result("c1", "the actual important result " * 4),
            Message("user", "q recent"),
            Message("assistant", "final answer"),
        ]
        cm = _make_cm(history, max_tokens=140, min_keep=2)
        cm._smart_compress()
        # The tool result (or its whole cluster) must have survived — it is
        # the single highest-value non-recent message in native mode.
        has_tool_result = any(m.role == "tool" for m in cm.history)
        assert has_tool_result, (
            "native tool_result was dropped despite being high-value "
            "(scoring ignores role='tool' in native mode)"
        )

    def test_error_display_type_scored_high(self):
        """display_type='error' must score high (D-017), not a substring match."""
        history = [
            Message("user", "filler " * 5),
            Message("assistant", "filler reply"),
            Message("user", "filler2 " * 5),
            Message("assistant", "filler reply2"),
            # Explicit error-tagged message, no 'error' substring in content
            Message("assistant", "boom trace: AAA", display_type="error"),
            Message("user", "q recent"),
            Message("assistant", "final answer"),
        ]
        cm = _make_cm(history, max_tokens=120, min_keep=2)
        cm._smart_compress()
        has_error = any(m.display_type == "error" for m in cm.history)
        assert has_error, "display_type='error' message was dropped despite being high-value"

    def test_reasoning_with_error_word_not_over_scored(self):
        """An assistant reasoning message that merely mentions 'error' must not
        be kept purely on the substring match when budget is tight and real
        high-value content exists. (Regression for the old +25 'error' heuristic.)"""
        history = [
            Message(
                "assistant",
                "I see an error in the logs, let me think about it " * 6,
                display_type="response",
            ),
            _assistant_with_call("c1"),
            _tool_result("c1", "real important data " * 6),
            Message("user", "q recent"),
            Message("assistant", "final answer"),
        ]
        cm = _make_cm(history, max_tokens=140, min_keep=2)
        cm._smart_compress()
        # The substring-based reasoning message should not displace the tool
        # result. At minimum, if kept, it must not be the sole survivor over
        # the tool result. Verify tool result survived (parity: with its owner).
        has_tool_result = any(m.role == "tool" for m in cm.history)
        assert has_tool_result, (
            "tool_result dropped in favor of an 'error'-word reasoning message "
            "(old +25 'error' heuristic regression)"
        )


# ── C1 (code-review-2026-06-23): budget must count the real API payload ──


class TestSmartCompressionBudgetWithToolCalls:
    """_smart_compress underestimates the budget when messages carry
    tool_calls/tool_call_id overhead, because the old implementation counted
    only msg.content. After compression the history MUST be at or below
    max_tokens (measured via count_api_payload, the real API shape — D-031),
    unless the only remaining messages are protected parity pairs (D-036)."""

    def test_drops_tool_call_clusters_to_fit_real_payload(self):
        tc = TokenCounter(model="some-unknown-local")
        # Each assistant has a FAT tool_calls JSON but tiny content.
        # Content-only counting massively underestimates the real payload,
        # so a naive compressor thinks everything already fits.
        history = [Message("user", "q0")]
        for i in range(20):
            history.append(
                Message(
                    role="assistant",
                    content=f"t{i}",
                    tool_calls=[
                        {
                            "id": f"c{i:03d}",
                            "type": "function",
                            "function": {
                                "name": "code-edit",
                                "arguments": '{"content":"' + ("A" * 300) + '"}',
                            },
                        }
                    ],
                    display_type="response",
                )
            )
            history.append(
                Message(
                    role="tool",
                    content="B" * 50,
                    tool_call_id=f"c{i:03d}",
                    name="code-edit",
                    display_type="tool_result",
                )
            )
        cm = _make_cm(history, max_tokens=400, min_keep=2, protected_steps=2)
        # Tools schema must be set BEFORE compressing: count_api_payload counts it.
        tools_schema = [
            {
                "type": "function",
                "function": {"name": "x", "description": "d" * 200, "parameters": {}},
            }
        ]
        cm.set_tools_schema(tools_schema)

        cm._smart_compress()

        real_tokens = tc.count_api_payload(cm.get_messages_for_api(), tools_schema)
        assert real_tokens <= cm.config.max_tokens, (
            f"history {real_tokens} tokens still exceeds max_tokens "
            f"{cm.config.max_tokens} after smart_compress (parity-safe trimming expected)"
        )
