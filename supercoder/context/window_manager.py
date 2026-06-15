"""Context window management for LLMs."""

from dataclasses import dataclass
from typing import Literal

from rich.console import Console

from ..llm.base import Message, UsageStats
from .token_counter import TokenCounter

console = Console()


@dataclass
class ContextConfig:
    """Configuration for context window management."""

    max_tokens: int = 32000  # Total context window size
    reserved_for_response: int = 4096  # Reserved for model response
    system_prompt_tokens: int = 500  # Estimated system prompt size
    compression_threshold: float = 0.95  # Emergency fallback trimming threshold
    auto_compact: bool = True  # Prefer LLM summarization before trimming
    auto_compact_threshold: float = 0.75  # Trigger auto-compact at this total-context utilization
    protected_recent_steps: int = 6  # Exact recent messages to keep after compact
    min_messages_to_keep: int = 4  # Always keep at least this many messages
    compression_strategy: Literal["sliding", "summarize", "smart"] = "sliding"


@dataclass
class ContextStats:
    """Current context utilization statistics."""

    total_tokens: int
    used_tokens: int
    available_tokens: int
    message_count: int
    utilization_percent: float

    def __str__(self) -> str:
        return (
            f"Context: {self.used_tokens:,}/{self.total_tokens:,} tokens "
            f"({self.utilization_percent:.1f}%), "
            f"{self.message_count} messages"
        )


class ContextWindowManager:
    """Manages the context window for LLM conversations.

    Handles:
    - Token counting for all messages
    - Automatic history compression when approaching limits
    - Statistics and monitoring
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        self.counter = TokenCounter()
        self.history: list[Message] = []
        self._system_prompt: str = ""
        self._system_tokens: int = 0
        self._last_response_total_tokens: int | None = None
        self._tools_schema: list[dict] | None = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and calculate its tokens."""
        self._system_prompt = prompt
        self._system_tokens = self.counter.count(prompt)

    def set_tools_schema(self, tools_schema: list[dict] | None) -> None:
        """Set the native tools schema used for fallback request-size estimation."""
        self._tools_schema = tools_schema

    def add_message(self, message: Message) -> None:
        """Add a message to history without changing latest API usage."""
        self.history.append(message)

    def get_messages(self) -> list[Message]:
        """Get all messages in history."""
        return self.history.copy()

    def get_messages_for_api(self) -> list[Message]:
        """Get messages formatted for API call (with system prompt).

        Filters out display-only messages (e.g. reasoning) that should
        not be sent back to the model.
        """
        messages = []
        if self._system_prompt:
            messages.append(Message("system", self._system_prompt))
        messages.extend(m for m in self.history if m.display_type != "thinking")
        return messages

    def get_stats(self) -> ContextStats:
        """Get current context utilization statistics."""
        if self._last_response_total_tokens is not None:
            used = self._last_response_total_tokens
        else:
            used = self._estimate_current_payload_tokens()
        total = max(1, self.config.max_tokens)

        return ContextStats(
            total_tokens=self.config.max_tokens,
            used_tokens=used,
            available_tokens=self.config.max_tokens - used,
            message_count=len(self.history),
            utilization_percent=(used / total) * 100,
        )

    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []
        self._last_response_total_tokens = None

    def set_max_tokens(self, max_tokens: int) -> None:
        """Update the maximum context token limit at runtime."""
        self.config.max_tokens = max_tokens

    def update_actual_usage(
        self,
        usage: UsageStats | int | None = None,
        *,
        fallback_total_tokens: int | None = None,
    ) -> None:
        """Update context usage from the latest API response.

        The primary value is usage.total_tokens because it reflects the actual
        request plus completion as seen by the API/proxy. When a compatible
        backend omits total_tokens, prompt_tokens + completion_tokens is used.
        If the backend omits usage entirely, the caller can provide a fallback
        estimate of the same request/response pair.
        """
        if isinstance(usage, int):
            self._last_response_total_tokens = max(0, usage)
            return
        if usage:
            if usage.total_tokens:
                self._last_response_total_tokens = usage.total_tokens
                return
            estimated_total = usage.prompt_tokens + usage.completion_tokens
            if estimated_total:
                self._last_response_total_tokens = estimated_total
                return
        if fallback_total_tokens is not None:
            self._last_response_total_tokens = max(0, fallback_total_tokens)
            return
        self._last_response_total_tokens = None

    def reset_actual_usage(self) -> None:
        """Reset actual usage, forcing fallback to estimation."""
        self._last_response_total_tokens = None

    def usable_tokens(self) -> int:
        """Return the context budget available before the reserved response space."""
        return max(1, self.config.max_tokens - self.config.reserved_for_response)

    def should_auto_compact(self) -> bool:
        """Return True when history should be compacted at the next safe boundary."""
        if not self.config.auto_compact:
            return False

        compactable_count = sum(
            1
            for msg in self.history
            if msg.display_type not in ("thinking", "mode_policy")
            and not self._is_compact_summary(msg)
        )
        if compactable_count <= self.config.protected_recent_steps:
            return False

        stats = self.get_stats()
        return stats.used_tokens >= self.config.max_tokens * self.config.auto_compact_threshold

    def should_emergency_compress(self) -> bool:
        """Return True when hard trimming is needed to avoid overflowing the context."""
        stats = self.get_stats()
        return stats.used_tokens >= self.config.max_tokens * self.config.compression_threshold

    def force_compress(self) -> None:
        """Run the configured compression strategy immediately."""
        self._compress()

    def get_protected_recent_messages(self, steps: int | None = None) -> list[Message]:
        """Return the recent exact messages to keep after compaction.

        The count is based on API-visible messages, excluding thinking and old compact
        summaries. If a selected message is part of a native tool-call exchange, the
        corresponding assistant/tool messages are kept together so replay remains valid.
        """
        limit = self.config.protected_recent_steps if steps is None else steps
        if limit <= 0:
            return []

        eligible_indices = [
            idx
            for idx, msg in enumerate(self.history)
            if msg.display_type not in ("thinking", "mode_policy")
            and not self._is_compact_summary(msg)
        ]
        selected = set(eligible_indices[-limit:])
        if not selected:
            return []

        call_owner, call_results = self._build_tool_call_index(self.history)

        changed = True
        while changed:
            changed = False
            for idx in list(selected):
                msg = self.history[idx]
                if msg.display_type == "user_input":
                    prev_idx = idx - 1
                    while (
                        prev_idx >= 0
                        and self.history[prev_idx].display_type == "context_attachment"
                    ):
                        if prev_idx not in selected:
                            selected.add(prev_idx)
                            changed = True
                        prev_idx -= 1
                if msg.display_type == "context_attachment":
                    next_idx = idx + 1
                    if (
                        next_idx < len(self.history)
                        and self.history[next_idx].display_type == "user_input"
                        and next_idx not in selected
                    ):
                        selected.add(next_idx)
                        changed = True
                if msg.role == "tool" and msg.tool_call_id:
                    owner_idx = call_owner.get(msg.tool_call_id)
                    if owner_idx is not None and owner_idx not in selected:
                        selected.add(owner_idx)
                        changed = True
                if msg.role == "assistant" and msg.tool_calls:
                    for raw_call in msg.tool_calls:
                        call_id = raw_call.get("id")
                        if isinstance(call_id, str):
                            for result_idx in call_results.get(call_id, []):
                                if result_idx not in selected:
                                    selected.add(result_idx)
                                    changed = True

        return [self.history[idx] for idx in sorted(selected)]

    def set_initial_summary(
        self, summary: str, recent_messages: list[Message] | None = None
    ) -> None:
        """Set a summary as the initial context after clearing history.

        This is used by the /compact command to preserve key context
        after clearing the conversation history. Recent messages can be kept
        verbatim after the summary to preserve the exact current working state.

        Note: We use 'user' role so the model treats this as input context
        to remember, not as its own previous response.
        """
        self._last_response_total_tokens = None
        self.history = [
            Message(
                "user",
                f"[Previous Context Summary - remember this information]\n\n{summary}",
                display_type="compact_summary",
            )
        ]
        if recent_messages:
            self.history.extend(
                msg
                for msg in recent_messages
                if msg.display_type not in ("thinking", "mode_policy")
                and not self._is_compact_summary(msg)
            )

    def _is_compact_summary(self, msg: Message) -> bool:
        """Return True for compact summary messages, including older saved sessions."""
        return msg.display_type == "compact_summary" or msg.content.startswith("[Previous Context")

    def _compress(self) -> None:
        """Compress history to free up context space."""
        self._last_response_total_tokens = None
        if self.config.compression_strategy == "sliding":
            self._sliding_window_compress()
        elif self.config.compression_strategy == "summarize":
            self._summarize_compress()
        else:
            self._smart_compress()

    def _build_tool_call_index(
        self, history: list[Message]
    ) -> tuple[dict[str, int], dict[str, list[int]]]:
        """Build call_id -> owner-assistant index and call_id -> result indices.

        Shared between protected-recent selection and sliding compression so that
        assistant(tool_calls) and their tool(result) messages stay together.
        """
        call_owner: dict[str, int] = {}
        call_results: dict[str, list[int]] = {}
        for idx, msg in enumerate(history):
            if msg.role == "assistant" and msg.tool_calls:
                for raw_call in msg.tool_calls:
                    call_id = raw_call.get("id")
                    if call_id:
                        call_owner[call_id] = idx
            elif msg.role == "tool" and msg.tool_call_id:
                call_results.setdefault(msg.tool_call_id, []).append(idx)
        return call_owner, call_results

    def _sliding_window_compress(self) -> None:
        """Remove oldest messages to fit in context, keeping tool-call pairs intact."""
        target = self.config.max_tokens * 0.5  # Compress to 50%

        while len(self.history) > self.config.min_messages_to_keep:
            stats = self.get_stats()
            if stats.used_tokens <= target:
                break
            if len(self.history) <= 1:
                break

            # Drop a cluster, not a single message: if history[0] is an assistant
            # with tool_calls, drop it together with its results; if it is a tool
            # result, drop it together with its owner and the owner's other
            # results. Otherwise just drop the single oldest message.
            call_owner, call_results = self._build_tool_call_index(self.history)
            first = self.history[0]
            to_remove = {0}
            if first.role == "assistant" and first.tool_calls:
                for raw_call in first.tool_calls:
                    call_id = raw_call.get("id")
                    if isinstance(call_id, str):
                        for result_idx in call_results.get(call_id, []):
                            to_remove.add(result_idx)
            elif first.role == "tool" and first.tool_call_id:
                owner_idx = call_owner.get(first.tool_call_id)
                if owner_idx is not None:
                    to_remove.add(owner_idx)
                    for raw_call in self.history[owner_idx].tool_calls or []:
                        call_id = raw_call.get("id")
                        if isinstance(call_id, str):
                            for result_idx in call_results.get(call_id, []):
                                to_remove.add(result_idx)

            for idx in sorted(to_remove, reverse=True):
                if len(self.history) > self.config.min_messages_to_keep:
                    self.history.pop(idx)

    def _summarize_compress(self) -> None:
        """Summarize old messages instead of removing them.

        Note: Full implementation would use LLM to summarize.
        For now, falls back to sliding window.
        """
        # This would require an LLM call to summarize
        # For MVP, fall back to sliding window
        self._sliding_window_compress()

    def _smart_compress(self) -> None:
        """Smart compression that keeps important messages.

        Prioritizes:
        - Recent messages
        - Messages with tool results
        - Messages with code
        """
        if len(self.history) <= self.config.min_messages_to_keep:
            return

        # Score messages by importance
        scored = []
        for i, msg in enumerate(self.history):
            score = 0

            # Recent messages are more important
            recency_score = i / len(self.history) * 50
            score += recency_score

            # Tool results are important
            if "<@TOOL_RESULT>" in msg.content:
                score += 30

            # Code blocks are important
            if "```" in msg.content or "def " in msg.content or "class " in msg.content:
                score += 20

            # Errors are important
            if "error" in msg.content.lower() or "Error" in msg.content:
                score += 25

            scored.append((score, i, msg))

        # Sort by score (keep highest)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep top messages that fit
        target = self.config.max_tokens * 0.5
        kept_indices = set()
        current_tokens = self._system_tokens

        for _score, idx, msg in scored:
            msg_tokens = self.counter.count(msg.content)
            if current_tokens + msg_tokens <= target:
                kept_indices.add(idx)
                current_tokens += msg_tokens

        # Always keep the most recent min_messages_to_keep messages
        for i in range(min(self.config.min_messages_to_keep, len(self.history))):
            kept_indices.add(len(self.history) - 1 - i)

        # Preserve message pairs so the conversation stays coherent:
        # - if we keep an assistant message, also keep the preceding user message
        # - if we keep a user message, also keep the following assistant message
        history_len = len(self.history)
        paired: set[int] = set()
        for idx in kept_indices:
            if idx > 0 and self.history[idx].role == "assistant":
                paired.add(idx - 1)
            if idx + 1 < history_len and self.history[idx].role == "user":
                paired.add(idx + 1)
        kept_indices.update(paired)

        # Rebuild history in order
        self.history = [msg for i, msg in enumerate(self.history) if i in kept_indices]

    def estimate_response_fit(self, response_tokens: int) -> bool:
        """Check if a response of given size would fit."""
        stats = self.get_stats()
        return stats.available_tokens >= response_tokens

    def _estimate_current_payload_tokens(self) -> int:
        """Estimate the current chat_with_tools() request payload."""
        return self.counter.count_api_payload(self.get_messages_for_api(), self._tools_schema)
