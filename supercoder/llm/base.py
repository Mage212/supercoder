"""Base LLM client interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..abort_controller import AbortController


@dataclass
class Message:
    """Chat message.

    Supports standard roles (system, user, assistant) and native tool calling:
    - Assistant messages with tool calls: set ``tool_calls`` to the raw API list.
    - Tool result messages: set ``role="tool"``, ``tool_call_id``, and ``name``.
    """

    role: str  # "system", "user", "assistant", "tool"
    content: str
    # For assistant messages that contain tool calls (raw API format)
    tool_calls: list[dict] | None = field(default=None, repr=False)
    # For tool result messages (role="tool")
    tool_call_id: str | None = None
    name: str | None = None  # tool name (used with role="tool")

    def to_api_dict(self) -> dict:
        """Serialize to a dict suitable for the OpenAI messages array."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name and self.role == "tool":
            d["name"] = self.name
        return d


@dataclass
class NativeToolCall:
    """A single tool call extracted from the API response."""

    id: str  # tool_call ID from API (required for multi-turn)
    name: str
    arguments: dict


@dataclass
class UsageStats:
    """Token usage reported by the API response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResult:
    """Structured response from LLM (non-streaming)."""

    content: str  # Text part of the response
    tool_calls: list[NativeToolCall]  # Native tool calls (already parsed dicts)
    reasoning: str = ""  # Reasoning / thinking content (GLM, DeepSeek, etc.)
    raw_tool_calls: list[dict] | None = field(
        default=None, repr=False
    )  # Raw API tool_calls for context
    usage: UsageStats | None = None  # Actual token usage from API


@dataclass
class StreamChunk:
    """Streaming response chunk.

    .. deprecated::
        Streaming mode is deprecated. Use ``chat_with_tools()`` instead.
    """

    content: str
    reasoning: str = ""  # For reasoning_content (GLM, DeepSeek, etc.)
    is_done: bool = False


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""

    model: str = ""

    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Send messages and get complete response."""
        pass

    @abstractmethod
    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> CompletionResult:
        """Send messages with native tool support (non-streaming).

        Args:
            messages: Conversation history.
            tools: OpenAI-compatible tool schemas (from ToolDefinition.to_openai_schema()).

        Returns:
            CompletionResult with content, tool_calls, and reasoning.
        """
        pass

    def chat_with_tools_interruptible(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        abort_controller: AbortController | None = None,
        on_chunk: Callable[[int], None] | None = None,
        max_completion_tokens: int = 16000,
    ) -> CompletionResult:
        """Interruptible variant of chat_with_tools using streaming internally.

        Falls back to ``chat_with_tools()`` if not overridden.
        """
        return self.chat_with_tools(messages, tools)

    @abstractmethod
    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """Send messages and stream response chunks.

        .. deprecated::
            Streaming mode is deprecated. Use ``chat_with_tools()`` instead.
        """
        pass
