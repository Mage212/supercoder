"""OpenAI-compatible LLM client (works with OpenAI, OpenRouter, Ollama, etc.)."""

import contextlib
import json
from collections.abc import Callable, Iterator

from openai import OpenAI

from ..abort_controller import AbortController
from ..agent.streaming_tool_parser import StreamingToolCallParser
from ..config import Config, ModelProfile
from .base import BaseLLM, CompletionResult, Message, NativeToolCall, StreamChunk, UsageStats


class OpenAIClient(BaseLLM):
    """OpenAI-compatible API client.

    Works with:
    - OpenAI API
    - OpenRouter (https://openrouter.ai/api/v1)
    - LM Studio (http://localhost:1234/v1)
    - Ollama (http://localhost:11434/v1)
    - Any OpenAI-compatible endpoint
    """

    # App identification for OpenRouter statistics
    APP_NAME = "SuperCoder CLI"
    APP_URL = "https://github.com/Mage212/supercoder"

    def __init__(self, config: Config):
        self.config = config

        # Add headers for OpenRouter app identification
        default_headers = {
            "HTTP-Referer": self.APP_URL,
            "X-Title": self.APP_NAME,
        }

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=default_headers,
            timeout=config.request_timeout,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.debug = config.debug

    def switch_model(self, profile: ModelProfile) -> None:
        """Switch to a different model profile.

        Reinitializes the OpenAI client with new credentials.
        """
        default_headers = {
            "HTTP-Referer": self.APP_URL,
            "X-Title": self.APP_NAME,
        }
        self.client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.endpoint,
            default_headers=default_headers,
            timeout=profile.request_timeout,
        )
        self.model = profile.model
        self.temperature = profile.temperature
        self.top_p = profile.top_p if profile.top_p is not None else self.config.top_p

    # ------------------------------------------------------------------
    # Primary API: non-streaming with native tool calling
    # ------------------------------------------------------------------

    def chat(self, messages: list[Message]) -> str:
        """Send messages and get complete response."""
        api_messages = [m.to_api_dict() for m in messages]
        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,  # type: ignore[arg-type]
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        """Non-streaming call with native tool support.

        Sends the request with ``tools`` parameter so the server (LM Studio,
        OpenAI, etc.) handles tool call parsing natively. Returns a structured
        ``CompletionResult`` with content, tool_calls, and reasoning already
        separated into clean fields.
        """
        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": self.temperature,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        # --- Extract native tool calls ---
        native_calls: list[NativeToolCall] = []
        raw_tool_calls: list[dict] | None = None

        if msg.tool_calls:
            raw_tool_calls = []
            for tc in msg.tool_calls:
                # Preserve raw API format for context replay
                raw_tc = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                raw_tool_calls.append(raw_tc)

                # Parse arguments JSON
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}

                native_calls.append(
                    NativeToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        # --- Extract reasoning (GLM / DeepSeek models) ---
        reasoning = getattr(msg, "reasoning_content", None) or ""

        # --- Extract token usage ---
        usage = None
        if response.usage:
            usage = UsageStats(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return CompletionResult(
            content=msg.content or "",
            tool_calls=native_calls,
            reasoning=reasoning,
            raw_tool_calls=raw_tool_calls,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Interruptible streaming: same result as chat_with_tools but abortable
    # ------------------------------------------------------------------

    def chat_with_tools_interruptible(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        abort_controller: AbortController | None = None,
        on_chunk: Callable[[int], None] | None = None,
        max_completion_tokens: int = 16000,
        tool_choice: str | dict | None = None,
    ) -> CompletionResult:
        """Streaming variant of chat_with_tools that checks abort between chunks.

        Uses ``stream=True`` internally but assembles the full response before
        returning, so the caller gets the same ``CompletionResult`` as
        ``chat_with_tools()``.  Between chunks the abort controller is polled;
        if aborted the stream is closed and ``AgentAbortedError`` is raised.

        Args:
            on_chunk: Optional callback invoked with approx token count on each chunk.
            max_completion_tokens: Hard limit on generated tokens (default 16,000).
        """
        from ..abort_controller import AgentAbortedError

        kwargs: dict = {
            "model": self.model,
            "messages": [m.to_api_dict() for m in messages],
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        stream = self.client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_parser = StreamingToolCallParser()
        usage: UsageStats | None = None
        truncated = False
        total_chars = 0  # Running character count for token estimation

        try:
            for chunk in stream:
                # Check abort
                if abort_controller is not None and abort_controller.is_aborted:
                    stream.close()
                    raise AgentAbortedError("Agent execution aborted by user")

                if not chunk.choices:
                    # Usage-only chunk (last chunk with stream_options)
                    if chunk.usage:
                        usage = UsageStats(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            completion_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )
                    continue

                delta = chunk.choices[0].delta
                chunk_chars = 0

                # Accumulate content
                if delta.content:
                    content_parts.append(delta.content)
                    chunk_chars += len(delta.content)

                # Accumulate reasoning
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
                    chunk_chars += len(rc)

                # Accumulate tool calls via streaming parser
                if delta.tool_calls:
                    tc_parser.feed_chunk(delta.tool_calls)
                    for tc_delta in delta.tool_calls:
                        fn = tc_delta.function
                        if fn:
                            if fn.arguments:
                                chunk_chars += len(fn.arguments)
                            if fn.name:
                                chunk_chars += len(fn.name)

                # Update running count and notify
                if chunk_chars:
                    total_chars += chunk_chars
                    approx_tokens = total_chars // 4
                    if on_chunk:
                        on_chunk(approx_tokens)

                    # Truncate if over limit
                    if approx_tokens >= max_completion_tokens:
                        truncated = True
                        break

            # Final usage (some providers send it on the last content chunk)
            if not truncated and chunk.usage and usage is None:
                usage = UsageStats(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

        finally:
            with contextlib.suppress(Exception):
                stream.close()

        # Detect truncated tool calls (provider says 'stop' but JSON incomplete)
        if not truncated and tc_parser.has_incomplete_tool_calls():
            truncated = True

        # Assemble tool calls from streaming parser
        native_calls: list[NativeToolCall] = []
        raw_tool_calls: list[dict] | None = None

        native_calls_data, raw_tool_calls = tc_parser.finalize()
        if native_calls_data:
            for tc_data in native_calls_data:
                native_calls.append(
                    NativeToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=tc_data["arguments"],
                    )
                )

        content_text = "".join(content_parts)
        if truncated:
            content_text += f"\n\n[RESPONSE TRUNCATED — exceeded {max_completion_tokens:,} token generation limit]"

        return CompletionResult(
            content=content_text,
            tool_calls=native_calls,
            reasoning="".join(reasoning_parts),
            raw_tool_calls=raw_tool_calls or None,
            usage=usage,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Deprecated: streaming
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """Send messages and stream response chunks.

        .. deprecated::
            Streaming mode is deprecated. Use ``chat_with_tools()`` for
            reliable native tool calling instead.
        """
        api_messages = [m.to_api_dict() for m in messages]
        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,  # type: ignore[arg-type]
            "temperature": self.temperature,
            "stream": True,
        }
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        stream = self.client.chat.completions.create(**kwargs)

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                # Extract reasoning_content for GLM/DeepSeek models
                reasoning = getattr(delta, "reasoning_content", None) or ""

                if content or reasoning:
                    yield StreamChunk(content=content, reasoning=reasoning)

        yield StreamChunk(content="", is_done=True)
