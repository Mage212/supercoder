"""OpenAI-compatible LLM client (works with OpenAI, OpenRouter, Ollama, etc.)."""

import json
from collections.abc import Iterator

from openai import OpenAI

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

    # ------------------------------------------------------------------
    # Primary API: non-streaming with native tool calling
    # ------------------------------------------------------------------

    def chat(self, messages: list[Message]) -> str:
        """Send messages and get complete response."""
        api_messages = [m.to_api_dict() for m in messages]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
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
        if tools:
            kwargs["tools"] = tools

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
    # Deprecated: streaming
    # ------------------------------------------------------------------

    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """Send messages and stream response chunks.

        .. deprecated::
            Streaming mode is deprecated. Use ``chat_with_tools()`` for
            reliable native tool calling instead.
        """
        api_messages = [m.to_api_dict() for m in messages]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=api_messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                # Extract reasoning_content for GLM/DeepSeek models
                reasoning = getattr(delta, "reasoning_content", None) or ""

                if content or reasoning:
                    yield StreamChunk(content=content, reasoning=reasoning)

        yield StreamChunk(content="", is_done=True)
