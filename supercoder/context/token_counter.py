"""Token counting utilities."""

import json
from typing import Any


class TokenCounter:
    """Count tokens in text, with optional tiktoken support."""

    def __init__(self, use_tiktoken: bool = True, model: str = "gpt-4"):
        self.encoder = None
        self.model = model

        if use_tiktoken:
            try:
                import tiktoken

                # Try to get encoding for specific model
                try:
                    self.encoder = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fallback to cl100k_base (GPT-4 encoding)
                    self.encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # tiktoken not installed, will use estimation
                pass

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        if self.encoder:
            return len(self.encoder.encode(text))

        # Fallback estimation: ~4 chars per token for English/code
        return self._estimate_tokens(text)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count without tiktoken."""
        # More accurate estimation for code:
        # - Count words (split on whitespace and punctuation)
        # - Account for code-specific patterns

        # Basic word count
        words = len(text.split())

        # Character-based estimate
        chars = len(text)
        char_estimate = chars // 4

        # Use the higher of the two estimates
        return max(words, char_estimate)

    def count_serialized(self, value: Any) -> int:
        """Count tokens in compact JSON-serialized structured data."""
        if value is None:
            return 0
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        return self.count(text)

    def _message_to_api_dict(self, msg: Any) -> dict:
        """Convert a Message-like object to the shape sent to the chat API."""
        if hasattr(msg, "to_api_dict"):
            return msg.to_api_dict()
        if isinstance(msg, dict):
            return msg
        return {
            "role": getattr(msg, "role", "user"),
            "content": getattr(msg, "content", ""),
        }

    def count_api_messages(self, messages: list) -> int:
        """Count tokens in serialized API message payloads."""
        return self.count_serialized([self._message_to_api_dict(msg) for msg in messages])

    def count_tools_schema(self, tools_schema: list[dict] | None) -> int:
        """Count tokens in the serialized native tools schema."""
        if not tools_schema:
            return 0
        return self.count_serialized(tools_schema)

    def count_api_payload(self, messages: list, tools_schema: list[dict] | None = None) -> int:
        """Count tokens in the request payload shape used by chat_with_tools()."""
        payload: dict[str, Any] = {"messages": [self._message_to_api_dict(msg) for msg in messages]}
        if tools_schema:
            payload["tools"] = tools_schema
        return self.count_serialized(payload)

    def count_messages(self, messages: list) -> int:
        """Count tokens in a list of messages."""
        return self.count_api_messages(messages)

    @property
    def has_accurate_counting(self) -> bool:
        """Check if we have tiktoken for accurate counting."""
        return self.encoder is not None


# Global instance for convenience
_default_counter = None


def get_token_counter(use_tiktoken: bool = True) -> TokenCounter:
    """Get or create a global token counter."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter(use_tiktoken=use_tiktoken)
    return _default_counter


def count_tokens(text: str) -> int:
    """Convenience function to count tokens."""
    return get_token_counter().count(text)
