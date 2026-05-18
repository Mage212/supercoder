"""LLM clients."""

from typing import TYPE_CHECKING

from .base import BaseLLM, Message, StreamChunk

if TYPE_CHECKING:
    from .openai_client import OpenAIClient

__all__ = ["BaseLLM", "Message", "OpenAIClient", "StreamChunk"]


def __getattr__(name: str):
    """Lazy-load the OpenAI client to avoid import cycles during package init."""
    if name == "OpenAIClient":
        from .openai_client import OpenAIClient

        return OpenAIClient
    raise AttributeError(name)
