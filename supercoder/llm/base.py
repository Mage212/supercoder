"""Base LLM client interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str
    reasoning: str = ""  # For reasoning_content (GLM, DeepSeek, etc.)
    is_done: bool = False


class BaseLLM(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: list[Message]) -> str:
        """Send messages and get complete response."""
        pass
    
    @abstractmethod
    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """Send messages and stream response chunks."""
        pass
