"""Context window management for LLMs."""

from dataclasses import dataclass, field
from typing import Literal
from rich.console import Console

from ..llm.base import Message
from .token_counter import TokenCounter

console = Console()


@dataclass
class ContextConfig:
    """Configuration for context window management."""
    max_tokens: int = 32000           # Total context window size
    reserved_for_response: int = 4096  # Reserved for model response
    system_prompt_tokens: int = 500    # Estimated system prompt size
    compression_threshold: float = 0.7 # Start compression at this utilization
    min_messages_to_keep: int = 4      # Always keep at least this many messages
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
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt and calculate its tokens."""
        self._system_prompt = prompt
        self._system_tokens = self.counter.count(prompt)
    
    def add_message(self, message: Message) -> None:
        """Add a message to history, compressing if needed."""
        self.history.append(message)
        
        # Check if we need to compress
        stats = self.get_stats()
        threshold = self.config.max_tokens * self.config.compression_threshold
        
        if stats.used_tokens > threshold:
            self._compress()
    
    def get_messages(self) -> list[Message]:
        """Get all messages in history."""
        return self.history.copy()
    
    def get_messages_for_api(self) -> list[Message]:
        """Get messages formatted for API call (with system prompt)."""
        messages = []
        if self._system_prompt:
            messages.append(Message("system", self._system_prompt))
        messages.extend(self.history)
        return messages
    
    def get_stats(self) -> ContextStats:
        """Get current context utilization statistics."""
        history_tokens = self.counter.count_messages(self.history)
        used = self._system_tokens + history_tokens
        available = self.config.max_tokens - self.config.reserved_for_response
        
        return ContextStats(
            total_tokens=self.config.max_tokens,
            used_tokens=used,
            available_tokens=available - used,
            message_count=len(self.history),
            utilization_percent=(used / available) * 100 if available > 0 else 100
        )
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.history = []
    
    def set_max_tokens(self, max_tokens: int) -> None:
        """Update the maximum context token limit at runtime."""
        self.config.max_tokens = max_tokens
    
    def set_initial_summary(self, summary: str) -> None:
        """Set a summary as the initial context after clearing history.
        
        This is used by the /compact command to preserve key context
        after clearing the conversation history.
        
        Note: We use 'user' role so the model treats this as input context
        to remember, not as its own previous response.
        """
        self.history = [Message("user", f"[Previous Context Summary - remember this information]\n\n{summary}")]
    
    def _compress(self) -> None:
        """Compress history to free up context space."""
        if self.config.compression_strategy == "sliding":
            self._sliding_window_compress()
        elif self.config.compression_strategy == "summarize":
            self._summarize_compress()
        else:
            self._smart_compress()
    
    def _sliding_window_compress(self) -> None:
        """Remove oldest messages to fit in context."""
        target = self.config.max_tokens * 0.5  # Compress to 50%
        
        while len(self.history) > self.config.min_messages_to_keep:
            stats = self.get_stats()
            if stats.used_tokens <= target:
                break
            
            # Remove the oldest message (after any potential system context)
            if len(self.history) > 1:
                removed = self.history.pop(0)
                if self.config.compression_strategy == "sliding":
                    # Optionally log what was removed
                    pass
    
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
        
        for score, idx, msg in scored:
            msg_tokens = self.counter.count(msg.content)
            if current_tokens + msg_tokens <= target:
                kept_indices.add(idx)
                current_tokens += msg_tokens
        
        # Always keep min messages
        for i in range(min(self.config.min_messages_to_keep, len(self.history))):
            kept_indices.add(len(self.history) - 1 - i)
        
        # Rebuild history in order
        self.history = [
            msg for i, msg in enumerate(self.history) 
            if i in kept_indices
        ]
    
    def estimate_response_fit(self, response_tokens: int) -> bool:
        """Check if a response of given size would fit."""
        stats = self.get_stats()
        return stats.available_tokens >= response_tokens
