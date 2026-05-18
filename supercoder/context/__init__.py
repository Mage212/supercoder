"""Context management module."""

from .references import (
    ContextAttachment,
    ContextReferenceItem,
    expand_context_references,
    extract_context_references,
    summarize_attachment_content,
    summarize_context_attachment,
)
from .session_manager import ChatSession, SessionManager
from .token_counter import TokenCounter, count_tokens, get_token_counter
from .window_manager import ContextConfig, ContextStats, ContextWindowManager

__all__ = [
    "ChatSession",
    "ContextAttachment",
    "ContextConfig",
    "ContextReferenceItem",
    "ContextStats",
    "ContextWindowManager",
    "SessionManager",
    "TokenCounter",
    "count_tokens",
    "expand_context_references",
    "extract_context_references",
    "get_token_counter",
    "summarize_attachment_content",
    "summarize_context_attachment",
]
