"""Streaming display buffer with safe-prefix tag detection.

.. deprecated::
    This module is only used in legacy streaming mode (``streaming: true``).
    Native tool calling mode does not require tag filtering.
    Will be removed in a future version.

Ensures tool call tags (e.g. <@TOOL>...</@TOOL>) are never shown to the user
during live markdown streaming. Only "safe" text — guaranteed free of tool call
markup — is released for display.

Works with all supported tool_calling_type formats by tracking format-specific
opening/closing tag patterns.
"""

from __future__ import annotations


# Tag signatures for each supported tool calling format.
# 'opener' is the string that marks the START of a tool call block.
# 'closer' is the string that marks the END (None = consume to end of line).
TAG_SIGNATURES: dict[str, dict[str, str | None]] = {
    "supercoder": {
        "opener": "<@TOOL>",
        "closer": "</@TOOL>",
    },
    "qwen_like": {
        "opener": "to=tool:",
        "closer": None,  # No closing tag; consume to end of line
    },
    "json_block": {
        "opener": "```json",
        "closer": "```",
    },
    "xml_function": {
        "opener": "<function_call",
        "closer": "</function_call>",
    },
    "glm_tool_call": {
        "opener": "<tool_call>",
        "closer": "</tool_call>",
    },
}

# Maximum buffer size (chars) before forcing a flush even without
# a paragraph boundary.  Prevents indefinite holding of long text.
_MAX_HOLD = 200


class StreamingDisplayBuffer:
    """Buffer that separates displayable text from tool call tags.

    Usage::

        buf = StreamingDisplayBuffer("supercoder")
        for token in llm_stream:
            chunk = buf.add(token)
            if chunk:
                md_stream.update(accumulated + chunk)
        remaining = buf.flush()
    """

    def __init__(self, tool_calling_type: str = "supercoder"):
        sig = TAG_SIGNATURES.get(tool_calling_type, TAG_SIGNATURES["supercoder"])
        self._opener: str = sig["opener"]  # type: ignore[assignment]
        self._closer: str | None = sig["closer"]
        self._tool_calling_type = tool_calling_type

        # State
        self._buffer: str = ""  # Pending text not yet released
        self._in_tag: bool = False  # Currently inside a tool call tag?
        self._tag_buffer: str = ""  # Accumulator for tag content (not displayed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, token: str) -> str | None:
        """Add a streaming token.  Return displayable text or ``None``."""
        if self._in_tag:
            return self._consume_tag(token)

        self._buffer += token

        # Check: does the buffer contain a COMPLETE opener?
        tag_idx = self._buffer.find(self._opener)
        if tag_idx >= 0:
            # Everything before the tag is safe to show
            safe = self._buffer[:tag_idx]
            # The tag opener + everything after goes into tag accumulation
            self._tag_buffer = self._buffer[tag_idx:]
            self._buffer = ""
            self._in_tag = True
            return safe if safe else None

        # Check: could the END of the buffer be the START of an opener?
        safe, held = self._split_at_potential_tag()

        if safe:
            self._buffer = held
            # Prefer paragraph / line boundaries for smoother markdown rendering
            return self._batch_by_boundary(safe)

        return None

    def flush(self) -> str:
        """Finalize: return ALL remaining text (end of stream)."""
        result = self._buffer + self._tag_buffer
        self._buffer = ""
        self._tag_buffer = ""
        self._in_tag = False
        return result

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._buffer = ""
        self._tag_buffer = ""
        self._in_tag = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _consume_tag(self, token: str) -> str | None:
        """Handle a token while inside a tag (tag-mode).

        Returns displayable text only if text appears AFTER the closing tag.
        """
        self._tag_buffer += token

        if self._closer is None:
            # No explicit closer (e.g. qwen_like) — consume until newline
            nl_idx = self._tag_buffer.find("\n")
            if nl_idx >= 0:
                # Tag ended at newline — anything after goes back to normal buffer
                after = self._tag_buffer[nl_idx + 1 :]
                self._tag_buffer = ""
                self._in_tag = False
                if after:
                    self._buffer = after
                    # Try to extract displayable content from what's left
                    return self.add("")  # re-enter normal path with empty token
            return None

        # Look for the closing tag
        close_idx = self._tag_buffer.find(self._closer)
        if close_idx >= 0:
            # Found closer — discard the entire tag block
            after_close = self._tag_buffer[close_idx + len(self._closer) :]
            self._tag_buffer = ""
            self._in_tag = False
            if after_close:
                self._buffer = after_close
                return self.add("")  # re-enter normal path
        return None

    def _split_at_potential_tag(self) -> tuple[str, str]:
        """Split buffer into (safe_prefix, held_suffix).

        The held_suffix is text that COULD be the beginning of an opener tag.
        We hold it back to avoid showing partial tags.
        """
        # Check progressively shorter suffixes of the buffer against
        # progressively longer prefixes of the opener.
        max_check = min(len(self._opener), len(self._buffer))
        for i in range(max_check, 0, -1):
            suffix = self._buffer[-i:]
            prefix = self._opener[:i]
            if suffix == prefix:
                # This suffix matches an opener prefix — hold it back
                return self._buffer[:-i], suffix

        # Nothing suspicious — everything is safe
        return self._buffer, ""

    def _batch_by_boundary(self, text: str) -> str | None:
        """Prefer returning text at paragraph/line boundaries.

        This produces smoother markdown rendering (complete headings,
        lists, code blocks).  Falls back to returning everything if the
        text exceeds _MAX_HOLD or has no boundaries.
        """
        if not text:
            return None

        # If text is short, try to wait for a natural boundary
        if len(text) < _MAX_HOLD:
            # Prefer paragraph breaks
            last_para = text.rfind("\n\n")
            if last_para > 0:
                # Return up to the paragraph break, push the rest back
                self._buffer = text[last_para + 2 :] + self._buffer
                return text[: last_para + 2]

            # Prefer line breaks
            last_nl = text.rfind("\n")
            if last_nl > 0:
                self._buffer = text[last_nl + 1 :] + self._buffer
                return text[: last_nl + 1]

        # Long text or no boundaries — return everything
        return text
