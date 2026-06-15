"""Streaming tool call parser with bracket-depth state tracking.

Handles incremental accumulation of tool call arguments from streaming
API chunks (SSE deltas). Tracks JSON structure state (bracket depth,
string boundaries, escape sequences) on a per-index basis to enable:

1. Early detection of truncated tool calls (depth > 0 at stream end)
2. Collision resolution when same index gets different tool call IDs
3. 3-level JSON recovery: exact parse -> auto-close strings -> json-repair
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class _ToolCallBuffer:
    """Per-index state for a single streaming tool call."""

    buffer: str = ""
    depth: int = 0
    in_string: bool = False
    escape_next: bool = False
    tool_id: str = ""
    tool_name: str = ""
    has_content: bool = False
    unbalanced: bool = False

    def feed_char(self, ch: str) -> None:
        """Process a single character and update parsing state."""
        self.has_content = True

        if self.escape_next:
            self.buffer += ch
            self.escape_next = False
            return

        if ch == "\\" and self.in_string:
            self.buffer += ch
            self.escape_next = True
            return

        if ch == '"':
            self.in_string = not self.in_string
            self.buffer += ch
            return

        self.buffer += ch

        if not self.in_string:
            if ch in ("{", "["):
                self.depth += 1
            elif ch in ("}", "]"):
                self.depth -= 1
                if self.depth < 0:
                    self.unbalanced = True

    def is_complete(self) -> bool:
        """Check if the JSON arguments are structurally complete."""
        return self.has_content and self.depth == 0 and not self.in_string and not self.unbalanced


class StreamingToolCallParser:
    """Stateful parser for streaming tool call fragments.

    Usage::

        parser = StreamingToolCallParser()
        for chunk in stream:
            parser.feed_chunk(chunk.delta.tool_calls)
        native_calls, raw_tool_calls = parser.finalize()
    """

    def __init__(self) -> None:
        self._buffers: dict[int, _ToolCallBuffer] = {}
        self._id_to_index: dict[str, int] = {}

    def _get_or_create_buffer(self, index: int, tool_id: str = "") -> _ToolCallBuffer:
        """Get existing buffer or create new one, handling index collisions."""
        # Prefer routing by tool_id when known — some providers reuse indices
        # across calls but keep the id stable.
        if tool_id and tool_id in self._id_to_index:
            return self._buffers[self._id_to_index[tool_id]]

        if index not in self._buffers:
            buf = _ToolCallBuffer()
            self._buffers[index] = buf
            if tool_id:
                buf.tool_id = tool_id
                self._id_to_index[tool_id] = index
            return buf

        buf = self._buffers[index]

        # Collision: same index, different tool call ID
        if tool_id and buf.tool_id and tool_id != buf.tool_id:
            if tool_id in self._id_to_index:
                return self._buffers[self._id_to_index[tool_id]]

            # Existing buffer complete → remap new call to next free index
            if buf.is_complete():
                try:
                    json.loads(buf.buffer)
                    new_index = max(self._buffers.keys()) + 1
                    new_buf = _ToolCallBuffer(tool_id=tool_id)
                    self._buffers[new_index] = new_buf
                    self._id_to_index[tool_id] = new_index
                    return new_buf
                except json.JSONDecodeError:
                    pass

            # Existing buffer incomplete → also remap
            new_index = max(self._buffers.keys()) + 1
            new_buf = _ToolCallBuffer(tool_id=tool_id)
            self._buffers[new_index] = new_buf
            self._id_to_index[tool_id] = new_index
            return new_buf

        # Normal: update ID if not yet set
        if tool_id and not buf.tool_id:
            buf.tool_id = tool_id
            self._id_to_index[tool_id] = index

        return buf

    def feed_chunk(self, tool_calls_deltas: list) -> None:
        """Process a list of tool call deltas from a streaming chunk.

        Each delta is expected to have: .index (int), .id (str|None),
        .function.name (str|None), .function.arguments (str|None)
        """
        for delta in tool_calls_deltas:
            idx = delta.index
            buf = self._get_or_create_buffer(idx, getattr(delta, "id", "") or "")

            fn = delta.function
            if fn:
                if fn.name:
                    buf.tool_name += fn.name
                if fn.arguments:
                    for ch in fn.arguments:
                        buf.feed_char(ch)

    def has_incomplete_tool_calls(self) -> bool:
        """Check if any buffer has a started but incomplete tool call.

        Detects truncated responses where the provider reports
        finish_reason='stop' but the JSON arguments are still open.
        """
        for buf in self._buffers.values():
            if (
                buf.tool_name
                and buf.has_content
                and (buf.depth > 0 or buf.in_string or buf.unbalanced)
            ):
                return True
        return False

    def _try_parse_arguments(self, buf: _ToolCallBuffer) -> dict:
        """3-level JSON recovery for tool call arguments.

        Level 1: Standard json.loads
        Level 2: Auto-close unclosed strings or braces + json.loads
        Level 3: json-repair library
        """
        text = buf.buffer.strip()
        if not text:
            return {}

        # Level 1: Direct parse
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            pass

        # Level 2a: Auto-close unclosed strings
        if buf.in_string:
            try:
                return json.loads(text + '"', strict=False)
            except json.JSONDecodeError:
                pass

        # Level 2b: Auto-close unclosed braces
        if buf.depth > 0:
            closed = text + ("}" * buf.depth)
            try:
                return json.loads(closed, strict=False)
            except json.JSONDecodeError:
                pass

        # Level 3: json-repair
        try:
            import json_repair

            result = json_repair.loads(text)
            return result if isinstance(result, dict) else {"_raw": text}
        except Exception:
            pass

        return {"_raw": text}

    def finalize(self) -> tuple[list[dict], list[dict] | None]:
        """Finalize all buffers and return parsed tool calls.

        Returns:
            Tuple of (native_calls_list, raw_tool_calls).
            native_calls_list: list of dicts with id, name, arguments.
            raw_tool_calls: list in API format or None if no tool calls.
        """
        if not self._buffers:
            return [], None

        raw_tool_calls = []
        native_calls = []

        for idx in sorted(self._buffers):
            buf = self._buffers[idx]
            if not buf.tool_name and not buf.has_content:
                continue

            args = self._try_parse_arguments(buf)
            tool_id = buf.tool_id or f"call_{idx}"

            raw_tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": buf.tool_name,
                        "arguments": buf.buffer,
                    },
                }
            )

            native_calls.append(
                {
                    "id": tool_id,
                    "name": buf.tool_name,
                    "arguments": args,
                }
            )

        if not raw_tool_calls:
            return [], None

        return native_calls, raw_tool_calls

    def reset(self) -> None:
        """Clear all state for a new streaming session."""
        self._buffers.clear()
        self._id_to_index.clear()
