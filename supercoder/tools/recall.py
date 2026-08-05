"""Recall tool: search past events and recover compacted tool outputs.

The JSONL session log (``~/.supercoder/logs/session_*.jsonl``) records tool
calls (full arguments), tool results (truncated), commands, and errors. It is
host-owned and append-only, but historically write-only — the agent had no way
to search it, so anything compacted out of the context window was gone.

This tool turns that log into a retrieval surface. It also recovers the full
text of large tool outputs (>8000 chars) that were offloaded to
``.supercoder/tool-outputs/`` and replaced with a head+tail digest.

Trust model:
- The JSONL log lives outside the repo (``~/.supercoder/logs/``) and is always
  readable.
- Offloaded tool-output files live inside the repo (``.supercoder/tool-outputs/``)
  and are attacker-controllable in a cloned malicious repo (R2-7-class prompt
  injection). Reading them is gated behind ``allow_offload_read``, threaded
  from the repo trust decision exactly like session loading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..logging import LOG_DIR, get_logger
from ..permissions import PermissionPolicy
from .base import BaseTool, ToolDefinition
from .tool_utils import resolve_within_root

# Per-event content fields searched for a substring query.
_CONTENT_FIELDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "tool_call": ("tool", "arguments"),
    "tool_result": ("tool", "result"),
    "error": ("error",),
    "tool_output_masked": ("tool",),
    "user": ("content",),
    "assistant": ("content",),
    "permission_decision": ("tool", "subject", "reason"),
    "edit_confirm": ("filepath",),
    "default": ("content", "result", "arguments", "error"),
}


class RecallTool(BaseTool):
    """Search past events and recover compacted tool outputs."""

    DEFAULT_LIMIT = 10
    HARD_LIMIT = 50
    PREVIEW_CHARS = 500

    def __init__(
        self,
        allowed_root: Path | None = None,
        permission_policy: PermissionPolicy | None = None,
        allow_offload_read: bool = False,
        log_dir: Path | None = None,
    ):
        self.allowed_root = allowed_root
        self.permission_policy = permission_policy
        self.allow_offload_read = allow_offload_read
        self.log_dir = log_dir or LOG_DIR

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="recall",
            description=(
                "Search past events from this and previous sessions: tool calls, "
                "tool results, commands, errors. Recover the full output of large "
                "tool calls that were compacted out of context. Use this when the "
                "information you need is no longer in the recent conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to search for in event content, arguments, or results",
                    },
                    "type": {
                        "type": "string",
                        "description": "Filter by event type",
                        "enum": ["tool_call", "tool_result", "error", "user", "assistant"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": f"Max events to return (default {self.DEFAULT_LIMIT}, max {self.HARD_LIMIT})",
                        "default": self.DEFAULT_LIMIT,
                    },
                    "session": {
                        "type": "string",
                        "description": "Which sessions to search: current or all (default current)",
                        "enum": ["current", "all"],
                        "default": "current",
                    },
                    "offload": {
                        "type": "string",
                        "description": (
                            "Read the full text of a previously offloaded tool output by its path "
                            "(as shown in 'Full output saved to: ...'). Repo-local; requires trust."
                        ),
                    },
                },
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"

        # Offload-read mode: recover a single full tool output.
        offload_path = args.get("offload")
        if offload_path:
            return self._read_offload(offload_path)

        # Search mode.
        query = args.get("query", "")
        event_type = args.get("type")
        limit = self._coerce_limit(args.get("limit", self.DEFAULT_LIMIT))
        session_scope = args.get("session", "current")

        log_files = self._resolve_log_files(session_scope)
        if not log_files:
            return self._no_logs_message(session_scope)

        matches = self._search_logs(log_files, query=query, event_type=event_type, limit=limit)
        if not matches:
            return "No matching events found. Try a different query, a broader type filter, or session=all."

        return self._render_matches(matches)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _resolve_log_files(self, scope: str) -> list[Path]:
        """Return the JSONL files to search, newest first."""
        if scope == "all":
            files = sorted(self.log_dir.glob("session_*.jsonl"), key=lambda p: p.name, reverse=True)
            return files

        # current: the active session log, if logging is enabled.
        logger = get_logger()
        if not getattr(logger, "enabled", False):
            return []
        log_file = getattr(logger, "log_file", None)
        return [log_file] if log_file and log_file.exists() else []

    def _no_logs_message(self, scope: str) -> str:
        if scope == "current":
            return (
                "Session logging is disabled, so the current session has no searchable log. "
                "Re-launch without --no-log to enable logging, or use session=all to search past sessions."
            )
        return "No session logs found."

    def _search_logs(
        self,
        log_files: list[Path],
        *,
        query: str,
        event_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        query_lower = query.lower() if query else ""
        for log_file in log_files:
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not self._entry_matches(
                            entry, query_lower=query_lower, event_type=event_type
                        ):
                            continue
                        matches.append(entry)
            except OSError:
                continue

        # Newest first by timestamp (stable sort preserves file order on ties).
        matches.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return matches[:limit]

    def _entry_matches(
        self, entry: dict[str, Any], *, query_lower: str, event_type: str | None
    ) -> bool:
        etype = entry.get("type")
        if event_type and etype != event_type:
            return False
        if not query_lower:
            return True  # type filter only
        fields = (
            _CONTENT_FIELDS_BY_TYPE.get(etype, _CONTENT_FIELDS_BY_TYPE["default"])
            if isinstance(etype, str)
            else _CONTENT_FIELDS_BY_TYPE["default"]
        )
        # Also always search the union of common fields as a fallback.
        search_fields = set(fields) | {"content", "result", "arguments", "error"}
        for key in search_fields:
            value = entry.get(key)
            if isinstance(value, str) and query_lower in value.lower():
                return True
        return False

    def _render_matches(self, matches: list[dict[str, Any]]) -> str:
        lines = [f"# Recall: {len(matches)} matching event(s)"]
        for entry in matches:
            lines.append("")
            lines.append(self._render_entry(entry))
        return "\n".join(lines)

    def _render_entry(self, entry: dict[str, Any]) -> str:
        ts = entry.get("timestamp", "?")
        etype = entry.get("type", "?")
        tool = entry.get("tool", "")
        header = f"[{ts}] {etype}"
        if tool:
            header += f" tool={tool}"

        body = self._entry_body(entry)
        preview = body[: self.PREVIEW_CHARS]
        if len(body) > self.PREVIEW_CHARS:
            preview += " ..."

        out = [header]
        if preview:
            out.append(preview)

        # Hint at recoverable full output.
        offload = entry.get("offload_path")
        if offload and entry.get("masked"):
            out.append(f"Full output saved to: {offload}")
        return "\n".join(out)

    @staticmethod
    def _entry_body(entry: dict[str, Any]) -> str:
        for key in ("result", "error", "content", "arguments"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    # ------------------------------------------------------------------
    # Offload read (trust-gated)
    # ------------------------------------------------------------------

    def _read_offload(self, offload_input: str) -> str:
        if not self.allow_offload_read:
            return (
                "Error: Reading tool-output files requires trusting this repo "
                "(they live under .supercoder/tool-outputs/ and could be planted by a "
                "cloned malicious repo). Trust the repo and re-launch, or read host-owned "
                "logs via the search mode instead."
            )

        path, error = resolve_within_root(offload_input, self.allowed_root)
        if error or path is None:
            return error or "Error: Invalid offload path"
        if not path.exists():
            return f"Error: Offloaded output not found: {offload_input}"

        if self.permission_policy:
            decision = self.permission_policy.check_path(path, "read")
            if decision.denied:
                get_logger().log_permission_decision(
                    tool_name=self.definition.name,
                    subject=self.permission_policy.relative_path(path),
                    action=decision.action.value,
                    reason=decision.reason,
                    source=decision.source,
                    matched_rule=decision.matched_rule,
                )
                return self.permission_policy.format_denial(offload_input, decision)

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading offloaded output: {e}"
        return f"# Offloaded output: {path}\n\n{text}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_limit(self, raw) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return self.DEFAULT_LIMIT
        return max(1, min(value, self.HARD_LIMIT))
