"""Conversation logging for debugging and analysis."""

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Log directory
LOG_DIR = Path.home() / ".supercoder" / "logs"


def ensure_log_dir() -> Path:
    """Create logs directory if it doesn't exist."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


class ConversationLogger:
    """Logs user inputs and model responses to files for debugging."""

    def __init__(self, model_name: str = "unknown", enabled: bool = True):
        self.model_name = model_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = LOG_DIR / f"session_{self.session_id}.jsonl"
        self.enabled = enabled

        # Write session header
        if self.enabled:
            self._write_entry(
                {
                    "type": "session_start",
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def set_model(self, model_name: str) -> None:
        """Update the current model name (e.g., after switching)."""
        self.model_name = model_name
        self._write_entry(
            {
                "type": "model_switch",
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_user_input(self, message: str) -> None:
        """Log user input."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "user",
                "content": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_model_response(self, response: str, model: str | None = None) -> None:
        """Log model response."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "assistant",
                "model": model or self.model_name,
                "content": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_reasoning(self, reasoning: str, stage: str = "") -> None:
        """Log reasoning/thinking content from model."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "reasoning",
                "stage": stage,
                "content": reasoning,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_stream_event(self, event_type: str, content: str, meta: dict | None = None) -> None:
        """Log individual streaming event for debugging."""
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "type": "stream_event",
            "event_type": event_type,
            "content": content[:500] if content else "",  # Truncate
            "timestamp": datetime.now().isoformat(),
        }
        if meta:
            entry["meta"] = meta
        self._write_entry(entry)

    def log_system_prompt(self, prompt: str) -> None:
        """Log the current system prompt."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "system_prompt",
                "content": prompt,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_messages(self, messages: list) -> None:
        """Log the full list of messages sent to the API."""
        if not self.enabled:
            return
        # Convert Message objects to dicts if needed
        serializable_messages = []
        for msg in messages:
            if hasattr(msg, "to_api_dict"):
                serializable_messages.append(msg.to_api_dict())
            elif isinstance(msg, dict):
                serializable_messages.append(msg)
            else:
                serializable_messages.append(
                    {
                        "role": getattr(msg, "role", "unknown"),
                        "content": getattr(msg, "content", ""),
                    }
                )

        self._write_entry(
            {
                "type": "api_request",
                "messages": serializable_messages,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_tool_call(self, tool_name: str, arguments: str) -> None:
        """Log tool call."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_tool_result(self, tool_name: str, result: str) -> None:
        """Log tool result."""
        if not self.enabled:
            return
        # Truncate long results
        truncated = result[:2000] + "..." if len(result) > 2000 else result
        self._write_entry(
            {
                "type": "tool_result",
                "tool": tool_name,
                "result": truncated,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_tool_output_masked(
        self,
        tool_name: str,
        tool_call_id: str | None,
        masked: bool,
        original_chars: int,
        model_chars: int,
        offload_path: str | None = None,
    ) -> None:
        """Log how tool output was prepared for model context."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "tool_output_masked",
                "tool": tool_name,
                "tool_call_id": tool_call_id,
                "masked": masked,
                "original_chars": original_chars,
                "model_chars": model_chars,
                "offload_path": offload_path,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_context_attachment(self, summary: dict[str, Any]) -> None:
        """Log metadata for @path context attachment expansion."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "context_attachment",
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_permission_decision(
        self,
        *,
        tool_name: str,
        subject: str,
        action: str,
        reason: str,
        source: str,
        matched_rule: str | None = None,
    ) -> None:
        """Log host-side permission decisions without exposing file contents."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "permission_decision",
                "tool": tool_name,
                "subject": subject[:300],
                "action": action,
                "reason": reason,
                "source": source,
                "matched_rule": matched_rule,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_permission_rule_change(
        self,
        *,
        action: str,
        scope: str,
        rule_action: str,
        rule: str,
        source: str,
    ) -> None:
        """Log user-driven changes to session or persistent permission rules."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "permission_rule_change",
                "action": action,
                "scope": scope,
                "rule_action": rule_action,
                "rule": rule[:300],
                "source": source,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_edit_confirmation(
        self,
        *,
        mode: str,
        filepath: str,
        operation: str,
        approved: bool,
    ) -> None:
        """Log host-side file edit confirmations without file contents."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "edit_confirm",
                "mode": mode,
                "filepath": filepath[:300],
                "operation": operation,
                "approved": approved,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_mode_policy(
        self,
        *,
        mode: str,
        tool_name: str,
        action: str,
        reason: str,
        subject: str = "",
    ) -> None:
        """Log host-side mode policy announcements and tool decisions."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "mode_policy",
                "mode": mode,
                "tool": tool_name,
                "action": action,
                "reason": reason,
                "subject": subject[:300],
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_freshness_check(
        self,
        *,
        path: str,
        source: str,
        action: str,
        status: str,
        reason: str,
        size: int | None = None,
        hash_present: bool = False,
    ) -> None:
        """Log read-before-edit freshness decisions without file contents."""
        if not self.enabled:
            return
        self._write_entry(
            {
                "type": "freshness_check",
                "path": path[:300],
                "source": source,
                "action": action,
                "status": status,
                "reason": reason,
                "size": size,
                "hash_present": hash_present,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_error(self, error: str | Exception, *, include_traceback: bool = True) -> None:
        """Log error with optional full traceback.

        Args:
            error: Error message string or Exception object.
            include_traceback: If True and error is an Exception, include full traceback.
        """
        if not self.enabled:
            return
        entry: dict[str, Any] = {
            "type": "error",
            "error": str(error),
            "timestamp": datetime.now().isoformat(),
        }
        if include_traceback and isinstance(error, BaseException):
            entry["traceback"] = traceback.format_exception(error)
        self._write_entry(entry)

    def log_exception(self, context: str = "") -> None:
        """Log the current exception with full traceback.

        Call this from inside an except block to capture the active exception.

        Args:
            context: Optional description of where the error occurred.
        """
        if not self.enabled:
            return
        tb = traceback.format_exc()
        entry: dict[str, Any] = {
            "type": "error",
            "error": context or "Unhandled exception",
            "traceback": tb,
            "timestamp": datetime.now().isoformat(),
        }
        self._write_entry(entry)

    def _write_entry(self, entry: dict) -> None:
        """Write a log entry to file."""
        if not self.enabled:
            return
        try:
            ensure_log_dir()
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Fail silently - logging should not break the app

    @property
    def log_path(self) -> Path:
        """Return path to current log file."""
        return self.log_file


# Global logger instance
_logger: ConversationLogger | None = None


def get_logger() -> ConversationLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = ConversationLogger(enabled=False)
    return _logger


def init_logger(model_name: str, enabled: bool = True) -> ConversationLogger:
    """Initialize logger with model name."""
    global _logger
    _logger = ConversationLogger(model_name, enabled=enabled)
    return _logger
