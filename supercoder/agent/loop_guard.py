"""Host-side loop detection for the native agent chain."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm.base import NativeToolCall


@dataclass(frozen=True)
class LoopGuardConfig:
    """Runtime loop detection settings."""

    enabled: bool = True
    identical_tool_call_threshold: int = 3
    identical_tool_error_threshold: int = 3
    no_progress_edit_threshold: int = 3
    assistant_repeat_threshold: int = 3
    corrective_attempts: int = 1

    @classmethod
    def from_mapping(cls, data: dict | bool | None) -> LoopGuardConfig:
        """Build config from user settings."""
        if data is False:
            return cls(enabled=False)
        if data is True or data is None:
            return cls()
        if not isinstance(data, dict):
            return cls()

        def int_value(key: str, default: int, minimum: int = 1) -> int:
            try:
                return max(minimum, int(data.get(key, default)))
            except (TypeError, ValueError):
                return default

        return cls(
            enabled=bool(data.get("enabled", True)),
            identical_tool_call_threshold=int_value("identical_tool_call_threshold", 3),
            identical_tool_error_threshold=int_value("identical_tool_error_threshold", 3),
            no_progress_edit_threshold=int_value("no_progress_edit_threshold", 3),
            assistant_repeat_threshold=int_value("assistant_repeat_threshold", 3),
            corrective_attempts=int_value("corrective_attempts", 1, minimum=0),
        )


@dataclass(frozen=True)
class LoopGuardDecision:
    """Decision returned when a loop pattern is detected."""

    kind: str
    reason: str
    message: str
    stop: bool


class AgentLoopGuard:
    """Detect repeated model behavior before the hard tool-iteration cap."""

    def __init__(self, config: LoopGuardConfig):
        self.config = config
        self._last_tool_call_key: str | None = None
        self._tool_call_streak = 0
        self._last_assistant_key: str | None = None
        self._assistant_streak = 0
        self._tool_error_counts: dict[str, int] = {}
        self._no_progress_edit_counts: dict[str, int] = {}
        self._corrective_counts: dict[str, int] = {}

    @classmethod
    def from_config(cls, data: dict | bool | None) -> AgentLoopGuard:
        """Create guard from user-facing config data."""
        return cls(LoopGuardConfig.from_mapping(data))

    @property
    def enabled(self) -> bool:
        """Whether loop detection is active."""
        return self.config.enabled

    def observe_assistant(
        self,
        content: str,
        tool_calls: list[NativeToolCall],
    ) -> LoopGuardDecision | None:
        """Track repeated assistant responses before tool execution."""
        if not self.enabled:
            return None
        if not content and not tool_calls:
            return None

        call_keys = [self.tool_call_fingerprint(tc.name, tc.arguments) for tc in tool_calls]
        normalized_content = re.sub(r"\s+", " ", content).strip()
        key = self._hash_json({"content": normalized_content, "tool_calls": call_keys})

        if key == self._last_assistant_key:
            self._assistant_streak += 1
        else:
            self._last_assistant_key = key
            self._assistant_streak = 1

        if self._assistant_streak >= self.config.assistant_repeat_threshold:
            return self._decision(
                "assistant_repeat",
                key,
                f"assistant response repeated {self._assistant_streak} times",
                "The assistant response repeated without new progress.",
            )
        return None

    def observe_tool_call(self, tool_name: str, arguments: dict) -> LoopGuardDecision | None:
        """Track repeated tool calls before execution."""
        if not self.enabled:
            return None

        edit_key = self._edit_progress_key(tool_name, arguments)
        if (
            edit_key
            and self._no_progress_edit_counts.get(edit_key, 0)
            >= self.config.no_progress_edit_threshold
        ):
            return self._decision(
                "no_progress_edit",
                edit_key,
                "code-edit repeatedly failed or made no progress",
                "The same file edit has repeatedly failed or made no progress.",
            )

        key = self.tool_call_fingerprint(tool_name, arguments)
        if key == self._last_tool_call_key:
            self._tool_call_streak += 1
        else:
            self._last_tool_call_key = key
            self._tool_call_streak = 1

        if self._tool_call_streak >= self.config.identical_tool_call_threshold:
            return self._decision(
                "identical_tool_call",
                key,
                f"identical tool call repeated {self._tool_call_streak} times",
                "The model repeated the same tool call with identical arguments.",
            )
        return None

    def observe_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
    ) -> LoopGuardDecision | None:
        """Track repeated tool failures and no-progress edits after execution."""
        if not self.enabled:
            return None

        result_class = self._result_class(result)
        if result_class is None:
            self._clear_no_progress_edit(tool_name, arguments)
            return None

        edit_key = self._edit_progress_key(tool_name, arguments)
        if edit_key:
            self._no_progress_edit_counts[edit_key] = (
                self._no_progress_edit_counts.get(edit_key, 0) + 1
            )
            if self._no_progress_edit_counts[edit_key] >= self.config.no_progress_edit_threshold:
                return self._decision(
                    "no_progress_edit",
                    edit_key,
                    "code-edit repeatedly failed or made no progress",
                    "The same file edit has repeatedly failed or made no progress.",
                )

        error_key = self._hash_json(
            {
                "tool": tool_name,
                "call": self.tool_call_fingerprint(tool_name, arguments),
                "result": result_class,
            }
        )
        self._tool_error_counts[error_key] = self._tool_error_counts.get(error_key, 0) + 1
        if self._tool_error_counts[error_key] >= self.config.identical_tool_error_threshold:
            return self._decision(
                "identical_tool_error",
                error_key,
                "same tool call produced the same error repeatedly",
                "The same tool call has repeatedly produced the same error.",
            )

        return None

    def tool_call_fingerprint(self, tool_name: str, arguments: dict) -> str:
        """Return a stable non-reversible fingerprint for a tool call."""
        return self._hash_json({"tool": tool_name, "arguments": arguments})

    def _decision(
        self,
        kind: str,
        key: str,
        reason: str,
        user_message: str,
    ) -> LoopGuardDecision:
        decision_key = f"{kind}:{key}"
        corrective_count = self._corrective_counts.get(decision_key, 0)
        stop = corrective_count >= self.config.corrective_attempts
        if not stop:
            self._corrective_counts[decision_key] = corrective_count + 1

        action = "Stopping this turn." if stop else "This call was not executed."
        message = (
            f"Loop detected ({kind}): {user_message} {action} "
            "Do not retry the same approach. Choose a different strategy, "
            "inspect the latest tool result, or explain the blocker to the user."
        )
        return LoopGuardDecision(kind=kind, reason=reason, message=message, stop=stop)

    def _result_class(self, result: str) -> str | None:
        text = (result or "").strip()
        if not text:
            return "empty_result"

        lower = text.lower()
        first_line = lower.splitlines()[0][:240]
        if lower.startswith("error") or lower.startswith("error executing tool"):
            return f"error:{first_line}"
        if "unknown tool" in lower:
            return "error:unknown_tool"
        if "permission denied" in lower or "denied" in lower or "not allowed" in lower:
            return f"denied:{first_line}"
        if "cancelled" in lower or "canceled" in lower:
            return f"cancelled:{first_line}"
        if "no changes" in lower or "no edits made" in lower:
            return f"no_progress:{first_line}"
        if lower.startswith("failed") or " failed" in lower:
            return f"failed:{first_line}"
        return None

    def _edit_progress_key(self, tool_name: str, arguments: dict) -> str | None:
        if tool_name != "code-edit":
            return None
        filepath = str(arguments.get("filepath") or arguments.get("fileName") or "")
        operation = str(arguments.get("operation") or "")
        if not filepath and not operation:
            return None
        return self._hash_json({"tool": tool_name, "filepath": filepath, "operation": operation})

    def _clear_no_progress_edit(self, tool_name: str, arguments: dict) -> None:
        edit_key = self._edit_progress_key(tool_name, arguments)
        if edit_key:
            self._no_progress_edit_counts.pop(edit_key, None)

    def _hash_json(self, value: Any) -> str:
        raw = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
