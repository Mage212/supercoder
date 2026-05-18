"""Host-side permission policy for tools and file paths."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class PermissionAction(StrEnum):
    """Permission decision for an attempted action."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission policy check."""

    action: PermissionAction
    reason: str
    matched_rule: str | None = None
    source: str = "default"

    @property
    def allowed(self) -> bool:
        return self.action == PermissionAction.ALLOW

    @property
    def denied(self) -> bool:
        return self.action == PermissionAction.DENY

    @property
    def requires_approval(self) -> bool:
        return self.action == PermissionAction.ASK


BUILTIN_COMMAND_DENY = [
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf ~/*",
    "rm -rf *",
    "sudo *",
    "mkfs*",
    "dd if=*",
    ":(){:|:&};:*",
    "chmod -R 777 /",
    "chmod -R 777 /*",
    "curl * | sh",
    "curl *| sh",
    "curl * | bash",
    "curl *| bash",
    "wget * | sh",
    "wget *| sh",
    "wget * | bash",
    "wget *| bash",
]

BUILTIN_PATH_ALLOW = [
    ".env.example",
    "**/.env.example",
]

BUILTIN_PATH_DENY = [
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "*.pem",
    "**/*.pem",
    "*.key",
    "**/*.key",
    "*.p12",
    "**/*.p12",
    "*.pfx",
    "**/*.pfx",
    "credentials.json",
    "**/credentials.json",
    ".aws/credentials",
    "**/.aws/credentials",
    ".ssh/id_*",
    "**/.ssh/id_*",
]


class PermissionPolicy:
    """Evaluate host-side permissions for commands and paths."""

    def __init__(self, repo_root: Path, config: dict[str, Any] | None = None):
        self.repo_root = repo_root.resolve()
        self.config = config or {}
        self.command_rules = self._section("command-exec")
        self.path_rules = self._section("paths")

    def check_command(self, command: str) -> PermissionDecision:
        """Return allow/ask/deny for a shell command."""
        normalized = " ".join(command.strip().split())
        lowered = normalized.lower()

        for rule in BUILTIN_COMMAND_DENY:
            if self._matches(lowered, rule.lower()):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Command is blocked by built-in safety rule: {rule}",
                    matched_rule=rule,
                    source="builtin",
                )

        for rule in self._rules(self.command_rules, "deny"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Command is blocked by configured deny rule: {rule}",
                    matched_rule=rule,
                    source="config",
                )

        for rule in self._rules(self.command_rules, "allow"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Command allowed by configured rule: {rule}",
                    matched_rule=rule,
                    source="config",
                )

        for rule in self._rules(self.command_rules, "ask"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.ASK,
                    f"Command requires approval by configured rule: {rule}",
                    matched_rule=rule,
                    source="config",
                )

        return PermissionDecision(
            PermissionAction.ASK,
            "Command requires approval by default",
            source="default",
        )

    def check_path(self, path: Path | str, operation: str = "access") -> PermissionDecision:
        """Return allow/deny for a filesystem path."""
        rel = self.relative_path(path)

        for rule in BUILTIN_PATH_ALLOW:
            if self._matches_path(rel, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Path allowed by built-in safe exception: {rule}",
                    matched_rule=rule,
                    source="builtin",
                )

        for rule in BUILTIN_PATH_DENY:
            if self._matches_path(rel, rule):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Path denied for {operation}: sensitive path matches built-in rule {rule}",
                    matched_rule=rule,
                    source="builtin",
                )

        for rule in self._rules(self.path_rules, "deny"):
            if self._matches_path(rel, rule):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Path denied for {operation}: matches configured deny rule {rule}",
                    matched_rule=rule,
                    source="config",
                )

        for rule in self._rules(self.path_rules, "allow"):
            if self._matches_path(rel, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Path allowed by configured rule: {rule}",
                    matched_rule=rule,
                    source="config",
                )

        return PermissionDecision(
            PermissionAction.ALLOW,
            f"Path allowed by default for {operation}",
            source="default",
        )

    def relative_path(self, path: Path | str) -> str:
        """Return repo-relative POSIX path when possible."""
        raw = Path(path)
        resolved = (self.repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            return resolved.relative_to(self.repo_root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def format_denial(self, subject: str, decision: PermissionDecision) -> str:
        """Format a denial message for tool results."""
        return f"Error: Permission denied for {subject}. {decision.reason}"

    def _section(self, name: str) -> dict[str, Any]:
        value = self.config.get(name, {})
        return value if isinstance(value, dict) else {}

    def _rules(self, section: dict[str, Any], key: str) -> list[str]:
        value = section.get(key, [])
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    def _matches(self, value: str, pattern: str) -> bool:
        return fnmatch.fnmatchcase(value, pattern) or value == pattern

    def _matches_path(self, rel_path: str, pattern: str) -> bool:
        pattern = pattern.replace("\\", "/")
        return (
            fnmatch.fnmatchcase(rel_path, pattern)
            or fnmatch.fnmatchcase(Path(rel_path).name, pattern)
            or rel_path == pattern
        )
