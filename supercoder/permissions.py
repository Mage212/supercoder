"""Host-side permission policy for tools and file paths."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from .utils.atomic_writer import AtomicFileWriter


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


@dataclass(frozen=True)
class PermissionRule:
    """A user-manageable command permission rule."""

    id: str
    scope: str
    action: PermissionAction
    pattern: str


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
        self.persistent_path = self.repo_root / ".supercoder" / "permissions.yaml"
        self.persistent_config = self._load_persistent_config()
        self.session_command_rules: dict[str, list[str]] = {"allow": [], "deny": []}
        self.command_rules = self._section(self.config, "command-exec")
        self.persistent_command_rules = self._section(self.persistent_config, "command-exec")
        self.path_rules = self._section(self.config, "paths")

    def check_command(self, command: str) -> PermissionDecision:
        """Return allow/ask/deny for a shell command."""
        normalized = self.normalize_command(command)
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

        for rule in self._rules(self.persistent_command_rules, "deny"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Command is blocked by persistent deny rule: {rule}",
                    matched_rule=rule,
                    source="persistent",
                )

        for rule in self._rules(self.session_command_rules, "deny"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.DENY,
                    f"Command is blocked by session deny rule: {rule}",
                    matched_rule=rule,
                    source="session",
                )

        for rule in self._rules(self.command_rules, "allow"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Command allowed by configured rule: {rule}",
                    matched_rule=rule,
                    source="config",
                )

        for rule in self._rules(self.persistent_command_rules, "allow"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Command allowed by persistent rule: {rule}",
                    matched_rule=rule,
                    source="persistent",
                )

        for rule in self._rules(self.session_command_rules, "allow"):
            if self._matches(normalized, rule):
                return PermissionDecision(
                    PermissionAction.ALLOW,
                    f"Command allowed by session rule: {rule}",
                    matched_rule=rule,
                    source="session",
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

    def add_command_rule(
        self,
        action: PermissionAction,
        command: str,
        *,
        scope: str = "session",
    ) -> PermissionRule:
        """Add a session or persistent exact-command rule."""
        if action not in {PermissionAction.ALLOW, PermissionAction.DENY}:
            raise ValueError("Only allow and deny command rules can be persisted")
        if scope not in {"session", "persistent"}:
            raise ValueError("Permission rule scope must be 'session' or 'persistent'")

        pattern = self.normalize_command(command)
        target = self.session_command_rules if scope == "session" else self.persistent_command_rules
        rules = target.setdefault(action.value, [])
        if pattern not in rules:
            rules.append(pattern)
            if scope == "persistent":
                self._save_persistent_config()

        return self._find_command_rule(scope, action, pattern)

    def list_command_rules(self, scope: str | None = None) -> list[PermissionRule]:
        """Return user-manageable command rules."""
        if scope not in {None, "session", "persistent"}:
            raise ValueError("Permission rule scope must be 'session', 'persistent', or None")

        rules: list[PermissionRule] = []
        if scope in {None, "persistent"}:
            rules.extend(self._list_rules_for_scope("persistent", self.persistent_command_rules))
        if scope in {None, "session"}:
            rules.extend(self._list_rules_for_scope("session", self.session_command_rules))
        return rules

    def remove_persistent_command_rule(self, identifier: str) -> PermissionRule | None:
        """Remove one persistent command rule by displayed id or exact pattern."""
        target_id = identifier.strip()
        if target_id.isdigit():
            target_id = f"p{target_id}"

        target = None
        for rule in self.list_command_rules("persistent"):
            if rule.id == target_id or rule.pattern == identifier:
                target = rule
                break
        if target is None:
            return None

        rules = self.persistent_command_rules.setdefault(target.action.value, [])
        self.persistent_command_rules[target.action.value] = [
            rule for rule in rules if rule != target.pattern
        ]
        self._save_persistent_config()
        return target

    def clear_persistent_command_rules(self) -> int:
        """Clear all persistent command permission rules."""
        count = len(self.list_command_rules("persistent"))
        self.persistent_command_rules["allow"] = []
        self.persistent_command_rules["deny"] = []
        if count:
            self._save_persistent_config()
        return count

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

    def normalize_command(self, command: str) -> str:
        """Normalize shell command whitespace before rule matching/storage."""
        return " ".join(command.strip().split())

    def _load_persistent_config(self) -> dict[str, Any]:
        if not self.persistent_path.exists():
            return {}
        try:
            raw = self.persistent_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid persistent permissions file: {self.persistent_path}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"Persistent permissions file must contain a mapping: {self.persistent_path}"
            )
        return data

    def _save_persistent_config(self) -> None:
        data = dict(self.persistent_config)
        data["command-exec"] = {
            "allow": self._rules(self.persistent_command_rules, "allow"),
            "deny": self._rules(self.persistent_command_rules, "deny"),
        }
        content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        AtomicFileWriter.write(self.persistent_path, content)
        self.persistent_config = data

    def _section(self, config: dict[str, Any], name: str) -> dict[str, Any]:
        value = config.get(name, {})
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

    def _list_rules_for_scope(
        self, scope: str, rules_by_action: dict[str, Any]
    ) -> list[PermissionRule]:
        prefix = "p" if scope == "persistent" else "s"
        rules: list[PermissionRule] = []
        index = 1
        for action in (PermissionAction.ALLOW, PermissionAction.DENY):
            for pattern in self._rules(rules_by_action, action.value):
                rules.append(
                    PermissionRule(
                        id=f"{prefix}{index}",
                        scope=scope,
                        action=action,
                        pattern=pattern,
                    )
                )
                index += 1
        return rules

    def _find_command_rule(
        self, scope: str, action: PermissionAction, pattern: str
    ) -> PermissionRule:
        for rule in self.list_command_rules(scope):
            if rule.action == action and rule.pattern == pattern:
                return rule
        raise RuntimeError("Added permission rule was not found")
