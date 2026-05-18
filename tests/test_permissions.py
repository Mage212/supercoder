"""Tests for host-side permission policy."""

from supercoder.permissions import PermissionAction, PermissionPolicy


def test_command_builtin_dangerous_denied(tmp_path):
    policy = PermissionPolicy(tmp_path)

    decision = policy.check_command("sudo rm -rf /tmp/example")

    assert decision.action == PermissionAction.DENY
    assert decision.source == "builtin"


def test_command_config_allow_and_default_ask(tmp_path):
    policy = PermissionPolicy(
        tmp_path,
        {
            "command-exec": {
                "allow": ["uv run pytest*"],
                "ask": ["git push*"],
            }
        },
    )

    assert policy.check_command("uv run pytest tests/").action == PermissionAction.ALLOW
    assert policy.check_command("git push origin main").action == PermissionAction.ASK
    assert policy.check_command("python script.py").action == PermissionAction.ASK


def test_command_config_deny_precedes_allow(tmp_path):
    policy = PermissionPolicy(
        tmp_path,
        {
            "command-exec": {
                "allow": ["git *"],
                "deny": ["git push*"],
            }
        },
    )

    decision = policy.check_command("git push origin main")

    assert decision.action == PermissionAction.DENY
    assert decision.source == "config"


def test_sensitive_paths_denied_by_default(tmp_path):
    policy = PermissionPolicy(tmp_path)

    assert policy.check_path(tmp_path / ".env", "read").action == PermissionAction.DENY
    assert policy.check_path(tmp_path / "nested" / "secret.pem", "read").action == (
        PermissionAction.DENY
    )
    assert policy.check_path(tmp_path / "credentials.json", "read").action == (
        PermissionAction.DENY
    )


def test_env_example_allowed_by_default(tmp_path):
    policy = PermissionPolicy(tmp_path)

    decision = policy.check_path(tmp_path / ".env.example", "read")

    assert decision.action == PermissionAction.ALLOW
    assert decision.source == "builtin"


def test_config_path_deny(tmp_path):
    policy = PermissionPolicy(tmp_path, {"paths": {"deny": ["private/*"]}})

    decision = policy.check_path(tmp_path / "private" / "notes.txt", "read")

    assert decision.action == PermissionAction.DENY
    assert decision.source == "config"
