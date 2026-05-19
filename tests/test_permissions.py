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


def test_persistent_allow_is_loaded_from_project_file(tmp_path):
    policy = PermissionPolicy(tmp_path)

    rule = policy.add_command_rule(
        PermissionAction.ALLOW,
        "  printf   allowed  ",
        scope="persistent",
    )
    reloaded = PermissionPolicy(tmp_path)
    decision = reloaded.check_command("printf allowed")

    assert rule.pattern == "printf allowed"
    assert decision.action == PermissionAction.ALLOW
    assert decision.source == "persistent"
    assert (tmp_path / ".supercoder" / "permissions.yaml").exists()


def test_persistent_allow_can_resolve_config_ask(tmp_path):
    policy = PermissionPolicy(tmp_path, {"command-exec": {"ask": ["git push*"]}})

    policy.add_command_rule(PermissionAction.ALLOW, "git push origin main", scope="persistent")

    decision = policy.check_command("git push origin main")
    assert decision.action == PermissionAction.ALLOW
    assert decision.source == "persistent"


def test_session_allow_is_not_persisted(tmp_path):
    policy = PermissionPolicy(tmp_path)

    policy.add_command_rule(PermissionAction.ALLOW, "printf session", scope="session")

    assert policy.check_command("printf session").source == "session"
    assert PermissionPolicy(tmp_path).check_command("printf session").action == PermissionAction.ASK


def test_persistent_deny_blocks_command(tmp_path):
    policy = PermissionPolicy(tmp_path)

    policy.add_command_rule(PermissionAction.DENY, "printf blocked", scope="persistent")

    decision = PermissionPolicy(tmp_path).check_command("printf blocked")
    assert decision.action == PermissionAction.DENY
    assert decision.source == "persistent"


def test_builtin_deny_precedes_persistent_allow(tmp_path):
    policy = PermissionPolicy(tmp_path)

    policy.add_command_rule(PermissionAction.ALLOW, "sudo echo nope", scope="persistent")

    decision = policy.check_command("sudo echo nope")
    assert decision.action == PermissionAction.DENY
    assert decision.source == "builtin"


def test_persistent_command_rules_do_not_duplicate(tmp_path):
    policy = PermissionPolicy(tmp_path)

    policy.add_command_rule(PermissionAction.ALLOW, "printf once", scope="persistent")
    policy.add_command_rule(PermissionAction.ALLOW, "printf once", scope="persistent")

    assert len(policy.list_command_rules("persistent")) == 1


def test_remove_and_clear_persistent_command_rules(tmp_path):
    policy = PermissionPolicy(tmp_path)
    first = policy.add_command_rule(PermissionAction.ALLOW, "printf first", scope="persistent")
    policy.add_command_rule(PermissionAction.DENY, "printf second", scope="persistent")

    removed = policy.remove_persistent_command_rule(first.id)

    assert removed == first
    assert policy.check_command("printf first").action == PermissionAction.ASK
    assert policy.clear_persistent_command_rules() == 1
    assert policy.list_command_rules("persistent") == []


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
