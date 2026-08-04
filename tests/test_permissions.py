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


def test_command_allow_rule_does_not_auto_allow_shell_chaining(tmp_path):
    policy = PermissionPolicy(
        tmp_path,
        {"command-exec": {"allow": ["uv run pytest*"]}},
    )

    risky_commands = [
        "uv run pytest tests/; python cleanup.py",
        "uv run pytest tests/ && python cleanup.py",
        "uv run pytest tests/ || python cleanup.py",
        "uv run pytest tests/ | tee out.txt",
        "uv run pytest $(python pick_tests.py)",
        "uv run pytest tests/\npython cleanup.py",
    ]

    for command in risky_commands:
        decision = policy.check_command(command)
        assert decision.action == PermissionAction.ASK
        assert decision.matched_rule == "uv run pytest*"


def test_shell_chaining_cannot_be_saved_as_allow_rule(tmp_path):
    policy = PermissionPolicy(tmp_path)

    try:
        policy.add_command_rule(
            PermissionAction.ALLOW,
            "uv run pytest tests/ && python cleanup.py",
            scope="persistent",
        )
    except ValueError as exc:
        assert "shell control operators" in str(exc)
    else:
        raise AssertionError("Expected ValueError for risky allow rule")


def test_shell_control_detector_backslash_in_single_quote_bypass(tmp_path):
    """Backslash must NOT act as an escape inside single quotes (POSIX).

    Regression for M1: the detector honored ``escaped`` globally, so a command
    like ``git status'\\'; rm -rf ~`` was parsed as keeping the ``;`` inside the
    (already-closed-by-shell) single quotes, yielding has_shell_control=False and
    an ALLOW decision. The real shell closes the quote at the ``\\`` idiom and
    executes the payload after ``;``. The detector must reflect that.
    """
    policy = PermissionPolicy(
        tmp_path,
        {"command-exec": {"allow": ["git status*"]}},
    )

    # The bypass vector: matches "git status*" allow rule, but the shell executes
    # the part after `;` because `'\''` closes+reopens the single quote.
    bypass = "git status'\\'; rm -rf ~"
    decision = policy.check_command(bypass)
    assert decision.action == PermissionAction.ASK, (
        f"backslash-in-single-quote bypass: expected ASK, got {decision.action}/{decision.source}; "
        f"has_shell_control={policy._has_shell_control_operator(bypass)}"
    )

    # Benign single-quote usages must still ALLOW (the idiom is legitimate in shell).
    for benign in [
        "git status",
        "git status -sb",
    ]:
        d = policy.check_command(benign)
        assert d.action == PermissionAction.ALLOW, f"benign {benign!r} should still be ALLOW"


def test_shell_control_detector_plain_control_operators_still_caught(tmp_path):
    """Sanity: ordinary control operators (no quote tricks) are still detected."""
    policy = PermissionPolicy(
        tmp_path,
        {"command-exec": {"allow": ["git status*"]}},
    )
    for cmd in [
        "git status; rm -rf ~",
        "git status && echo x",
        "git status | tee out",
        "git status$(rm -rf ~)",
        "git status`rm -rf ~`",
    ]:
        assert policy._has_shell_control_operator(cmd), f"missed control op in {cmd!r}"


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
