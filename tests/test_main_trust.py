"""Tests for the repo-trust resolution logic in main.py (BYPASS A regression).

The trust decision used to be a single ``repo_trusted`` boolean computed as
``trust_store.is_trusted(repo) or not has_local_perms``. When a malicious repo
planted ONLY a .supercoder.yaml (no permissions.yaml), ``has_local_perms`` was
False, so ``repo_trusted`` became True by default and the sensitive config
(endpoint/credentials/models) was reloaded with allow_sensitive_local=True —
honoring the attacker's endpoint redirect with NO prompt, in both interactive
and non-interactive modes.

The fix separates the two independent trust axes:
- config_trusted  — gates sensitive .supercoder.yaml fields
- perms_trusted   — gates .supercoder/permissions.yaml persistent rules
Both default to ``trust_store.is_trusted(repo)`` and are only flipped to True by
an explicit user trust grant.
"""

from supercoder.main import resolve_repo_trust


class TestResolveRepoTrust:
    """Decision table for the trust resolution used at startup."""

    def test_untrusted_config_only_repo_does_not_honor_sensitive(self):
        # The BYPASS A scenario: config-only repo, never trusted.
        # Sensitive fields must stay filtered (config_trusted=False) and perms
        # must stay off, regardless of TTY.
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=False,
            local_config_sensitive=True,
            is_tty=True,
            user_trusts=False,
        )
        assert decision.config_trusted is False, (
            "config-only untrusted repo must NOT honor sensitive fields"
        )
        assert decision.perms_trusted is False

    def test_untrusted_config_only_repo_non_interactive_stays_safe(self):
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=False,
            local_config_sensitive=True,
            is_tty=False,
            user_trusts=None,
        )
        assert decision.config_trusted is False
        assert decision.perms_trusted is False
        assert decision.prompt_needed is False  # non-interactive: no prompt

    def test_untrusted_config_only_repo_interactive_needs_prompt(self):
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=False,
            local_config_sensitive=True,
            is_tty=True,
            user_trusts=None,  # not yet answered
        )
        assert decision.prompt_needed is True
        assert decision.config_trusted is False  # until user answers

    def test_user_trust_grants_both_axes(self):
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=True,
            local_config_sensitive=True,
            is_tty=True,
            user_trusts=True,
        )
        assert decision.config_trusted is True
        assert decision.perms_trusted is True

    def test_already_trusted_repo_honors_everything_no_prompt(self):
        decision = resolve_repo_trust(
            trusted_in_store=True,
            has_local_perms=True,
            local_config_sensitive=True,
            is_tty=True,
            user_trusts=None,
        )
        assert decision.config_trusted is True
        assert decision.perms_trusted is True
        assert decision.prompt_needed is False

    def test_perms_only_repo_untrusted_keeps_perms_off(self):
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=True,
            local_config_sensitive=False,
            is_tty=True,
            user_trusts=False,
        )
        assert decision.perms_trusted is False
        assert decision.config_trusted is False

    def test_no_local_untrusted_files_no_prompt(self):
        # Clean repo, nothing planted — no prompt, no trust granted.
        decision = resolve_repo_trust(
            trusted_in_store=False,
            has_local_perms=False,
            local_config_sensitive=False,
            is_tty=True,
            user_trusts=None,
        )
        assert decision.prompt_needed is False
        # perms_trusted is irrelevant when has_local_perms is False (no file to
        # load), but it must not be spuriously True.
        assert decision.perms_trusted is False
