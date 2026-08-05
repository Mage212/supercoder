"""SuperCoder CLI entry point."""

import sys
from dataclasses import dataclass

import click
from rich.console import Console

from . import __version__
from .agent.coder_agent import CoderAgent
from .config import Config
from .context import ContextConfig
from .llm.openai_client import OpenAIClient
from .logging import init_logger
from .tools import ALL_TOOLS

console = Console()


@dataclass(frozen=True)
class TrustDecision:
    """Outcome of resolving per-repository trust at startup.

    Two independent axes (BYPASS A regression): ``config_trusted`` gates
    sensitive .supercoder.yaml fields (endpoint/credentials/models), and
    ``perms_trusted`` gates .supercoder/permissions.yaml persistent command
    rules. Both default to False for an untrusted repo and flip to True only on
    an explicit user trust grant (or a prior trust-store entry).
    """

    config_trusted: bool
    perms_trusted: bool
    prompt_needed: bool


def resolve_repo_trust(
    *,
    trusted_in_store: bool,
    has_local_perms: bool,
    local_config_sensitive: bool,
    is_tty: bool,
    user_trusts: bool | None,
) -> TrustDecision:
    """Pure decision function for the startup trust resolution.

    Args:
        trusted_in_store: whether the repo is already in the trust store.
        has_local_perms: whether .supercoder/permissions.yaml exists in the repo.
        local_config_sensitive: whether .supercoder.yaml carries sensitive keys.
        is_tty: whether the session is interactive (can show a prompt).
        user_trusts: the user's answer if a prompt was shown (None before asking).

    Returns:
        TrustDecision with the two independent trust axes and whether a prompt
        is needed to decide. The caller re-asks resolve_repo_trust with
        user_trusts set after prompting.

    Semantics:
        - A repo is fully trusted only via the trust store or an explicit grant.
        - When there is nothing untrusted to honor (no sensitive config, no
          local perms, or already trusted), no prompt is needed.
        - A prompt is shown only in interactive sessions with something at stake.
    """
    if trusted_in_store:
        # Already trusted: honor everything, no prompt.
        return TrustDecision(config_trusted=True, perms_trusted=True, prompt_needed=False)

    # Untrusted repo: both axes start False.
    something_at_stake = local_config_sensitive or has_local_perms
    if not something_at_stake:
        # Nothing planted that requires trust — stay safe, no prompt.
        return TrustDecision(config_trusted=False, perms_trusted=False, prompt_needed=False)

    if not is_tty:
        # Non-interactive: cannot prompt, stay safe.
        return TrustDecision(config_trusted=False, perms_trusted=False, prompt_needed=False)

    if user_trusts is None:
        # Interactive, something at stake, not yet answered: prompt needed.
        return TrustDecision(config_trusted=False, perms_trusted=False, prompt_needed=True)

    # User answered.
    if user_trusts:
        return TrustDecision(config_trusted=True, perms_trusted=True, prompt_needed=False)
    return TrustDecision(config_trusted=False, perms_trusted=False, prompt_needed=False)


def _prompt_repo_trust(repo_path, *, sensitive_config: bool, has_local_perms: bool) -> bool:
    """Ask the user whether to trust local config from ``repo_path``.

    Returns True if the user trusts the repo (sensitive local config / persistent
    command rules will be honored on this and future runs). Returns False if the
    user declines or dismisses the prompt — safe tuning fields are still applied,
    but endpoint/credential/permission overrides are dropped.
    """
    import questionary

    console.print(
        f"\n[yellow]⚠ Local configuration detected in:[/] [cyan]{repo_path}[/]\n"
        "This repository contains files that can redirect credentials or override\n"
        "command permissions. Review them before trusting.\n"
    )
    if sensitive_config:
        console.print("  • [cyan].supercoder.yaml[/] overrides endpoint / model / permissions")
    if has_local_perms:
        console.print("  • [cyan].supercoder/permissions.yaml[/] adds persistent command rules")
    console.print()

    try:
        choice = questionary.select(
            "Trust this repository's local configuration?",
            choices=[
                "Trust (honor local config now and in future runs)",
                "Do not trust (ignore sensitive overrides this session)",
                "Show local config files",
            ],
            use_arrow_keys=True,
        ).ask()
    except (KeyboardInterrupt, EOFError):
        return False

    if choice is None:
        return False
    if choice.startswith("Show"):
        _show_local_config(repo_path)
        # Re-ask once after showing.
        try:
            choice = questionary.select(
                "Trust this repository's local configuration?",
                choices=[
                    "Trust (honor local config now and in future runs)",
                    "Do not trust (ignore sensitive overrides this session)",
                ],
                use_arrow_keys=True,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            return False
        if choice is None:
            return False
    return choice.startswith("Trust")


def _show_local_config(repo_path) -> None:
    """Print the contents of local config files for review."""
    from pathlib import Path

    for rel in (".supercoder.yaml", ".supercoder/permissions.yaml"):
        p = Path(repo_path) / rel
        if p.exists():
            console.print(f"\n[cyan]── {rel} ──[/]")
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                console.print(f"[red]  (could not read: {exc})[/]")
                continue
            for line in content.splitlines():
                console.print(f"  {line}")


def resolve_log_enabled(debug: bool, no_log: bool) -> bool:
    """Decide whether session logging to ~/.supercoder/logs/ is on.

    Logging is on by default (so the recall tool can retrieve past events that
    were compacted out of context). ``--debug`` forces it on and overrides
    ``--no-log``; ``--no-log`` opts out when not debugging. Extracted as a pure
    function so the default-on inversion is unit-testable without booting the
    REPL.
    """
    return debug or not no_log


@click.command()
@click.option("--model", "-m", default="", help="Model to use for the agent")
@click.option("--endpoint", "-e", default="", help="LLM API endpoint (base URL)")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.option("--temperature", "-t", type=float, default=None, help="Temperature for LLM")
@click.option(
    "--max-context", "-c", type=int, default=None, help="Max context tokens (default: from config)"
)
@click.option("--repo-map/--no-repo-map", default=True, help="Enable/disable RepoMap")
@click.option(
    "--stream/--no-stream",
    default=False,
    help="Enable deprecated streaming mode (default: off, uses native tool calls)",
)
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    help="Skip the animated startup banner",
)
@click.option(
    "--no-log",
    is_flag=True,
    default=False,
    help="Disable session logging to ~/.supercoder/logs/ (enabled by default)",
)
@click.version_option(version=__version__)
def main(
    model: str,
    endpoint: str,
    debug: bool,
    temperature: float | None,
    max_context: int,
    repo_map: bool,
    stream: bool,
    no_banner: bool,
    no_log: bool,
):
    """SuperCoder - AI Coding Assistant for the Terminal."""

    # C1/C2: load config defensively. A local .supercoder.yaml in the cwd is
    # untrusted (a cloned malicious repo can plant one to redirect credentials
    # or override permissions). Load with sensitive local fields filtered out,
    # and re-load with them honored only once the user trusts the repo.
    from pathlib import Path

    from .permissions import PermissionPolicy
    from .trust import RepoTrustStore

    repo_path = Path.cwd()
    trust_store = RepoTrustStore()
    # A persistent permissions file in the repo is also untrusted; detect it
    # without loading (loading would honor the rules).
    has_local_perms = PermissionPolicy(
        repo_path, allow_persistent=False
    ).has_persistent_rules_file()

    config = Config.load(allow_sensitive_local=False)

    # Resolve per-repository trust (BYPASS A): config_trusted and perms_trusted
    # are INDEPENDENT axes. Previously a single ``repo_trusted`` defaulted to
    # True when no permissions.yaml existed, which silently honored a planted
    # .supercoder.yaml endpoint redirect with no prompt. See resolve_repo_trust.
    decision = resolve_repo_trust(
        trusted_in_store=trust_store.is_trusted(repo_path),
        has_local_perms=has_local_perms,
        local_config_sensitive=config.local_config_sensitive,
        is_tty=sys.stdin.isatty(),
        user_trusts=None,
    )
    if decision.prompt_needed:
        granted = _prompt_repo_trust(
            repo_path,
            sensitive_config=config.local_config_sensitive,
            has_local_perms=has_local_perms,
        )
        decision = resolve_repo_trust(
            trusted_in_store=trust_store.is_trusted(repo_path),
            has_local_perms=has_local_perms,
            local_config_sensitive=config.local_config_sensitive,
            is_tty=sys.stdin.isatty(),
            user_trusts=granted,
        )
        if granted:
            trust_store.trust(repo_path)
    elif (
        (config.local_config_sensitive or has_local_perms)
        and not sys.stdin.isatty()
        and not decision.config_trusted
    ):
        # Non-interactive with untrusted local files: inform the user.
        console.print(
            "[yellow]Warning:[/] local config overrides ignored (untrusted repo, "
            "non-interactive session). Trust it via an interactive run.\n"
        )

    # Honor sensitive local config only when the config axis is trusted.
    if decision.config_trusted and config.local_config_sensitive:
        config = Config.load(allow_sensitive_local=True)
    # The permissions axis is threaded into the agent (allow_persistent_permissions).
    perms_trusted = decision.perms_trusted
    if model:  # noqa: SIM102
        # If the model name matches an existing profile, switch to it
        if not config.switch_to_model(model):
            # Otherwise just override the model name
            config.model = model
    if endpoint:
        config.base_url = endpoint
    if temperature is not None:
        config.temperature = temperature
    config.debug = debug
    # Only override context if explicitly provided via CLI
    if max_context is not None:
        config.max_context_tokens = max_context

    # Validate config — launch interactive wizard if API key is missing
    errors = config.validate()
    if errors:
        # Check if the only error is a missing API key (wizard can fix that)
        api_key_errors = [e for e in errors if "API key" in e]
        other_errors = [e for e in errors if "API key" not in e]

        # Print non-fixable errors immediately
        for error in other_errors:
            console.print(f"[red]Error: {error}[/]")
        if other_errors:
            return

        if api_key_errors:
            from .config import is_first_run

            if is_first_run():
                # Genuine first run: launch interactive wizard
                from .setup_wizard import run_setup_wizard

                ok = run_setup_wizard()
                if not ok:
                    return
                # Reload config after successful setup
                config = Config.load()
                if model and not config.switch_to_model(model):
                    config.model = model
                if endpoint:
                    config.base_url = endpoint
                if temperature is not None:
                    config.temperature = temperature
                config.debug = debug
                if max_context is not None:
                    config.max_context_tokens = max_context

                # Re-validate — abort if still broken
                remaining = config.validate()
                if remaining:
                    for error in remaining:
                        console.print(f"[red]Error: {error}[/]")
                    return
            else:
                # Existing config with profiles: show diagnostic, do NOT launch wizard
                console.print("\n[red]Error: API key not configured for the active profile.[/]\n")
                available = config.get_available_models()
                console.print(f"  Active profile: [cyan]{config.current_profile_name}[/]")
                if available:
                    console.print(f"  Available profiles: [cyan]{', '.join(available)}[/]")
                console.print(
                    "\n[yellow]To fix this, either:[/]\n"
                    "  1. Edit your config:   [dim]nano ~/.supercoder/config.yaml[/]\n"
                    "  2. Set the API key:    [dim]export SUPERCODER_API_KEY=<your-key>[/]\n"
                    "  3. Switch to a profile: [dim]supercoder -m <profile-name>[/]"
                )
                return

    # Initialize logger
    # Session logging is on by default so the recall tool can retrieve past
    # events (tool calls, results, commands, errors) that have been compacted
    # out of the context window. --no-log disables it; --debug forces verbose
    # logging regardless.
    log_enabled = resolve_log_enabled(config.debug, no_log)
    logger = init_logger(config.model, enabled=log_enabled)

    # Banner is displayed by the REPL (see repl.py run() method)

    # Context configuration
    context_config = ContextConfig(
        max_tokens=config.max_context_tokens,
        reserved_for_response=config.reserved_for_response,
        auto_compact=config.auto_compact,
        auto_compact_threshold=config.auto_compact_threshold,
        protected_recent_steps=config.protected_recent_steps,
        compression_threshold=config.compression_threshold,
        compression_strategy="smart",
    )

    # Initialize LLM and agent
    try:
        llm = OpenAIClient(config)

        # Get tool_calling_type from current model profile
        profile = config.get_model_profile(config.current_profile_name)
        tool_calling_type = profile.tool_calling_type if profile else "supercoder"
        lean = profile.lean if profile else False

        # Resolve streaming mode: CLI flag > model profile > global config
        use_streaming = stream  # CLI flag takes precedence
        if not stream and profile and profile.streaming:
            use_streaming = profile.streaming
        if not use_streaming:
            use_streaming = config.streaming

        agent = CoderAgent(
            llm,
            tools=ALL_TOOLS,
            context_config=context_config,
            use_repo_map=repo_map,
            repo_root=".",
            tool_calling_type=tool_calling_type,
            streaming=use_streaming,
            lean=lean,
            permissions=config.permissions,
            loop_detection=config.loop_detection,
            allow_persistent_permissions=perms_trusted,
            allow_project_rules=decision.config_trusted,
            allow_session_load=decision.config_trusted,
            allow_offload_read=decision.config_trusted,
        )
        agent.set_debug(debug)
    except Exception as e:
        logger.log_error(e)
        console.print(f"[red]Failed to initialize: {e}[/]")
        return

    # Start REPL
    from .repl import SuperCoderREPL

    repl = SuperCoderREPL(agent, no_banner=no_banner)
    repl.run()


if __name__ == "__main__":
    main()
