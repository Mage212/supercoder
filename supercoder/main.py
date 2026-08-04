"""SuperCoder CLI entry point."""

import sys

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
    # repo_trusted gates whether the agent may honor .supercoder/permissions.yaml.
    repo_trusted = trust_store.is_trusted(repo_path) or not has_local_perms

    if config.local_config_sensitive or has_local_perms:
        if repo_trusted and config.local_config_sensitive:
            # Already trusted on a previous run: honor sensitive local config.
            config = Config.load(allow_sensitive_local=True)
        elif sys.stdin.isatty():
            # Interactive: ask the user whether to trust this repository.
            if _prompt_repo_trust(
                repo_path,
                sensitive_config=config.local_config_sensitive,
                has_local_perms=has_local_perms,
            ):
                trust_store.trust(repo_path)
                repo_trusted = True
                config = Config.load(allow_sensitive_local=True)
            else:
                repo_trusted = not has_local_perms
        else:
            # Non-interactive (e.g. piped input): stay safe, keep filtering.
            repo_trusted = not has_local_perms
            console.print(
                "[yellow]Warning:[/] local config overrides ignored (untrusted repo, "
                "non-interactive session). Trust it via an interactive run.\n"
            )
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
    logger = init_logger(config.model, enabled=config.debug)

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
            allow_persistent_permissions=repo_trusted,
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
