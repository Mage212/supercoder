"""SuperCoder CLI entry point."""

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
    "--stream/--no-stream", default=False,
    help="Enable deprecated streaming mode (default: off, uses native tool calls)",
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
):
    """SuperCoder - AI Coding Assistant for the Terminal."""

    # Load config
    config = Config.load()
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
                console.print(
                    "\n[red]Error: API key not configured for the active profile.[/]\n"
                )
                available = config.get_available_models()
                console.print(f"  Active profile: [cyan]{config.current_profile_name}[/]")
                if available:
                    console.print(
                        f"  Available profiles: [cyan]{', '.join(available)}[/]"
                    )
                console.print(
                    "\n[yellow]To fix this, either:[/]\n"
                    "  1. Edit your config:   [dim]nano ~/.supercoder/config.yaml[/]\n"
                    "  2. Set the API key:    [dim]export SUPERCODER_API_KEY=<your-key>[/]\n"
                    "  3. Switch to a profile: [dim]supercoder -m <profile-name>[/]"
                )
                return


    # Initialize logger
    logger = init_logger(config.model)

    # Banner is displayed by the REPL (see repl.py run() method)

    # Context configuration
    context_config = ContextConfig(
        max_tokens=config.max_context_tokens,
        reserved_for_response=4096,
        compression_threshold=0.7,
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
        )
        agent.set_debug(debug)
    except Exception as e:
        logger.log_error(e)
        console.print(f"[red]Failed to initialize: {e}[/]")
        return

    # Start REPL
    from .repl import SuperCoderREPL

    repl = SuperCoderREPL(agent)
    repl.run()


if __name__ == "__main__":
    main()
