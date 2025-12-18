"""SuperCoder CLI entry point."""

import click
from rich.console import Console
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from . import __version__
from .config import Config
from .llm.openai_client import OpenAIClient
from .agent.coder_agent import CoderAgent
from .context import ContextConfig
from .tools import ALL_TOOLS
from .logging import init_logger

console = Console()


@click.command()
@click.option("--model", "-m", default="", help="Model to use for the agent")
@click.option("--endpoint", "-e", default="", help="LLM API endpoint (base URL)")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode")
@click.option("--temperature", "-t", type=float, default=0.2, help="Temperature for LLM")
@click.option("--max-context", "-c", type=int, default=32000, help="Max context tokens")
@click.option("--repo-map/--no-repo-map", default=True, help="Enable/disable RepoMap")
@click.version_option(version=__version__)
def main(model: str, endpoint: str, debug: bool, temperature: float, max_context: int, repo_map: bool):
    """SuperCoder - AI Coding Assistant for the Terminal."""
    
    # Load config
    config = Config.load()
    if model:
        config.model = model
    if endpoint:
        config.base_url = endpoint
    if temperature != 0.2:
        config.temperature = temperature
    config.debug = debug
    config.max_context_tokens = max_context
    
    # Validate config
    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[red]Error: {error}[/]")
        return
    
    # Initialize logger
    logger = init_logger(config.model)
    
    # Print banner
    console.print(f"[bold green]SuperCoder v{__version__}[/]")
    console.print(f"[dim]Model: {config.model}[/]")
    console.print(f"[dim]Endpoint: {config.base_url}[/]")
    console.print(f"[dim]Context: {max_context:,} tokens[/]")
    console.print(f"[dim]RepoMap: {'Enabled' if repo_map else 'Disabled'}[/]")
    console.print(f"[dim]Tools: {len(ALL_TOOLS)} available[/]")
    console.print(f"[dim]Logs: {logger.log_path}[/]")
    console.print()
    
    # Context configuration
    context_config = ContextConfig(
        max_tokens=config.max_context_tokens,
        reserved_for_response=4096,
        compression_threshold=0.7,
        compression_strategy="smart"
    )
    
    # Initialize LLM and agent
    try:
        llm = OpenAIClient(config)
        agent = CoderAgent(
            llm, 
            tools=ALL_TOOLS, 
            context_config=context_config,
            use_repo_map=repo_map,
            repo_root="."  # Default to current directory
        )
        agent.set_debug(debug)
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/]")
        return
    
    # Start REPL
    from .repl import SuperCoderREPL
    repl = SuperCoderREPL(agent)
    repl.run()


if __name__ == "__main__":
    main()
