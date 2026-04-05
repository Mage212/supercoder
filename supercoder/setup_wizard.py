"""Interactive first-run setup wizard for configuring the LLM provider."""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

console = Console()

# Well-known provider presets
PROVIDERS = [
    {
        "name": "OpenAI",
        "endpoint": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "default_model": "gpt-4o-mini",
        "tool_calling_type": "supercoder",
        "key_hint": "sk-...",
        "key_url": "https://platform.openai.com/api-keys",
    },
    {
        "name": "OpenRouter",
        "endpoint": "https://openrouter.ai/api/v1",
        "models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.0-flash-001",
            "openai/gpt-oss-20b:free",
        ],
        "default_model": "openai/gpt-4o-mini",
        "tool_calling_type": "supercoder",
        "key_hint": "sk-or-v1-...",
        "key_url": "https://openrouter.ai/settings/keys",
    },
    {
        "name": "Anthropic (via OpenRouter)",
        "endpoint": "https://openrouter.ai/api/v1",
        "models": [
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-3-5",
        ],
        "default_model": "anthropic/claude-sonnet-4-5",
        "tool_calling_type": "supercoder",
        "key_hint": "sk-or-v1-...",
        "key_url": "https://openrouter.ai/settings/keys",
    },
    {
        "name": "Ollama (local)",
        "endpoint": "http://localhost:11434/v1",
        "models": ["llama3.2", "qwen2.5-coder:7b", "deepseek-coder-v2", "codellama"],
        "default_model": "llama3.2",
        "tool_calling_type": "supercoder",
        "key_hint": "ollama",
        "key_url": None,
    },
    {
        "name": "Custom endpoint",
        "endpoint": "",
        "models": [],
        "default_model": "",
        "tool_calling_type": "supercoder",
        "key_hint": "...",
        "key_url": None,
    },
]


def _print_header() -> None:
    header = Text()
    header.append("🚀 SuperCoder Setup\n", style="bold green")
    header.append("No API key configured. Let's set up your first provider!\n", style="dim")
    header.append(
        "Config will be saved to ", style="dim"
    )
    header.append(str(Path.home() / ".supercoder" / "config.yaml"), style="cyan")
    console.print(Panel(header, border_style="green", box=box.ROUNDED))


def _pick_provider() -> dict:
    """Ask user to select a provider from the list."""
    console.print("\n[bold]Choose a provider:[/]\n")
    for i, p in enumerate(PROVIDERS, 1):
        tag = f"  [cyan]{i}[/]. [bold]{p['name']}[/]"
        if p["endpoint"]:
            tag += f"  [dim]({p['endpoint']})[/]"
        console.print(tag)

    while True:
        raw = Prompt.ask(
            "\n[bold green]Provider[/]",
            default="1",
        )
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(PROVIDERS):
                return PROVIDERS[idx]
        except ValueError:
            pass
        console.print("[red]Please enter a number between 1 and {len(PROVIDERS)}[/]")


def _pick_model(provider: dict) -> str:
    """Ask user to choose or type a model name."""
    models = provider["models"]
    default = provider["default_model"]

    if models:
        console.print("\n[bold]Common models for this provider:[/]\n")
        for i, m in enumerate(models, 1):
            marker = " [dim](default)[/]" if m == default else ""
            console.print(f"  [cyan]{i}[/]. {m}{marker}")
        console.print(f"  [cyan]{len(models)+1}[/]. Enter custom model name")

        while True:
            raw = Prompt.ask("\n[bold green]Model[/]", default="1")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(models):
                    return models[idx]
                if idx == len(models):
                    break  # fall through to custom
            except ValueError:
                # Maybe they typed a model name directly
                if raw.strip():
                    return raw.strip()
            console.print("[red]Invalid choice[/]")

    return Prompt.ask("[bold green]Model name[/]", default=default or "gpt-4o-mini")


def _get_api_key(provider: dict) -> str:
    """Prompt user for the API key."""
    is_local = "localhost" in provider.get("endpoint", "") or "127.0.0.1" in provider.get(
        "endpoint", ""
    )

    if is_local:
        console.print(
            "\n[dim]Local endpoint detected — no API key needed. "
            "Using 'ollama' as placeholder.[/]"
        )
        return "ollama"

    key_url = provider.get("key_url")
    if key_url:
        console.print(f"\n[dim]Get your API key at: [link={key_url}]{key_url}[/link][/]")

    hint = provider.get("key_hint", "...")
    while True:
        key = Prompt.ask(
            f"[bold green]API Key[/] [dim]({hint})[/]",
            password=True,
        )
        if key.strip():
            return key.strip()
        console.print("[red]API key cannot be empty[/]")


def _get_endpoint(provider: dict) -> str:
    """Prompt for endpoint, pre-filled from provider preset."""
    default = provider.get("endpoint", "https://api.openai.com/v1")
    if default:
        return Prompt.ask("[bold green]API Endpoint[/]", default=default)
    return Prompt.ask("[bold green]API Endpoint[/]")


def _write_config(
    api_key: str,
    endpoint: str,
    model: str,
    tool_calling_type: str,
    profile_name: str = "default",
) -> Path:
    """Write the config file with the given profile."""
    from .config import CONFIG_DIR, CONFIG_FILE

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Build the tool_calling_type comment only when non-default
    tct_comment = ""
    if tool_calling_type != "supercoder":
        tct_comment = f"\n    tool_calling_type: \"{tool_calling_type}\""

    content = f"""# SuperCoder Configuration
# Documentation: https://github.com/your-repo/supercoder

# Default model profile to use on startup
default_model: "{profile_name}"

# Model profiles
models:
  {profile_name}:
    api_key: "{api_key}"
    endpoint: "{endpoint}"
    model: "{model}"{tct_comment}
    # max_context_tokens: 128000  # Uncomment and adjust for your model

# Shared settings
temperature: 0.2
top_p: 0.1
max_context_tokens: 32000
reserved_for_response: 4096
request_timeout: 60.0
debug: false
"""
    CONFIG_FILE.write_text(content, encoding="utf-8")
    return CONFIG_FILE


def run_setup_wizard() -> bool:
    """Run the interactive setup wizard.

    Returns:
        True if setup completed successfully and config was written.
        False if user cancelled.
    """
    _print_header()

    try:
        provider = _pick_provider()

        console.print(f"\n[bold]Setting up: [green]{provider['name']}[/][/]")

        # Custom provider needs endpoint input first
        endpoint = _get_endpoint(provider)
        model = _pick_model(provider)
        api_key = _get_api_key(provider)

        tool_calling_type = provider.get("tool_calling_type", "supercoder")

        # Summary
        console.print("\n[bold]Configuration summary:[/]")
        console.print(f"  Provider : [cyan]{provider['name']}[/]")
        console.print(f"  Endpoint : [dim]{endpoint}[/]")
        console.print(f"  Model    : [cyan]{model}[/]")
        console.print(f"  API Key  : [dim]{'*' * min(len(api_key), 8)}...[/]")

        if not Confirm.ask("\n[bold]Save this configuration?[/]", default=True):
            console.print("[yellow]Setup cancelled.[/]")
            return False

        config_path = _write_config(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            tool_calling_type=tool_calling_type,
        )

        console.print(
            f"\n[bold green]✓ Configuration saved to:[/] [cyan]{config_path}[/]\n"
        )
        return True

    except (KeyboardInterrupt, EOFError):
        console.print("\n\n[yellow]Setup cancelled.[/]")
        return False
