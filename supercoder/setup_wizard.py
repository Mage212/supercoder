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


def _sanitize_key(raw: str) -> str:
    """Strip any non-printable / ANSI escape characters from a string.

    Some terminals inject escape sequences when capturing input with Rich's
    password prompt on macOS. Using getpass avoids the issue, but we also
    sanitize as a defence-in-depth measure.
    """
    import re
    # Remove ANSI escape sequences (ESC + anything up to a letter)
    cleaned = re.sub(r"\x1b[\[\]()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><~]", "", raw)
    # Remove other non-printable characters (keep normal ASCII/Unicode)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())
    return cleaned.strip()


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
        import getpass
        console.print(f"\n[bold green]API Key[/] [dim]({hint})[/]: ", end="")
        try:
            raw = getpass.getpass(prompt="")
        except (KeyboardInterrupt, EOFError):
            raise
        key = _sanitize_key(raw)
        if key:
            return key
        console.print("[red]API key cannot be empty[/]")


def _get_context_tokens() -> int:
    """Ask user for max context window size."""
    console.print("\n[dim]Context window size (tokens). Check your model's specs.[/]")
    common = [
        ("32 000  — safe default for most models", 32000),
        ("128 000 — GPT-4o, Claude 3.x, Gemini", 128000),
        ("200 000 — Claude 3.5+", 200000),
        ("8 000   — smaller local models", 8000),
        ("Custom", 0),
    ]
    for i, (label, _) in enumerate(common, 1):
        console.print(f"  [cyan]{i}[/]. {label}")
    while True:
        raw = Prompt.ask("[bold green]Context window[/]", default="1")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(common) - 1:
                return common[idx][1]
            if idx == len(common) - 1:
                custom = Prompt.ask("[bold green]Enter token count[/]", default="32000")
                return max(1000, int(custom))
        except ValueError:
            pass
        console.print("[red]Invalid choice[/]")


def _pick_tool_calling_type(suggested: str = "supercoder") -> str:
    """Ask user to select the tool calling format for the model."""
    formats = [
        ("supercoder", "Native format — best for most instruction-following models (default)"),
        ("qwen_like",  "Qwen / GPT-OSS style:  to=tool:name {...}"),
        ("json_block", "JSON code block:  ```json {\"tool\": \"...\", ...} ```"),
        ("xml_function", "XML function:  <function_call name=\"...\">...</function_call>"),
        ("glm_tool_call", "GLM-4 style:  <tool_call>name<arg_key>k</arg_key>...</tool_call>"),
    ]
    console.print("\n[bold]Tool calling format[/] [dim](how the model sends tool calls)[/]\n")
    for i, (key, desc) in enumerate(formats, 1):
        marker = " [green](suggested)[/]" if key == suggested else ""
        console.print(f"  [cyan]{i}[/]. [bold]{key}[/] — {desc}{marker}")

    # Find the default index
    default_idx = next((i for i, (k, _) in enumerate(formats, 1) if k == suggested), 1)

    while True:
        raw = Prompt.ask("[bold green]Format[/]", default=str(default_idx))
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(formats):
                return formats[idx][0]
        except ValueError:
            pass
        console.print("[red]Invalid choice[/]")


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
    max_context_tokens: int = 32000,
    profile_name: str = "default",
) -> Path:
    """Write config file, merging the new profile into any existing config."""
    import yaml

    from .config import CONFIG_DIR, CONFIG_FILE
    from .utils.atomic_writer import AtomicFileWriter

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing config data (if any)
    config_data: dict = {}
    if CONFIG_FILE.exists():
        try:
            raw = CONFIG_FILE.read_text(encoding="utf-8")
            if raw.strip():
                loaded = yaml.safe_load(raw)
                if isinstance(loaded, dict):
                    config_data = loaded
        except Exception:
            config_data = {}

    # Ensure top-level structure
    if not isinstance(config_data.get("models"), dict):
        config_data["models"] = {}

    config_data.setdefault("temperature", 0.2)
    config_data.setdefault("top_p", 0.1)
    config_data.setdefault("max_context_tokens", max_context_tokens)
    config_data.setdefault("reserved_for_response", 4096)
    config_data.setdefault("request_timeout", 300.0)
    config_data.setdefault("debug", False)

    # Merge new profile (overwrites only the named profile)
    config_data["models"][profile_name] = {
        "api_key": api_key,
        "endpoint": endpoint,
        "model": model,
        "max_context_tokens": max_context_tokens,
        "tool_calling_type": tool_calling_type,
    }

    config_data["default_model"] = profile_name

    header = "# SuperCoder Configuration\n# Documentation: https://github.com/your-repo/supercoder\n\n"
    content = header + yaml.dump(config_data, default_flow_style=False, sort_keys=False)
    AtomicFileWriter.write(CONFIG_FILE, content)
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
        max_context_tokens = _get_context_tokens()

        # Ask for tool calling format; pre-select provider suggestion, always confirmable
        suggested_tct = provider.get("tool_calling_type", "supercoder")
        tool_calling_type = _pick_tool_calling_type(suggested_tct)

        # Summary
        console.print("\n[bold]Configuration summary:[/]")
        console.print(f"  Provider   : [cyan]{provider['name']}[/]")
        console.print(f"  Endpoint   : [dim]{endpoint}[/]")
        console.print(f"  Model      : [cyan]{model}[/]")
        console.print(f"  Context    : [cyan]{max_context_tokens:,} tokens[/]")
        console.print(f"  Format     : [cyan]{tool_calling_type}[/]")
        console.print(f"  API Key    : [dim]{'*' * min(len(api_key), 8)}...[/]")

        if not Confirm.ask("\n[bold]Save this configuration?[/]", default=True):
            console.print("[yellow]Setup cancelled.[/]")
            return False

        config_path = _write_config(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            tool_calling_type=tool_calling_type,
            max_context_tokens=max_context_tokens,
        )

        console.print(
            f"\n[bold green]✓ Configuration saved to:[/] [cyan]{config_path}[/]\n"
        )
        return True

    except (KeyboardInterrupt, EOFError):
        console.print("\n\n[yellow]Setup cancelled.[/]")
        return False
