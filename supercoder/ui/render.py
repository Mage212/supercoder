"""Unified rich renderables for SuperCoder messages (Task 1.2).

Pure functions: data in, rich renderable out. No imports from
``supercoder.agent``, ``supercoder.context``, or ``supercoder.repl``.

Each function here is designed to map 1:1 to a Textual widget when the REPL
migrates (see docs/ui-redesign-2026-08-03.md, Phase 3).
"""

from __future__ import annotations

import json

from rich import box
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from ..mdstream import NoInsetMarkdown
from . import theme

# ---------------------------------------------------------------------------
# Message-role renderers
# ---------------------------------------------------------------------------


def render_user_message(content: str, live: bool = False) -> Panel:
    """Render a user message as a rounded panel with a cyan left-accent title.

    ``live=True`` is used during active input display; ``live=False`` for
    session-history restore. The visual style is identical either way, which
    fixes the previous inconsistency where live input used a grey background
    and restored history used a cyan Panel.

    Args:
        content: The user's message text.
        live: Whether this is the live input echo (True) or a restored
            history message (False). Currently does not change rendering.

    Returns:
        A rounded Panel with a cyan title.
    """
    # `live` is accepted for API symmetry and to document intent; the unified
    # style is identical so scrollback and live echo match.
    del live
    title = Text.assemble(("👤 ", "cyan"), ("You", f"bold {theme.ROLE_COLORS['user']}"))
    return Panel(
        Text(content, style="bold"),
        title=title,
        title_align="left",
        border_style=theme.ROLE_COLORS["user"],
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_assistant_message(
    content: str,
    model: str = "",
    elapsed_s: float | None = None,
    interrupted: bool = False,
) -> Panel:
    """Render an assistant response in a rounded panel with a meta header.

    The header line shows the avatar, an optional model tag, optional
    duration, and an interrupted marker. The body uses ``NoInsetMarkdown``
    so code blocks render without thick padding and h1 headings get a heavy
    frame (reusing the existing customization from mdstream.py).

    Border: a full ``box.ROUNDED`` in the brand color. The spec (Task 2.5)
    called for a left-only accent bar (opencode SplitBorder look) on the
    assistant panel specifically. That is deferred to the Textual migration:
    rich's custom ``Box`` cannot render a title against a left-only border
    cleanly (the title sits in the blanked top edge), and an ``_ACCENT_BAR``
    box already works for the title-less banner. A Textual widget can apply
    ``border-left`` via CSS while keeping the title, so the accent-bar look
    lands there without a rich compromise.

    Args:
        content: The assistant response text (markdown).
        model: Optional model name tag for the header.
        elapsed_s: Optional response duration in seconds.
        interrupted: Whether the response was interrupted (double-ESC).

    Returns:
        A rounded Panel with a brand-colored title.
    """
    header_parts: list[tuple[str, str]] = [("✨ ", f"bold {theme.BRAND}")]
    header_parts.append(("SuperCoder", f"bold {theme.BRAND}"))
    if model:
        short_model = model.split("/")[-1]
        header_parts.append((f" · {short_model}", "dim"))
    if elapsed_s is not None:
        header_parts.append((f" · {elapsed_s:.0f}s", "dim"))
    if interrupted:
        header_parts.append((" · interrupted", "bold red"))
    title = Text.assemble(*header_parts)

    body: Markdown | str = NoInsetMarkdown(content) if content else ""
    return Panel(
        body,
        title=title,
        title_align="left",
        border_style=theme.BRAND,
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_reasoning(content: str, streaming: bool = False) -> Panel:
    """Render a reasoning/thinking block as a magenta rounded panel.

    The content is rendered in italic magenta (not dim) so the block is
    visually distinct from a plain assistant response: the shared magenta
    hue ties it to the "💭 Reasoning" header, while the italic slant signals
    "internal monologue" rather than a direct answer. ``dim`` was previously
    used but made the block too faint to read and easy to mistake for a
    muted fragment of the response.

    Args:
        content: The reasoning text.
        streaming: If True, append a "thinking…" suffix to the title.

    Returns:
        A rounded Panel with a magenta border.
    """
    suffix = " thinking…" if streaming else ""
    title = Text.assemble(
        ("💭 ", theme.ROLE_COLORS["reasoning"]),
        ("Reasoning", "bold magenta"),
        (suffix, "dim magenta"),
    )
    return Panel(
        Text(content, style="italic magenta"),
        title=title,
        title_align="left",
        border_style=theme.ROLE_COLORS["reasoning"],
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Tool call / result renderers
# ---------------------------------------------------------------------------


def _tool_summary(name: str, args: dict) -> str:
    """Build a one-line argument summary for a tool call."""
    # Reuse the path-key heuristic from repl._tool_argument_path for parity.
    path_keys = ("fileName", "filename", "filepath", "file_path", "path", "file")
    for key in path_keys:
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
    # Fall back to the first string-valued argument, truncated.
    for val in args.values():
        if isinstance(val, str) and val:
            return val[:100]
    return ""


def render_tool_call_compact(name: str, args: dict) -> Text:
    """Render a compact one-line tool-call summary with a per-tool icon.

    The icon comes from ``theme.TOOL_ICONS`` (with a default fallback).
    """
    icon = theme.TOOL_ICONS.get(name, theme.TOOL_ICON_DEFAULT)
    summary = _tool_summary(name, args)
    if summary:
        return Text.assemble(
            (f"{icon} ", "yellow"),
            (f"{name} ", "bold yellow"),
            (summary, "yellow"),
        )
    return Text.assemble((f"{icon} ", "yellow"), (name, "bold yellow"))


def render_tool_call_expanded(name: str, args: dict) -> Panel:
    """Render a full JSON view of a tool call (for /debug mode)."""
    args_str = json.dumps(args, indent=2, ensure_ascii=False)
    body = Syntax(
        args_str,
        "json",
        theme=theme.SYNTAX_THEME,
        word_wrap=True,
        background_color="default",
    )
    title = Text.assemble(("🔧 ", "yellow"), (f"Tool Call: {name}", "bold yellow"))
    return Panel(
        body,
        title=title,
        title_align="left",
        border_style=theme.ROLE_COLORS["tool"],
        box=box.ROUNDED,
        padding=(0, 1),
    )


def render_tool_result(
    summary: str,
    *,
    policy: str = "compact",
    name: str = "",
    diff_text: str | None = None,
    code_text: str | None = None,
    lexer: str | None = None,
) -> Panel | Text:
    """Unified tool-result renderer driven by a display policy.

    Args:
        summary: Short one-line summary of the result.
        policy: One of "compact", "expanded", "hidden", "error".
        name: Tool name, used in the panel title for expanded/error policies.
        diff_text: Optional unified-diff text; rendered with the "diff" lexer.
        code_text: Optional code text; rendered with ``lexer``.
        lexer: Lexer for ``code_text`` (e.g. "text", "bash", "python").

    Returns:
        A Text for compact/hidden policies, or a Panel for expanded/error.
    """
    if policy == "hidden":
        return Text.assemble(("· ", "dim"), (summary, "dim"))
    if policy == "error":
        title = Text.assemble(("❌ ", "red"), (f"Error: {name}", "bold red"))
        return Panel(
            Text(summary),
            title=title,
            title_align="left",
            border_style=theme.ROLE_COLORS["error"],
            box=box.ROUNDED,
            padding=(0, 1),
        )
    if policy == "compact":
        return Text.assemble(
            ("✔ ", theme.ROLE_COLORS["success"]), (summary, theme.ROLE_COLORS["success"])
        )

    # expanded
    title = Text.assemble(
        ("✔ ", theme.ROLE_COLORS["success"]),
        (f"Result: {name}", f"bold {theme.ROLE_COLORS['success']}"),
    )
    if diff_text is not None:
        body: Syntax | Text = Syntax(
            diff_text,
            "diff",
            theme=theme.SYNTAX_THEME,
            background_color="default",
        )
    elif code_text is not None:
        body = Syntax(
            code_text,
            lexer or "text",
            theme=theme.SYNTAX_THEME,
            line_numbers=True,
            background_color="default",
        )
    else:
        body = Text(summary, style="dim")
    return Panel(
        body,
        title=title,
        title_align="left",
        border_style=theme.ROLE_COLORS["success"],
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Status / progress renderers
# ---------------------------------------------------------------------------


def _bar_color(percent: float) -> str:
    if percent < theme.BAR_THRESHOLDS["green"]:
        return theme.ROLE_COLORS["success"]
    if percent < theme.BAR_THRESHOLDS["yellow"]:
        return theme.ROLE_COLORS["warning"]
    return theme.ROLE_COLORS["error"]


def render_context_bar(used: int, total: int, width: int = theme.BAR_WIDTH_FOOTER) -> Text:
    """Render a unified context-usage progress bar.

    Replaces the three ad-hoc bar renderers in repl.py (footer, /stats,
    streaming) with a single function. Color thresholds live in theme.py.

    Args:
        used: Tokens used so far.
        total: Maximum tokens in the context window.
        width: Bar width in characters (footer vs /stats).

    Returns:
        A Text containing the bar and the ``used/total tokens`` label.
    """
    if total <= 0:
        percent = 0.0
        width = max(width, 1)
        filled = 0
    else:
        percent = (used / total) * 100
        filled = max(0, min(width, int(width * used / total)))
    empty = width - filled
    color = _bar_color(percent)
    return Text.assemble(
        (theme.BAR_FILL * filled, color),
        (theme.BAR_FILL * empty, "dim"),
        (" ", ""),
        (f"{used:,}/{total:,} tokens", "dim"),
    )


def render_mode_prompt(mode: str, model_tag: str) -> Text:
    """Render a mode-colored prompt as a rich Text.

    Note: prompt_toolkit does NOT interpret rich markup, so when this is used
    in the live prompt_toolkit session the caller must apply the mode color
    via PromptStyle instead. This function exists for tests and for any
    rich-rendered prompt preview.

    Args:
        mode: Mode key (must be an AgentMode member name, e.g. "CODE").
        model_tag: Short model label (already truncated by the caller).

    Returns:
        A Text with the model tag (dim) and mode icon (mode-colored).
    """
    style = theme.MODE_STYLE.get(mode, {"color": theme.BRAND, "icon": ">"})
    return Text.assemble(
        (f"[{model_tag}] ", "dim"),
        (f"{style['icon']} ", f"bold {style['color']}"),
    )
