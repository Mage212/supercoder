"""Animated ASCII startup banner for SuperCoder (Task 2.3).

Pure data + functions: no imports from agent/context/repl. The banner art
and the color-cycling animate() port cleanly to a Textual widget later.

Design
------
- ``BANNER_ART`` is a hand-crafted ASCII logo (fits ~80 columns). It is kept
  as a single multi-line string so it renders the same in any font.
- ``render_banner()`` produces a static branded Panel (model/context/tools
  meta). Used as the non-animated fallback and after the animation settles.
- ``animate_banner()`` runs a short color-cycling wave over the ASCII art via
  ``rich.live.Live``. On non-TTY or monochrome terminals it falls back to
  printing the static panel.
"""

from __future__ import annotations

import os
import time

from rich.box import Box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import theme

# Custom box: only a left accent bar (the opencode/mistral-vibe "SplitBorder"
# look). rich's Box expects exactly 8 lines following the same layout as the
# built-in boxes (see rich.box.ROUNDED). We blank every glyph except the left
# border so the panel renders a single vertical accent bar with no frame.
_ACCENT_BAR = Box(
    "    \n"  # top edge (4 blanks)
    "    \n"  # top junction
    "    \n"  # top-right corner
    "    \n"  # right border
    "    \n"  # bottom-right corner
    "    \n"  # bottom edge
    "    \n"  # bottom junction
    "│   \n",  # left border (the only visible glyph)
)

# Hand-crafted ASCII logo. ~44 columns wide; fits comfortably in 80-col terms.
BANNER_ART = r"""
  ____             _         ____                 _ _
 / ___|  ___   ___| | _____ / ___| ___   ___   __| | |__
 \___ \ / _ \ / __| |/ / __| |  _ / _ \ / _ \ / _` | '_ \
  ___) | (_) | (__|   <\__ \ |_| | (_) | (_) | (_| | |_) |
 |____/ \___/ \___|_|\_\___/\____|\___/ \___/ \__,_|_.__/
""".strip("\n")


def _meta_text(version: str, model: str, context_tokens: int, tools_count: int) -> Text:
    """Build the dim meta line shown under the banner art."""
    meta = Text()
    meta.append(f"v{version}  ", style="dim")
    meta.append("Model: ", style="dim")
    meta.append(f"{model}  ", style=f"bold {theme.BRAND}")
    meta.append(f"Context: {context_tokens:,}  Tools: {tools_count}", style="dim")
    return meta


def render_banner(
    version: str,
    model: str,
    context_tokens: int,
    tools_count: int,
    *,
    animated_art: Text | None = None,
) -> Panel:
    """Render the static startup banner Panel.

    Args:
        version: Package version string (without leading 'v').
        model: Active model name.
        context_tokens: Configured max context tokens.
        tools_count: Number of registered tools.
        animated_art: If provided (a pre-painted Text of BANNER_ART), use it
            instead of the plain art. Used by animate_banner() to swap in the
            color-cycled frame while keeping the surrounding Panel layout.

    Returns:
        A Panel with a brand-colored left accent bar.
    """
    art_renderable: Text = (
        animated_art if animated_art is not None else Text(BANNER_ART, style=f"bold {theme.BRAND}")
    )
    body = Group(art_renderable, Text(""), _meta_text(version, model, context_tokens, tools_count))
    return Panel(
        body,
        box=_ACCENT_BAR,
        border_style=theme.BRAND,
        padding=(0, 1),
    )


def _paint_art_with_wave(frame: int, supports_truecolor: bool) -> Text:
    """Paint each non-space character of BANNER_ART with the brand wave.

    Spaces and newlines are passed through unstyled so the art keeps its
    shape; only the glyph characters participate in the wave.
    """
    if not supports_truecolor:
        return Text(BANNER_ART, style=f"bold {theme.BRAND}")

    out = Text()
    n = len(theme.BRAND_RAMP)
    col = 0  # x-position across the whole art (newlines reset visually but the
    # wave continues, which looks fine).
    for ch in BANNER_ART:
        if ch == "\n":
            out.append(ch)
            continue
        if ch == " ":
            out.append(ch)
        else:
            color = theme.BRAND_RAMP[(col + frame) % n]
            out.append(ch, style=f"bold {color}")
        col += 1
    return out


def animate_banner(
    console: Console,
    version: str,
    model: str,
    context_tokens: int,
    tools_count: int,
    *,
    duration: float = 1.5,
    fps: int = theme.GRADIENT_REFRESH_PER_SECOND,
) -> None:
    """Run a short color-cycling animation over the banner, then leave it shown.

    On non-TTY output or monochrome terminals, the animation is skipped and
    the static banner is printed once. On Ctrl+C / interruption the Live
    context exits cleanly (the banner remains in scrollback).

    Args:
        console: The REPL's Console.
        version/model/context_tokens/tools_count: Same as render_banner().
        duration: Total animation time in seconds (default 1.5s).
        fps: Frames per second (default from theme).
    """
    # Skip animation entirely when the output is not an interactive terminal
    # or has no color support: a recorded/piped console would capture every
    # intermediate frame as separate output, which is noise.
    if not console.is_terminal or os.environ.get("TERM") == "dumb":
        console.print(render_banner(version, model, context_tokens, tools_count))
        return

    supports_truecolor = theme.terminal_supports_truecolor(console)
    frame_delay = 1.0 / fps
    total_frames = max(1, int(duration * fps))

    with Live(
        console=console,
        refresh_per_second=fps,
        transient=False,
    ) as live:
        for f in range(total_frames):
            art = _paint_art_with_wave(f, supports_truecolor)
            live.update(
                render_banner(version, model, context_tokens, tools_count, animated_art=art)
            )
            time.sleep(frame_delay)
        # Final frame stays rendered (transient=False), so the banner remains
        # visible in the terminal after the Live block exits.
