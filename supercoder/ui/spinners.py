"""Spinner frames and phase config for SuperCoder TUI (Tasks 2.1, 2.2, 2.4).

Pure data + functions: no imports from agent/context/repl. The frames and the
wave-gradient function port cleanly to a Textual Spinner widget later.

Design notes
------------
- ``BRAILLE_FRAMES`` mirrors rich's built-in ``"dots"`` spinner (10 frames);
  exported here so the spinner name in repl.py is not a magic string.
- ``PULSE_FRAMES`` (filled/empty squares) is intended for the "generating"
  phase. ``console.status()`` only accepts a spinner *name*, not a frame list,
  so PULSE cannot be used through ``console.status()`` today. It is kept here
  as data for a future Live-based spinner (or the Textual migration). For now
  the loader distinguishes phases via the gradient text, not a different glyph.
- ``wave_gradient`` paints each character of the loading text with a cyclic
  offset into ``BRAND_RAMP`` so the brand color appears to travel left to
  right. This is the simplified cyclic form of mistral-vibe's ping-pong wave.
"""

from __future__ import annotations

from rich.text import Text

from . import theme

# ---------------------------------------------------------------------------
# Spinner frames
# ---------------------------------------------------------------------------

# 10-frame braille spinner. Identical to rich's "dots"; exported so repl.py
# references theme/spinners rather than a literal string.
BRAILLE_FRAMES: list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# PULSE frames: six filled squares then four empty (a "breathing" pulse).
# Reserved for the generating phase; not wired into console.status() yet
# (see module docstring). Kept as data for a Live-based spinner.
PULSE_FRAMES: list[str] = ["■", "■", "■", "■", "■", "■", "□", "□", "□", "□"]

# The rich spinner NAME used for each agent phase. console.status() takes a
# name only, so we keep two visually distinct built-ins: "dots" for thinking
# and executing, "dots12" for generating. If PULSE is ever wired in via a
# Live-based spinner, this mapping is the single place to change.
SPINNER_BY_PHASE: dict[str, str] = {
    "thinking": "dots",
    "executing": "dots",
    "generating": "dots12",
}

DEFAULT_SPINNER_NAME = "dots"


def phase_spinner_name(phase: str) -> str:
    """Return the rich spinner name for an agent phase.

    Falls back to DEFAULT_SPINNER_NAME for unknown phases (e.g. MCP tools)
    so a novel phase never crashes the loader.
    """
    return SPINNER_BY_PHASE.get(phase, DEFAULT_SPINNER_NAME)


# ---------------------------------------------------------------------------
# Wave-gradient loading text (Task 2.2)
# ---------------------------------------------------------------------------


def wave_gradient(
    text: str,
    frame: int,
    palette: list[str] | None = None,
    *,
    supports_truecolor: bool = True,
) -> Text:
    """Paint ``text`` with a traveling wave of brand colors.

    Each character's color is ``palette[(i + frame) % len(palette)]``. As the
    caller increments ``frame`` per tick, the wave appears to move left to
    right.

    Args:
        text: The loading string to paint.
        frame: Monotonic tick counter (the caller increments it).
        palette: Color ramp; defaults to ``theme.BRAND_RAMP``.
        supports_truecolor: If False (16-color/monochrome terminal), return a
            solid brand-colored Text instead of per-character colors so the
            line does not flicker as default-color glyphs.

    Returns:
        A ``rich.text.Text`` with per-character styling.
    """
    ramp = palette if palette is not None else theme.BRAND_RAMP
    if not supports_truecolor:
        # Solid brand color reads cleanly even when per-character hex colors
        # are not available.
        return Text(text, style=f"bold {theme.BRAND}")

    out = Text()
    n = len(ramp)
    for i, ch in enumerate(text):
        color = ramp[(i + frame) % n]
        out.append(ch, style=color)
    return out


def wave_gradient_for(console, text: str, frame: int, palette: list[str] | None = None) -> Text:
    """Convenience wrapper: detect color support from ``console`` and paint.

    Use this from call sites that already hold a Console (e.g. the REPL tick
    thread) so they don't repeat the color-system check.
    """
    return wave_gradient(
        text,
        frame,
        palette=palette,
        supports_truecolor=theme.terminal_supports_truecolor(console),
    )


# ---------------------------------------------------------------------------
# Easter eggs (Task 2.4)
# ---------------------------------------------------------------------------
# Whimsical loading labels that replace "Generating..." with a small chance.
# Seasonal pools (Halloween / December) are layered on top of the base pool
# on their dates. Pattern adapted from mistral-vibe's loading.py.

EASTER_EGGS: list[str] = [
    "Compiling the universe",
    "Petting the rubber duck",
    "Counting semicolons",
    "Asking the compiler nicely",
    "Herding threads",
    "Untangling the call graph",
    "Negotiating with the linker",
    "Reticulating splines",
    "Defragmenting the stack",
    "Optimizing the optimizer",
]

# (month, day) -> seasonal phrases. day=None means the whole month.
SEASONAL_EGGS: dict[tuple[int, int | None], list[str]] = {
    (10, 31): [  # Halloween
        "Summoning spirits",
        "Carving pumpkins",
        "Petting the rubber bat",
        "Brewing potions",
    ],
    (12, None): [  # All of December
        "Wrapping presents",
        "Decorating the tree",
        "Drinking hot cocoa",
        "Building snowmen",
    ],
}


def _seasonal_pool(now) -> list[str]:
    """Return the seasonal phrase additions active on the given date."""
    pool: list[str] = []
    for (month, day), eggs in SEASONAL_EGGS.items():
        if now.month == month and (day is None or now.day == day):
            pool.extend(eggs)
    return pool


def maybe_easter_egg(rng=None, *, now=None) -> str | None:
    """Return a whimsical loading label, or None.

    With probability ``theme.EASTER_EGG_PROBABILITY`` (default 10%) a phrase
    is chosen from the base pool, plus any seasonal pool active on ``now``.
    Returns None the rest of the time so the caller keeps the generic label.

    Args:
        rng: Optional random.Random instance (inject for deterministic tests).
        now: Optional datetime (inject for seasonal tests). Defaults to now.
    """
    import random
    from datetime import datetime

    rng = rng if rng is not None else random
    now = now if now is not None else datetime.now()

    if rng.random() >= theme.EASTER_EGG_PROBABILITY:
        return None

    pool = list(EASTER_EGGS)
    pool.extend(_seasonal_pool(now))
    return rng.choice(pool)
