"""SuperCoder TUI design system.

All colors and visual constants live here. Forward-compatible with Textual:
each value will become a CSS variable (``$brand``, ``$role-user``, etc.) verbatim
when the REPL migrates to Textual (see docs/ui-textual-migration-plan.md).

This module is a pure data layer. It must NOT import from ``supercoder.agent``,
``supercoder.context``, or ``supercoder.repl`` — only standard library and
the tools registry (read-only, for ``TOOL_ICONS`` coverage).
"""

# ---------------------------------------------------------------------------
# Brand accent
# ---------------------------------------------------------------------------
# SuperCoder teal/cyan — recalls the "coder/code" identity and is visually
# distinct from mistral-vibe's orange (#FF8205) and opencode's per-theme primary.
BRAND = "#00d7af"
BRAND_DIM = "#00875f"
BRAND_BRIGHT = "#5fffdf"

# 5-step teal ramp for the wave-gradient loading text (Task 2.2) and the
# animated startup banner (Task 2.3). Each step cycles a per-character color
# offset, so a single ramp yields a smooth traveling wave. The ramp is the
# brand palette itself (no imported orange like mistral-vibe), so the
# gradient reinforces the SuperCoder identity rather than borrowing one.
BRAND_RAMP: list[str] = [BRAND_BRIGHT, "#00d7af", BRAND, "#00997a", BRAND_DIM]

# Semantic background tones (used for message containers).
BG_PANEL = "grey15"
BG_ELEMENT = "grey23"

# ---------------------------------------------------------------------------
# Role colors — one per content type. These drive all message rendering.
# Keys are semantic roles, not specific colors, so themes can swap them later.
# ---------------------------------------------------------------------------
ROLE_COLORS: dict[str, str] = {
    "user": "cyan",
    "assistant": "white",  # bare text; the avatar/label carries the accent
    "tool": "yellow",
    "reasoning": "magenta",
    "error": "red",
    "success": "green",
    "warning": "yellow",
    "muted": "grey50",
}

# ---------------------------------------------------------------------------
# Per-mode presentation (prompt color + glyph + toolbar wording).
# Keys MUST match AgentMode enum member names (ASK/PLAN/CODE/ACCEPT_EDITS).
# ---------------------------------------------------------------------------
MODE_STYLE: dict[str, dict[str, str]] = {
    "ASK": {"color": "blue", "icon": "?", "label": "ASK"},
    "PLAN": {"color": "magenta", "icon": "☰", "label": "PLAN"},
    "CODE": {"color": "green", "icon": "❯", "label": "CODE"},  # noqa: RUF001
    "ACCEPT_EDITS": {"color": "yellow", "icon": "⚡", "label": "ACCEPT"},
}

# ---------------------------------------------------------------------------
# Per-tool icons for compact tool-call summaries.
# Tool names match supercoder/tools/__init__.py ALL_TOOLS definition names.
# Unknown/MCP tools fall back to the default icon in renderers.
# ---------------------------------------------------------------------------
TOOL_ICONS: dict[str, str] = {
    "file-read": "📖",
    "code-search": "🔍",
    "glob": "🌐",
    "code-edit": "✏️",
    "project-structure": "🌲",
    "command-exec": "$",
}

# Default icon for tools not listed above (MCP tools, future builtins).
TOOL_ICON_DEFAULT = "⚙"

# ---------------------------------------------------------------------------
# Unified progress-bar configuration.
# Replaces three ad-hoc bar renderers in repl.py (footer, /stats, streaming).
# ---------------------------------------------------------------------------
BAR_WIDTH_FOOTER = 10
BAR_WIDTH_STATS = 20
BAR_FILL = "━"

# Context-utilization thresholds for progress-bar color (percent of max).
BAR_THRESHOLDS = {"green": 50, "yellow": 80}  # <50 green, <80 yellow, >=80 red

# ---------------------------------------------------------------------------
# Syntax highlighting theme.
# Kept as a single knob in theme.py so a future spec can rotate themes
# per-mode or per-model without touching renderer code.
# ---------------------------------------------------------------------------
SYNTAX_THEME = "monokai"

# ---------------------------------------------------------------------------
# Animation configuration (Phase 2 — personality).
# Wave-gradient loader, banner color-cycling, and spinner phase refresh all
# share these knobs. Kept here so tuning is a one-line change.
# ---------------------------------------------------------------------------
# Refresh rate for the wave-gradient loader and banner color-cycling.
# Matches mistral-vibe's 10 FPS; 0.1s feels animated without burning CPU.
GRADIENT_REFRESH_PER_SECOND = 10

# Probability (per turn) that a whimsical "easter egg" replaces the generic
# "Generating..." label. Seasonal eggs (Halloween/December) are added on top
# of this pool on their dates.
EASTER_EGG_PROBABILITY = 0.10


def terminal_supports_truecolor(console) -> bool:
    """Return True if the console can render per-character hex colors.

    Used for graceful degradation: the wave gradient and banner color-cycling
    fall back to a solid brand color on 16-color ANSI / monochrome terminals
    so they don't render as flickering default-color text.
    """
    color_system = getattr(console, "color_system", None)
    return color_system in ("truecolor", "256")
