"""Tests for the central UI theme module (Task 1.1).

These tests enforce structural invariants so the theme stays in sync with
AgentMode enum values and ALL_TOOLS names. If a new mode or tool is added
without a corresponding theme entry, these tests fail loudly.
"""

from supercoder.agent.agent_modes import AgentMode
from supercoder.tools import ALL_TOOLS
from supercoder.ui import theme


class TestModeStyleCoversAllAgentModes:
    """MODE_STYLE must have an entry for every AgentMode member."""

    def test_mode_style_keys_match_agent_mode_names(self):
        expected = {m.name for m in AgentMode}
        assert set(theme.MODE_STYLE.keys()) == expected, (
            f"MODE_STYLE keys {set(theme.MODE_STYLE.keys())} do not match "
            f"AgentMode names {expected}"
        )

    def test_every_mode_style_entry_has_required_fields(self):
        for mode_name, style in theme.MODE_STYLE.items():
            assert "color" in style, f"{mode_name} missing 'color'"
            assert "icon" in style, f"{mode_name} missing 'icon'"
            assert "label" in style, f"{mode_name} missing 'label'"
            assert isinstance(style["color"], str) and style["color"], (
                f"{mode_name} 'color' must be a non-empty string"
            )
            assert isinstance(style["icon"], str) and style["icon"], (
                f"{mode_name} 'icon' must be a non-empty string"
            )


class TestToolIconsCoverAllBuiltinTools:
    """TOOL_ICONS must cover every builtin tool name in ALL_TOOLS."""

    def test_tool_icons_cover_all_tools(self):
        builtin_names = {t.definition.name for t in ALL_TOOLS}
        missing = builtin_names - set(theme.TOOL_ICONS.keys())
        assert not missing, f"TOOL_ICONS missing entries for: {missing}"

    def test_tool_icon_default_exists(self):
        assert isinstance(theme.TOOL_ICON_DEFAULT, str)
        assert theme.TOOL_ICON_DEFAULT  # non-empty


class TestRoleColors:
    def test_all_semantic_roles_present(self):
        required = {
            "user",
            "assistant",
            "tool",
            "reasoning",
            "error",
            "success",
            "warning",
            "muted",
        }
        assert required <= set(theme.ROLE_COLORS.keys()), (
            f"ROLE_COLORS missing roles: {required - set(theme.ROLE_COLORS.keys())}"
        )

    def test_every_role_color_is_non_empty_string(self):
        for role, color in theme.ROLE_COLORS.items():
            assert isinstance(color, str) and color, f"role '{role}' must map to a non-empty string"


class TestBrandAndBarTokens:
    def test_brand_tokens_are_hex_colors(self):
        for token in (theme.BRAND, theme.BRAND_DIM, theme.BRAND_BRIGHT):
            assert token.startswith("#"), f"{token} should be a hex color"
            assert len(token) == 7, f"{token} should be #RRGGBB"

    def test_bar_widths_are_positive_integers(self):
        assert isinstance(theme.BAR_WIDTH_FOOTER, int) and theme.BAR_WIDTH_FOOTER > 0
        assert isinstance(theme.BAR_WIDTH_STATS, int) and theme.BAR_WIDTH_STATS > 0
        # Stats bar should be at least as wide as the footer.
        assert theme.BAR_WIDTH_STATS >= theme.BAR_WIDTH_FOOTER

    def test_bar_thresholds_ordered(self):
        assert theme.BAR_THRESHOLDS["green"] < theme.BAR_THRESHOLDS["yellow"]

    def test_bar_fill_is_non_empty(self):
        assert isinstance(theme.BAR_FILL, str) and theme.BAR_FILL

    def test_syntax_theme_is_string(self):
        assert isinstance(theme.SYNTAX_THEME, str) and theme.SYNTAX_THEME


class TestBrandRamp:
    """BRAND_RAMP drives the wave-gradient loader and banner color-cycling."""

    def test_ramp_has_enough_steps_for_smooth_wave(self):
        # At least 3 steps so the cyclic offset produces a visible wave rather
        # than an alternating two-color blink.
        assert len(theme.BRAND_RAMP) >= 3

    def test_every_step_is_hex_color(self):
        for step in theme.BRAND_RAMP:
            assert isinstance(step, str), f"{step} must be a string"
            assert step.startswith("#"), f"{step} should be a hex color"
            assert len(step) == 7, f"{step} should be #RRGGBB"

    def test_ramp_is_brand_palette(self):
        # The ramp must be built from the brand colors (not an imported palette
        # like mistral-vibe orange) so the gradient reinforces the identity.
        assert theme.BRAND in theme.BRAND_RAMP
        assert theme.BRAND_DIM in theme.BRAND_RAMP
        assert theme.BRAND_BRIGHT in theme.BRAND_RAMP


class TestAnimationConfig:
    def test_gradient_refresh_rate_is_positive(self):
        assert isinstance(theme.GRADIENT_REFRESH_PER_SECOND, int)
        assert theme.GRADIENT_REFRESH_PER_SECOND > 0

    def test_easter_egg_probability_in_range(self):
        assert 0.0 <= theme.EASTER_EGG_PROBABILITY <= 1.0

    def test_terminal_supports_truecolor_detects_color_system(self):
        # A truecolor console reports it; a monochrome one does not.
        class TruecolorConsole:
            color_system = "truecolor"

        class StandardConsole:
            color_system = "standard"

        class MonoConsole:
            color_system = None

        assert theme.terminal_supports_truecolor(TruecolorConsole()) is True
        assert theme.terminal_supports_truecolor(StandardConsole()) is False
        assert theme.terminal_supports_truecolor(MonoConsole()) is False
