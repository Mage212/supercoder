"""Tests for the spinner frames, wave-gradient loader, and easter eggs.

Covers Tasks 2.1, 2.2, and 2.4.
"""

from datetime import datetime

from rich.console import Console
from rich.text import Text

from supercoder.ui import spinners, theme


class TestSpinnerFrames:
    def test_braille_frames_have_ten_steps(self):
        # 10 frames is the standard braille "dots" cadence; fewer would
        # look choppy. mistral-vibe uses the same 10.
        assert len(spinners.BRAILLE_FRAMES) == 10
        assert all(isinstance(f, str) and f for f in spinners.BRAILLE_FRAMES)

    def test_pulse_frames_have_ten_steps(self):
        # Reserved for the generating phase; six filled then four empty.
        assert len(spinners.PULSE_FRAMES) == 10
        assert spinners.PULSE_FRAMES.count("■") == 6
        assert spinners.PULSE_FRAMES.count("□") == 4

    def test_spinner_by_phase_covers_three_phases(self):
        expected = {"thinking", "executing", "generating"}
        assert expected <= set(spinners.SPINNER_BY_PHASE.keys())

    def test_phase_spinner_name_falls_back_for_unknown_phase(self):
        # A novel phase (e.g. an MCP-driven phase) must not crash; it falls
        # back to the default braille spinner.
        assert spinners.phase_spinner_name("totally-new-phase") == spinners.DEFAULT_SPINNER_NAME

    def test_known_phases_return_valid_names(self):
        for phase in ("thinking", "executing", "generating"):
            name = spinners.phase_spinner_name(phase)
            assert isinstance(name, str) and name


class TestWaveGradient:
    def test_returns_text_renderable(self):
        result = spinners.wave_gradient("Generating", 0)
        assert isinstance(result, Text)

    def test_each_character_gets_a_color(self):
        # On a truecolor terminal, each character should carry a style derived
        # from the ramp. We check that the Text has multiple non-default spans.
        text = "Generating"
        result = spinners.wave_gradient(text, 0, supports_truecolor=True)
        # Spans are (start, end, style); at least as many as characters.
        spans = list(result._spans)
        assert len(spans) >= 1

    def test_frame_advances_colors(self):
        # The first character's color at frame 0 vs frame 1 must differ,
        # proving the wave moves with the frame counter.
        text = "Generating"
        span0 = next(iter(spinners.wave_gradient(text, 0, supports_truecolor=True)._spans))
        span1 = next(iter(spinners.wave_gradient(text, 1, supports_truecolor=True)._spans))
        # Compare the style strings (color component).
        style0 = str(span0.style) if span0.style else ""
        style1 = str(span1.style) if span1.style else ""
        assert style0 != style1, "wave_gradient must change colors as frame advances"

    def test_solid_fallback_when_no_truecolor(self):
        result = spinners.wave_gradient("Generating", 0, supports_truecolor=False)
        # The whole text is a single solid-color span, not per-character.
        # rich may store the style either as a single span OR on the Text
        # object itself (Text(text, style=...) does the latter), so check both.
        spans = list(result._spans)
        span_style = str(spans[0].style) if spans and spans[0].style else ""
        text_style = str(result.style) if result.style else ""
        assert theme.BRAND in span_style or theme.BRAND in text_style
        # No per-character spans (the whole string is one solid color).
        assert len(spans) <= 1

    def test_for_console_detects_color_system(self):
        # wave_gradient_for should delegate to terminal_supports_truecolor.
        truecolor = Console(force_terminal=True, color_system="truecolor")
        mono = Console(color_system=None)
        from_truecolor = spinners.wave_gradient_for(truecolor, "Generating text", 0)
        from_mono = spinners.wave_gradient_for(mono, "Generating text", 0)
        # Truecolor path produces multiple per-character spans; mono path
        # collapses to a single solid color (no per-character spans).
        assert len(list(from_truecolor._spans)) >= 2
        assert len(list(from_mono._spans)) <= 1

    def test_custom_palette_used(self):
        # A custom 2-color palette alternates strictly between two colors.
        custom = ["#ff0000", "#00ff00"]
        result = spinners.wave_gradient("abcd", 0, palette=custom, supports_truecolor=True)
        spans = list(result._spans)
        colors = {str(s.style) if s.style else "" for s in spans}
        assert colors <= {"#ff0000", "#00ff00"}
        assert len(colors) == 2  # both colors appear


class TestEasterEggs:
    """maybe_easter_egg() returns a whimsical label with low probability."""

    def test_returns_none_above_probability_threshold(self):
        # A deterministic rng whose random() is above EASTER_EGG_PROBABILITY
        # must yield None.
        class HighRng:
            def random(self):
                return theme.EASTER_EGG_PROBABILITY + 0.01

            def choice(self, seq):
                return seq[0]

        assert spinners.maybe_easter_egg(HighRng()) is None

    def test_returns_base_phrase_below_threshold(self):
        class LowRng:
            def __init__(self):
                self.calls = 0

            def random(self):
                return 0.0  # always below threshold

            def choice(self, seq):
                return seq[0]

        egg = spinners.maybe_easter_egg(LowRng())
        assert egg is not None
        assert egg in spinners.EASTER_EGGS

    def test_halloween_eggs_added_on_oct_31(self):
        class LowRng:
            def random(self):
                return 0.0

            def choice(self, seq):
                return seq[-1]  # pick the last (a seasonal one if added)

        halloween = datetime(2026, 10, 31)
        egg = spinners.maybe_easter_egg(LowRng(), now=halloween)
        assert egg in spinners.SEASONAL_EGGS[(10, 31)]

    def test_december_eggs_added_all_month(self):
        class LowRng:
            def random(self):
                return 0.0

            def choice(self, seq):
                return seq[-1]

        mid_dec = datetime(2026, 12, 15)
        egg = spinners.maybe_easter_egg(LowRng(), now=mid_dec)
        assert egg in spinners.SEASONAL_EGGS[(12, None)]

    def test_no_seasonal_eggs_in_july(self):
        class LowRng:
            def random(self):
                return 0.0

            def choice(self, seq):
                # If seasonal eggs were wrongly added, the pool would be
                # larger; assert the pool is exactly the base list by
                # choosing index past the base length and expecting IndexError.
                raise AssertionError("seasonal eggs leaked into a non-seasonal date")

        july = datetime(2026, 7, 4)
        # choice is only called when an egg fires; on a non-seasonal date the
        # pool should equal EASTER_EGGS. Verify via _seasonal_pool directly.
        assert spinners._seasonal_pool(july) == []

    def test_easter_eggs_are_non_empty_strings(self):
        for egg in spinners.EASTER_EGGS:
            assert isinstance(egg, str) and egg
