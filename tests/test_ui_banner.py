"""Tests for the animated startup banner (Task 2.3)."""

from rich.console import Console

from supercoder.ui import banner, theme


def _render_to_text(renderable, width: int = 100) -> str:
    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


class TestRenderBanner:
    def test_renders_panel_without_error(self):
        result = banner.render_banner(
            version="0.4.2",
            model="gpt-4o",
            context_tokens=128000,
            tools_count=6,
        )
        text = _render_to_text(result)
        # The Standard figlet logo renders "Supercoder" as stroke ASCII art
        # (not a literal substring), so check for a distinctive art row + meta.
        assert "____" in text  # top row of the figlet Standard "S"
        assert "0.4.2" in text
        assert "gpt-4o" in text

    def test_meta_line_shows_context_and_tools(self):
        result = banner.render_banner("1.0", "m", 32000, 6)
        text = _render_to_text(result)
        assert "32,000" in text
        assert "6" in text

    def test_banner_art_is_multiline_ascii(self):
        assert isinstance(banner.BANNER_ART, str)
        assert banner.BANNER_ART.count("\n") >= 3  # several rows
        # Pure ASCII so the logo renders on every terminal regardless of
        # font support (no box-drawing / filled-block glyphs).
        assert all(ord(ch) < 128 for ch in banner.BANNER_ART.replace("\n", ""))
        # Fits an 80-column terminal (the most common width).
        assert max(len(line) for line in banner.BANNER_ART.split("\n")) <= 80


class TestAnimateBanner:
    def test_static_fallback_on_non_terminal(self):
        # A recorded (non-TTY) console must NOT animate; it prints the static
        # banner once. animate_banner returns None; we verify no exception and
        # that something was written.
        console = Console(record=True, width=100)  # not a real terminal
        banner.animate_banner(console, "0.4.2", "gpt-4o", 128000, 6, duration=0.1)
        text = console.export_text()
        assert "____" in text  # figlet Standard art row present

    def test_paint_art_with_wave_truecolor_produces_spans(self):
        art = banner._paint_art_with_wave(frame=0, supports_truecolor=True)
        # Multiple per-character spans on a truecolor terminal.
        spans = list(art._spans)
        assert len(spans) >= 2

    def test_paint_art_with_wave_mono_is_solid(self):
        art = banner._paint_art_with_wave(frame=0, supports_truecolor=False)
        # Solid color: at most one span, brand color on the Text style.
        spans = list(art._spans)
        assert len(spans) <= 1
        style_str = str(art.style) if art.style else ""
        assert theme.BRAND in style_str

    def test_paint_art_frame_advances_colors(self):
        # Adjacent frames must shift the color of the first glyph.
        a0 = banner._paint_art_with_wave(frame=0, supports_truecolor=True)
        a1 = banner._paint_art_with_wave(frame=1, supports_truecolor=True)
        s0 = next(iter(a0._spans))
        s1 = next(iter(a1._spans))
        style0 = str(s0.style) if s0.style else ""
        style1 = str(s1.style) if s1.style else ""
        assert style0 != style1
