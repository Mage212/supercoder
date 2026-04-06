"""Streaming markdown renderer that progressively displays content.

.. deprecated::
    This module is only used in legacy streaming mode (``streaming: true``).
    Native mode renders complete responses as Markdown.
    Will be removed in a future version.

Adapted from Aider's mdstream.py - provides smooth live-updating markdown output
using Rich's Live display with a sliding window approach.
"""

import contextlib
import io
import time
from typing import ClassVar

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import CodeBlock, Heading, Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


def find_paragraph_boundary(text: str) -> int:
    """Find safe boundary for streaming output.

    Returns index AFTER the boundary (position to split at),
    or 0 if no safe boundary found.

    Safe boundaries (in priority order):
    - "\\n\\n" paragraph break
    - "```\\n" code block end
    - Sentence end with newline (. or ! or ? followed by \\n)
    """
    # Look for paragraph breaks (most reliable)
    idx = text.rfind("\n\n")
    if idx > 0:
        return idx + 2

    # Look for code block end
    idx = text.rfind("```\n")
    if idx > 0:
        return idx + 4

    # Look for sentence end with newline
    for ending in [".\n", "!\n", "?\n"]:
        idx = text.rfind(ending)
        if idx > 0:
            return idx + 2

    return 0


class NoInsetCodeBlock(CodeBlock):
    """A code block with syntax highlighting and no padding."""

    def __rich_console__(self, console, options):
        code = str(self.text).rstrip()
        syntax = Syntax(
            code,
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=(1, 0),
            background_color="default",
        )
        yield syntax


class LeftHeading(Heading):
    """A heading class that renders left-justified."""

    def __rich_console__(self, console, options):
        text = self.text
        text.justify = "left"
        if self.tag == "h1":
            yield Panel(
                text,
                box=box.HEAVY,
                style="markdown.h1.border",
            )
        else:
            if self.tag == "h2":
                yield Text("")
            yield text


class NoInsetMarkdown(Markdown):
    """Markdown with code blocks that have no padding and left-justified headings."""

    elements: ClassVar[dict] = {
        **Markdown.elements,
        "fence": NoInsetCodeBlock,
        "code_block": NoInsetCodeBlock,
        "heading_open": LeftHeading,
    }


class MarkdownStream:
    """Streaming markdown renderer with live-updating window.

    Uses Rich's Live display to show markdown content with smooth scrolling.
    Splits output into "stable" older lines (printed to console) and
    "live" recent lines (updated in a live window).

    This approach works better with terminal scrollback buffers than
    keeping everything in the Live window.
    """

    live = None
    when = 0
    min_delay = 1.0 / 20  # 20fps max update rate
    live_window = 6  # Lines to keep in live area

    def __init__(self, mdargs=None, style=None):
        """Initialize the markdown stream.

        Args:
            mdargs: Additional arguments for Rich Markdown renderer
            style: Optional style for the markdown (e.g., color)
        """
        self.printed = []
        self.mdargs = mdargs or {}
        if style:
            self.mdargs["style"] = style
        self.live = None
        self._live_started = False

    def _render_markdown_to_lines(self, text):
        """Render markdown text to a list of lines.

        Args:
            text: Markdown text to render

        Returns:
            List of rendered lines with line endings
        """
        string_io = io.StringIO()
        console = Console(file=string_io, force_terminal=True)
        markdown = NoInsetMarkdown(text, **self.mdargs)
        console.print(markdown)
        output = string_io.getvalue()
        return output.splitlines(keepends=True)

    def __del__(self):
        """Ensure Live display is cleaned up."""
        if self.live:
            with contextlib.suppress(Exception):
                self.live.stop()

    def update(self, text, final=False):
        """Update the displayed markdown content.

        Args:
            text: The markdown text received so far
            final: If True, this is the final update - clean up
        """
        if not text:
            return

        # Start Live display on first update
        if not self._live_started:
            self.live = Live(Text(""), refresh_per_second=20)
            self.live.start()
            self._live_started = True

        now = time.time()
        # Throttle updates
        if not final and now - self.when < self.min_delay:
            return
        self.when = now

        # Measure render time and adjust throttle
        start = time.time()
        lines = self._render_markdown_to_lines(text)
        render_time = time.time() - start
        self.min_delay = min(max(render_time * 10, 1.0 / 20), 2)

        num_lines = len(lines)

        # How many lines are "stable" (left the live window)?
        if not final:
            num_lines -= self.live_window

        assert self.live is not None

        # Print stable lines above live window
        num_printed = len(self.printed)
        show = 0
        if final or num_lines > 0:
            show = num_lines - num_printed

            if show <= 0 and not final:
                return

        if show > 0:
            show_lines = lines[num_printed:num_lines]
            show_text = "".join(show_lines)
            show_text = Text.from_ansi(show_text)
            self.live.console.print(show_text)
            self.printed = lines[:num_lines]

        # Final cleanup
        if final:
            self.live.update(Text(""))
            self.live.stop()
            self.live = None
            self._live_started = False
            return

        # Update live window with remaining lines
        rest = lines[num_lines:]
        rest_text = "".join(rest)
        rest_text = Text.from_ansi(rest_text)
        self.live.update(rest_text)
