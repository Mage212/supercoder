"""SuperCoder TUI rendering layer.

This package is a pure rendering layer: data in, rich renderables out.
It must NOT import from ``supercoder.agent``, ``supercoder.context``, or
``supercoder.repl`` — that isolation is what makes the future Textual
migration a 1:1 widget swap (see docs/ui-redesign-2026-08-03.md).
"""

from . import render, theme

__all__ = ["render", "theme"]
