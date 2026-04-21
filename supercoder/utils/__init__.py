"""Utility modules for SuperCoder."""

from datetime import datetime

from .atomic_writer import AtomicFileWriter

__all__ = ["AtomicFileWriter", "format_relative_time"]


def format_relative_time(iso_timestamp: str) -> str:
    """Convert ISO timestamp to relative time like '5m ago', '2h ago'."""
    if not iso_timestamp:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        delta = datetime.now() - dt
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        if seconds < 604800:
            return f"{seconds // 86400}d ago"
        return f"{seconds // 604800}w ago"
    except (ValueError, TypeError):
        return "unknown"
