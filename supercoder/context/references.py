"""Host-side expansion of @path references in user prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Literal

from ..logging import get_logger
from ..permissions import PermissionPolicy
from ..tools.tool_utils import (
    find_similar_files,
    format_size,
    is_binary_file,
    is_ignored_path,
    relative_display_path,
    resolve_within_root,
)
from .freshness import FileFreshnessTracker

REFERENCE_RE = re.compile(
    r"(?<!\\)(?:^|(?<=[\s([{<,;:]))@([A-Za-z0-9_./-]+)",
    re.MULTILINE,
)


@dataclass
class ContextReferenceItem:
    """One resolved @path reference."""

    ref: str
    kind: Literal["file", "directory", "skipped"]
    status: Literal["attached", "skipped"]
    path: str | None = None
    reason: str | None = None
    size_bytes: int | None = None
    bytes_returned: int = 0
    lines_returned: int = 0
    entries_returned: int = 0
    entries_seen: int = 0
    truncated: bool = False
    suggestions: list[str] = field(default_factory=list)

    def to_log_dict(self) -> dict:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "status": self.status,
            "path": self.path,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
            "bytes_returned": self.bytes_returned,
            "lines_returned": self.lines_returned,
            "entries_returned": self.entries_returned,
            "entries_seen": self.entries_seen,
            "truncated": self.truncated,
            "suggestions": self.suggestions,
        }


@dataclass
class ContextAttachment:
    """Expanded context payload derived from @path references."""

    content: str
    items: list[ContextReferenceItem]
    estimated_tokens: int
    model_bytes: int

    @property
    def files(self) -> int:
        return sum(1 for item in self.items if item.kind == "file" and item.status == "attached")

    @property
    def directories(self) -> int:
        return sum(
            1 for item in self.items if item.kind == "directory" and item.status == "attached"
        )

    @property
    def skipped(self) -> int:
        return sum(1 for item in self.items if item.status == "skipped")

    @property
    def truncated(self) -> bool:
        return any(item.truncated for item in self.items)

    def to_log_dict(self) -> dict:
        return {
            "files": self.files,
            "directories": self.directories,
            "skipped": self.skipped,
            "truncated": self.truncated,
            "model_bytes": self.model_bytes,
            "estimated_tokens": self.estimated_tokens,
            "items": [item.to_log_dict() for item in self.items],
        }


def extract_context_references(message: str) -> list[str]:
    """Return unique @path references from a user message."""
    refs: list[str] = []
    seen: set[str] = set()
    for match in REFERENCE_RE.finditer(message):
        ref = match.group(1).rstrip(".")
        if not ref or ref in seen:
            continue
        refs.append(ref)
        seen.add(ref)
    return refs


def expand_context_references(
    message: str,
    repo_root: Path,
    *,
    permission_policy: PermissionPolicy | None = None,
    freshness_tracker: FileFreshnessTracker | None = None,
    max_file_bytes: int = 64_000,
    max_dir_entries: int = 200,
    max_total_tokens: int = 12_000,
) -> ContextAttachment | None:
    """Expand @path references into a bounded user-message attachment."""
    refs = extract_context_references(message)
    if not refs:
        return None

    root = repo_root.resolve()
    max_total_bytes = max(1, max_total_tokens * 4)
    remaining_bytes = max_total_bytes
    blocks: list[str] = []
    items: list[ContextReferenceItem] = []

    for ref in refs:
        if remaining_bytes <= 0:
            item, block = _skipped(ref, "total attachment budget exhausted")
            items.append(item)
            blocks.append(block)
            continue

        path, error = resolve_within_root(ref, root)
        if error or path is None:
            item, block = _skipped(ref, "outside repository")
        elif (
            permission_policy and (decision := permission_policy.check_path(path, "attach")).denied
        ):
            rel = permission_policy.relative_path(path)
            get_logger().log_permission_decision(
                tool_name="context_attachment",
                subject=rel,
                action=decision.action.value,
                reason=decision.reason,
                source=decision.source,
                matched_rule=decision.matched_rule,
            )
            item, block = _skipped(ref, "permission denied", path=rel)
        elif not path.exists():
            suggestions = find_similar_files(path, root)
            item, block = _skipped(ref, "not found", suggestions=suggestions)
        elif is_ignored_path(path, root):
            item, block = _skipped(ref, "ignored path")
        elif path.is_dir():
            item, block = _attach_directory(
                ref,
                path,
                root,
                permission_policy=permission_policy,
                max_entries=max_dir_entries,
                max_bytes=remaining_bytes,
            )
        elif path.is_file():
            item, block = _attach_file(
                ref,
                path,
                root,
                freshness_tracker=freshness_tracker,
                max_bytes=min(max_file_bytes, remaining_bytes),
            )
        else:
            item, block = _skipped(ref, "unsupported path type")

        items.append(item)
        blocks.append(block)
        remaining_bytes -= len(block.encode("utf-8", errors="replace"))

    content = "[Attached context from @ references]\n\n" + "\n\n".join(blocks)
    model_bytes = len(content.encode("utf-8", errors="replace"))
    estimated_tokens = max(1, model_bytes // 4)
    return ContextAttachment(
        content=content,
        items=items,
        estimated_tokens=estimated_tokens,
        model_bytes=model_bytes,
    )


def summarize_context_attachment(summary: dict) -> str:
    """Format a compact, display-safe attachment summary."""
    parts: list[str] = []
    if summary.get("files"):
        parts.append(f"{summary['files']} file(s)")
    if summary.get("directories"):
        parts.append(f"{summary['directories']} dir(s)")
    if summary.get("skipped"):
        parts.append(f"{summary['skipped']} skipped")
    if not parts:
        parts.append("no paths attached")

    size = format_size(int(summary.get("model_bytes") or 0))
    tokens = int(summary.get("estimated_tokens") or 0)
    suffix = " truncated" if summary.get("truncated") else ""
    return f"Attached context: {', '.join(parts)}; {size}, ~{tokens} tokens{suffix}"


def summarize_attachment_content(content: str) -> str:
    """Summarize a stored context_attachment message without exposing file contents."""
    summary = {
        "files": content.count("<attached_file "),
        "directories": content.count("<attached_directory "),
        "skipped": content.count("<skipped_reference "),
        "model_bytes": len(content.encode("utf-8", errors="replace")),
        "estimated_tokens": max(1, len(content.encode("utf-8", errors="replace")) // 4),
        "truncated": 'truncated="true"' in content,
    }
    return summarize_context_attachment(summary)


def _attach_file(
    ref: str,
    path: Path,
    root: Path,
    *,
    freshness_tracker: FileFreshnessTracker | None = None,
    max_bytes: int,
) -> tuple[ContextReferenceItem, str]:
    display_path = relative_display_path(path, root)
    if is_binary_file(path):
        item = ContextReferenceItem(
            ref=ref,
            kind="skipped",
            status="skipped",
            path=display_path,
            reason="binary file",
            size_bytes=path.stat().st_size,
        )
        return item, _format_skipped_block(item)

    selected: list[tuple[int, str]] = []
    used_bytes = 0
    truncated = False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                if used_bytes >= max_bytes:
                    truncated = True
                    break

                encoded = line.encode("utf-8", errors="replace")
                if used_bytes + len(encoded) > max_bytes:
                    available = max(max_bytes - used_bytes, 0)
                    if available == 0:
                        truncated = True
                        break
                    line = encoded[:available].decode("utf-8", errors="replace")
                    encoded = line.encode("utf-8", errors="replace")
                    truncated = True

                selected.append((line_number, line.rstrip("\n")))
                used_bytes += len(encoded)
                if truncated:
                    break
    except OSError as e:
        item = ContextReferenceItem(
            ref=ref,
            kind="skipped",
            status="skipped",
            path=display_path,
            reason=f"read error: {e}",
        )
        return item, _format_skipped_block(item)

    size_bytes = path.stat().st_size
    lines_returned = len(selected)
    if not selected and size_bytes > 0:
        truncated = True

    item = ContextReferenceItem(
        ref=ref,
        kind="file",
        status="attached",
        path=display_path,
        size_bytes=size_bytes,
        bytes_returned=used_bytes,
        lines_returned=lines_returned,
        truncated=truncated,
    )
    if freshness_tracker:
        freshness_tracker.mark_read(path, source="context_attachment")
    body = "\n".join(f"{number:4d}: {text}" for number, text in selected)
    lines_label = f"1-{selected[-1][0]}" if selected else "0"
    block = (
        f'<attached_file path="{escape(display_path)}" '
        f'size="{format_size(size_bytes)}" '
        f'bytes_returned="{used_bytes}" '
        f'lines="{lines_label}" '
        f'truncated="{str(truncated).lower()}">\n'
        f"{body}\n"
        "</attached_file>"
    )
    return item, block


def _attach_directory(
    ref: str,
    path: Path,
    root: Path,
    *,
    permission_policy: PermissionPolicy | None = None,
    max_entries: int,
    max_bytes: int,
) -> tuple[ContextReferenceItem, str]:
    display_path = relative_display_path(path, root)
    entries: list[str] = []
    used_bytes = 0
    seen = 0
    truncated = False

    try:
        for candidate in path.rglob("*"):
            if not candidate.is_file() or is_ignored_path(candidate, root):
                continue
            if permission_policy and permission_policy.check_path(candidate, "attach").denied:
                continue
            seen += 1
            rel = relative_display_path(candidate, root)
            line_bytes = len((rel + "\n").encode("utf-8", errors="replace"))
            if len(entries) >= max_entries or used_bytes + line_bytes > max_bytes:
                truncated = True
                break
            entries.append(rel)
            used_bytes += line_bytes
    except OSError as e:
        item = ContextReferenceItem(
            ref=ref,
            kind="skipped",
            status="skipped",
            path=display_path,
            reason=f"read error: {e}",
        )
        return item, _format_skipped_block(item)

    entries = sorted(entries)
    item = ContextReferenceItem(
        ref=ref,
        kind="directory",
        status="attached",
        path=display_path,
        bytes_returned=used_bytes,
        entries_returned=len(entries),
        entries_seen=seen,
        truncated=truncated,
    )
    body = "\n".join(entries)
    total_label = f">={seen}" if truncated else str(seen)
    block = (
        f'<attached_directory path="{escape(display_path)}" '
        f'entries_returned="{len(entries)}" '
        f'entries_seen="{total_label}" '
        f'truncated="{str(truncated).lower()}">\n'
        f"{body}\n"
        "</attached_directory>"
    )
    return item, block


def _skipped(
    ref: str,
    reason: str,
    *,
    path: str | None = None,
    suggestions: list[str] | None = None,
) -> tuple[ContextReferenceItem, str]:
    item = ContextReferenceItem(
        ref=ref,
        kind="skipped",
        status="skipped",
        path=path,
        reason=reason,
        suggestions=suggestions or [],
    )
    return item, _format_skipped_block(item)


def _format_skipped_block(item: ContextReferenceItem) -> str:
    suggestion_text = ""
    if item.suggestions:
        suggestion_text = "\nDid you mean:\n" + "\n".join(f"- {s}" for s in item.suggestions)
    path_attr = f' path="{escape(item.path)}"' if item.path else ""
    return (
        f'<skipped_reference ref="@{escape(item.ref)}"{path_attr} '
        f'reason="{escape(item.reason or "unknown")}">'
        f"{suggestion_text}\n"
        "</skipped_reference>"
    )
