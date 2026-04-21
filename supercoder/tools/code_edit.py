"""Diff-based code editing tool."""

import difflib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..utils.atomic_writer import AtomicFileWriter
from .base import BaseTool, ToolDefinition

if TYPE_CHECKING:
    from ..checkpoint import CheckpointManager


class CodeEditTool(BaseTool):
    """Edit code using diff-based operations with atomic writes."""

    def __init__(
        self,
        checkpoint_manager: Optional["CheckpointManager"] = None,
        allowed_root: Path | None = None,
    ):
        """Initialize with optional checkpoint manager and allowed root directory.

        Args:
            checkpoint_manager: Optional CheckpointManager for backup/rollback support
            allowed_root: If set, file paths must be within this directory (path traversal guard)
        """
        self.checkpoint = checkpoint_manager
        self.allowed_root = allowed_root

    def _safe_write(self, path: Path, content: str) -> None:
        """Write file with backup and atomic write.

        Args:
            path: Target file path
            content: Content to write
        """
        # Backup before modifying (if checkpoint is active)
        if self.checkpoint:
            self.checkpoint.backup_file(path)

        # Atomic write
        AtomicFileWriter.write(path, content)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code-edit",
            description=(
                "Edit code files. Operations: "
                "search_replace (find and replace text), "
                "insert_after (insert content after a matching line), "
                "insert_before (insert content before a matching line), "
                "replace_lines (replace a line range), "
                "append (append to end of file), "
                "create (create a new file)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to edit or create"},
                    "operation": {
                        "type": "string",
                        "description": "Edit operation to perform",
                        "enum": ["search_replace", "insert_after", "insert_before", "replace_lines", "append", "create"],
                    },
                    "search": {"type": "string", "description": "Text to find (for search_replace)"},
                    "replace": {"type": "string", "description": "Replacement text (for search_replace)"},
                    "after": {"type": "string", "description": "Line to insert after (for insert_after)"},
                    "before": {"type": "string", "description": "Line to insert before (for insert_before)"},
                    "content": {"type": "string", "description": "New content (for create, insert_after, insert_before, replace_lines, append)"},
                    "startLine": {"type": "integer", "description": "Start line number (for replace_lines)"},
                    "endLine": {"type": "integer", "description": "End line number (for replace_lines)"},
                },
                "required": ["filepath", "operation"],
            },
        )

    def execute(self, arguments: str) -> str:
        args = self.parse_args(arguments)
        if args.get("_parse_error"):
            return f"Error: Invalid JSON arguments: {args.get('raw', '')}"
        filepath = args.get("filepath", args.get("fileName", ""))
        operation = args.get("operation", "search_replace")

        if not filepath:
            return "Error: filepath is required"

        path = Path(filepath)

        # Validate path stays within the allowed root (when configured)
        if self.allowed_root is not None:
            try:
                path.resolve().relative_to(self.allowed_root)
            except ValueError:
                return f"Error: Path '{filepath}' is outside the project directory"

        # Handle create operation separately
        if operation == "create":
            return self._create_file(path, args.get("content", ""))

        # For other operations, file must exist
        if not path.exists():
            return f"Error: File '{filepath}' not found"

        try:
            if operation == "search_replace":
                return self._search_replace(path, args.get("search", ""), args.get("replace", ""))
            elif operation == "insert_after":
                return self._insert_after(path, args.get("after", ""), args.get("content", ""))
            elif operation == "insert_before":
                return self._insert_before(path, args.get("before", ""), args.get("content", ""))
            elif operation == "replace_lines":
                return self._replace_lines(
                    path, args.get("startLine", 1), args.get("endLine", 1), args.get("content", "")
                )
            elif operation == "append":
                return self._append(path, args.get("content", ""))
            else:
                return f"Error: Unknown operation '{operation}'"

        except Exception as e:
            return f"Error: {e}"

    def _generate_diff(self, before: str, after: str, filepath: Path) -> str:
        """Generate unified diff between before and after content."""
        diff_lines = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(filepath),
            tofile=str(filepath),
            lineterm="",
        )
        return "\n".join(diff_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace per line for fuzzy matching."""
        return "\n".join(re.sub(r"\s+", " ", line.strip()) for line in text.splitlines())

    def _find_best_match(self, content: str, search: str, threshold: float = 0.85) -> dict:
        """Find the best match for search in content using a cascade of strategies.

        Returns dict with: found, match_type ("exact"/"whitespace_normalized"/"fuzzy"/"none"),
        start, end, matched_text, ratio, best_ratio (for error reporting).
        """
        result = {"found": False, "match_type": "none", "start": -1, "end": -1,
                  "matched_text": "", "ratio": 0.0, "best_ratio": 0.0}

        # 1. Exact match
        idx = content.find(search)
        if idx != -1:
            result.update(found=True, match_type="exact", start=idx, end=idx + len(search),
                          matched_text=search, ratio=1.0, best_ratio=1.0)
            return result

        # 2. Whitespace-normalized match (line-based)
        search_lines_norm = self._normalize_whitespace(search).splitlines()
        content_lines = content.splitlines()
        content_norm_lines = [re.sub(r"\s+", " ", line.strip()) for line in content_lines]

        if search_lines_norm and content_norm_lines:
            norm_match_len = len(search_lines_norm)
            for i in range(len(content_norm_lines) - norm_match_len + 1):
                if content_norm_lines[i:i + norm_match_len] == search_lines_norm:
                    start_char = sum(len(ln) + 1 for ln in content_lines[:i])
                    end_char = start_char + sum(len(ln) + 1 for ln in content_lines[i:i + norm_match_len]) - 1
                    matched_text = content[start_char:end_char]
                    ratio = difflib.SequenceMatcher(None, search, matched_text).ratio()
                    result.update(found=True, match_type="whitespace_normalized",
                                  start=start_char, end=end_char,
                                  matched_text=matched_text, ratio=ratio, best_ratio=ratio)
                    return result

        # 3. Fuzzy match using SequenceMatcher on lines
        search_lines = search.splitlines()
        content_lines_for_fuzzy = content.splitlines()
        best_i = -1
        best_j = -1
        best_ratio = 0.0

        if search_lines and content_lines_for_fuzzy:
            s = difflib.SequenceMatcher(None, search_lines, content_lines_for_fuzzy)
            _ = s.get_matching_blocks()  # Force full computation
            # Use find_longest_match as anchor, then expand
            match = s.find_longest_match(0, len(search_lines), 0, len(content_lines_for_fuzzy))
            if match.size > 0:
                # Expand around the longest match to cover full search span
                start_line = match.b - match.a
                end_line = match.b + (len(search_lines) - match.a)
                start_line = max(0, start_line)
                end_line = min(len(content_lines_for_fuzzy), end_line)

                # Try multiple window sizes around the anchor
                for offset in range(-2, 3):
                    sl = max(0, start_line + offset)
                    el = sl + len(search_lines)
                    if el > len(content_lines_for_fuzzy):
                        continue
                    candidate_lines = content_lines_for_fuzzy[sl:el]
                    candidate_text = "\n".join(candidate_lines)
                    r = difflib.SequenceMatcher(None, search, candidate_text).ratio()
                    if r > best_ratio:
                        best_ratio = r
                        best_i = sl
                        best_j = el

                if best_ratio >= threshold and best_i >= 0:
                    matched_text = "\n".join(content_lines_for_fuzzy[best_i:best_j])
                    start_char = sum(len(ln) + 1 for ln in content_lines_for_fuzzy[:best_i])
                    end_char = start_char + len(matched_text)
                    result.update(found=True, match_type="fuzzy",
                                  start=start_char, end=end_char,
                                  matched_text=matched_text, ratio=best_ratio,
                                  best_ratio=best_ratio)
                    return result

                # Store best ratio for error reporting
                result["best_ratio"] = best_ratio
                if best_i >= 0:
                    result["matched_text"] = "\n".join(content_lines_for_fuzzy[best_i:best_j])
                    result["start"] = sum(len(ln) + 1 for ln in content_lines_for_fuzzy[:best_i])
                    result["end"] = result["start"] + len(result["matched_text"])

        return result

    def _build_match_error(self, path: Path, content: str, search: str, match_info: dict) -> str:
        """Build detailed error message when no match is found."""
        lines = content.splitlines()
        parts = [f"Error: Search string not found in {path}"]

        best_ratio = match_info.get("best_ratio", 0.0)
        best_text = match_info.get("matched_text", "")

        # Show context around the best fuzzy candidate
        if best_ratio > 0.4 and best_text:
            # Find the line number where best match starts
            best_start = match_info.get("start", -1)
            if best_start >= 0:
                # Convert char offset to line number
                char_count = 0
                best_line_idx = 0
                for i, line in enumerate(lines):
                    if char_count >= best_start:
                        best_line_idx = i
                        break
                    char_count += len(line) + 1

                start = max(0, best_line_idx - 2)
                end = min(len(lines), best_line_idx + 3)
                context_lines = []
                for i in range(start, end):
                    marker = ">>>" if i == best_line_idx else "   "
                    context_lines.append(f"{marker} {i + 1:4d} | {lines[i]}")
                parts.append(f"\nClosest match (similarity: {best_ratio:.0%}):")
                parts.append("\n".join(context_lines))

                # Show diff between search and closest match
                diff_lines = list(difflib.unified_diff(
                    search.splitlines(),
                    best_text.splitlines(),
                    fromfile="search_string",
                    tofile="actual_content",
                    lineterm="",
                ))
                if diff_lines:
                    parts.append("\nDiff (search vs actual):\n" + "\n".join(diff_lines))

        # Show similar lines (full content, not truncated)
        search_fragment = search[:30].strip()
        if search_fragment:
            similar = [line for line in lines if search_fragment in line][:3]
            if similar:
                parts.append("\nLines containing '" + search_fragment + "':")
                for line in similar:
                    parts.append(f"  {line}")

        return "\n".join(parts)

    def _create_file(self, path: Path, content: str) -> str:
        """Create a new file."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            # Track created file for rollback
            if self.checkpoint:
                self.checkpoint.track_created_file(path)

            AtomicFileWriter.write(path, content)
            # For new files, show the content as all additions
            diff = self._generate_diff("", content, path)
            return f"✅ Created file: {path}\n\n{diff}" if diff else f"✅ Created file: {path}"
        except Exception as e:
            return f"Error creating file: {e}"

    def _search_replace(self, path: Path, search: str, replace: str) -> str:
        """Search and replace text in file with fuzzy matching fallback."""
        if not search:
            return "Error: search string is required"

        content_before = path.read_text()
        match = self._find_best_match(content_before, search)

        if not match["found"]:
            return self._build_match_error(path, content_before, search, match)

        matched_text = match["matched_text"]

        # For exact matches, check uniqueness
        if match["match_type"] == "exact":
            count = content_before.count(search)
            if count > 1:
                return (
                    f"Error: Search string found {count} times in {path}. "
                    f"Provide a more specific search string with additional context."
                )

        content_after = content_before.replace(matched_text, replace, 1)
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)

        if match["match_type"] == "exact":
            return f"✅ Replaced 1 occurrence in {path}\n\n{diff}"
        else:
            match_diff = self._generate_diff(search, match["matched_text"], path)
            return (
                f"✅ Replaced 1 occurrence in {path} "
                f"({match['match_type']}, similarity: {match['ratio']:.0%})\n\n"
                f"Match diff (searched vs found):\n{match_diff}\n\n"
                f"Applied changes:\n{diff}"
            )

    def _insert_after(self, path: Path, after: str, content: str) -> str:
        """Insert content after a matching line."""
        if not after:
            return "Error: 'after' string is required"

        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()

        matching = [i for i, line in enumerate(lines) if after in line]

        # Fuzzy fallback: find best matching line
        if not matching:
            best_idx, best_ratio = -1, 0.0
            for i, line in enumerate(lines):
                ratio = difflib.SequenceMatcher(None, after, line).ratio()
                if ratio > best_ratio:
                    best_idx, best_ratio = i, ratio
            if best_ratio >= 0.8:
                matching = [best_idx]
            else:
                # Build detailed error
                parts = [f"Error: Line containing '{after[:60]}' not found in {path}"]
                if best_ratio > 0.4 and best_idx >= 0:
                    parts.append(f"\nClosest match (line {best_idx + 1}, similarity: {best_ratio:.0%}):")
                    start = max(0, best_idx - 1)
                    end = min(len(lines), best_idx + 2)
                    for i in range(start, end):
                        marker = ">>>" if i == best_idx else "   "
                        parts.append(f"{marker} {i + 1:4d} | {lines[i]}")
                return "\n".join(parts)

        if len(matching) > 1:
            return (
                f"Error: '{after[:50]}' found on {len(matching)} lines. "
                f"Provide a more specific string to uniquely identify the target line."
            )

        i = matching[0]
        new_lines = content.splitlines()
        lines = lines[: i + 1] + new_lines + lines[i + 1 :]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Inserted {len(new_lines)} line(s) after line {i + 1} in {path}\n\n{diff}"

    def _insert_before(self, path: Path, before: str, content: str) -> str:
        """Insert content before a matching line."""
        if not before:
            return "Error: 'before' string is required"

        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()

        matching = [i for i, line in enumerate(lines) if before in line]

        # Fuzzy fallback: find best matching line
        if not matching:
            best_idx, best_ratio = -1, 0.0
            for i, line in enumerate(lines):
                ratio = difflib.SequenceMatcher(None, before, line).ratio()
                if ratio > best_ratio:
                    best_idx, best_ratio = i, ratio
            if best_ratio >= 0.8:
                matching = [best_idx]
            else:
                parts = [f"Error: Line containing '{before[:60]}' not found in {path}"]
                if best_ratio > 0.4 and best_idx >= 0:
                    parts.append(f"\nClosest match (line {best_idx + 1}, similarity: {best_ratio:.0%}):")
                    start = max(0, best_idx - 1)
                    end = min(len(lines), best_idx + 2)
                    for i in range(start, end):
                        marker = ">>>" if i == best_idx else "   "
                        parts.append(f"{marker} {i + 1:4d} | {lines[i]}")
                return "\n".join(parts)

        if len(matching) > 1:
            return (
                f"Error: '{before[:50]}' found on {len(matching)} lines. "
                f"Provide a more specific string to uniquely identify the target line."
            )

        i = matching[0]
        new_lines = content.splitlines()
        lines = lines[:i] + new_lines + lines[i:]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Inserted {len(new_lines)} line(s) before line {i + 1} in {path}\n\n{diff}"

    def _replace_lines(self, path: Path, start: int, end: int, content: str) -> str:
        """Replace a range of lines."""
        content_before = path.read_text()
        had_trailing_newline = content_before.endswith("\n")
        lines = content_before.splitlines()
        total = len(lines)

        if start < 1 or start > total:
            return f"Error: startLine {start} out of range (1-{total})"
        if end < start or end > total:
            return f"Error: endLine {end} invalid (must be {start}-{total})"

        new_lines = content.splitlines() if content else []
        lines = lines[: start - 1] + new_lines + lines[end:]
        content_after = "\n".join(lines)
        if had_trailing_newline:
            content_after += "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Replaced lines {start}-{end} with {len(new_lines)} line(s) in {path}\n\n{diff}"

    def _append(self, path: Path, content: str) -> str:
        """Append content to end of file."""
        content_before = path.read_text()
        if not content_before.endswith("\n"):
            content_before_normalized = content_before + "\n"
        else:
            content_before_normalized = content_before
        content_after = content_before_normalized + content + "\n"
        self._safe_write(path, content_after)

        diff = self._generate_diff(content_before, content_after, path)
        return f"✅ Appended to {path}\n\n{diff}"
