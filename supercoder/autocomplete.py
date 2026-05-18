"""Enhanced autocomplete for SuperCoder CLI.

Provides intelligent completion for:
- Slash commands (/help, /code, etc.)
- File paths from repository
- Code symbols (optional tokenization)
"""

import os
from collections import defaultdict
from pathlib import Path

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.completion import Completer, Completion

from .tools.tool_utils import is_ignored_path, relative_display_path


class SlashCommandAutoSuggest(AutoSuggest):
    """Inline gray-text auto-suggestion for slash commands."""

    def __init__(self, commands):
        self.commands = sorted(commands)

    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return None
        for cmd in self.commands:
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text) :])
        return None


class AutoCompleter(Completer):
    """Intelligent autocompleter for files, commands, and symbols.

    Provides context-aware completion:
    - Commands starting with /
    - File paths starting with ./ or containing /
    - Code symbols after 3+ characters
    """

    def __init__(self, repo_root, commands, rel_fnames=None, encoding="utf-8"):
        """Initialize the autocompleter.

        Args:
            repo_root: Root path of the repository
            commands: List of slash commands (e.g., ['/help', '/exit'])
            rel_fnames: List of relative file paths in the repo
            encoding: File encoding for reading files
        """
        self.repo_root = Path(repo_root) if repo_root else Path(".")
        self.commands = sorted(commands)
        self.rel_fnames = rel_fnames or []
        self.encoding = encoding

        # Build filename -> full path mapping for basename completion
        self.fname_to_paths = defaultdict(list)
        self.path_entries: list[tuple[str, bool]] = []
        for rel_fname in self.rel_fnames:
            basename = os.path.basename(rel_fname)
            if basename != rel_fname:
                self.fname_to_paths[basename].append(rel_fname)
            self.path_entries.append((rel_fname, False))

        # Collect all completable words
        self.words = set(self.rel_fnames)
        self.tokenized = False

    def _scan_repo_files(self):
        """Lazily scan repository for files if not already done."""
        if self.path_entries and self.rel_fnames:
            return

        seen_entries = set(self.path_entries)
        for root, dirs, files in os.walk(self.repo_root):
            root_path = Path(root)
            kept_dirs = []
            for dirname in dirs:
                dir_path = root_path / dirname
                if is_ignored_path(dir_path, self.repo_root):
                    continue
                rel_dir = relative_display_path(dir_path, self.repo_root) + "/"
                entry = (rel_dir, True)
                if entry not in seen_entries:
                    self.path_entries.append(entry)
                    seen_entries.add(entry)
                    self.words.add(rel_dir)
                kept_dirs.append(dirname)
            dirs[:] = kept_dirs

            for filename in files:
                full_path = root_path / filename
                if is_ignored_path(full_path, self.repo_root):
                    continue
                rel_path = relative_display_path(full_path, self.repo_root)
                entry = (rel_path, False)
                if entry in seen_entries:
                    continue
                self.rel_fnames.append(rel_path)
                self.words.add(rel_path)
                self.path_entries.append(entry)
                seen_entries.add(entry)
                self.fname_to_paths[filename].append(rel_path)

    def get_completions(self, document, complete_event):
        """Get completions for current input.

        Args:
            document: prompt_toolkit Document object
            complete_event: Complete event

        Yields:
            Completion objects
        """
        text = document.text_before_cursor
        words = text.split()

        if not words:
            return

        # Don't complete after a space (user hasn't started typing next word)
        if text and text[-1].isspace():
            return

        # Command completion: starts with /
        if text.lstrip().startswith("/"):
            yield from self._complete_commands(text, words)
            return

        # File completion: has path-like characters
        last_word = words[-1] if words else ""
        context_ref = self._context_ref_partial(last_word)
        if context_ref is not None:
            partial, replace_len = context_ref
            yield from self._complete_context_refs(partial, replace_len)
            return

        if "/" in last_word or last_word.startswith("."):
            yield from self._complete_files(last_word)
            return

        # Symbol/path word completion: only on explicit Tab completion to avoid
        # noisy popups while typing normal prose.
        if len(last_word) >= 3 and getattr(complete_event, "completion_requested", False):
            yield from self._complete_words(last_word)

    def _complete_commands(self, text, words):
        """Complete slash commands."""
        if len(words) == 1 and not text.endswith(" "):
            partial = words[0].lower()
            for cmd in self.commands:
                if cmd.lower().startswith(partial):
                    yield Completion(cmd, start_position=-len(partial))

    def _complete_files(self, partial):
        """Complete file paths from repository."""
        self._scan_repo_files()

        partial_lower = partial.lower()
        completions = []

        for rel_fname, is_dir in self.path_entries:
            # Match anywhere in path
            if partial_lower in rel_fname.lower():
                display = f"{rel_fname} (dir)" if is_dir else rel_fname
                completions.append((rel_fname, display))

        # Also check basenames
        for basename, paths in self.fname_to_paths.items():
            if partial_lower in basename.lower():
                for path in paths:
                    if (path, path) not in completions:
                        completions.append((path, f"{basename} ({path})"))

        # Sort and yield
        for path, display in sorted(completions, key=lambda x: x[0]):
            yield Completion(path, start_position=-len(partial), display=display)

    def _context_ref_partial(self, token: str) -> tuple[str, int] | None:
        """Return partial @path token and replacement length for context refs."""
        at_index = token.rfind("@")
        if at_index < 0:
            return None
        if at_index > 0 and token[at_index - 1] not in "([{<,;:":
            return None
        partial = token[at_index + 1 :]
        return partial, len(token) - at_index

    def _complete_context_refs(self, partial: str, replace_len: int):
        """Complete @path references from repository files and directories."""
        self._scan_repo_files()

        partial_lower = partial.lower()
        ranked: list[tuple[tuple[int, int, int, str], str, str]] = []
        for path, is_dir in self.path_entries:
            rank = self._rank_context_ref(path, partial_lower)
            if rank is None:
                continue
            display = f"{path} (dir)" if is_dir else path
            ranked.append((rank, path, display))

        for _rank, path, display in sorted(ranked, key=lambda item: item[0])[:50]:
            yield Completion(f"@{path}", start_position=-replace_len, display=display)

    def _rank_context_ref(self, path: str, partial_lower: str) -> tuple[int, int, int, str] | None:
        """Rank @path completion candidates by basename and path match quality."""
        normalized = path[:-1] if path.endswith("/") else path
        basename = os.path.basename(normalized)
        path_lower = path.lower()
        basename_lower = basename.lower()

        if not partial_lower:
            return (4, 0, len(path), path)
        if basename_lower.startswith(partial_lower):
            return (0, 0, len(basename), path)
        basename_index = basename_lower.find(partial_lower)
        if basename_index >= 0:
            return (1, basename_index, len(basename), path)
        if path_lower.startswith(partial_lower):
            return (2, 0, len(path), path)
        path_index = path_lower.find(partial_lower)
        if path_index >= 0:
            return (3, path_index, len(path), path)
        return None

    def _complete_words(self, partial):
        """Complete from collected words (files, symbols)."""
        self._scan_repo_files()

        partial_lower = partial.lower()

        for word in sorted(self.words):
            if word.lower().startswith(partial_lower):
                yield Completion(word, start_position=-len(partial))


class CommandCompleter(Completer):
    """Simple completer for just slash commands.

    Lightweight alternative when you only need command completion.
    """

    def __init__(self, commands):
        """Initialize with list of commands."""
        self.commands = sorted(commands)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor.strip()

        if not text.startswith("/"):
            return

        partial = text.lower()
        for cmd in self.commands:
            if cmd.lower().startswith(partial):
                yield Completion(cmd, start_position=-len(text))
