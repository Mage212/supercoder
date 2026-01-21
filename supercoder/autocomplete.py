"""Enhanced autocomplete for SuperCoder CLI.

Provides intelligent completion for:
- Slash commands (/help, /code, etc.)
- File paths from repository
- Code symbols (optional tokenization)
"""

import os
from pathlib import Path
from collections import defaultdict

from prompt_toolkit.completion import Completer, Completion


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
        for rel_fname in self.rel_fnames:
            basename = os.path.basename(rel_fname)
            if basename != rel_fname:
                self.fname_to_paths[basename].append(rel_fname)
        
        # Collect all completable words
        self.words = set(self.rel_fnames)
        self.tokenized = False
        
    def _scan_repo_files(self):
        """Lazily scan repository for files if not already done."""
        if self.rel_fnames:
            return
            
        try:
            for root, dirs, files in os.walk(self.repo_root):
                # Skip hidden and common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') 
                          and d not in ('node_modules', '__pycache__', 'venv', '.git')]
                
                for f in files:
                    if not f.startswith('.'):
                        full_path = Path(root) / f
                        try:
                            rel_path = full_path.relative_to(self.repo_root)
                            self.rel_fnames.append(str(rel_path))
                            self.words.add(str(rel_path))
                            
                            # Also add basename for quick access
                            self.fname_to_paths[f].append(str(rel_path))
                        except ValueError:
                            pass
        except Exception:
            pass  # Gracefully handle permission errors etc.
    
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
        if '/' in last_word or last_word.startswith('.'):
            yield from self._complete_files(last_word)
            return
        
        # Symbol completion: 3+ characters
        if len(last_word) >= 3:
            yield from self._complete_words(last_word)
    
    def _complete_commands(self, text, words):
        """Complete slash commands."""
        if len(words) == 1 and not text.endswith(' '):
            partial = words[0].lower()
            for cmd in self.commands:
                if cmd.lower().startswith(partial):
                    yield Completion(cmd, start_position=-len(partial))
    
    def _complete_files(self, partial):
        """Complete file paths from repository."""
        self._scan_repo_files()
        
        partial_lower = partial.lower()
        completions = []
        
        for rel_fname in self.rel_fnames:
            # Match anywhere in path
            if partial_lower in rel_fname.lower():
                completions.append((rel_fname, rel_fname))
                
        # Also check basenames
        for basename, paths in self.fname_to_paths.items():
            if partial_lower in basename.lower():
                for path in paths:
                    if (path, path) not in completions:
                        completions.append((path, f"{basename} ({path})"))
        
        # Sort and yield
        for path, display in sorted(completions, key=lambda x: x[0]):
            yield Completion(path, start_position=-len(partial), display=display)
    
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
        
        if not text.startswith('/'):
            return
            
        partial = text.lower()
        for cmd in self.commands:
            if cmd.lower().startswith(partial):
                yield Completion(cmd, start_position=-len(text))
