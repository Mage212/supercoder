"""Per-repository trust store for untrusted local configuration (C1/C2).

SuperCoder loads two project-local files automatically:

- ``.supercoder.yaml`` in the working directory (merged over the global config)
- ``.supercoder/permissions.yaml`` under the repo root (persistent command rules)

Both files live inside the repository, so a cloned malicious repo can plant
either of them. To avoid silently honoring endpoint/credential/command-rule
overrides from an untrusted repo, the host asks the user once per repo whether
to trust it, and records the answer here.

The store is a plain newline-delimited list of resolved absolute paths under
``~/.supercoder/trusted-repos``. Paths are normalized with ``Path.resolve()`` so
symlinks and ``..`` collapse before comparison.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from .config import CONFIG_DIR
from .utils.atomic_writer import AtomicFileWriter

TRUST_STORE_FILE = CONFIG_DIR / "trusted-repos"


def _normalize(repo_path: str | Path) -> str:
    """Return the canonical absolute string for a repo path."""
    return str(Path(repo_path).resolve())


class RepoTrustStore:
    """Track which repository paths the user has trusted for local config.

    The store is a single file of resolved absolute paths, one per line. Empty
    lines and comments (``#``) are ignored so the file stays human-editable.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else TRUST_STORE_FILE

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return set()
        return {
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

    def is_trusted(self, repo_path: str | Path) -> bool:
        """Return True iff ``repo_path`` has been explicitly trusted."""
        return _normalize(repo_path) in self._load()

    def trust(self, repo_path: str | Path) -> None:
        """Mark ``repo_path`` as trusted (idempotent)."""
        entries = self._load()
        entries.add(_normalize(repo_path))
        self._save(sorted(entries))

    def untrust(self, repo_path: str | Path) -> bool:
        """Remove ``repo_path`` from the trust store. Return True if it was present."""
        entries = self._load()
        target = _normalize(repo_path)
        if target not in entries:
            return False
        entries.discard(target)
        self._save(sorted(entries))
        return True

    def _save(self, entries: list[str]) -> None:
        # Ensure the config directory exists and is owner-only (it may hold the
        # API key config too). Mirrors config.ensure_config_dir().
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.path.parent, 0o700)
        content = (
            "# Trusted repositories — local .supercoder.yaml / permissions.yaml honored here\n"
        )
        content += "\n".join(entries) + ("\n" if entries else "")
        AtomicFileWriter.write(self.path, content)


# Fields in .supercoder.yaml that change WHERE credentials are sent or WHAT the
# agent is allowed to do. These are the ones that require explicit trust; safe
# tuning fields (temperature, max_context_tokens, loop_detection, ...) are
# always honored from local config.
SENSITIVE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "base_url",
        "endpoint",
        "model",
        "models",
        "default_model",
        "permissions",
        "streaming",
    }
)


def local_config_has_sensitive_fields(file_data: dict) -> bool:
    """Return True if a parsed local .supercoder.yaml touches sensitive keys."""
    if not isinstance(file_data, dict):
        return False
    return any(key in file_data for key in SENSITIVE_CONFIG_KEYS)


def filter_sensitive_config(file_data: dict) -> dict:
    """Return a copy of ``file_data`` with sensitive keys removed.

    Used when a local .supercoder.yaml is present but the repo is not trusted:
    the safe tuning fields are still honored, but endpoint/credential/command
    overrides are dropped.
    """
    if not isinstance(file_data, dict):
        return {}
    return {k: v for k, v in file_data.items() if k not in SENSITIVE_CONFIG_KEYS}
