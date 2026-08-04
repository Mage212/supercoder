"""Secret scrubbing for persistent storage (logs, sessions, tool-output dumps).

Applies regex-only redaction of known secret formats before content is written
to disk, so a tool reading ``~/.aws/credentials`` or an exception echoing an
API key cannot leak into ``~/.supercoder/logs/`` or ``.supercoder/sessions/``.

The scrubber is intentionally conservative: only well-known secret shapes are
matched (provider-prefixed tokens, PEM blocks, Bearer headers, and explicit
``key=value`` assignments). High-entropy detection is deliberately avoided to
keep false positives low — a masked commit/PR description is worse than a
missed random token.
"""

from __future__ import annotations

import re
from typing import Any

MASK = "[REDACTED]"

# Compiled once at module load. Order matters only for readability — every
# pattern is applied independently via re.sub on the whole string.
_PATTERNS: list[re.Pattern[str]] = [
    # OpenAI / Anthropic keys. Covers the legacy `sk-<alnum>` shape and the
    # modern `sk-proj-...` / `sk-ant-api03-...` formats (default since 2024),
    # which embed hyphens and underscores after the provider segment. The tail
    # still requires >= 20 chars so short identifiers like `sk-learn` are safe.
    re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-or-v1-[A-Za-z0-9-]+"),  # OpenRouter
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub PAT
    re.compile(r"gho_[A-Za-z0-9]{36}"),  # GitHub OAuth
    re.compile(r"github_pat_[A-Za-z0-9_]{82}"),  # GitHub fine-grained
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    # AWS secret access key (40 chars, base64 incl. /+=) — anchored on key name
    re.compile(
        r"(?i)(?:aws_secret_access_key|secret_access_key)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9/+=]{20,}[\"']?"
    ),
    re.compile(r"xoxb-[0-9]+-[0-9]+-[A-Za-z0-9]+"),  # Slack bot token
    # PEM private key block (multi-line, non-greedy)
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----"),
    # Bearer authorization header
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    # Generic key=value assignment (api_key, secret, token, password, ...).
    # Requires mixed letters+digits AND length >= 10 to look like a real
    # generated key. This avoids masking ordinary source code (function-call
    # RHS like get_token_from_request, all-letter placeholders, short literals
    # such as "abcdef12"/"changeme") per the module docstring: a masked
    # commit/PR description is worse than a missed random token. Format-specific
    # regexes above handle exact shapes (OpenAI, GitHub, AWS, PEM, Bearer).
    re.compile(
        r"(?i)(?:api_key|apikey|secret|token|password|passwd|pwd)\s*[:=]\s*[\"']?"
        r"(?=[A-Za-z0-9_\-]{10,})(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9_\-]{10,}"
        r"[\"']?"
    ),
]

# Recursion backstop — message trees are shallow (<5 levels) but guard anyway.
_MAX_DEPTH = 10


def _scrub_str(text: str) -> str:
    scrubbed = text
    for pattern in _PATTERNS:
        scrubbed = pattern.sub(MASK, scrubbed)
    return scrubbed


def scrub_secrets(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact known secret patterns from strings within ``value``.

    - ``str`` → patterns applied, matches replaced with ``[REDACTED]``.
    - ``dict`` → new dict with scrubbed values (keys are preserved — they are
      field names, not secrets, and keeping them readable aids debugging).
    - ``list`` / ``tuple`` → new sequence with scrubbed items.
    - any other type (``int``, ``bool``, ``None``, ``float``) → returned
      unchanged (secrets are always strings).
    """
    if _depth > _MAX_DEPTH:
        return value
    if isinstance(value, str):
        return _scrub_str(value)
    if isinstance(value, dict):
        return {k: scrub_secrets(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_secrets(v, _depth=_depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub_secrets(v, _depth=_depth + 1) for v in value)
    return value
