"""Tests for the command execution tool (env sanitization, M3)."""

import json

from supercoder.tools.command_exec import CommandExecutionTool

_SECRET_ENV_KEYS = (
    "OPENAI_API_KEY",
    "SUPERCODER_API_KEY",
    "ANTHROPIC_API_KEY",
    "MY_SERVICE_TOKEN",
    "DATABASE_PASSWORD",
)


def _run(tool: CommandExecutionTool, command: str) -> str:
    """Run a command via execute_streaming and collect the textual output."""
    args = json.dumps({"command": command, "timeout": 10})
    parts = []
    for event in tool.execute_streaming(args):
        if event["type"] in {"output", "done"}:
            parts.append(event["content"])
        elif event["type"] == "error":
            parts.append(event["content"])
    return "\n".join(parts)


class TestEnvSanitization:
    """M3: spawned commands must not inherit API keys / secret env vars.

    CommandExecutionTool spawned subprocess with no ``env=`` override, so every
    model-executed command inherited ``SUPERCODER_API_KEY`` / ``OPENAI_API_KEY``
    etc. A command as innocuous as ``env`` or ``printenv`` would dump live
    credentials into tool output, which flows into model context and persistent
    storage (where scrubber gaps then apply). The child env must strip known
    secret variables while preserving PATH/HOME/etc.
    """

    def test_printenv_does_not_leak_secret_env(self, monkeypatch):
        for key in _SECRET_ENV_KEYS:
            monkeypatch.setenv(key, f"leak-{key}-value")

        tool = CommandExecutionTool()
        output = _run(tool, "printenv")

        for key in _SECRET_ENV_KEYS:
            value = f"leak-{key}-value"
            assert value not in output, f"secret env var {key}={value!r} leaked into command output"

    def test_non_secret_env_preserved(self, monkeypatch):
        """PATH/HOME must still be available to the child process."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-leak-1234567890")
        monkeypatch.setenv("MY_TEST_NONSECRET", "keepme")

        tool = CommandExecutionTool()
        output = _run(tool, "printenv MY_TEST_NONSECRET")

        assert "keepme" in output
        assert "sk-leak" not in output

    def test_direct_echo_of_secret_not_leaked_via_env(self, monkeypatch):
        """Even ``echo $VAR`` style must not expand because the var is absent."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-leak1234567890abc")

        tool = CommandExecutionTool()
        output = _run(tool, 'echo "$OPENAI_API_KEY"')

        assert "sk-proj-leak" not in output

    def test_key_suffix_without_api_stripped(self, monkeypatch):
        """R2-5: *_KEY without _API_ (STRIPE_SECRET_KEY, SIGNING_KEY) must be
        stripped — these are common secret-naming conventions that previously
        leaked into child env."""
        for key in ("STRIPE_SECRET_KEY", "SIGNING_KEY", "ENCRYPTION_KEY"):
            monkeypatch.setenv(key, f"leak-{key}-value")

        tool = CommandExecutionTool()
        output = _run(tool, "printenv")

        for key in ("STRIPE_SECRET_KEY", "SIGNING_KEY", "ENCRYPTION_KEY"):
            value = f"leak-{key}-value"
            assert value not in output, f"{key}={value!r} leaked into child env"
