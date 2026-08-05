"""Tests for the recall retrieval tool (epoch-memory-arch #3).

The recall tool searches the JSONL session log (current and past sessions) and
can recover the full text of large tool outputs that were compacted out of the
context window. The JSONL log is host-owned (always readable); the offloaded
tool-output files live under ``.supercoder/tool-outputs/`` in the repo and are
trust-gated like sessions (R2-7-class prompt-injection vector).
"""

import json
from pathlib import Path

import pytest

from supercoder import logging as logging_mod
from supercoder.tools.recall import RecallTool


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point the logging module at a temp dir and reset the global logger."""
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)
    # Reset module-global logger so get_logger() picks up a fresh one.
    monkeypatch.setattr(logging_mod, "_logger", None)
    return tmp_path


def _write_session(log_dir: Path, name: str, entries: list[dict]) -> Path:
    """Write a fake session JSONL file with the given entries."""
    p = log_dir / name
    p.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return p


class TestRecallJSONLSearch:
    """Searching the JSONL log by content/type/time.

    These tests use session=all because they run without an active in-process
    logger; the current-session resolver relies on the live logger singleton.
    session=all scans files on disk directly, which is what we exercise here.
    """

    def test_finds_matching_tool_result(self, log_dir):
        _write_session(
            log_dir,
            "session_20260805_080000.jsonl",
            [
                {
                    "type": "tool_call",
                    "tool": "command-exec",
                    "arguments": "pytest -x",
                    "timestamp": "2026-08-05T08:00:01",
                },
                {
                    "type": "tool_result",
                    "tool": "command-exec",
                    "result": "3 passed in 1.2s",
                    "timestamp": "2026-08-05T08:00:02",
                },
                {"type": "error", "error": "unrelated crash", "timestamp": "2026-08-05T08:00:03"},
            ],
        )
        tool = RecallTool(log_dir=log_dir)
        out = tool.execute(json.dumps({"query": "passed", "session": "all"}))

        assert "3 passed in 1.2s" in out
        assert "command-exec" in out
        # Unrelated error should not appear when querying for "passed"
        assert "unrelated crash" not in out

    def test_filter_by_type(self, log_dir):
        _write_session(
            log_dir,
            "session_20260805_080000.jsonl",
            [
                {
                    "type": "tool_result",
                    "tool": "x",
                    "result": "pytest passed",
                    "timestamp": "2026-08-05T08:00:02",
                },
                {"type": "error", "error": "pytest failure", "timestamp": "2026-08-05T08:00:03"},
            ],
        )
        tool = RecallTool(log_dir=log_dir)

        errors_only = tool.execute(
            json.dumps({"query": "pytest", "type": "error", "session": "all"})
        )
        assert "pytest failure" in errors_only
        assert "pytest passed" not in errors_only

    def test_limit_respected(self, log_dir):
        entries = [
            {
                "type": "tool_result",
                "tool": "x",
                "result": f"pytest match {i}",
                "timestamp": f"2026-08-05T08:00:0{i}",
            }
            for i in range(10)
        ]
        _write_session(log_dir, "session_20260805_080000.jsonl", entries)

        tool = RecallTool(log_dir=log_dir)
        out = tool.execute(json.dumps({"query": "pytest match", "limit": 3, "session": "all"}))

        # Three result entries rendered (each "pytest match N" is one line)
        assert out.count("pytest match") == 3

    def test_session_all_searches_multiple_files(self, log_dir):
        _write_session(
            log_dir,
            "session_20260805_070000.jsonl",
            [
                {
                    "type": "tool_result",
                    "tool": "x",
                    "result": "old pytest run",
                    "timestamp": "2026-08-05T07:00:00",
                }
            ],
        )
        _write_session(
            log_dir,
            "session_20260805_090000.jsonl",
            [
                {
                    "type": "tool_result",
                    "tool": "x",
                    "result": "new pytest run",
                    "timestamp": "2026-08-05T09:00:00",
                }
            ],
        )
        tool = RecallTool(log_dir=log_dir)
        out = tool.execute(json.dumps({"query": "pytest", "session": "all"}))

        assert "old pytest run" in out
        assert "new pytest run" in out

    def test_no_matches_returns_helpful_message(self, log_dir):
        _write_session(
            log_dir,
            "session_20260805_080000.jsonl",
            [
                {
                    "type": "tool_result",
                    "tool": "x",
                    "result": "nothing relevant",
                    "timestamp": "2026-08-05T08:00:00",
                }
            ],
        )
        tool = RecallTool(log_dir=log_dir)
        out = tool.execute(json.dumps({"query": "nonexistent_query_xyz", "session": "all"}))

        assert "No matching events" in out or "no matches" in out.lower()


class TestRecallDisabledLog:
    """Graceful handling when logging is disabled."""

    def test_disabled_log_informs_user(self, tmp_path, monkeypatch):
        # A fresh RecallTool with an empty log dir and no active logger
        monkeypatch.setattr(logging_mod, "_logger", None)
        tool = RecallTool(log_dir=tmp_path)
        out = tool.execute(json.dumps({"query": "anything"}))

        assert "disabled" in out.lower() or "no log" in out.lower()


class TestRecallOffloadRead:
    """Reading full offloaded tool outputs (trust-gated, repo-local)."""

    def test_read_offload_when_trusted(self, tmp_path):
        # offload file under repo .supercoder/tool-outputs/
        offload_dir = tmp_path / ".supercoder" / "tool-outputs"
        offload_dir.mkdir(parents=True)
        offload_file = offload_dir / "20260805-080000-command-exec-abc.txt"
        offload_file.write_text("FULL OUTPUT LINE 1\nFULL OUTPUT LINE 2\n")

        tool = RecallTool(allowed_root=tmp_path, allow_offload_read=True)
        out = tool.execute(json.dumps({"offload": str(offload_file.relative_to(tmp_path))}))

        assert "FULL OUTPUT LINE 1" in out
        assert "FULL OUTPUT LINE 2" in out

    def test_refuse_offload_when_untrusted(self, tmp_path):
        offload_dir = tmp_path / ".supercoder" / "tool-outputs"
        offload_dir.mkdir(parents=True)
        offload_file = offload_dir / "20260805-080000-command-exec-abc.txt"
        offload_file.write_text("INJECTED PROMPT PAYLOAD")

        tool = RecallTool(allowed_root=tmp_path, allow_offload_read=False)
        out = tool.execute(json.dumps({"offload": str(offload_file.relative_to(tmp_path))}))

        assert "FULL OUTPUT LINE 1" not in out
        assert "trusting" in out.lower() or "trust" in out.lower()
        # The planted payload must NOT be returned when untrusted
        assert "INJECTED PROMPT PAYLOAD" not in out

    def test_offload_escape_refused(self, tmp_path):
        # Even when trusted, an offload path escaping the repo is rejected
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("SECRET")

        tool = RecallTool(allowed_root=tmp_path, allow_offload_read=True)
        out = tool.execute(json.dumps({"offload": str(outside)}))

        assert "SECRET" not in out
        assert "outside" in out.lower() or "error" in out.lower()

    def test_no_trust_store_attribute(self):
        # Regression guard: the trust_store parameter/attribute was removed as
        # dead code (never injected, never read; allow_offload_read covers the
        # gate). Ensure it does not silently come back.
        tool = RecallTool()
        assert not hasattr(tool, "trust_store")

    def test_offload_read_logs_permission_denial(self, tmp_path, monkeypatch):
        """When permission_policy denies an offload read, log_permission_decision
        must be called (mirrors the file-read precedent)."""
        from supercoder import logging as logging_mod
        from supercoder.permissions import PermissionPolicy

        offload_dir = tmp_path / ".supercoder" / "tool-outputs"
        offload_dir.mkdir(parents=True)
        offload_file = offload_dir / "20260805-080000-command-exec-abc.txt"
        offload_file.write_text("SECRET CONTENT")

        # Deny the offload directory via a configured path rule.
        policy = PermissionPolicy(tmp_path, {"paths": {"deny": [".supercoder/tool-outputs/*"]}})
        tool = RecallTool(allowed_root=tmp_path, permission_policy=policy, allow_offload_read=True)

        calls = []
        fake_logger = type(
            "FakeLogger",
            (),
            {
                "enabled": True,
                "log_permission_decision": lambda self_, **kw: calls.append(kw),
            },
        )()
        monkeypatch.setattr(logging_mod, "get_logger", lambda: fake_logger)
        monkeypatch.setattr("supercoder.tools.recall.get_logger", lambda: fake_logger)

        out = tool.execute(json.dumps({"offload": str(offload_file.relative_to(tmp_path))}))

        # Content was not returned (denied)...
        assert "SECRET CONTENT" not in out
        # ...and the denial was logged exactly once with the recall tool name.
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "recall"


class TestRecallParseError:
    """Malformed arguments return a clean error, not a traceback."""

    def test_invalid_json(self, log_dir):
        tool = RecallTool(log_dir=log_dir)
        out = tool.execute("not valid json {")
        assert "Error" in out
