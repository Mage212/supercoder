"""Tests for privacy-safe debug logging defaults."""

import json

from supercoder import logging as logging_mod
from supercoder.llm.base import Message


def test_logger_disabled_does_not_write_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=False)
    logger.log_user_input("secret user prompt")
    logger.log_system_prompt("secret system prompt")

    assert list(tmp_path.glob("*.jsonl")) == []


def test_logger_enabled_writes_debug_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_user_input("debug user prompt")

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    assert "debug user prompt" in files[0].read_text()


def test_log_messages_preserves_native_tool_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "file-read", "arguments": '{"fileName":"x.py"}'},
                    }
                ],
            ),
            Message(
                role="tool",
                content="result",
                tool_call_id="call_1",
                name="file-read",
            ),
        ]
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    api_request = next(entry for entry in entries if entry["type"] == "api_request")

    assert api_request["messages"][0]["tool_calls"][0]["id"] == "call_1"
    assert api_request["messages"][1]["tool_call_id"] == "call_1"
    assert api_request["messages"][1]["name"] == "file-read"


def test_log_tool_output_masked_event(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_tool_output_masked(
        "file-read",
        "call_1",
        masked=True,
        original_chars=12000,
        model_chars=5200,
        offload_path=".supercoder/tool-outputs/output.txt",
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "tool_output_masked")

    assert event["tool"] == "file-read"
    assert event["tool_call_id"] == "call_1"
    assert event["masked"] is True
    assert event["original_chars"] == 12000
    assert event["model_chars"] == 5200
    assert event["offload_path"] == ".supercoder/tool-outputs/output.txt"


def test_log_context_attachment_event(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_context_attachment(
        {
            "files": 1,
            "directories": 0,
            "skipped": 1,
            "model_bytes": 1200,
            "estimated_tokens": 300,
            "items": [{"ref": "main.py", "status": "attached"}],
        }
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "context_attachment")

    assert event["summary"]["files"] == 1
    assert event["summary"]["skipped"] == 1
    assert "content" not in event


def test_log_freshness_check_event_without_file_content_or_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_freshness_check(
        path="main.py",
        source="file-read",
        action="mark_read",
        status="recorded",
        reason="file content exposed to model",
        size=42,
        hash_present=True,
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "freshness_check")

    assert event["path"] == "main.py"
    assert event["source"] == "file-read"
    assert event["status"] == "recorded"
    assert event["size"] == 42
    assert event["hash_present"] is True
    assert "content" not in event
    assert "sha256" not in event


def test_log_mode_policy_event(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_mode_policy(
        mode="plan",
        tool_name="code-edit",
        action="deny",
        reason="PLAN mode cannot edit project files.",
        subject="src/app.py",
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "mode_policy")

    assert event["mode"] == "plan"
    assert event["tool"] == "code-edit"
    assert event["action"] == "deny"
    assert event["subject"] == "src/app.py"


def test_log_permission_rule_change_event(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_permission_rule_change(
        action="add",
        scope="persistent",
        rule_action="allow",
        rule="printf ok",
        source="command_confirm",
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "permission_rule_change")

    assert event["action"] == "add"
    assert event["scope"] == "persistent"
    assert event["rule_action"] == "allow"
    assert event["rule"] == "printf ok"
    assert event["source"] == "command_confirm"


def test_log_edit_confirmation_event_without_file_content(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_mod, "LOG_DIR", tmp_path)

    logger = logging_mod.ConversationLogger("model", enabled=True)
    logger.log_edit_confirmation(
        mode="code",
        filepath="main.py",
        operation="create",
        approved=True,
    )

    entries = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    event = next(entry for entry in entries if entry["type"] == "edit_confirm")

    assert event["mode"] == "code"
    assert event["filepath"] == "main.py"
    assert event["operation"] == "create"
    assert event["approved"] is True
    assert "content" not in event
