"""Tests for privacy-safe debug logging defaults."""

from supercoder import logging as logging_mod


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
