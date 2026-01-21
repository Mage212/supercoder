"""Conversation logging for debugging and analysis."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Log directory
LOG_DIR = Path.home() / ".supercoder" / "logs"


def ensure_log_dir() -> Path:
    """Create logs directory if it doesn't exist."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


class ConversationLogger:
    """Logs user inputs and model responses to files for debugging."""
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = ensure_log_dir() / f"session_{self.session_id}.jsonl"
        self.enabled = True
        
        # Write session header
        self._write_entry({
            "type": "session_start",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
        })
    
    def set_model(self, model_name: str) -> None:
        """Update the current model name (e.g., after switching)."""
        self.model_name = model_name
        self._write_entry({
            "type": "model_switch",
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_user_input(self, message: str) -> None:
        """Log user input."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_model_response(self, response: str, model: Optional[str] = None) -> None:
        """Log model response."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "assistant",
            "model": model or self.model_name,
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_reasoning(self, reasoning: str, stage: str = "") -> None:
        """Log reasoning/thinking content from model."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "reasoning",
            "stage": stage,
            "content": reasoning,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_stream_event(self, event_type: str, content: str, meta: dict = None) -> None:
        """Log individual streaming event for debugging."""
        if not self.enabled:
            return
        entry = {
            "type": "stream_event",
            "event_type": event_type,
            "content": content[:500] if content else "",  # Truncate
            "timestamp": datetime.now().isoformat(),
        }
        if meta:
            entry["meta"] = meta
        self._write_entry(entry)
    
    def log_system_prompt(self, prompt: str) -> None:
        """Log the current system prompt."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "system_prompt",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_messages(self, messages: list) -> None:
        """Log the full list of messages sent to the API."""
        if not self.enabled:
            return
        # Convert Message objects to dicts if needed
        serializable_messages = []
        for msg in messages:
            if hasattr(msg, "to_dict"):
                serializable_messages.append(msg.to_dict())
            else:
                serializable_messages.append({
                    "role": getattr(msg, "role", "unknown"),
                    "content": getattr(msg, "content", "")
                })
                
        self._write_entry({
            "type": "api_request",
            "messages": serializable_messages,
            "timestamp": datetime.now().isoformat(),
        })

    
    def log_tool_call(self, tool_name: str, arguments: str) -> None:
        """Log tool call."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "tool_call",
            "tool": tool_name,
            "arguments": arguments,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_tool_result(self, tool_name: str, result: str) -> None:
        """Log tool result."""
        if not self.enabled:
            return
        # Truncate long results
        truncated = result[:2000] + "..." if len(result) > 2000 else result
        self._write_entry({
            "type": "tool_result",
            "tool": tool_name,
            "result": truncated,
            "timestamp": datetime.now().isoformat(),
        })
    
    def log_error(self, error: str) -> None:
        """Log error."""
        if not self.enabled:
            return
        self._write_entry({
            "type": "error",
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
    
    def _write_entry(self, entry: dict) -> None:
        """Write a log entry to file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Fail silently - logging should not break the app
    
    @property
    def log_path(self) -> Path:
        """Return path to current log file."""
        return self.log_file


# Global logger instance
_logger: Optional[ConversationLogger] = None


def get_logger() -> ConversationLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger


def init_logger(model_name: str) -> ConversationLogger:
    """Initialize logger with model name."""
    global _logger
    _logger = ConversationLogger(model_name)
    return _logger
