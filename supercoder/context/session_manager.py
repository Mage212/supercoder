"""Chat session management for persistence across restarts."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm.base import Message


@dataclass
class ChatSession:
    """Represents a saved chat session."""

    id: str
    title: str
    created_at: str  # ISO format
    last_modified: str  # ISO format
    messages: list[Message] = field(default_factory=list)
    is_compacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        serialized_messages = []
        for msg in self.messages:
            m: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.name:
                m["name"] = msg.name
            if msg.display_type:
                m["display_type"] = msg.display_type
            serialized_messages.append(m)

        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "is_compacted": self.is_compacted,
            "messages": serialized_messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatSession":
        """Create session from dictionary."""
        messages = []
        for m in data.get("messages", []):
            messages.append(
                Message(
                    role=m["role"],
                    content=m["content"],
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                    name=m.get("name"),
                    display_type=m.get("display_type"),
                )
            )
        return cls(
            id=data["id"],
            title=data.get("title", "Untitled"),
            created_at=data.get("created_at", ""),
            last_modified=data.get("last_modified", ""),
            messages=messages,
            is_compacted=data.get("is_compacted", False),
        )


class SessionManager:
    """Manages chat session persistence.

    Sessions are stored as JSON files in .supercoder/sessions/ directory.
    Maximum of 10 sessions are kept; oldest sessions are automatically deleted.
    """

    MAX_SESSIONS = 10
    SESSIONS_DIR = "sessions"

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.sessions_dir = self.project_root / ".supercoder" / self.SESSIONS_DIR
        self._ensure_sessions_dir()

    def _ensure_sessions_dir(self) -> None:
        """Create sessions directory if it doesn't exist."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_gitignore()

    def _ensure_gitignore(self) -> None:
        """Ensure .supercoder/ is listed in the project's .gitignore."""
        gitignore = self.project_root / ".gitignore"
        entry = ".supercoder/\n"
        try:
            if gitignore.exists():
                content = gitignore.read_text()
                if ".supercoder/" not in content and ".supercoder" not in content:
                    with gitignore.open("a") as f:
                        f.write(entry)
            else:
                gitignore.write_text(entry)
        except Exception:
            pass  # Non-critical; don't fail session management over gitignore

    def _get_session_path(self, session_id: str) -> Path:
        """Get path to session file."""
        return self.sessions_dir / f"{session_id}.json"

    def create_new_session(self) -> ChatSession:
        """Create a new empty session."""
        now = datetime.now().isoformat()
        session = ChatSession(
            id=str(uuid.uuid4())[:8],  # Short UUID for readability
            title="New Session",
            created_at=now,
            last_modified=now,
            messages=[],
            is_compacted=False,
        )
        return session

    def save_session(self, session: ChatSession) -> None:
        """Save session to JSON file.

        Also triggers cleanup if MAX_SESSIONS is exceeded.
        """
        session.last_modified = datetime.now().isoformat()

        # Update title from last user message
        user_messages = [
            m
            for m in session.messages
            if m.role == "user"
            and not m.content.startswith("<@TOOL_RESULT>")
            and not m.content.startswith("[Previous Context")
        ]
        if user_messages:
            last_msg = user_messages[-1].content
            # Truncate and clean title
            session.title = (last_msg[:50] + "...") if len(last_msg) > 50 else last_msg
            session.title = session.title.replace("\n", " ").strip()

        # Save to file
        session_path = self._get_session_path(session.id)
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        # Cleanup old sessions
        self._cleanup_old_sessions()

    def load_session(self, session_id: str) -> ChatSession | None:
        """Load session from JSON file."""
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        try:
            with open(session_path, encoding="utf-8") as f:
                data = json.load(f)
            return ChatSession.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            # Corrupted session file
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all available sessions with metadata.

        Returns list of dicts with id, title, last_modified, is_compacted.
        Sorted by last_modified (newest first).
        """
        sessions = []

        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "id": data.get("id", session_file.stem),
                        "title": data.get("title", "Untitled"),
                        "created_at": data.get("created_at", ""),
                        "last_modified": data.get("last_modified", ""),
                        "is_compacted": data.get("is_compacted", False),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted files
                continue

        # Sort by last_modified (newest first)
        sessions.sort(key=lambda s: s.get("last_modified", ""), reverse=True)

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        session_path = self._get_session_path(session_id)

        if session_path.exists():
            session_path.unlink()
            return True
        return False

    def update_session_after_compact(self, session: ChatSession, summary: str) -> None:
        """Update session after context compaction.

        Replaces all messages with the summary and marks as compacted.
        """
        session.is_compacted = True
        session.messages = [
            Message(
                "user", f"[Previous Context Summary]\n\n{summary}", display_type="compact_summary"
            )
        ]
        session.last_modified = datetime.now().isoformat()

        # Save updated session
        session_path = self._get_session_path(session.id)
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

    def _cleanup_old_sessions(self) -> None:
        """Remove oldest sessions if we exceed MAX_SESSIONS."""
        sessions = self.list_sessions()

        if len(sessions) <= self.MAX_SESSIONS:
            return

        # Sessions are sorted newest first, so remove from the end
        sessions_to_delete = sessions[self.MAX_SESSIONS :]

        for session in sessions_to_delete:
            self.delete_session(session["id"])
