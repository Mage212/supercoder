"""Test SessionManager functionality."""

from datetime import datetime

from supercoder.context.session_manager import ChatSession, SessionManager
from supercoder.llm.base import Message


class TestChatSession:
    """Tests for ChatSession dataclass."""

    def test_session_creation(self):
        """Test creating a ChatSession."""
        now = datetime.now().isoformat()
        session = ChatSession(
            id="test123",
            title="Test Session",
            created_at=now,
            last_modified=now,
            messages=[],
            is_compacted=False,
        )

        assert session.id == "test123"
        assert session.title == "Test Session"
        assert session.messages == []
        assert session.is_compacted is False

    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        now = datetime.now().isoformat()
        session = ChatSession(
            id="test123",
            title="Test Session",
            created_at=now,
            last_modified=now,
            messages=[Message("user", "Hello"), Message("assistant", "Hi!")],
            is_compacted=False,
        )

        data = session.to_dict()

        assert data["id"] == "test123"
        assert data["title"] == "Test Session"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"

    def test_session_from_dict(self):
        """Test creating session from dictionary."""
        now = datetime.now().isoformat()
        data = {
            "id": "test456",
            "title": "From Dict",
            "created_at": now,
            "last_modified": now,
            "is_compacted": True,
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"},
            ],
        }

        session = ChatSession.from_dict(data)

        assert session.id == "test456"
        assert session.title == "From Dict"
        assert session.is_compacted is True
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_session_manager_initialization(self, tmp_path):
        """Test SessionManager creates sessions directory."""
        SessionManager(tmp_path)

        sessions_dir = tmp_path / ".supercoder" / "sessions"
        assert sessions_dir.exists()
        assert sessions_dir.is_dir()

    def test_create_new_session(self, tmp_path):
        """Test creating a new session."""
        manager = SessionManager(tmp_path)
        session = manager.create_new_session()

        assert session.id is not None
        assert len(session.id) == 8  # Short UUID
        assert session.title == "New Session"
        assert session.messages == []
        assert session.is_compacted is False

    def test_save_and_load_session(self, tmp_path):
        """Test saving and loading a session."""
        manager = SessionManager(tmp_path)

        # Create and save session
        session = manager.create_new_session()
        session.messages = [Message("user", "Hello"), Message("assistant", "Hi there!")]
        manager.save_session(session)

        # Load session
        loaded = manager.load_session(session.id)

        assert loaded is not None
        assert loaded.id == session.id
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello"

    def test_save_session_uses_atomic_writer(self, tmp_path, monkeypatch):
        """Session JSON is written through AtomicFileWriter."""
        from supercoder.context import session_manager

        manager = SessionManager(tmp_path)
        session = manager.create_new_session()
        original_write = session_manager.AtomicFileWriter.write
        calls = []

        def spy(path, content, encoding="utf-8"):
            calls.append((path, content))
            return original_write(path, content, encoding)

        monkeypatch.setattr(session_manager.AtomicFileWriter, "write", spy)

        manager.save_session(session)

        assert calls
        assert calls[0][0] == manager._get_session_path(session.id)

    def test_update_session_after_compact_uses_atomic_writer(self, tmp_path, monkeypatch):
        """Compacted session JSON is written through AtomicFileWriter."""
        from supercoder.context import session_manager

        manager = SessionManager(tmp_path)
        session = manager.create_new_session()
        original_write = session_manager.AtomicFileWriter.write
        calls = []

        def spy(path, content, encoding="utf-8"):
            calls.append((path, content))
            return original_write(path, content, encoding)

        monkeypatch.setattr(session_manager.AtomicFileWriter, "write", spy)

        manager.update_session_after_compact(session, "Summary")

        assert calls
        assert calls[0][0] == manager._get_session_path(session.id)

    def test_save_session_updates_title(self, tmp_path):
        """Test that saving updates title from last user message."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        session.messages = [
            Message("user", "How do I create a Python function?"),
            Message("assistant", "Here's how..."),
        ]
        manager.save_session(session)

        loaded = manager.load_session(session.id)
        assert loaded.title == "How do I create a Python function?"

    def test_save_session_title_ignores_context_attachment(self, tmp_path):
        """Attached @path context should not replace the human session title."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        session.messages = [
            Message(
                "user", "[Attached context from @ references]", display_type="context_attachment"
            ),
            Message("user", "Review @main.py", display_type="user_input"),
        ]
        manager.save_session(session)

        loaded = manager.load_session(session.id)
        assert loaded is not None
        assert loaded.title == "Review @main.py"

    def test_list_sessions(self, tmp_path):
        """Test listing all sessions."""
        manager = SessionManager(tmp_path)

        # Create multiple sessions
        for i in range(3):
            session = manager.create_new_session()
            session.messages = [Message("user", f"Session {i}")]
            manager.save_session(session)

        sessions = manager.list_sessions()

        assert len(sessions) == 3
        # Should have expected keys
        assert "id" in sessions[0]
        assert "title" in sessions[0]
        assert "last_modified" in sessions[0]

    def test_delete_session(self, tmp_path):
        """Test deleting a session."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        manager.save_session(session)

        # Verify it exists
        assert manager.load_session(session.id) is not None

        # Delete it
        result = manager.delete_session(session.id)

        assert result is True
        assert manager.load_session(session.id) is None

    def test_cleanup_old_sessions(self, tmp_path):
        """Test that old sessions are cleaned up when exceeding MAX_SESSIONS."""
        manager = SessionManager(tmp_path)

        # Create more sessions than MAX_SESSIONS
        session_ids = []
        for i in range(manager.MAX_SESSIONS + 3):
            session = manager.create_new_session()
            session.messages = [Message("user", f"Session {i}")]
            manager.save_session(session)
            session_ids.append(session.id)

        sessions = manager.list_sessions()

        # Should only have MAX_SESSIONS
        assert len(sessions) <= manager.MAX_SESSIONS

    def test_load_nonexistent_session(self, tmp_path):
        """Test loading a session that doesn't exist."""
        manager = SessionManager(tmp_path)

        result = manager.load_session("nonexistent")

        assert result is None

    def test_update_session_after_compact(self, tmp_path):
        """Test updating session after context compaction."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        session.messages = [
            Message("user", "Original message 1"),
            Message("assistant", "Response 1"),
            Message("user", "Original message 2"),
            Message("assistant", "Response 2"),
        ]
        manager.save_session(session)

        # Compact
        summary = "This is a summary of the previous conversation."
        manager.update_session_after_compact(session, summary)

        # Load and verify
        loaded = manager.load_session(session.id)

        assert loaded.is_compacted is True
        assert len(loaded.messages) == 1
        assert summary in loaded.messages[0].content

    def test_update_session_after_compact_keeps_recent_messages(self, tmp_path):
        """Compact persistence keeps the protected tail after the summary."""
        manager = SessionManager(tmp_path)
        session = manager.create_new_session()
        recent = [Message("user", "Recent exact step", display_type="user_input")]

        manager.update_session_after_compact(session, "Summary", recent)
        loaded = manager.load_session(session.id)

        assert loaded is not None
        assert loaded.is_compacted is True
        assert len(loaded.messages) == 2
        assert loaded.messages[0].display_type == "compact_summary"
        assert loaded.messages[1].content == "Recent exact step"

    def test_display_type_roundtrip(self, tmp_path):
        """Test that display_type survives save/load cycle."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        session.messages = [
            Message("user", "Hello", display_type="user_input"),
            Message("assistant", "Let me think...", display_type="thinking"),
            Message("assistant", "Hi!", display_type="response"),
            Message("user", "Attached", display_type="context_attachment"),
            Message("assistant", "", display_type="tool_call"),
            Message(
                "tool", "result", tool_call_id="tc1", name="file-read", display_type="tool_result"
            ),
            Message("tool", "ERROR", tool_call_id="tc2", name="code-edit", display_type="error"),
        ]
        manager.save_session(session)

        loaded = manager.load_session(session.id)
        assert loaded is not None
        types = [m.display_type for m in loaded.messages]
        assert types == [
            "user_input",
            "thinking",
            "response",
            "context_attachment",
            "tool_call",
            "tool_result",
            "error",
        ]

    def test_display_type_backward_compat(self, tmp_path):
        """Test loading old session without display_type."""
        manager = SessionManager(tmp_path)

        session = manager.create_new_session()
        session.messages = [Message("user", "Hello"), Message("assistant", "Hi")]
        manager.save_session(session)

        loaded = manager.load_session(session.id)
        assert all(m.display_type is None for m in loaded.messages)

    def test_display_type_not_in_api_dict(self):
        """Test that to_api_dict() does NOT include display_type."""
        msg = Message("user", "Hello", display_type="user_input")
        d = msg.to_api_dict()
        assert "display_type" not in d
        assert d == {"role": "user", "content": "Hello"}
