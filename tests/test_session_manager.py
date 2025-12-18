"""Test SessionManager functionality."""

import pytest
import json
from pathlib import Path
from datetime import datetime

from supercoder.context.session_manager import SessionManager, ChatSession
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
            is_compacted=False
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
            is_compacted=False
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
                {"role": "assistant", "content": "Test response"}
            ]
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
        manager = SessionManager(tmp_path)
        
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
        session.messages = [
            Message("user", "Hello"),
            Message("assistant", "Hi there!")
        ]
        manager.save_session(session)
        
        # Load session
        loaded = manager.load_session(session.id)
        
        assert loaded is not None
        assert loaded.id == session.id
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "Hello"
    
    def test_save_session_updates_title(self, tmp_path):
        """Test that saving updates title from last user message."""
        manager = SessionManager(tmp_path)
        
        session = manager.create_new_session()
        session.messages = [
            Message("user", "How do I create a Python function?"),
            Message("assistant", "Here's how...")
        ]
        manager.save_session(session)
        
        loaded = manager.load_session(session.id)
        assert loaded.title == "How do I create a Python function?"
    
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
