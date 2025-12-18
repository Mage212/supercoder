"""Test context management functionality."""

import pytest
from supercoder.context import TokenCounter, ContextWindowManager, ContextConfig
from supercoder.llm.base import Message


class TestTokenCounter:
    """Tests for TokenCounter class."""
    
    def test_token_counter_estimation(self):
        """Test that token counter provides reasonable estimates."""
        tc = TokenCounter(use_tiktoken=False)
        
        text = "Hello, this is a test message for token counting."
        tokens = tc.count(text)
        
        # Rough estimate: ~4 chars per token
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count
    
    def test_token_counter_with_code(self):
        """Test token counting for code."""
        tc = TokenCounter(use_tiktoken=False)
        
        code = '''def hello_world():
    print("Hello, World!")
    return 42'''
        
        tokens = tc.count(code)
        assert tokens > 0
    
    def test_tiktoken_availability(self):
        """Test that tiktoken-based counter reports accurate counting."""
        tc = TokenCounter(use_tiktoken=True)
        # Should have accurate counting if tiktoken is available
        assert tc.has_accurate_counting is True


class TestContextWindowManager:
    """Tests for ContextWindowManager class."""
    
    def test_context_manager_initialization(self):
        """Test ContextWindowManager initializes correctly."""
        config = ContextConfig(
            max_tokens=1000,
            reserved_for_response=200,
            compression_threshold=0.5
        )
        cm = ContextWindowManager(config)
        
        assert cm is not None
    
    def test_add_messages(self):
        """Test adding messages to context."""
        config = ContextConfig(max_tokens=1000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("You are a helpful assistant.")
        
        cm.add_message(Message("user", "Hello"))
        cm.add_message(Message("assistant", "Hi there!"))
        
        stats = cm.get_stats()
        assert stats.message_count == 2
    
    def test_context_stats(self):
        """Test context statistics tracking."""
        config = ContextConfig(
            max_tokens=1000,
            reserved_for_response=200
        )
        cm = ContextWindowManager(config)
        cm.set_system_prompt("You are a helpful assistant.")
        
        for i in range(5):
            cm.add_message(Message("user", f"Message {i}: test content"))
            cm.add_message(Message("assistant", f"Response {i}"))
        
        stats = cm.get_stats()
        assert stats.message_count == 10
        assert stats.used_tokens > 0
        assert stats.utilization_percent >= 0
    
    def test_context_clear(self):
        """Test clearing context."""
        config = ContextConfig(max_tokens=1000)
        cm = ContextWindowManager(config)
        cm.set_system_prompt("System prompt")
        
        cm.add_message(Message("user", "Hello"))
        cm.add_message(Message("assistant", "Hi"))
        
        cm.clear()
        stats = cm.get_stats()
        assert stats.message_count == 0
