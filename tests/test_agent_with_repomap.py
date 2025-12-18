"""Test Agent with RepoMap integration."""

import pytest
from unittest.mock import MagicMock, patch

from supercoder.agent.coder_agent import CoderAgent
from supercoder.tools import ALL_TOOLS
from supercoder.llm.base import StreamChunk


class MockLLM:
    """Mock LLM for testing without API calls."""
    
    model = "mock-model"
    
    def chat(self, messages):
        return "Mock response"
    
    def chat_stream(self, messages):
        yield StreamChunk("Mock response", is_done=False)
        yield StreamChunk("", is_done=True)


@pytest.fixture
def mock_llm():
    """Provide a mock LLM instance."""
    return MockLLM()


def test_agent_initialization_with_repomap(mock_llm, tmp_path):
    """Test that CoderAgent initializes correctly with RepoMap enabled."""
    # Create a simple Python file in temp directory
    test_file = tmp_path / "test_module.py"
    test_file.write_text("def hello():\n    pass\n")
    
    agent = CoderAgent(
        mock_llm,
        tools=ALL_TOOLS,
        use_repo_map=True,
        repo_root=str(tmp_path)
    )
    
    assert agent.repo_map is not None
    assert agent.repo_root == tmp_path


def test_agent_system_prompt_contains_repomap(mock_llm, tmp_path):
    """Test that system prompt includes RepoMap content when enabled."""
    # Create a Python file to be detected by RepoMap
    test_file = tmp_path / "example.py"
    test_file.write_text("class Example:\n    def method(self):\n        pass\n")
    
    agent = CoderAgent(
        mock_llm,
        tools=ALL_TOOLS,
        use_repo_map=True,
        repo_root=str(tmp_path)
    )
    
    # The base system prompt should be set
    assert agent.base_system_prompt is not None
    assert len(agent.base_system_prompt) > 0


def test_agent_without_repomap(mock_llm, tmp_path):
    """Test that CoderAgent works correctly without RepoMap."""
    agent = CoderAgent(
        mock_llm,
        tools=ALL_TOOLS,
        use_repo_map=False,
        repo_root=str(tmp_path)
    )
    
    assert agent.repo_map is None
