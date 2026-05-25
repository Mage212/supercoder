"""Tests for OpenAI-compatible client request shaping."""

from types import SimpleNamespace

from supercoder.config import Config
from supercoder.llm.base import Message
from supercoder.llm.openai_client import OpenAIClient


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=FakeCompletions(response))


def test_chat_with_tools_passes_top_p(monkeypatch):
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning_content="")
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
    )
    fake_client = FakeOpenAI(response)
    monkeypatch.setattr("supercoder.llm.openai_client.OpenAI", lambda **_kwargs: fake_client)
    client = OpenAIClient(Config(api_key="test", top_p=0.3))

    client.chat_with_tools([Message("user", "hello")])

    assert fake_client.chat.completions.calls[0]["top_p"] == 0.3


def test_interruptible_chat_with_tools_passes_top_p(monkeypatch):
    delta = SimpleNamespace(content="ok", reasoning_content="", tool_calls=None)
    stream = [
        SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        ),
    ]
    fake_client = FakeOpenAI(stream)
    monkeypatch.setattr("supercoder.llm.openai_client.OpenAI", lambda **_kwargs: fake_client)
    client = OpenAIClient(Config(api_key="test", top_p=0.4))

    client.chat_with_tools_interruptible([Message("user", "hello")])

    assert fake_client.chat.completions.calls[0]["top_p"] == 0.4
