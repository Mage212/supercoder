"""OpenAI-compatible LLM client (works with OpenAI, OpenRouter, Ollama, etc.)."""

from typing import Iterator
from openai import OpenAI

from .base import BaseLLM, Message, StreamChunk
from ..config import Config, ModelProfile


class OpenAIClient(BaseLLM):
    """OpenAI-compatible API client.
    
    Works with:
    - OpenAI API
    - OpenRouter (https://openrouter.ai/api/v1)
    - Ollama (http://localhost:11434/v1)
    - Any OpenAI-compatible endpoint
    """
    
    # App identification for OpenRouter statistics
    APP_NAME = "SuperCoder CLI"
    APP_URL = "https://github.com/Mage212/supercoder"
    
    def __init__(self, config: Config):
        self.config = config
        
        # Add headers for OpenRouter app identification
        default_headers = {
            "HTTP-Referer": self.APP_URL,
            "X-Title": self.APP_NAME,
        }
        
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            default_headers=default_headers,
            timeout=config.request_timeout,
        )
        self.model = config.model
        self.temperature = config.temperature
        self.debug = config.debug
    
    def switch_model(self, profile: ModelProfile) -> None:
        """Switch to a different model profile.
        
        Reinitializes the OpenAI client with new credentials.
        """
        default_headers = {
            "HTTP-Referer": self.APP_URL,
            "X-Title": self.APP_NAME,
        }
        self.client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.endpoint,
            default_headers=default_headers,
            timeout=profile.request_timeout,
        )
        self.model = profile.model
    
    def chat(self, messages: list[Message]) -> str:
        """Send messages and get complete response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""
    
    def chat_stream(self, messages: list[Message]) -> Iterator[StreamChunk]:
        """Send messages and stream response chunks."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(content=chunk.choices[0].delta.content)
            
            yield StreamChunk(content="", is_done=True)
            
        except Exception as e:
            yield StreamChunk(content=f"\n[Error: {e}]", is_done=True)

