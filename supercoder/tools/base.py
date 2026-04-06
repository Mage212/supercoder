"""Base tool interface."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Tool metadata with optional JSON Schema for native function calling."""

    name: str
    description: str
    parameters: dict | None = field(default=None, repr=False)

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI-compatible function tool schema.

        Returns a dict suitable for the ``tools`` parameter of
        ``chat.completions.create()``.
        """
        func: dict = {
            "name": self.name,
            "description": self.description,
        }
        if self.parameters:
            func["parameters"] = self.parameters
        return {"type": "function", "function": func}


class BaseTool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        pass

    @abstractmethod
    def execute(self, arguments: str) -> str:
        """Execute tool with JSON arguments string."""
        pass

    def parse_args(self, arguments: str) -> dict:
        """Parse JSON arguments safely.

        Returns {"_parse_error": True, "raw": ...} on JSON decode failure
        so callers can return a meaningful error to the LLM instead of
        silently proceeding with empty arguments.
        """
        if not arguments or arguments == '""':
            return {}
        try:
            # Handle double-encoded JSON
            if arguments.startswith('"') and arguments.endswith('"'):
                arguments = json.loads(arguments)
            return json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return {"_parse_error": True, "raw": arguments[:200]}
