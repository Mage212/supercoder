"""Base tool interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


@dataclass
class ToolDefinition:
    """Tool metadata."""
    name: str
    description: str


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
        """Parse JSON arguments safely."""
        if not arguments or arguments == '""':
            return {}
        try:
            # Handle double-encoded JSON
            if arguments.startswith('"') and arguments.endswith('"'):
                arguments = json.loads(arguments)
            return json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return {}
