"""Configuration and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Global config directory
CONFIG_DIR = Path.home() / ".supercoder"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

# Template for new config file with multi-model support
CONFIG_TEMPLATE = """# SuperCoder Configuration
# Documentation: https://github.com/your-repo/supercoder

# Default model profile to use on startup
default_model: "default"

# Model profiles - define multiple LLM configurations
models:
  default:
    api_key: ""
    endpoint: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    
  # Example: OpenRouter with free models
  # openrouter-free:
  #   api_key: "sk-or-v1-..."
  #   endpoint: "https://openrouter.ai/api/v1"
  #   model: "openai/gpt-oss-20b:free"
  #   tool_calling_type: "qwen_like"  # Format for tool calls: supercoder, qwen_like, json_block, xml_function
  
  # Example: Local Ollama
  # ollama:
  #   api_key: "ollama"
  #   endpoint: "http://localhost:11434/v1"
  #   model: "llama3.2"
  #   tool_calling_type: "supercoder"  # default

# Shared settings (applied to all models)
temperature: 0.2
top_p: 0.1
max_context_tokens: 32000
reserved_for_response: 4096
request_timeout: 60.0
debug: false
"""


def ensure_config_dir() -> Path:
    """Create config directory if it doesn't exist."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def ensure_config_file() -> Path:
    """Create template config file if it doesn't exist."""
    ensure_config_dir()
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(CONFIG_TEMPLATE)
    return CONFIG_FILE


@dataclass
class ModelProfile:
    """A named LLM configuration profile."""
    name: str
    api_key: str = ""
    endpoint: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    request_timeout: float = 60.0
    tool_calling_type: str = "supercoder"  # supercoder, qwen_like, json_block, xml_function
    
    @property
    def base_url(self) -> str:
        """Alias for endpoint to match OpenAI client."""
        return self.endpoint


@dataclass
class Config:
    """Application configuration with multi-model support."""
    
    # Current active model profile
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    
    # Shared settings
    temperature: float = 0.2
    top_p: float = 0.1
    debug: bool = False
    max_context_tokens: int = 32000
    reserved_for_response: int = 4096
    request_timeout: float = 60.0
    
    # Multi-model support
    default_model: str = "default"
    models: dict = field(default_factory=dict)
    _current_profile: str = ""
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from file and environment variables.
        
        Config priority (later overrides earlier):
        1. ~/.supercoder/config.yaml (global)
        2. .supercoder.yaml (local project)
        3. Environment variables
        """
        config_data = {}
        models_data = {}
        
        # Ensure global config exists (creates template on first run)
        ensure_config_file()
        
        # Load from config files (global first, then local overrides)
        config_paths = [
            str(CONFIG_FILE),  # ~/.supercoder/config.yaml
            os.path.join(os.getcwd(), ".supercoder.yaml")  # local override
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    import yaml
                    with open(path, "r") as f:
                        file_data = yaml.safe_load(f) or {}
                        
                        # Extract models dict
                        if "models" in file_data:
                            models_data.update(file_data.pop("models"))
                        
                        # Support both 'base_url' and 'endpoint'
                        if "endpoint" in file_data and "base_url" not in file_data:
                            file_data["base_url"] = file_data["endpoint"]
                        
                        config_data.update(file_data)
                except Exception:
                    pass  # Ignore config file errors
        
        # Build ModelProfile objects
        models = {}
        for name, profile_data in models_data.items():
            if isinstance(profile_data, dict):
                models[name] = ModelProfile(
                    name=name,
                    api_key=profile_data.get("api_key", ""),
                    endpoint=profile_data.get("endpoint", profile_data.get("base_url", "https://api.openai.com/v1")),
                    model=profile_data.get("model", "gpt-4o-mini"),
                    request_timeout=float(profile_data.get("request_timeout", config_data.get("request_timeout", 60.0))),
                    tool_calling_type=profile_data.get("tool_calling_type", "supercoder"),
                )
        
        # Create instance with shared settings
        valid_fields = {k: v for k, v in config_data.items() 
                       if hasattr(cls, k) and k != "models"}
        config = cls(**valid_fields)
        config.models = models
        
        # Set default model profile as active
        default_name = config_data.get("default_model", "default")
        config.default_model = default_name
        
        if default_name in models:
            profile = models[default_name]
            config.api_key = profile.api_key
            config.base_url = profile.endpoint
            config.model = profile.model
            config.request_timeout = profile.request_timeout
            config._current_profile = default_name
        
        # Override with environment variables (highest priority)
        env_api_key = os.getenv("SUPERCODER_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        if env_api_key:
            config.api_key = env_api_key
            
        env_base_url = os.getenv("SUPERCODER_ENDPOINT", os.getenv("SUPERCODER_BASE_URL", os.getenv("OPENAI_BASE_URL", "")))
        if env_base_url:
            config.base_url = env_base_url
            
        env_model = os.getenv("SUPERCODER_MODEL", os.getenv("OPENAI_MODEL", ""))
        if env_model:
            config.model = env_model
            
        if os.getenv("SUPERCODER_DEBUG"):
            config.debug = os.getenv("SUPERCODER_DEBUG", "").lower() == "true"
            
        return config
    
    @classmethod
    def from_env(cls) -> "Config":
        """Deprecated: Use load() instead."""
        return cls.load()
    
    def get_model_profile(self, name: str) -> Optional[ModelProfile]:
        """Get a model profile by name."""
        return self.models.get(name)
    
    def get_available_models(self) -> list[str]:
        """Get list of available model profile names."""
        return list(self.models.keys())
    
    def switch_to_model(self, name: str) -> bool:
        """Switch to a different model profile. Returns True if successful."""
        if name not in self.models:
            return False
        
        profile = self.models[name]
        self.api_key = profile.api_key
        self.base_url = profile.endpoint
        self.model = profile.model
        self.request_timeout = profile.request_timeout
        self._current_profile = name
        return True
    
    @property
    def current_profile_name(self) -> str:
        """Get the name of the current active profile."""
        return self._current_profile or self.default_model
    
    def validate(self) -> list[str]:
        """Validate configuration, return list of errors."""
        errors = []
        if not self.api_key:
            errors.append(
                f"API key not set. Edit {CONFIG_FILE} or set SUPERCODER_API_KEY env var"
            )
        return errors
    
    @staticmethod
    def get_config_path() -> Path:
        """Return path to global config file."""
        return CONFIG_FILE
