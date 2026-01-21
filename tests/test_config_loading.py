
import os
import yaml
import pytest
from unittest.mock import patch
from supercoder.config import Config

def test_load_config_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    # Create a dummy config file
    config_data = {
        "api_key": "test-key-from-yaml",
        "model": "model-from-yaml",
        "debug": True
    }
    
    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    # Mock os.getcwd to return tmp_path
    # We strip the original method to avoid recursion if needed, 
    # but here just patching for the scope of the test
    
    original_getcwd = os.getcwd
    
    try:
        os.getcwd = lambda: str(tmp_path)
        
        # Load config - we need to patch os.path.exists for the global config to avoid loading real values
        with patch('os.path.exists', side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml")):
            config = Config.load()
    
        # Verify values
        assert config.api_key == "test-key-from-yaml"
        assert config.model == "model-from-yaml"
        assert config.debug is True
        
    finally:
        os.getcwd = original_getcwd

def test_env_override_yaml(tmp_path):
    """Test that environment variables override YAML config."""
    config_data = {
        "model": "model-from-yaml",
    }
    
    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
        
    original_getcwd = os.getcwd
    os.environ["SUPERCODER_MODEL"] = "model-from-env"
    
    try:
        os.getcwd = lambda: str(tmp_path)
        config = Config.load()
        assert config.model == "model-from-env"
    finally:
        os.getcwd = original_getcwd
        del os.environ["SUPERCODER_MODEL"]


def test_model_profile_with_context_limit(tmp_path):
    """Test that model-specific max_context_tokens loads correctly."""
    config_data = {
        "default_model": "custom",
        "max_context_tokens": 32000,  # Global default
        "models": {
            "custom": {
                "api_key": "test-key",
                "model": "test-model",
                "max_context_tokens": 128000  # Model-specific
            }
        }
    }
    
    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    original_getcwd = os.getcwd
    
    try:
        os.getcwd = lambda: str(tmp_path)
        
        with patch('os.path.exists', side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml")):
            config = Config.load()
        
        # Verify model profile has correct context limit
        profile = config.get_model_profile("custom")
        assert profile is not None
        assert profile.max_context_tokens == 128000
        
        # Verify config was updated on load (since it's the default model)
        assert config.max_context_tokens == 128000
        
    finally:
        os.getcwd = original_getcwd


def test_switch_model_applies_context_limit(tmp_path):
    """Test that switch_to_model updates max_context_tokens."""
    config_data = {
        "default_model": "small",
        "max_context_tokens": 32000,
        "models": {
            "small": {
                "api_key": "key1",
                "model": "small-model",
                "max_context_tokens": 8000
            },
            "large": {
                "api_key": "key2",
                "model": "large-model",
                "max_context_tokens": 128000
            }
        }
    }
    
    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    original_getcwd = os.getcwd
    
    try:
        os.getcwd = lambda: str(tmp_path)
        
        with patch('os.path.exists', side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml")):
            config = Config.load()
        
        # Initial state (small model is default)
        assert config.max_context_tokens == 8000
        
        # Switch to large model
        result = config.switch_to_model("large")
        assert result is True
        assert config.max_context_tokens == 128000
        assert config.model == "large-model"
        
        # Switch back to small
        config.switch_to_model("small")
        assert config.max_context_tokens == 8000
        
    finally:
        os.getcwd = original_getcwd
