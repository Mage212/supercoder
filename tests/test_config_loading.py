
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

