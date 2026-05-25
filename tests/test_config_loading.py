import os
from unittest.mock import patch

import yaml

from supercoder.config import Config


def test_load_config_from_yaml(tmp_path):
    """Test loading configuration from a YAML file."""
    # Create a dummy config file
    config_data = {"api_key": "test-key-from-yaml", "model": "model-from-yaml", "debug": True}

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
        with patch(
            "os.path.exists",
            side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml"),
        ):
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
                "max_context_tokens": 128000,  # Model-specific
            }
        },
    }

    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    original_getcwd = os.getcwd

    try:
        os.getcwd = lambda: str(tmp_path)

        with patch(
            "os.path.exists",
            side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml"),
        ):
            config = Config.load()

        # Verify model profile has correct context limit
        profile = config.get_model_profile("custom")
        assert profile is not None
        assert profile.max_context_tokens == 128000

        # Verify config was updated on load (since it's the default model)
        assert config.max_context_tokens == 128000

    finally:
        os.getcwd = original_getcwd


def test_model_profile_sampling_and_streaming_fields_load(tmp_path):
    """Profile-specific sampling and streaming fields are honored."""
    config_data = {
        "default_model": "custom",
        "temperature": 0.2,
        "top_p": 0.9,
        "streaming": False,
        "models": {
            "custom": {
                "api_key": "test-key",
                "model": "test-model",
                "temperature": 0.7,
                "top_p": 0.3,
                "streaming": True,
            }
        },
    }

    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    original_getcwd = os.getcwd

    try:
        os.getcwd = lambda: str(tmp_path)
        with patch(
            "os.path.exists",
            side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml"),
        ):
            config = Config.load()

        profile = config.get_model_profile("custom")
        assert profile is not None
        assert profile.temperature == 0.7
        assert profile.top_p == 0.3
        assert profile.streaming is True
        assert config.temperature == 0.7
        assert config.top_p == 0.3
        assert config.streaming is True

    finally:
        os.getcwd = original_getcwd


def test_permissions_config_loads_from_yaml(tmp_path):
    """Permission rules are loaded as shared configuration."""
    config_data = {
        "api_key": "test-key",
        "permissions": {
            "command-exec": {"allow": ["uv run pytest*"]},
            "paths": {"deny": ["private/*"]},
        },
    }

    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    original_getcwd = os.getcwd

    try:
        os.getcwd = lambda: str(tmp_path)
        with patch(
            "os.path.exists",
            side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml"),
        ):
            config = Config.load()

        assert config.permissions["command-exec"]["allow"] == ["uv run pytest*"]
        assert config.permissions["paths"]["deny"] == ["private/*"]

    finally:
        os.getcwd = original_getcwd


def test_switch_model_applies_context_limit(tmp_path):
    """Test that switch_to_model updates max_context_tokens."""
    config_data = {
        "default_model": "small",
        "max_context_tokens": 32000,
        "models": {
            "small": {"api_key": "key1", "model": "small-model", "max_context_tokens": 8000},
            "large": {"api_key": "key2", "model": "large-model", "max_context_tokens": 128000},
        },
    }

    config_file = tmp_path / ".supercoder.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    original_getcwd = os.getcwd

    try:
        os.getcwd = lambda: str(tmp_path)

        with patch(
            "os.path.exists",
            side_effect=lambda p: p == str(config_file) or p == str(tmp_path / ".supercoder.yaml"),
        ):
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
