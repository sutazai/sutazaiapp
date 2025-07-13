#!/usr/bin/env python3.11
"""Tests for the configuration management module."""

import pytest
import os
from pathlib import Path
from typing import Dict, Any

    Config,
    ConfigError,
    load_config,
    save_config,
    validate_config,
)


@pytest.fixture
def test_config_data() -> Dict[str, Any]:
    """Create test configuration data."""
    return {
        "name": "TestConfig",
        "version": "1.0.0",
        "description": "Test configuration",
        "settings": {
            "debug": True,
            "log_level": "INFO",
            "max_retries": 3,
        },
        "paths": {
            "data_dir": "/data",
            "log_dir": "/logs",
            "cache_dir": "/cache",
        },
        "api": {
            "base_url": "https://api.example.com",
            "timeout": 30,
            "retry_delay": 5,
        },
    }


@pytest.fixture
def test_config_file(tmp_path, test_config_data) -> Path:
    """Create a temporary configuration file."""
    config_path = tmp_path / "test_config.json"
    save_config(test_config_data, config_path)
    return config_path


def test_config_initialization(test_config_data):
    """Test configuration initialization."""
    config = Config(test_config_data)
    assert config.name == test_config_data["name"]
    assert config.version == test_config_data["version"]
    assert config.description == test_config_data["description"]
    assert config.settings == test_config_data["settings"]
    assert config.paths == test_config_data["paths"]
    assert config.api == test_config_data["api"]


def test_config_validation(test_config_data):
    """Test configuration validation."""
    # Test valid configuration
    assert validate_config(test_config_data) is True

    # Test missing required fields
    invalid_config = test_config_data.copy()
    del invalid_config["name"]
    with pytest.raises(ConfigError):
        validate_config(invalid_config)

    # Test invalid data types
    invalid_config = test_config_data.copy()
    invalid_config["settings"]["max_retries"] = "invalid"
    with pytest.raises(ConfigError):
        validate_config(invalid_config)


def test_config_loading(test_config_file):
    """Test loading configuration from file."""
    # Test successful loading
    config_data = load_config(test_config_file)
    assert config_data["name"] == "TestConfig"
    assert config_data["version"] == "1.0.0"

    # Test loading non-existent file
    with pytest.raises(ConfigError):
        load_config(Path("/non/existent/config.json"))

    # Test loading invalid JSON
    invalid_file = test_config_file.parent / "invalid.json"
    invalid_file.write_text("{invalid json}")
    with pytest.raises(ConfigError):
        load_config(invalid_file)


def test_config_saving(test_config_data, tmp_path):
    """Test saving configuration to file."""
    # Test successful saving
    config_path = tmp_path / "save_test.json"
    save_config(test_config_data, config_path)
    assert config_path.exists()

    # Test loading saved configuration
    loaded_data = load_config(config_path)
    assert loaded_data == test_config_data

    # Test saving to invalid path
    with pytest.raises(ConfigError):
        save_config(test_config_data, Path("/invalid/path/config.json"))


def test_config_environment_variables(test_config_data):
    """Test configuration environment variable handling."""
    # Set environment variables
    os.environ["TEST_CONFIG_NAME"] = "EnvConfig"
    os.environ["TEST_CONFIG_VERSION"] = "2.0.0"

    # Update config with environment variables
    config = Config(test_config_data)
    config.load_environment_variables(prefix="TEST_CONFIG_")

    assert config.name == "EnvConfig"
    assert config.version == "2.0.0"

    # Clean up environment variables
    del os.environ["TEST_CONFIG_NAME"]
    del os.environ["TEST_CONFIG_VERSION"]


def test_config_merging(test_config_data):
    """Test configuration merging."""
    # Create base config
    base_config = Config(test_config_data)

    # Create override config
    override_data = {
        "name": "OverrideConfig",
        "settings": {
            "debug": False,
        },
    }
    override_config = Config(override_data)

    # Merge configurations
    merged_config = base_config.merge(override_config)

    assert merged_config.name == "OverrideConfig"
    assert merged_config.version == test_config_data["version"]
    assert merged_config.settings["debug"] is False
    assert merged_config.settings["log_level"] == test_config_data["settings"]["log_level"]


def test_config_path_resolution(test_config_data):
    """Test configuration path resolution."""
    config = Config(test_config_data)

    # Test absolute path
    assert config.resolve_path("data_dir") == "/data"

    # Test relative path
    config.paths["data_dir"] = "data"
    assert config.resolve_path("data_dir").endswith("/data")

    # Test non-existent path
    with pytest.raises(ConfigError):
        config.resolve_path("non_existent_path")


def test_config_get_nested(test_config_data):
    """Test getting nested configuration values."""
    config = Config(test_config_data)

    # Test getting existing nested value
    assert config.get_nested("settings.log_level") == "INFO"

    # Test getting non-existent nested value with default
    assert config.get_nested("settings.non_existent", "DEFAULT") == "DEFAULT"

    # Test getting invalid nested path
    with pytest.raises(ConfigError):
        config.get_nested("invalid.path")
