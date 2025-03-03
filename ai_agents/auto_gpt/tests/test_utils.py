#!/usr/bin/env python3.11
"""Tests for the utility functions module."""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ai_agents.auto_gpt.src.utils import (
    load_json,
    save_json,
    format_timestamp,
    parse_timestamp,
    ensure_directory,
    get_file_hash,
    retry,
    timeout,
    validate_url,
    sanitize_filename,
    chunk_text,
    merge_dicts,
    flatten_dict,
    get_nested_value,
    set_nested_value,
)


@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Create test data for JSON operations."""
    return {
        "string": "test",
        "number": 42,
        "boolean": True,
        "null": None,
        "array": [1, 2, 3],
        "object": {
            "key": "value",
            "nested": {
                "key": "value",
            },
        },
    }


@pytest.fixture
def test_file(tmp_path, test_data) -> Path:
    """Create a temporary test file."""
    file_path = tmp_path / "test.json"
    save_json(test_data, file_path)
    return file_path


def test_json_operations(test_data, test_file):
    """Test JSON loading and saving operations."""
    # Test saving JSON
    assert test_file.exists()
    
    # Test loading JSON
    loaded_data = load_json(test_file)
    assert loaded_data == test_data
    
    # Test loading non-existent file
    with pytest.raises(FileNotFoundError):
        load_json(Path("/non/existent/file.json"))
    
    # Test loading invalid JSON
    invalid_file = test_file.parent / "invalid.json"
    invalid_file.write_text("{invalid json}")
    with pytest.raises(json.JSONDecodeError):
        load_json(invalid_file)


def test_timestamp_operations():
    """Test timestamp formatting and parsing."""
    # Test formatting timestamp
    now = datetime.now()
    formatted = format_timestamp(now)
    assert isinstance(formatted, str)
    
    # Test parsing timestamp
    parsed = parse_timestamp(formatted)
    assert isinstance(parsed, datetime)
    assert parsed.year == now.year
    assert parsed.month == now.month
    assert parsed.day == now.day
    
    # Test parsing invalid timestamp
    with pytest.raises(ValueError):
        parse_timestamp("invalid timestamp")


def test_directory_operations(tmp_path):
    """Test directory creation and validation."""
    # Test creating directory
    dir_path = tmp_path / "test_dir"
    ensure_directory(dir_path)
    assert dir_path.exists()
    assert dir_path.is_dir()
    
    # Test creating nested directory
    nested_path = dir_path / "nested" / "deep"
    ensure_directory(nested_path)
    assert nested_path.exists()
    assert nested_path.is_dir()


def test_file_hash(test_file):
    """Test file hash calculation."""
    # Test getting file hash
    hash_value = get_file_hash(test_file)
    assert isinstance(hash_value, str)
    assert len(hash_value) == 64  # SHA-256 hash length
    
    # Test getting hash of non-existent file
    with pytest.raises(FileNotFoundError):
        get_file_hash(Path("/non/existent/file.txt"))


def test_retry_decorator():
    """Test retry decorator functionality."""
    attempts = 0
    
    @retry(max_attempts=3, delay=0.1)
    def failing_function():
        nonlocal attempts
        attempts += 1
        raise ValueError("Test error")
    
    # Test retry behavior
    with pytest.raises(ValueError):
        failing_function()
    assert attempts == 3
    
    # Test successful execution
    attempts = 0
    
    @retry(max_attempts=3, delay=0.1)
    def succeeding_function():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ValueError("Test error")
        return "success"
    
    result = succeeding_function()
    assert result == "success"
    assert attempts == 2


def test_timeout_decorator():
    """Test timeout decorator functionality."""
    @timeout(1.0)
    def slow_function():
        import time
        time.sleep(2.0)
        return "success"
    
    # Test timeout behavior
    with pytest.raises(TimeoutError):
        slow_function()
    
    # Test successful execution
    @timeout(2.0)
    def fast_function():
        return "success"
    
    assert fast_function() == "success"


def test_url_validation():
    """Test URL validation."""
    # Test valid URLs
    assert validate_url("https://example.com")
    assert validate_url("http://example.com")
    assert validate_url("ftp://example.com")
    
    # Test invalid URLs
    assert not validate_url("invalid url")
    assert not validate_url("http://")
    assert not validate_url("https://")


def test_filename_sanitization():
    """Test filename sanitization."""
    # Test valid filenames
    assert sanitize_filename("test.txt") == "test.txt"
    assert sanitize_filename("test-file.txt") == "test-file.txt"
    
    # Test invalid characters
    assert sanitize_filename("test/file.txt") == "test_file.txt"
    assert sanitize_filename("test\\file.txt") == "test_file.txt"
    assert sanitize_filename("test:file.txt") == "test_file.txt"
    
    # Test empty filename
    assert sanitize_filename("") == "unnamed"


def test_text_chunking():
    """Test text chunking functionality."""
    text = "This is a test text that should be chunked into smaller pieces."
    
    # Test chunking with default size
    chunks = chunk_text(text, chunk_size=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    
    # Test chunking with overlap
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert all(len(chunk) <= 10 for chunk in chunks)
    
    # Test empty text
    assert chunk_text("") == []


def test_dict_operations():
    """Test dictionary operations."""
    # Test dictionary merging
    dict1 = {"a": 1, "b": {"c": 2}}
    dict2 = {"b": {"d": 3}, "e": 4}
    merged = merge_dicts(dict1, dict2)
    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
    
    # Test dictionary flattening
    nested_dict = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
    flattened = flatten_dict(nested_dict)
    assert flattened == {"a": 1, "b.c": 2, "b.d.e": 3}
    
    # Test getting nested value
    assert get_nested_value(nested_dict, "b.d.e") == 3
    assert get_nested_value(nested_dict, "b.d.f", default=4) == 4
    
    # Test setting nested value
    new_dict = {}
    set_nested_value(new_dict, "a.b.c", 1)
    assert new_dict == {"a": {"b": {"c": 1}}} 