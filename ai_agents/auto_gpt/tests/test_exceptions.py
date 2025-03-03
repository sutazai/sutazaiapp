#!/usr/bin/env python3.11
"""Tests for the custom exceptions module."""

import pytest
from typing import Dict, Any, List

from ai_agents.auto_gpt.src.exceptions import (
    AutoGPTError,
    ConfigError,
    ModelError,
    MemoryError,
    TaskError,
    ToolError,
    ValidationError,
    NetworkError,
    FileError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    ResourceNotFoundError,
    ResourceConflictError,
    ResourceValidationError,
)


def test_base_exception():
    """Test the base AutoGPTError exception."""
    # Test basic exception creation
    error = AutoGPTError("Test error")
    assert str(error) == "Test error"
    assert error.message == "Test error"
    
    # Test exception with details
    details = {"key": "value"}
    error = AutoGPTError("Test error", details=details)
    assert error.details == details
    
    # Test exception with cause
    cause = ValueError("Original error")
    error = AutoGPTError("Test error", cause=cause)
    assert error.cause == cause


def test_config_exception():
    """Test ConfigError exception."""
    # Test basic config error
    error = ConfigError("Invalid configuration")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Invalid configuration"
    
    # Test config error with details
    details = {"missing_fields": ["name", "version"]}
    error = ConfigError("Missing required fields", details=details)
    assert error.details == details


def test_model_exception():
    """Test ModelError exception."""
    # Test basic model error
    error = ModelError("Model initialization failed")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Model initialization failed"
    
    # Test model error with cause
    cause = ValueError("Invalid model parameters")
    error = ModelError("Model error", cause=cause)
    assert error.cause == cause


def test_memory_exception():
    """Test MemoryError exception."""
    # Test basic memory error
    error = MemoryError("Memory limit exceeded")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Memory limit exceeded"
    
    # Test memory error with details
    details = {"current_size": 1000, "max_size": 500}
    error = MemoryError("Memory limit exceeded", details=details)
    assert error.details == details


def test_task_exception():
    """Test TaskError exception."""
    # Test basic task error
    error = TaskError("Task execution failed")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Task execution failed"
    
    # Test task error with task details
    task_details = {"task_id": "123", "status": "failed"}
    error = TaskError("Task failed", details=task_details)
    assert error.details == task_details


def test_tool_exception():
    """Test ToolError exception."""
    # Test basic tool error
    error = ToolError("Tool execution failed")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Tool execution failed"
    
    # Test tool error with tool details
    tool_details = {"tool_name": "test_tool", "error_type": "validation"}
    error = ToolError("Tool error", details=tool_details)
    assert error.details == tool_details


def test_validation_exception():
    """Test ValidationError exception."""
    # Test basic validation error
    error = ValidationError("Invalid input")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Invalid input"
    
    # Test validation error with field errors
    field_errors = {
        "name": "Name is required",
        "age": "Age must be positive",
    }
    error = ValidationError("Validation failed", details=field_errors)
    assert error.details == field_errors


def test_network_exception():
    """Test NetworkError exception."""
    # Test basic network error
    error = NetworkError("Connection failed")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Connection failed"
    
    # Test network error with response details
    response_details = {"status_code": 500, "url": "https://example.com"}
    error = NetworkError("Server error", details=response_details)
    assert error.details == response_details


def test_file_exception():
    """Test FileError exception."""
    # Test basic file error
    error = FileError("File not found")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "File not found"
    
    # Test file error with file details
    file_details = {"path": "/test/file.txt", "operation": "read"}
    error = FileError("File operation failed", details=file_details)
    assert error.details == file_details


def test_timeout_exception():
    """Test TimeoutError exception."""
    # Test basic timeout error
    error = TimeoutError("Operation timed out")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Operation timed out"
    
    # Test timeout error with timing details
    timing_details = {"timeout": 30, "elapsed": 35}
    error = TimeoutError("Timeout exceeded", details=timing_details)
    assert error.details == timing_details


def test_authentication_exception():
    """Test AuthenticationError exception."""
    # Test basic authentication error
    error = AuthenticationError("Invalid credentials")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Invalid credentials"
    
    # Test authentication error with auth details
    auth_details = {"service": "api", "reason": "expired_token"}
    error = AuthenticationError("Auth failed", details=auth_details)
    assert error.details == auth_details


def test_rate_limit_exception():
    """Test RateLimitError exception."""
    # Test basic rate limit error
    error = RateLimitError("Rate limit exceeded")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Rate limit exceeded"
    
    # Test rate limit error with limit details
    limit_details = {"limit": 100, "remaining": 0, "reset": 3600}
    error = RateLimitError("Rate limit reached", details=limit_details)
    assert error.details == limit_details


def test_resource_not_found_exception():
    """Test ResourceNotFoundError exception."""
    # Test basic resource not found error
    error = ResourceNotFoundError("Resource not found")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Resource not found"
    
    # Test resource not found error with resource details
    resource_details = {"type": "model", "id": "123"}
    error = ResourceNotFoundError("Resource missing", details=resource_details)
    assert error.details == resource_details


def test_resource_conflict_exception():
    """Test ResourceConflictError exception."""
    # Test basic resource conflict error
    error = ResourceConflictError("Resource conflict")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Resource conflict"
    
    # Test resource conflict error with conflict details
    conflict_details = {"resource": "task", "conflict_type": "duplicate"}
    error = ResourceConflictError("Conflict detected", details=conflict_details)
    assert error.details == conflict_details


def test_resource_validation_exception():
    """Test ResourceValidationError exception."""
    # Test basic resource validation error
    error = ResourceValidationError("Invalid resource")
    assert isinstance(error, AutoGPTError)
    assert str(error) == "Invalid resource"
    
    # Test resource validation error with validation details
    validation_details = {
        "resource": "config",
        "errors": ["missing required field", "invalid value type"],
    }
    error = ResourceValidationError("Validation failed", details=validation_details)
    assert error.details == validation_details 