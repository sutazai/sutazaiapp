#!/usr/bin/env python3.11
"""Tests for the validation module."""

import pytest
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ai_agents.auto_gpt.src.validation import (
    validate_string,
    validate_number,
    validate_boolean,
    validate_datetime,
    validate_list,
    validate_dict,
    validate_email,
    validate_url,
    validate_ip_address,
    validate_uuid,
    validate_json,
    validate_schema,
    ValidationError,
    FieldValidator,
    SchemaValidator,
)


@dataclass
class TestUser:
    """Test data class for validation."""
    name: str
    age: int
    email: str
    is_active: bool
    created_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    optional_field: Optional[str] = None


def test_string_validation():
    """Test string validation."""
    # Test valid strings
    assert validate_string("test") == "test"
    assert validate_string("") == ""
    assert validate_string("123") == "123"
    
    # Test invalid strings
    with pytest.raises(ValidationError):
        validate_string(123)
    with pytest.raises(ValidationError):
        validate_string(None)
    with pytest.raises(ValidationError):
        validate_string([])


def test_number_validation():
    """Test number validation."""
    # Test valid numbers
    assert validate_number(42) == 42
    assert validate_number(42.5) == 42.5
    assert validate_number(0) == 0
    
    # Test invalid numbers
    with pytest.raises(ValidationError):
        validate_number("42")
    with pytest.raises(ValidationError):
        validate_number(None)
    with pytest.raises(ValidationError):
        validate_number([])


def test_boolean_validation():
    """Test boolean validation."""
    # Test valid booleans
    assert validate_boolean(True) is True
    assert validate_boolean(False) is False
    
    # Test invalid booleans
    with pytest.raises(ValidationError):
        validate_boolean(1)
    with pytest.raises(ValidationError):
        validate_boolean("true")
    with pytest.raises(ValidationError):
        validate_boolean(None)


def test_datetime_validation():
    """Test datetime validation."""
    # Test valid datetime
    now = datetime.now()
    assert validate_datetime(now) == now
    
    # Test string datetime
    dt_str = "2023-01-01T12:00:00"
    dt = datetime.fromisoformat(dt_str)
    assert validate_datetime(dt_str) == dt
    
    # Test invalid datetime
    with pytest.raises(ValidationError):
        validate_datetime("invalid")
    with pytest.raises(ValidationError):
        validate_datetime(123)
    with pytest.raises(ValidationError):
        validate_datetime(None)


def test_list_validation():
    """Test list validation."""
    # Test valid lists
    assert validate_list([1, 2, 3]) == [1, 2, 3]
    assert validate_list([]) == []
    assert validate_list(["a", "b", "c"]) == ["a", "b", "c"]
    
    # Test invalid lists
    with pytest.raises(ValidationError):
        validate_list("not a list")
    with pytest.raises(ValidationError):
        validate_list(123)
    with pytest.raises(ValidationError):
        validate_list(None)


def test_dict_validation():
    """Test dictionary validation."""
    # Test valid dictionaries
    assert validate_dict({"key": "value"}) == {"key": "value"}
    assert validate_dict({}) == {}
    assert validate_dict({"nested": {"key": "value"}}) == {"nested": {"key": "value"}}
    
    # Test invalid dictionaries
    with pytest.raises(ValidationError):
        validate_dict("not a dict")
    with pytest.raises(ValidationError):
        validate_dict(123)
    with pytest.raises(ValidationError):
        validate_dict(None)


def test_email_validation():
    """Test email validation."""
    # Test valid emails
    assert validate_email("test@example.com") == "test@example.com"
    assert validate_email("user.name@domain.co.uk") == "user.name@domain.co.uk"
    assert validate_email("test+tag@example.com") == "test+tag@example.com"
    
    # Test invalid emails
    with pytest.raises(ValidationError):
        validate_email("invalid")
    with pytest.raises(ValidationError):
        validate_email("test@")
    with pytest.raises(ValidationError):
        validate_email("@example.com")


def test_url_validation():
    """Test URL validation."""
    # Test valid URLs
    assert validate_url("https://example.com") == "https://example.com"
    assert validate_url("http://example.com") == "http://example.com"
    assert validate_url("ftp://example.com") == "ftp://example.com"
    
    # Test invalid URLs
    with pytest.raises(ValidationError):
        validate_url("invalid")
    with pytest.raises(ValidationError):
        validate_url("http://")
    with pytest.raises(ValidationError):
        validate_url("https://")


def test_ip_address_validation():
    """Test IP address validation."""
    # Test valid IP addresses
    assert validate_ip_address("192.168.1.1") == "192.168.1.1"
    assert validate_ip_address("10.0.0.0") == "10.0.0.0"
    assert validate_ip_address("172.16.0.0") == "172.16.0.0"
    
    # Test invalid IP addresses
    with pytest.raises(ValidationError):
        validate_ip_address("invalid")
    with pytest.raises(ValidationError):
        validate_ip_address("256.256.256.256")
    with pytest.raises(ValidationError):
        validate_ip_address("1.2.3")


def test_uuid_validation():
    """Test UUID validation."""
    # Test valid UUIDs
    valid_uuid = "123e4567-e89b-12d3-a456-426614174000"
    assert validate_uuid(valid_uuid) == valid_uuid
    
    # Test invalid UUIDs
    with pytest.raises(ValidationError):
        validate_uuid("invalid")
    with pytest.raises(ValidationError):
        validate_uuid("123e4567-e89b-12d3-a456")  # Incomplete UUID
    with pytest.raises(ValidationError):
        validate_uuid("123e4567-e89b-12d3-a456-42661417400g")  # Invalid character


def test_json_validation():
    """Test JSON validation."""
    # Test valid JSON
    json_str = '{"key": "value", "number": 42}'
    assert validate_json(json_str) == {"key": "value", "number": 42}
    
    # Test invalid JSON
    with pytest.raises(ValidationError):
        validate_json("invalid json")
    with pytest.raises(ValidationError):
        validate_json("{invalid}")
    with pytest.raises(ValidationError):
        validate_json("")


def test_schema_validation():
    """Test schema validation."""
    # Test valid schema
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    data = {"name": "John", "age": 30}
    assert validate_schema(data, schema) == data
    
    # Test invalid schema
    with pytest.raises(ValidationError):
        validate_schema({"name": "John"}, schema)  # Missing required field
    with pytest.raises(ValidationError):
        validate_schema({"name": 123, "age": 30}, schema)  # Invalid type


def test_field_validator():
    """Test FieldValidator functionality."""
    validator = FieldValidator()
    
    # Test string field validation
    assert validator.validate_field("name", "John", str) == "John"
    with pytest.raises(ValidationError):
        validator.validate_field("name", 123, str)
    
    # Test number field validation
    assert validator.validate_field("age", 30, int) == 30
    with pytest.raises(ValidationError):
        validator.validate_field("age", "30", int)
    
    # Test optional field validation
    assert validator.validate_field("optional", None, str, required=False) is None
    with pytest.raises(ValidationError):
        validator.validate_field("required", None, str, required=True)


def test_schema_validator():
    """Test SchemaValidator functionality."""
    validator = SchemaValidator()
    
    # Test data class validation
    user = TestUser(
        name="John",
        age=30,
        email="john@example.com",
        is_active=True,
        created_at=datetime.now(),
        tags=["test", "validation"],
        metadata={"key": "value"},
    )
    validated = validator.validate(user)
    assert validated == user
    
    # Test invalid data class
    with pytest.raises(ValidationError):
        validator.validate({
            "name": 123,  # Invalid type
            "age": "30",  # Invalid type
            "email": "invalid",  # Invalid email
            "is_active": 1,  # Invalid type
            "created_at": "invalid",  # Invalid datetime
            "tags": "not a list",  # Invalid type
            "metadata": "not a dict",  # Invalid type
        }) 