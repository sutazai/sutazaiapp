#!/usr/bin/env python3.11
"""Tests for the serialization module."""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

from ai_agents.auto_gpt.src.serialization import (
    Serializer,
    Deserializer,
    JSONSerializer,
    YAMLSerializer,
    PickleSerializer,
    SerializationError,
    SerializationManager,
)


@dataclass
class TestObject:
    """Test data class for serialization."""
    name: str
    value: int
    timestamp: datetime
    data: Dict[str, Any]
    items: List[str]


@pytest.fixture
def test_data():
    """Create test data for serialization."""
    return {
        "string": "test",
        "number": 42,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "dict": {
            "key": "value",
            "nested": {
                "key": "value",
            },
        },
        "datetime": datetime.now(),
    }


@pytest.fixture
def test_object():
    """Create a test object for serialization."""
    return TestObject(
        name="test",
        value=42,
        timestamp=datetime.now(),
        data={"key": "value"},
        items=["item1", "item2"],
    )


@pytest.fixture
def serialization_manager():
    """Create a test serialization manager."""
    return SerializationManager()


def test_json_serialization(test_data):
    """Test JSON serialization."""
    serializer = JSONSerializer()
    
    # Test serialization
    serialized = serializer.serialize(test_data)
    assert isinstance(serialized, str)
    assert json.loads(serialized) == test_data
    
    # Test deserialization
    deserialized = serializer.deserialize(serialized)
    assert deserialized == test_data
    
    # Test invalid JSON
    with pytest.raises(SerializationError):
        serializer.deserialize("invalid json")


def test_yaml_serialization(test_data):
    """Test YAML serialization."""
    serializer = YAMLSerializer()
    
    # Test serialization
    serialized = serializer.serialize(test_data)
    assert isinstance(serialized, str)
    
    # Test deserialization
    deserialized = serializer.deserialize(serialized)
    assert deserialized == test_data
    
    # Test invalid YAML
    with pytest.raises(SerializationError):
        serializer.deserialize("invalid: yaml: :")


def test_pickle_serialization(test_data):
    """Test Pickle serialization."""
    serializer = PickleSerializer()
    
    # Test serialization
    serialized = serializer.serialize(test_data)
    assert isinstance(serialized, bytes)
    
    # Test deserialization
    deserialized = serializer.deserialize(serialized)
    assert deserialized == test_data
    
    # Test invalid pickle
    with pytest.raises(SerializationError):
        serializer.deserialize(b"invalid pickle")


def test_object_serialization(test_object):
    """Test object serialization."""
    serializer = JSONSerializer()
    
    # Test object serialization
    serialized = serializer.serialize_object(test_object)
    assert isinstance(serialized, str)
    
    # Test object deserialization
    deserialized = serializer.deserialize_object(serialized, TestObject)
    assert isinstance(deserialized, TestObject)
    assert deserialized.name == test_object.name
    assert deserialized.value == test_object.value
    assert deserialized.items == test_object.items


def test_file_serialization(tmp_path, test_data):
    """Test file serialization."""
    serializer = JSONSerializer()
    file_path = tmp_path / "test.json"
    
    # Test serialization to file
    serializer.serialize_to_file(test_data, file_path)
    assert file_path.exists()
    
    # Test deserialization from file
    deserialized = serializer.deserialize_from_file(file_path)
    assert deserialized == test_data
    
    # Test non-existent file
    with pytest.raises(SerializationError):
        serializer.deserialize_from_file(tmp_path / "non_existent.json")


def test_serialization_manager(serialization_manager):
    """Test serialization manager functionality."""
    # Test serializer registration
    json_serializer = JSONSerializer()
    serialization_manager.register_serializer("json", json_serializer)
    assert "json" in serialization_manager.serializers
    
    # Test serialization through manager
    data = {"key": "value"}
    serialized = serialization_manager.serialize("json", data)
    assert isinstance(serialized, str)
    
    # Test deserialization through manager
    deserialized = serialization_manager.deserialize("json", serialized)
    assert deserialized == data
    
    # Test serializer unregistration
    serialization_manager.unregister_serializer("json")
    assert "json" not in serialization_manager.serializers


def test_serialization_error_handling(serialization_manager):
    """Test serialization error handling."""
    # Test non-existent serializer
    with pytest.raises(SerializationError):
        serialization_manager.serialize("non_existent", {"key": "value"})
    
    # Test invalid data
    json_serializer = JSONSerializer()
    serialization_manager.register_serializer("json", json_serializer)
    
    class Unserializable:
        def __init__(self):
            self.self_reference = self
    
    with pytest.raises(SerializationError):
        serialization_manager.serialize("json", Unserializable())


def test_custom_serialization():
    """Test custom serialization."""
    class CustomSerializer(Serializer):
        def serialize(self, data: Any) -> str:
            return str(data)
        
        def deserialize(self, data: str) -> Any:
            return eval(data)
    
    serializer = CustomSerializer()
    
    # Test custom serialization
    data = {"key": "value"}
    serialized = serializer.serialize(data)
    assert isinstance(serialized, str)
    
    # Test custom deserialization
    deserialized = serializer.deserialize(serialized)
    assert deserialized == data


def test_serialization_formats():
    """Test different serialization formats."""
    data = {
        "string": "test",
        "number": 42,
        "boolean": True,
        "null": None,
        "list": [1, 2, 3],
        "dict": {"key": "value"},
        "datetime": datetime.now(),
    }
    
    # Test JSON format
    json_serializer = JSONSerializer()
    json_data = json_serializer.serialize(data)
    assert isinstance(json_data, str)
    assert json.loads(json_data) == data
    
    # Test YAML format
    yaml_serializer = YAMLSerializer()
    yaml_data = yaml_serializer.serialize(data)
    assert isinstance(yaml_data, str)
    
    # Test Pickle format
    pickle_serializer = PickleSerializer()
    pickle_data = pickle_serializer.serialize(data)
    assert isinstance(pickle_data, bytes)


def test_serialization_performance():
    """Test serialization performance."""
    import time
    
    # Create large test data
    large_data = {
        "items": [
            {
                "id": i,
                "name": f"Item {i}",
                "value": i * 100,
                "timestamp": datetime.now(),
            }
            for i in range(1000)
        ]
    }
    
    # Test JSON serialization performance
    json_serializer = JSONSerializer()
    start_time = time.time()
    json_serializer.serialize(large_data)
    json_time = time.time() - start_time
    
    # Test YAML serialization performance
    yaml_serializer = YAMLSerializer()
    start_time = time.time()
    yaml_serializer.serialize(large_data)
    yaml_time = time.time() - start_time
    
    # Test Pickle serialization performance
    pickle_serializer = PickleSerializer()
    start_time = time.time()
    pickle_serializer.serialize(large_data)
    pickle_time = time.time() - start_time
    
    # Verify reasonable performance
    assert json_time < 1.0
    assert yaml_time < 1.0
    assert pickle_time < 1.0 