#!/usr/bin/env python3.11
"""Tests for the memory management module of the AutoGPT agent."""

import pytest
from datetime import datetime
from typing import Dict, List

from ai_agents.auto_gpt.src.memory import Memory, Message


@pytest.fixture
def test_memory(tmp_path) -> Memory:
    """Create a test memory instance with a temporary persist path."""
    persist_path = tmp_path / "memory.json"
    return Memory(max_messages=3, persist_path=str(persist_path))


@pytest.fixture
def test_messages() -> List[Dict]:
    """Create a list of test messages."""
    return [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "role": "user",
            "content": "How are you?",
            "timestamp": datetime.now().isoformat(),
        },
    ]


def test_message_creation():
    """Test creating a message."""
    message = Message(role="user", content="Test message")
    assert message.role == "user"
    assert message.content == "Test message"
    assert message.timestamp is not None
    assert isinstance(message.timestamp, datetime)


def test_message_to_dict():
    """Test converting a message to dictionary format."""
    message = Message(role="user", content="Test message")
    data = message.to_dict()
    assert data["role"] == "user"
    assert data["content"] == "Test message"
    assert "timestamp" in data


def test_message_from_dict():
    """Test creating a message from dictionary format."""
    data = {
        "role": "assistant",
        "content": "Test response",
        "timestamp": datetime.now().isoformat(),
    }
    message = Message.from_dict(data)
    assert message.role == "assistant"
    assert message.content == "Test response"
    assert isinstance(message.timestamp, datetime)


def test_memory_initialization(test_memory):
    """Test initializing memory manager."""
    assert test_memory.max_messages == 3
    assert test_memory.persist_path is not None
    assert isinstance(test_memory.messages, list)
    assert len(test_memory.messages) == 0


def test_add_message(test_memory):
    """Test adding a message to memory."""
    test_memory.add_message("user", "Test message")
    assert len(test_memory.messages) == 1
    assert test_memory.messages[0].role == "user"
    assert test_memory.messages[0].content == "Test message"


def test_memory_limit(test_memory):
    """Test that memory respects maximum message limit."""
    # Add more messages than the limit
    for i in range(5):
        test_memory.add_message("user", f"Message {i}")

    # Should only keep the last 3 messages
    assert len(test_memory.messages) == 3
    assert test_memory.messages[0].content == "Message 2"
    assert test_memory.messages[1].content == "Message 3"
    assert test_memory.messages[2].content == "Message 4"


def test_get_messages(test_memory, test_messages):
    """Test getting messages in API format."""
    for msg in test_messages:
        test_memory.add_message(msg["role"], msg["content"])

    messages = test_memory.get_messages()
    assert len(messages) == 3
    assert all(isinstance(msg, dict) for msg in messages)
    assert all("role" in msg and "content" in msg for msg in messages)


def test_clear_messages(test_memory, test_messages):
    """Test clearing message history."""
    for msg in test_messages:
        test_memory.add_message(msg["role"], msg["content"])

    test_memory.clear_messages()
    assert len(test_memory.messages) == 0


def test_memory_persistence(test_memory, test_messages):
    """Test saving and loading memory state."""
    # Add messages and save
    for msg in test_messages:
        test_memory.add_message(msg["role"], msg["content"])
    test_memory.save()

    # Create new memory instance with same persist path
    new_memory = Memory(
        max_messages=3,
        persist_path=test_memory.persist_path,
    )

    # Should load the saved messages
    assert len(new_memory.messages) == 3
    assert all(isinstance(msg, Message) for msg in new_memory.messages)


def test_memory_context(test_memory):
    """Test managing context dictionary."""
    # Update context
    test_memory.update_context("test_key", "test_value")
    assert test_memory.get_context("test_key") == "test_value"

    # Clear context
    test_memory.clear_context()
    assert test_memory.get_context("test_key") is None


def test_invalid_persist_path():
    """Test handling invalid persistence path."""
    # Try to create memory with invalid path
    memory = Memory(
        max_messages=3,
        persist_path="/invalid/path/memory.json",
    )

    # Should not raise error, but save() should fail silently
    memory.add_message("user", "Test message")
    memory.save()  # Should not raise error


def test_memory_with_function_calls(test_memory):
    """Test handling messages with function calls."""
    function_call = {
        "name": "test_function",
        "arguments": {
            "arg1": "value1"
        },
    }

    # Add message with function call
    test_memory.add_message("assistant", "Calling function", function_call)

    # Verify function call was stored
    assert len(test_memory.messages) == 1
    assert test_memory.messages[0].function_call == function_call

""""""
