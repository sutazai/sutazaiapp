#!/usr/bin/env python3.11
"""Tests for the tool management module of the AutoGPT agent."""

import pytest
from typing import Dict, Any

from ai_agents.auto_gpt.src.tools import Tool, ToolParameter, ToolRegistry


def test_tool_parameter_creation():
    """Test creating a tool parameter."""
    param = ToolParameter(
        name="test_param",
        description="A test parameter",
        type="str",
        required=True,
        default=None,
    )
    assert param.name == "test_param"
    assert param.description == "A test parameter"
    assert param.type == "str"
    assert param.required is True
    assert param.default is None


def test_tool_creation():
    """Test creating a tool."""

    def test_function(param1: str, param2: int = 42) -> str:
        return f"Result: {param1}, {param2}"

    parameters = [
        ToolParameter(
            name="param1",
            description="First parameter",
            type="str",
            required=True,
        ),
        ToolParameter(
            name="param2",
            description="Second parameter",
            type="int",
            required=False,
            default=42,
        ),
    ]

    tool = Tool(
        name="test_tool",
        description="A test tool",
        function=test_function,
        parameters=parameters,
    )

    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool.function == test_function
    assert len(tool.parameters) == 2


def test_tool_parameter_validation():
    """Test parameter validation in tools."""

    def test_function(param1: str, param2: int = 42) -> str:
        return f"Result: {param1}, {param2}"

    parameters = [
        ToolParameter(
            name="param1",
            description="First parameter",
            type="str",
            required=True,
        ),
        ToolParameter(
            name="param2",
            description="Second parameter",
            type="int",
            required=False,
            default=42,
        ),
    ]

    tool = Tool(
        name="test_tool",
        description="A test tool",
        function=test_function,
        parameters=parameters,
    )

    # Test valid parameters
    validated = tool.validate_params({"param1": "test", "param2": 100})
    assert validated["param1"] == "test"
    assert validated["param2"] == 100

    # Test missing required parameter
    with pytest.raises(ValueError, match="Missing required parameter"):
        tool.validate_params({"param2": 100})

    # Test invalid parameter type
    with pytest.raises(ValueError, match="Invalid value for parameter"):
        tool.validate_params({"param1": "test", "param2": "not_an_int"})


def test_tool_execution():
    """Test tool execution."""

    def test_function(param1: str, param2: int = 42) -> str:
        return f"Result: {param1}, {param2}"

    parameters = [
        ToolParameter(
            name="param1",
            description="First parameter",
            type="str",
            required=True,
        ),
        ToolParameter(
            name="param2",
            description="Second parameter",
            type="int",
            required=False,
            default=42,
        ),
    ]

    tool = Tool(
        name="test_tool",
        description="A test tool",
        function=test_function,
        parameters=parameters,
    )

    # Test successful execution
    result = tool.execute(param1="test", param2=100)
    assert result == "Result: test, 100"

    # Test execution with default parameter
    result = tool.execute(param1="test")
    assert result == "Result: test, 42"


def test_tool_registry():
    """Test tool registry functionality."""
    registry = ToolRegistry()

    # Create a test tool

    def test_function(param1: str) -> str:
        return f"Result: {param1}"

    parameters = [
        ToolParameter(
            name="param1",
            description="Test parameter",
            type="str",
            required=True,
        ),
    ]

    tool = Tool(
        name="test_tool",
        description="A test tool",
        function=test_function,
        parameters=parameters,
    )

    # Test tool registration
    registry.register(tool)
    assert "test_tool" in registry.tools

    # Test duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)

    # Test tool retrieval
    retrieved_tool = registry.get_tool("test_tool")
    assert retrieved_tool == tool

    # Test non-existent tool retrieval
    with pytest.raises(KeyError, match="not found"):
        registry.get_tool("non_existent_tool")

    # Test tool unregistration
    registry.unregister("test_tool")
    assert "test_tool" not in registry.tools

    # Test non-existent tool unregistration
    with pytest.raises(KeyError, match="not found"):
        registry.unregister("non_existent_tool")


def test_tool_registry_list_tools():
    """Test listing tools in the registry."""
    registry = ToolRegistry()

    # Create and register multiple tools

    def test_function1(param1: str) -> str:
        return f"Result 1: {param1}"

    def test_function2(param1: int, param2: str = "default") -> str:
        return f"Result 2: {param1}, {param2}"

    tool1 = Tool(
        name="tool1",
        description="First test tool",
        function=test_function1,
        parameters=[
            ToolParameter(
                name="param1",
                description="First parameter",
                type="str",
                required=True,
            ),
        ],
    )

    tool2 = Tool(
        name="tool2",
        description="Second test tool",
        function=test_function2,
        parameters=[
            ToolParameter(
                name="param1",
                description="First parameter",
                type="int",
                required=True,
            ),
            ToolParameter(
                name="param2",
                description="Second parameter",
                type="str",
                required=False,
                default="default",
            ),
        ],
    )

    registry.register(tool1)
    registry.register(tool2)

    # Test listing tools
    tools_list = registry.list_tools()
    assert len(tools_list) == 2
    assert any(t["name"] == "tool1" for t in tools_list)
    assert any(t["name"] == "tool2" for t in tools_list)
    assert all("parameters" in t for t in tools_list)


def test_tool_registry_execution():
    """Test tool execution through the registry."""
    registry = ToolRegistry()

    def test_function(param1: str) -> str:
        return f"Result: {param1}"

    tool = Tool(
        name="test_tool",
        description="A test tool",
        function=test_function,
        parameters=[
            ToolParameter(
                name="param1",
                description="Test parameter",
                type="str",
                required=True,
            ),
        ],
    )

    registry.register(tool)

    # Test successful execution
    result = registry.execute_tool("test_tool", param1="test")
    assert result == "Result: test"

    # Test execution of non-existent tool
    with pytest.raises(KeyError, match="not found"):
        registry.execute_tool("non_existent_tool", param1="test")
