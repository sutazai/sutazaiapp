"""
Tool management module for AutoGPT agent.

This module provides classes and utilities for managing tools that can be used by the agent,
including tool registration, validation, and execution.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import inspect
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Represents a parameter for a tool."""

    name: str
    description: str
    type: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    """Represents a tool that can be used by the agent."""

    name: str
    description: str
    function: Callable
    parameters: List[ToolParameter]

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against tool's parameter specifications.

        Args:
            params: Parameters to validate

        Returns:
            Dict[str, Any]: Validated and processed parameters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        validated = {}

        # Check for required parameters
        for param in self.parameters:
            if param.name not in params:
                if param.required:
                    raise ValueError(f"Missing required parameter: {param.name}")
                validated[param.name] = param.default
            else:
                value = params[param.name]
                try:
                    # Basic type conversion
                    if param.type == "str":
                        validated[param.name] = str(value)
                    elif param.type == "int":
                        validated[param.name] = int(value)
                    elif param.type == "float":
                        validated[param.name] = float(value)
                    elif param.type == "bool":
                        validated[param.name] = bool(value)
                    elif param.type == "list":
                        if not isinstance(value, list):
                            raise ValueError(f"Parameter {param.name} must be a list")
                        validated[param.name] = value
                    elif param.type == "dict":
                        if not isinstance(value, dict):
                            raise ValueError(f"Parameter {param.name} must be a dictionary")
                        validated[param.name] = value
                    else:
                        # For unknown types, pass through as-is
                        validated[param.name] = value
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for parameter {param.name}: {str(e)}")

        return validated

    def execute(self, params: Dict[str, Any]) -> Any:
        """
        Execute the tool with the given parameters.

        Args:
            params: Parameters to pass to the tool

        Returns:
            Any: Result of tool execution

        Raises:
            Exception: If tool execution fails
        """
        try:
            validated_params = self.validate_params(params)
            return self.function(**validated_params)
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
            raise


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a new tool.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self.tools[tool.name] = tool

    def register_function(self, name: str, description: str, parameters: List[ToolParameter]) -> Callable:
        """
        Decorator to register a function as a tool.

        Args:
            name: Name of the tool
            description: Description of what the tool does
            parameters: List of parameter specifications

        Returns:
            Callable: Decorator function
        """

        def decorator(func: Callable) -> Callable:
            self.register(Tool(name=name, description=description, function=func, parameters=parameters))
            return func

        return decorator

    def get_tool(self, name: str) -> Tool:
        """
        Get a registered tool by name.

        Args:
            name: Name of the tool to get

        Returns:
            Tool: The requested tool

        Raises:
            KeyError: If no tool is registered with the given name
        """
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")
        return self.tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all registered tools.

        Returns:
            List[Dict[str, Any]]: List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": param.name,
                        "description": param.description,
                        "type": param.type,
                        "required": param.required,
                        "default": param.default,
                    }
                    for param in tool.parameters
                ],
            }
            for tool in self.tools.values()
        ]


# Create a global tool registry instance
registry = ToolRegistry()


def register_tool(name: str, description: str, parameters: List[ToolParameter]) -> Callable:
    """
    Decorator to register a function as a tool in the global registry.

    Args:
        name: Name of the tool
        description: Description of what the tool does
        parameters: List of parameter specifications

    Returns:
        Callable: Decorator function
    """
    return registry.register_function(name, description, parameters)


# Example built-in tool
@register_tool(
    name="get_current_time",
    description="Get the current date and time",
    parameters=[
        ToolParameter(
            name="format",
            description="Format string for the datetime",
            type="str",
            required=False,
            default="%Y-%m-%d %H:%M:%S",
        )
    ],
)
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current date and time in the specified format."""
    return datetime.now().strftime(format)
