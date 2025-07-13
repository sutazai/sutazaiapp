#!/usr/bin/env python3.11
"""Tool management module for AutoGPT agent.

This module provides classes and utilities for managing tools that can be used by the agent,
including tool registration, validation, and execution.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

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
        """Validate parameters against tool's parameter specifications.

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
                        validated[param.name] = value
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for parameter {param.name}: {e!s}")

        return validated

    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Parameters to pass to the tool function

        Returns:
            Any: Result of tool execution

        Raises:
            ValueError: If parameters are invalid
            Exception: If tool execution fails
        """
        try:
            validated_params = self.validate_params(kwargs)
            return self.function(**validated_params)
        except Exception as e:
            logger.error(f"Tool execution failed: {e!s}", exc_info=True)
            raise


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool.

        Args:
            tool: Tool to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: Name of tool to unregister

        Raises:
            KeyError: If tool not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        del self.tools[name]

    def get_tool(self, name: str) -> Tool:
        """Get a tool by name.

        Args:
            name: Name of tool to get

        Returns:
            Tool: The requested tool

        Raises:
            KeyError: If tool not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """Get a list of all registered tools.

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

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name.

        Args:
            name: Name of tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            Any: Result of tool execution

        Raises:
            KeyError: If tool not found
            ValueError: If parameters are invalid
            Exception: If tool execution fails
        """
        tool = self.get_tool(name)
        return tool.execute(**kwargs)
