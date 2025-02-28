"""
Model interaction module for AutoGPT agent.

This module provides classes and utilities for interacting with language models,
including message formatting, response parsing, and error handling.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import openai

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation with the model."""

    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert message to dictionary format for API calls."""
        message = {"role": self.role, "content": self.content}
        if self.name:
            message["name"] = self.name
        if self.function_call:
            message["function_call"] = self.function_call
        return message


class ModelError(Exception):
    """Base class for model-related errors."""

    pass


class ModelConfig:
    """Configuration for model interactions."""

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        Initialize model configuration.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalty for token frequency
            presence_penalty: Penalty for token presence
            api_key: OpenAI API key (optional)
            api_base: OpenAI API base URL (optional)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        if api_key:
            openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format for API calls."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }


class ModelManager:
    """Manages interactions with language models."""

    def __init__(self, config: ModelConfig):
        """
        Initialize model manager.

        Args:
            config: Model configuration
        """
        self.config = config
        self.conversation_history: List[Message] = []

    def add_message(self, role: str, content: str, name: Optional[str] = None) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Role of the message sender
            content: Content of the message
            name: Name of the sender (optional)
        """
        message = Message(role=role, content=content, name=name)
        self.conversation_history.append(message)

    def get_messages(self) -> List[Dict]:
        """
        Get conversation history in format suitable for API calls.

        Returns:
            List[Dict]: List of message dictionaries
        """
        return [msg.to_dict() for msg in self.conversation_history]

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    async def get_response(
        self, system_prompt: Optional[str] = None, functions: Optional[List[Dict]] = None
    ) -> Union[str, Dict]:
        """
        Get a response from the model.

        Args:
            system_prompt: System prompt to prepend (optional)
            functions: Function definitions for function calling (optional)

        Returns:
            Union[str, Dict]: Model response or function call

        Raises:
            ModelError: If the API call fails
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.extend(self.get_messages())

        try:
            params = self.config.to_dict()
            params["messages"] = messages

            if functions:
                params["functions"] = functions
                params["function_call"] = "auto"

            response = await openai.ChatCompletion.acreate(**params)

            if not response.choices:
                raise ModelError("No response choices available")

            choice = response.choices[0]
            message = choice.message

            if message.get("function_call"):
                return {
                    "function": message["function_call"]["name"],
                    "arguments": json.loads(message["function_call"]["arguments"]),
                }

            self.add_message("assistant", message.content)
            return message.content

        except Exception as e:
            logger.error(f"Model API call failed: {str(e)}", exc_info=True)
            raise ModelError(f"Failed to get model response: {str(e)}")

    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with variables.

        Args:
            template: Prompt template string
            **kwargs: Variables to format into the template

        Returns:
            str: Formatted prompt
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ModelError(f"Missing required variable in prompt template: {str(e)}")
        except Exception as e:
            raise ModelError(f"Failed to format prompt template: {str(e)}")

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Text to count tokens in

        Returns:
            int: Approximate number of tokens

        Note:
            This is a very rough approximation. For accurate token counting,
            you should use the appropriate tokenizer for your model.
        """
        # Rough approximation: 4 characters per token
        return len(text) // 4
