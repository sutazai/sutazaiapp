#!/usr/bin/env python3.11
"""
Configuration module for AutoGPT Agent.

This module defines configuration classes and validation mechanisms
for the AutoGPT agent's model and runtime settings.
"""

from typing import Optional, Union

from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """
    Configuration for the language model used by AutoGPT.

    Attributes:
    model_name: Name of the language model to use
    temperature: Sampling temperature for model responses
    max_tokens: Maximum number of tokens to generate
    top_p: Nucleus sampling parameter
    frequency_penalty: Frequency penalty for token repetition
    presence_penalty: Presence penalty for token repetition
    """

    model_name: str = Field(
    default="gpt-4",
    description="Name of the language model to use",
    )
    temperature: float = Field(
    default=0.7,
    ge=0.0,
    le=2.0,
    description="Sampling temperature for model responses",
    )
    max_tokens: int = Field(
    default=2000,
    ge=1,
    description="Maximum number of tokens to generate",
    )
    top_p: float = Field(
    default=1.0,
    ge=0.0,
    le=1.0,
    description="Nucleus sampling parameter",
    )
    frequency_penalty: float = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Frequency penalty for token repetition",
    )
    presence_penalty: float = Field(
    default=0.0,
    ge=-2.0,
    le=2.0,
    description="Presence penalty for token repetition",
    )

    @validator("temperature", "top_p", "frequency_penalty", "presence_penalty")
    @classmethod
    def validate_model_params(cls, value):
        """
        Validate model parameters to ensure they are within acceptable ranges.

        Args:
        value: Parameter value to validate

        Returns:
        Validated parameter value

        Raises:
        ValueError: If parameter is outside acceptable range
        """
        return value


    class AutoGPTConfig(BaseModel):
        """
        Configuration for the AutoGPT agent.

        Attributes:
        model_config: Configuration for the language model
        memory_size: Maximum number of messages to keep in memory
        max_iterations: Maximum number of iterations for task execution
        verbose_mode: Enable verbose logging
        """

        model_config: ModelConfig = Field(
        default_factory=ModelConfig,
        description="Configuration for the language model",
        )
        memory_size: int = Field(
        default=10,
        ge=1,
        description="Maximum number of messages to keep in memory",
        )
        max_iterations: int = Field(
        default=5,
        ge=1,
        description="Maximum number of iterations for task execution",
        )
        verbose_mode: bool = Field(
        default=False,
        description="Enable verbose logging",
        )

        @validator("memory_size", "max_iterations")
        @classmethod
        def validate_agent_params(cls, value):
            """
            Validate agent parameters to ensure they are within acceptable ranges.

            Args:
            value: Parameter value to validate

            Returns:
            Validated parameter value

            Raises:
            ValueError: If parameter is outside acceptable range
            """
            return value


        def validate_config(config: Optional[Union[dict, AutoGPTConfig]] = None) -> AutoGPTConfig:
            """
            Validate and create an AutoGPT configuration.

            Args:
            config: Configuration dictionary or AutoGPTConfig instance

            Returns:
            Validated AutoGPTConfig instance
            """
            if config is None:
                return AutoGPTConfig()

            if isinstance(config, dict):
                return AutoGPTConfig(**config)

            if isinstance(config, AutoGPTConfig):
                return config

            raise ValueError("Invalid configuration type. Must be dict or AutoGPTConfig.")


            def main():
                """
                Demonstration of configuration validation.
                """
                # Example usage
                sample_config = {
                "model_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                },
                "memory_size": 15,
                "max_iterations": 10,
                "verbose_mode": True,
                }

                validated_config = validate_config(sample_config)
                print("Validated Configuration:", validated_config)


                if __name__ == "__main__":
                    main()
