from typing import Dict, List

#!/usr/bin/env python3.11
"""
Configuration module for AutoGPT agent.

This module defines the configuration schema and validation for AutoGPT agents.
"""

from typing import dict, list, Union
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the language model used by AutoGPT."""

    model_name: str = Field(
        default="gpt-4",
        description="Name of the language model to use")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the model")
    max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum number of tokens per API call")


    class ToolConfig(BaseModel):
        """Configuration for a tool that AutoGPT can use."""

        name: str = Field(description="Name of the tool")
        description: str = Field(
            description="Description of what the tool does")
        parameters: Dict[str, Dict] = Field(
            default_factory=dict,
            description="Parameters that the tool accepts")
        required: List[str] = Field(
            default_factory=list,
            description="List of required parameters")


        class AutoGPTConfig(BaseModel):
            """Main configuration for AutoGPT agent."""

            agent_name: str = Field(
                default="auto_gpt",
                description="Name of the AutoGPT agent instance")
            max_iterations: int = Field(
                default=10,
                gt=0,
                description="Maximum number of iterations for task execution")
            verbose_mode: bool = Field(
                default=False,
                description="Enable verbose logging")
            log_dir: str = Field(
                default="logs/auto_gpt",
                description="Directory for agent logs")
            model_config: ModelConfig = Field(
                default_factory=ModelConfig,
                description="Configuration for the language model")
            tools: List[ToolConfig] = Field(
                default_factory=list,
                description="List of available tools")
            memory_size: int = Field(
                default=10,
                gt=0,
                description="Number of previous interactions to keep in memory")

            class Config:
                """Pydantic model configuration."""

                arbitrary_types_allowed = True


                def validate_config(
                    config: Union[Dict, AutoGPTConfig]) -> AutoGPTConfig:
                    """
                                        Validate and \
                        convert configuration to AutoGPTConfig instance.

                    Args:
                                        config: Dictionary or \
                        AutoGPTConfig instance containing configuration

                    Returns:
                    AutoGPTConfig: Validated configuration instance

                    Raises:
                    ValueError: If configuration is invalid
                    """
                    if isinstance(config, dict):
                    return AutoGPTConfig(**config)
                    elif isinstance(config, AutoGPTConfig):
                return config
                else:
            raise ValueError(f"Invalid config type: {type(config)}")


            def get_default_config() -> AutoGPTConfig:
                """
                Get default configuration for AutoGPT agent.

                Returns:
                AutoGPTConfig: Default configuration instance
                """
            return AutoGPTConfig()
