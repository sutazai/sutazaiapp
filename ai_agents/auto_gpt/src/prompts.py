"""
Prompt template module for AutoGPT agent.

This module provides templates and utilities for generating prompts
used in interactions with language models.
"""

from typing import Dict, List, Optional
import json


# System prompt template for initializing the agent
SYSTEM_PROMPT = """You are an autonomous AI agent capable of breaking down and executing complex tasks.
Your responses should be clear, logical, and focused on achieving the given objective.

You have access to various tools that can help you complete tasks. When using tools:
1. Carefully validate inputs and handle errors appropriately
2. Maintain context between steps of a task
3. Keep track of progress and adjust plans as needed

Your responses should follow this format:
1. THOUGHT: Explain your reasoning about the current situation
2. PLAN: List the next steps to take (if starting a new task or changing plans)
3. ACTION: Specify the tool to use and its parameters
4. OBSERVATION: Analyze the results of your action

Remember:
- Stay focused on the main objective
- Be thorough but efficient
- Learn from failed attempts
- Ask for clarification if needed"""


# Template for planning task execution
TASK_PLANNING_TEMPLATE = """OBJECTIVE: {objective}

CONTEXT:
{context}

AVAILABLE TOOLS:
{tools}

Please analyze this task and create a plan for execution.
Break down complex steps into manageable subtasks.
Consider potential challenges and alternative approaches."""


# Template for executing a task step
TASK_EXECUTION_TEMPLATE = """OBJECTIVE: {objective}
CURRENT STEP: {step_description}

CONTEXT:
{context}

PREVIOUS STEPS:
{previous_steps}

AVAILABLE TOOLS:
{tools}

Please execute this step and report the results.
If any issues arise, explain them and suggest solutions."""


# Template for evaluating task progress
TASK_EVALUATION_TEMPLATE = """OBJECTIVE: {objective}

COMPLETED STEPS:
{completed_steps}

RESULTS SO FAR:
{results}

REMAINING STEPS:
{remaining_steps}

Please evaluate the progress and determine if any adjustments are needed.
Consider:
1. Are we on track to achieve the objective?
2. Have any new challenges or opportunities emerged?
3. Should the remaining plan be modified?"""


# Template for handling errors
ERROR_HANDLING_TEMPLATE = """ERROR OCCURRED:
{error_message}

CONTEXT:
{context}

CURRENT STATE:
{current_state}

Please analyze this error and suggest how to proceed.
Consider:
1. What caused the error?
2. Can it be resolved with available tools?
3. Should we try a different approach?
4. Do we need to modify the overall plan?"""


class PromptManager:
    """Manages prompt templates and their rendering."""

    def __init__(self):
        """Initialize prompt manager with default templates."""
        self.templates = {
            "system": SYSTEM_PROMPT,
            "task_planning": TASK_PLANNING_TEMPLATE,
            "task_execution": TASK_EXECUTION_TEMPLATE,
            "task_evaluation": TASK_EVALUATION_TEMPLATE,
            "error_handling": ERROR_HANDLING_TEMPLATE,
        }

    def add_template(self, name: str, template: str) -> None:
        """
        Add a new template or override an existing one.

        Args:
            name: Name of the template
            template: Template string
        """
        self.templates[name] = template

    def get_template(self, name: str) -> str:
        """
        Get a template by name.

        Args:
            name: Name of the template

        Returns:
            str: Template string

        Raises:
            KeyError: If template is not found
        """
        if name not in self.templates:
            raise KeyError(f"Template not found: {name}")
        return self.templates[name]

    def format_template(self, name: str, **kwargs) -> str:
        """
        Format a template with the given variables.

        Args:
            name: Name of the template
            **kwargs: Variables to format into the template

        Returns:
            str: Formatted template string

        Raises:
            KeyError: If template is not found
            ValueError: If required variables are missing
        """
        template = self.get_template(name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in template: {str(e)}")

    def format_tools_list(self, tools: List[Dict]) -> str:
        """
        Format a list of tools into a string for prompts.

        Args:
            tools: List of tool information dictionaries

        Returns:
            str: Formatted tools list
        """
        formatted = []
        for tool in tools:
            params = [f"- {p['name']}: {p['description']}" for p in tool.get("parameters", [])]
            params_str = "\n  ".join(params) if params else "  No parameters"

            formatted.append(f"{tool['name']}: {tool['description']}\n" f"Parameters:\n  {params_str}")

        return "\n\n".join(formatted)

    def format_steps_list(self, steps: List[Dict]) -> str:
        """
        Format a list of task steps into a string for prompts.

        Args:
            steps: List of step information dictionaries

        Returns:
            str: Formatted steps list
        """
        formatted = []
        for i, step in enumerate(steps, 1):
            status = step.get("status", "pending").upper()
            result = f"\nResult: {step['result']}" if step.get("result") else ""
            formatted.append(f"{i}. {step['description']} [{status}]{result}")

        return "\n".join(formatted)
