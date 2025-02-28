from typing import Dict, List, Optional

#!/usr/bin/env python3.11
"""
AutoGPT agent implementation.

This module provides an autonomous agent capable of executing complex tasks
using language models and various tools.
"""

from typing import dict, list, Optional, Union
import asyncio
import json
import logging
import os
from datetime import datetime

from ai_agents.base_agent import BaseAgent, AgentError
from .config import AutoGPTConfig, validate_config
from .memory import Memory
from .model import ModelConfig, ModelManager, ModelError
from .task import Task, TaskStatus
from .tools import Tool, ToolRegistry, registry
from .prompts import PromptManager

logger = logging.getLogger(__name__)


class AutoGPTAgent(BaseAgent):
    """
        An autonomous agent that can execute complex tasks using language models and \
        tools.
    """

    def __init__(
        self,
        config: Optional[Union[Dict, AutoGPTConfig]] = None,
        name: str = "auto_gpt",
        log_dir: str = "logs/auto_gpt",
        ):
        """
        Initialize the AutoGPT agent.

        Args:
        config: Agent configuration
        name: Name of the agent instance
        log_dir: Directory for agent logs
        """
        super().__init__(name=name, log_dir=log_dir)

        self.config = validate_config(config or {})
        self.model = ModelManager(ModelConfig(**self.config.model_config))
        self.memory = Memory(
            max_messages=self.config.memory_size,
            persist_path=os.path.join(log_dir, "memory.json"))
        self.prompts = PromptManager()
        self.current_task: Optional[Task] = None

        def _validate_config(self, config: Dict) -> None:
            """
            Validate the agent configuration.

            Args:
            config: Configuration dictionary to validate

            Raises:
            AgentError: If configuration is invalid
            """
            try:
                validate_config(config)
                except Exception as e:
                raise AgentError(f"Invalid configuration: {str(e)}")

                async def initialize(self) -> None:
                """
                Initialize the agent's resources.

                Raises:
                AgentError: If initialization fails
                """
                try:
                    # Register built-in tools
                    for tool_info in registry.list_tools():
                        self.register_tool(
                        Tool(
                        name=tool_info["name"],
                        description=tool_info["description"],
                        function=registry.get_tool(tool_info["name"]).function,
                        parameters=tool_info["parameters"],
                        )
                        )

                        # Initialize performance logging
                        self.start_performance_logging()

                        logger.info("Initialized %s agent", self.name)

                        except Exception as e:
                        raise AgentError(
                            f"Failed to initialize agent: {str(e)}")

                        async def shutdown(self) -> None:
                        """
                        Clean up the agent's resources.

                        Raises:
                        AgentError: If cleanup fails
                        """
                        try:
                            # Save memory state
                            if self.memory:
                                self.memory.save()

                                # Stop performance logging
                                self.stop_performance_logging()

                                logger.info("Shut down %s agent", self.name)

                                except Exception as e:
                                raise AgentError(
                                    f"Failed to shut down agent: {str(e)}")

                                async def execute(
                                    self,
                                    task: Union[str, Dict]) -> Dict:
                                """
                                Execute a task using the agent.

                                Args:
                                                                task: Task description or \
                                    configuration dictionary

                                Returns:
                                Dict: Task execution results

                                Raises:
                                AgentError: If task execution fails
                                """
                                try:
                                    # Create task instance
                                    if isinstance(task, str):
                                        self.current_task = Task(
                                        objective=task,
                                        max_steps=self.config.max_iterations,
                                        persist_path=os.path.join(
                                            self.log_dir,
                                            "task.json"),
                                        )
                                        else:
                                        self.current_task = Task(
                                        objective=task["objective"],
                                        context=task.get("context"),
                                        max_steps=task.get(
                                            "max_steps",
                                            self.config.max_iterations),
                                        persist_path=os.path.join(
                                            self.log_dir,
                                            "task.json"),
                                        )

                                        # Clear conversation history
                                        self.memory.clear_messages()

                                        # Get initial plan
                                        plan = await self._plan_task()

                                        # Execute steps
                                        iteration = 0
                                        while (
                                            self.current_task.status not in (
                                                TaskStatus.COMPLETED,
                                                TaskStatus.FAILED)
                                                                                        and \
                                                iteration < self.config.max_iterations
                                            ):
                                            try:
                                                # Execute next step
                                                result = await self._execute_step()

                                                # Evaluate progress
                                                if iteration % 3 == 0:  # Evaluate every 3 steps
                                                    await self._evaluate_progress()

                                                    iteration += 1

                                                    except Exception as e:
                                                        # Handle step execution error
                                                        await self._handle_error(
                                                            str(e))

                                                        # Return results
                                                    return {
                                                    "status": self.current_task.status.value,
                                                    "steps": [
                                                    {"description": step.description, "status": step.status.value, "result": step.result}
                                                                                                        for step in \
                                                        self.current_task.steps
                                                        ],
                                                        "completed_at": self.current_task.completed_at.isoformat() if self.current_task.completed_at else None,
                                                        }

                                                        except Exception as e:
                                                        raise AgentError(
                                                            f"Task execution failed: {str(e)}")

                                                        finally:
                                                        self.current_task = None

                                                        async def _plan_task(
                                                            self) -> List[str]:
                                                        """
                                                        Create an execution plan for the current task.

                                                        Returns:
                                                        List[str]: List of planned steps

                                                        Raises:
                                                        AgentError: If planning fails
                                                        """
                                                        try:
                                                            # Format tools list
                                                            tools_list = self.prompts.format_tools_list(
                                                                registry.list_tools())

                                                            # Get planning prompt
                                                            prompt = self.prompts.format_template(
                                                            "task_planning",
                                                            objective=self.current_task.objective,
                                                            context=(
                                                            json.dumps(
                                                                self.current_task.context,
                                                                indent=2)
                                                            if self.current_task.context
                                                                else "No additional context provided"
                                                                ),
                                                                tools=tools_list,
                                                                )

                                                                # Get model response
                                                                response = await self.model.get_response(
                                                                system_prompt=self.prompts.get_template(
                                                                    "system"), functions=registry.list_tools()
                                                                )

                                                                # Parse response and create steps
                                                                lines = response.strip(
                                                                    ).split("\n")
                                                                plan = []
                                                                                                                                for line in \
                                                                    lines:
                                                                    if line.startswith(
                                                                        "- "):
                                                                        step = line[2:].strip()
                                                                        self.current_task.add_step(
                                                                            step)
                                                                        plan.append(
                                                                            step)

                                                                    return plan

                                                                    except Exception as e:
                                                                    raise AgentError(
                                                                        f"Failed to create task plan: {str(e)}")

                                                                    async def _execute_step(
                                                                        self) -> Optional[str]:
                                                                    """
                                                                                                                                        Execute the next step in \
                                                                        the current task.

                                                                    Returns:
                                                                    Optional[str]: Step execution result

                                                                    Raises:
                                                                    AgentError: If step execution fails
                                                                    """
                                                                    try:
                                                                        # Find next pending step
                                                                        next_step = None
                                                                        step_index = 0
                                                                        for i, step in enumerate(
                                                                            self.current_task.steps):
                                                                            if step.status == TaskStatus.PENDING:
                                                                                next_step = step
                                                                                step_index = i
                                                                            break

                                                                            if not next_step:
                                                                            return None

                                                                            # Format previous steps
                                                                            previous_steps = self.prompts.format_steps_list(
                                                                                [s.to_dict() for s in self.current_task.steps[:step_index]])

                                                                            # Format tools list
                                                                            tools_list = self.prompts.format_tools_list(
                                                                                registry.list_tools())

                                                                            # Get execution prompt
                                                                            prompt = self.prompts.format_template(
                                                                            "task_execution",
                                                                            objective=self.current_task.objective,
                                                                            step_description=next_step.description,
                                                                            context=(
                                                                            json.dumps(
                                                                                self.current_task.context,
                                                                                indent=2)
                                                                            if self.current_task.context
                                                                                else "No additional context provided"
                                                                                ),
                                                                                                                                                                previous_steps=previous_steps or \
                                                                                    "No previous steps",
                                                                                tools=tools_list,
                                                                                )

                                                                                # Start step execution
                                                                                self.current_task.start_step(
                                                                                    step_index)

                                                                                # Get model response
                                                                                response = await self.model.get_response(
                                                                                system_prompt=self.prompts.get_template(
                                                                                    "system"), functions=registry.list_tools()
                                                                                )

                                                                                # Handle function calls
                                                                                if isinstance(
                                                                                    response,
                                                                                    dict):
                                                                                    tool = registry.get_tool(
                                                                                        response["function"])
                                                                                    result = tool.execute(
                                                                                        response["arguments"])
                                                                                    self.current_task.complete_step(
                                                                                        step_index,
                                                                                        str(result))
                                                                                return str(
                                                                                    result)

                                                                                # Handle text response
                                                                                self.current_task.complete_step(
                                                                                    step_index,
                                                                                    response)
                                                                            return response

                                                                            except Exception as e:
                                                                                if next_step:
                                                                                    self.current_task.fail_step(
                                                                                        step_index,
                                                                                        str(e))
                                                                                raise AgentError(
                                                                                    f"Failed to execute step: {str(e)}")

                                                                                async def _evaluate_progress(
                                                                                    self) -> None:
                                                                                """
                                                                                                                                                                Evaluate task progress and \
                                                                                    adjust plans if needed.

                                                                                Raises:
                                                                                AgentError: If evaluation fails
                                                                                """
                                                                                try:
                                                                                    # Get completed and remaining steps
                                                                                    completed = []
                                                                                    remaining = []
                                                                                                                                                                        for step in \
                                                                                        self.current_task.steps:
                                                                                        if step.status == TaskStatus.COMPLETED:
                                                                                            completed.append(
                                                                                                step.to_dict())
                                                                                            elif step.status == TaskStatus.PENDING:
                                                                                            remaining.append(
                                                                                                step.to_dict())

                                                                                            # Format steps lists
                                                                                            completed_steps = self.prompts.format_steps_list(
                                                                                                completed)
                                                                                            remaining_steps = self.prompts.format_steps_list(
                                                                                                remaining)

                                                                                            # Get evaluation prompt
                                                                                            prompt = self.prompts.format_template(
                                                                                            "task_evaluation",
                                                                                            objective=self.current_task.objective,
                                                                                                                                                                                        completed_steps=completed_steps or \
                                                                                                "No steps completed",
                                                                                            results="See individual step results above",
                                                                                                                                                                                        remaining_steps=remaining_steps or \
                                                                                                "No remaining steps",
                                                                                            )

                                                                                            # Get model response
                                                                                            response = await self.model.get_response(
                                                                                                system_prompt=self.prompts.get_template("system"))

                                                                                            # TODO: Parse response and adjust plan if needed

                                                                                            except Exception as e:
                                                                                            raise AgentError(
                                                                                                f"Failed to evaluate progress: {str(e)}")

                                                                                            async def _handle_error(
                                                                                                self,
                                                                                                error_message: str) -> None:
                                                                                            """
                                                                                            Handle an error that occurred during task execution.

                                                                                            Args:
                                                                                            error_message: Description of the error

                                                                                            Raises:
                                                                                            AgentError: If error handling fails
                                                                                            """
                                                                                            try:
                                                                                                # Get error handling prompt
                                                                                                prompt = self.prompts.format_template(
                                                                                                "error_handling",
                                                                                                error_message=error_message,
                                                                                                context=(
                                                                                                json.dumps(
                                                                                                    self.current_task.context,
                                                                                                    indent=2)
                                                                                                if self.current_task.context
                                                                                                    else "No additional context provided"
                                                                                                    ),
                                                                                                    current_state=json.dumps(
                                                                                                    {"status": self.current_task.status.value, "steps": [s.to_dict() for s in self.current_task.steps]},
                                                                                                    indent=2,
                                                                                                    ),
                                                                                                    )

                                                                                                    # Get model response
                                                                                                    response = await self.model.get_response(
                                                                                                        system_prompt=self.prompts.get_template("system"))

                                                                                                    # TODO: Parse response and take corrective action

                                                                                                    except Exception as e:
                                                                                                    raise AgentError(
                                                                                                        f"Failed to handle error: {str(e)}")

                                                                                                    def register_tool(
                                                                                                        self,
                                                                                                        tool: Tool) -> None:
                                                                                                        """
                                                                                                        Register a new tool with the agent.

                                                                                                        Args:
                                                                                                        tool: Tool to register

                                                                                                        Raises:
                                                                                                        AgentError: If tool registration fails
                                                                                                        """
                                                                                                        try:
                                                                                                            registry.register(
                                                                                                                tool)
                                                                                                            except Exception as e:
                                                                                                            raise AgentError(
                                                                                                                f"Failed to register tool: {str(e)}")

                                                                                                            def get_tool(
                                                                                                                self,
                                                                                                                name: str) -> Tool:
                                                                                                                """
                                                                                                                Get a registered tool by name.

                                                                                                                Args:
                                                                                                                name: Name of the tool

                                                                                                                Returns:
                                                                                                                Tool: The requested tool

                                                                                                                Raises:
                                                                                                                                                                                                                                AgentError: If tool is \
                                                                                                                    not found
                                                                                                                """
                                                                                                                try:
                                                                                                                return registry.get_tool(
                                                                                                                    name)
                                                                                                                except Exception as e:
                                                                                                                raise AgentError(
                                                                                                                    f"Failed to get tool: {str(e)}")

                                                                                                                def list_tools(
                                                                                                                    self) -> List[Dict]:
                                                                                                                    """
                                                                                                                    Get a list of all registered tools.

                                                                                                                    Returns:
                                                                                                                    List[Dict]: List of tool information dictionaries

                                                                                                                    Raises:
                                                                                                                    AgentError: If tool listing fails
                                                                                                                    """
                                                                                                                    try:
                                                                                                                    return registry.list_tools()
                                                                                                                    except Exception as e:
                                                                                                                    raise AgentError(
                                                                                                                        f"Failed to list tools: {str(e)}")
