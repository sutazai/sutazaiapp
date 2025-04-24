import logging
from typing import Dict, Any, Callable, Optional
import asyncio
import os
import shlex
import json
import tempfile
import shutil

# Import shared components
from sutazai_agi.core.config_loader import get_setting, PROJECT_ROOT
from sutazai_agi.utils.subprocess_utils import run_subprocess_with_workspace # Import the new utility

# Import components LocalAGI might need
# Assuming LocalAGI has a core class or function to interact with
# from localagi.main import LocalAGI # Example import - adjust based on actual structure

logger = logging.getLogger(__name__)

# Define base workspace for LocalAGI runs
LOCALAGI_BASE_WORKSPACE = os.path.join(PROJECT_ROOT, get_setting("workspace.localagi_dir", "./workspace/localagi_runs"))
os.makedirs(LOCALAGI_BASE_WORKSPACE, exist_ok=True)

# Placeholder for LocalAGI interaction logic
# This needs to be adapted based on LocalAGI's actual API/structure
class LocalAGIWrapper:
    def __init__(self, config: Dict[str, Any], tools: Dict[str, Callable]):
        self.config = config
        self.tools = tools
        self.agent_name = config.get("name", "LocalAGIAgent")
        # Initialize LocalAGI core here if possible
        # self.localagi_instance = LocalAGI(config=config, tools=tools) # Example
        logger.info(f"LocalAGIWrapper for {self.agent_name} initialized (Placeholder)")

    async def run(self, user_input: str) -> Dict[str, Any]:
        logger.warning(f"LocalAGIWrapper.run called for {self.agent_name}, but integration is a placeholder.")
        # Replace with actual call to LocalAGI instance
        # e.g., response = await self.localagi_instance.process_input(user_input)
        await asyncio.sleep(0.5) # Simulate work
        
        # Placeholder response
        return {
            "status": "error",
            "message": f"LocalAGI execution for agent '{self.agent_name}' is not yet implemented.",
            "output": f"(Placeholder response for LocalAGI regarding: {user_input[:50]}...)"
        }

# Refactored execution function using the utility
async def execute_localagi_task(
    agent_config: Dict[str, Any],
    user_input: str,
    available_tools: Dict[str, Callable], # available_tools is not used in subprocess mode directly
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes a LocalAGI task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this LocalAGI agent instance.
        user_input: The input message or task for the agent.
        available_tools: Dictionary of tools (currently unused in subprocess call).
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    command_key = "code_tools.localagi_command"
    default_command = "localagi_run_script.py" # Default command from settings
    workspace_key = "workspace.localagi_dir" # Base workspace directory setting
    default_workspace = "./workspace/localagi_runs"

    # Assuming LocalAGI takes config and input via files by default
    return await run_subprocess_with_workspace(
        process_name="LocalAGI",
        command_setting_key=command_key,
        default_command=default_command,
        workspace_setting_key=workspace_key,
        default_workspace_dir=default_workspace,
        agent_config=agent_config,
        user_input=user_input,
        timeout_seconds=timeout,
        # Default args assume --config config.json --input input.txt --workspace path
        pass_config=True,
        pass_input=True,
        pass_workspace_arg=True,
        run_in_workspace_cwd=True # Assume LocalAGI runs within its workspace
    )

# --- Original Subprocess Implementation (commented out or removed) ---
# async def execute_localagi_task_original(...): ... 