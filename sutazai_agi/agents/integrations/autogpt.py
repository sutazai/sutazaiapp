import logging
import os
import asyncio
import shlex
import tempfile
import shutil
from typing import Dict, Any, Optional
import json

# Import shared components
from sutazai_agi.core.config_loader import get_setting, PROJECT_ROOT
from sutazai_agi.utils.subprocess_utils import run_subprocess_with_workspace

logger = logging.getLogger(__name__)

# Define the base workspace for AutoGPT runs, relative to project root
# AUTOGPT_BASE_WORKSPACE = os.path.join(PROJECT_ROOT, get_setting("workspace.autogpt_dir", "./workspace/autogpt_runs"))
# os.makedirs(AUTOGPT_BASE_WORKSPACE, exist_ok=True)

async def run_autogpt(
    agent_config: Dict[str, Any],
    user_input: str, # This is the main goal/prompt for AutoGPT
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes an AutoGPT task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this AutoGPT agent instance.
        user_input: The main goal or prompt for AutoGPT.
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    command_key = "code_tools.autogpt_command"
    # Example: 'python -m autogpt' or just 'autogpt' if installed globally
    default_command = "autogpt run" # Adjust based on actual AutoGPT setup
    workspace_key = "workspace.autogpt_dir" # Setting for AutoGPT's base workspace
    default_workspace = "./workspace/autogpt_runs"

    # AutoGPT might take the main goal via command line args
    # Example: autogpt run --ai-goal "Your goal here"
    # Consult AutoGPT documentation for exact CLI flags
    extra_args = ["--ai-goal", user_input]
    # Add other flags like --continuous, --gpt3only (adapt for local models) if needed

    return await run_subprocess_with_workspace(
        process_name="AutoGPT",
        command_setting_key=command_key,
        default_command=default_command,
        workspace_setting_key=workspace_key,
        default_workspace_dir=default_workspace,
        agent_config=agent_config, # Pass config for potential use in workspace files
        user_input=user_input, # Pass user input for logging and potential file use
        timeout_seconds=timeout,
        extra_args=extra_args,
        pass_input=False, # Goal passed via args
        pass_config=True, # Config file might be used by AutoGPT for settings
        run_in_workspace_cwd=True # AutoGPT typically needs to run in its own workspace
    )

# --- Original Implementation (commented out or removed) ---
# async def run_autogpt_original(...): ...

# Note: Workspace cleanup might be desired on success, or handled externally.
# Consider adding logic here if needed: if result['status'] == 'success': shutil.rmtree(run_workspace) 