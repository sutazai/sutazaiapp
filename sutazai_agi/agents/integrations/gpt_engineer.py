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

# Base directory for GPT-Engineer outputs, might be configurable
BASE_WORKSPACE_DIR = os.path.abspath(get_setting("workspace.gpt_engineer_dir", "./workspace/gpt_engineer_runs"))
os.makedirs(BASE_WORKSPACE_DIR, exist_ok=True)

# Define base workspace for GPT-Engineer runs (redundant if using utility's handling)
# GPT_ENGINEER_BASE_WORKSPACE = os.path.join(PROJECT_ROOT, get_setting("workspace.gpt_engineer_dir", "./workspace/gpt_engineer_runs"))
# os.makedirs(GPT_ENGINEER_BASE_WORKSPACE, exist_ok=True)

# Refactored function using the utility
async def run_gpt_engineer(
    agent_config: Dict[str, Any],
    user_input: str, # This is the main prompt for GPT-Engineer
    output_path: Optional[str] = None, # Allow specifying output project dir
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes a GPT-Engineer task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this agent instance.
        user_input: The main prompt describing the software to build.
        output_path: Optional path within the main workspace for the generated project.
                     If None, a default path might be used by gpt-engineer.
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    command_key = "code_tools.gpt_engineer_command"
    default_command = "gpt-engineer" # Assumes gpt-engineer is in PATH
    workspace_key = "workspace.gpt_engineer_dir" # Base workspace for runs
    default_workspace = "./workspace/gpt_engineer_runs"

    # GPT-Engineer takes the output path and prompt file as args
    # We create the prompt file inside the temporary workspace
    prompt_filename = "prompt.txt"
    prompt_file_path = os.path.join("{run_workspace}", prompt_filename) # Path will be formatted by utility

    extra_args = []
    # Determine output directory relative to the main agent workspace
    main_agent_workspace = get_setting("agent_workspace", "./workspace")
    if output_path:
        # Ensure output path is within the main agent workspace for safety
        resolved_output_path = os.path.abspath(os.path.join(main_agent_workspace, output_path))
        if os.path.commonpath([main_agent_workspace, resolved_output_path]) == main_agent_workspace:
            extra_args.append(resolved_output_path)
        else:
            logger.warning(f"Requested output_path '{output_path}' is outside agent workspace. Using default.")
            # Let gpt-engineer use its default relative path within its CWD
            extra_args.append("generated_project") # Example default
    else:
         extra_args.append("generated_project") # Example default

    # The user_input (prompt) is passed via a file
    # run_subprocess_with_workspace handles creating input_file_path

    return await run_subprocess_with_workspace(
        process_name="GPT-Engineer",
        command_setting_key=command_key,
        default_command=default_command,
        workspace_setting_key=workspace_key,
        default_workspace_dir=default_workspace,
        agent_config=agent_config,
        user_input=user_input, # The prompt content to write to file
        timeout_seconds=timeout,
        extra_args=extra_args, # Contains the output path
        input_filename=prompt_filename, # Specify filename for prompt
        pass_input=True, # Pass prompt via file
        config_filename="gpt_engineer_config.json", # Optional config file if gpt-engineer uses one
        pass_config=True, # Pass agent config
        pass_workspace_arg=False, # gpt-engineer doesn't use --workspace
        run_in_workspace_cwd=True # Run in temp workspace where prompt file exists
    )

# --- Original Implementation (commented out or removed) ---
# async def run_gpt_engineer_original(...): ...