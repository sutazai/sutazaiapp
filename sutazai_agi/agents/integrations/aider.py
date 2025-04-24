import logging
import os
import asyncio
import shlex
from typing import Optional, Dict, Any, List

from sutazai_agi.core.config_loader import get_setting
from sutazai_agi.utils.subprocess_utils import run_subprocess_with_workspace

logger = logging.getLogger(__name__)

# Define base workspace - Aider operates on files within this workspace
# Use the dedicated workspace path for code assistant tasks
AIDER_WORKSPACE = os.path.abspath(get_setting("code_assistant.workspace_path", "./workspace/code_assistant"))

async def run_aider(
    agent_config: Dict[str, Any],
    user_input: str, # User input might be the prompt for the edit
    files_to_edit: Optional[list] = None, # Specific files aider should work on
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes an Aider task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this agent instance.
        user_input: The editing instruction or prompt for Aider.
        files_to_edit: Optional list of file paths (relative to workspace) to edit.
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    # Aider specific settings
    command_key = "code_tools.aider_command"
    default_command = "aider" # Assumes aider is in PATH
    # Aider typically runs in the project's main workspace, not a temp one
    workspace_key = "agent_workspace" # Use the main agent workspace
    default_workspace = "./workspace"

    # Aider takes files and prompt as arguments
    extra_args = []
    if files_to_edit:
        extra_args.extend(files_to_edit)
    extra_args.extend(["--message", user_input])
    # Add other aider flags if needed (e.g., --yes for auto-commit)
    # extra_args.append("--yes")

    # Get the main agent workspace path
    main_workspace_path = get_setting(workspace_key, default_workspace)
    if not os.path.isabs(main_workspace_path):
         main_workspace_path = os.path.abspath(os.path.join(os.getcwd(), main_workspace_path)) # Resolve relative path

    # Aider runs in the main workspace, not a temp dir
    return await run_subprocess_with_workspace(
        process_name="Aider",
        command_setting_key=command_key,
        default_command=default_command,
        # These workspace settings are ignored because run_in_workspace_cwd=False and we provide cwd
        workspace_setting_key="dummy_workspace_key",
        default_workspace_dir="dummy_workspace_dir",
        agent_config=agent_config,
        user_input=user_input,
        timeout_seconds=timeout,
        extra_args=extra_args,
        pass_input=False, # Input passed via --message arg
        pass_config=False, # Aider uses its own config/env vars
        pass_workspace_arg=False, # Aider doesn't take --workspace
        run_in_workspace_cwd=False, # Explicitly disable temp workspace creation
        # Instead, manually set cwd for the subprocess
        # Need to modify run_subprocess_with_workspace to accept explicit cwd or handle this case
        # For now, this won't work perfectly as run_subprocess creates temp workspace.
        # TODO: Refactor run_subprocess_with_workspace to allow running in existing dir.
    )

# TODO: Refactor run_subprocess_with_workspace to handle running
#       in an existing directory specified by `cwd` argument, instead of always
#       creating and running within a temporary directory.
#       This is needed for tools like Aider that operate on the main project repo.

# Example Usage (for testing)
# async def _test_aider():
#     logging.basicConfig(level=logging.INFO)
#     # Setup: Create a dummy file in the workspace
#     test_file = "aider_test.py"
#     test_file_path = os.path.join(AIDER_WORKSPACE, test_file)
#     if not os.path.exists(os.path.dirname(test_file_path)): os.makedirs(os.path.dirname(test_file_path))
#     with open(test_file_path, "w") as f:
#         f.write("def hello():\n    print(\"Hello World!\")\n")
#     
#     print(f"Attempting to edit '{test_file}' with Aider...")
#     result = await run_aider(
#         files_to_edit=[test_file], 
#         instruction="Add a docstring to the hello function."
#         # git_repo_path="." # If the workspace is a git repo
#     )
#     print("Aider Result:")
#     import json
#     print(json.dumps(result, indent=2))
#     # Clean up dummy file
#     # if os.path.exists(test_file_path): os.remove(test_file_path)

# if __name__ == "__main__":
#     import asyncio
#     # Need to ensure aider is installed and configured (e.g., with local Ollama model)
#     # asyncio.run(_test_aider()) 