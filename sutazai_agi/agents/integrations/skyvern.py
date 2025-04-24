import logging
import os
import asyncio
import shlex
import tempfile
import shutil
from typing import Dict, Any, Optional

# Import shared components
from sutazai_agi.core.config_loader import get_setting
from sutazai_agi.utils.subprocess_utils import run_subprocess_with_workspace

logger = logging.getLogger(__name__)

# Define the base workspace for Skyvern runs
SKYVERN_BASE_WORKSPACE = os.path.join(get_setting("agent_workspace", "/opt/v3/workspace"), "skyvern_runs")
os.makedirs(SKYVERN_BASE_WORKSPACE, exist_ok=True)

async def run_skyvern(
    agent_config: Dict[str, Any],
    user_input: str, # User input might be the URL or task description
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes a Skyvern task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this agent instance.
        user_input: The input message or task (e.g., URL, instructions).
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    # Skyvern specific settings - adjust keys/defaults as needed
    command_key = "code_tools.skyvern_command"
    default_command = "skyvern run" # Placeholder
    workspace_key = "workspace.skyvern_dir" # Assume a dedicated dir setting
    default_workspace = "./workspace/skyvern_runs"

    # Skyvern might take input/URL as args
    # Customize args as needed based on Skyvern's CLI
    extra_args = [user_input] # Example: pass input directly as arg

    return await run_subprocess_with_workspace(
        process_name="Skyvern",
        command_setting_key=command_key,
        default_command=default_command,
        workspace_setting_key=workspace_key,
        default_workspace_dir=default_workspace,
        agent_config=agent_config,
        user_input=user_input, # Keep user_input for logging/config writing if needed
        timeout_seconds=timeout,
        extra_args=extra_args,
        pass_input=False, # Assuming input passed via args
        pass_config=True, # Assuming config passed via file
        run_in_workspace_cwd=True
    )

async def run_skyvern_old(
    task_description: str, 
    agent_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Runs Skyvern to perform a task using a subprocess.

    Args:
        task_description: The description of the task for Skyvern (e.g., URL, actions).
        agent_config: The configuration dictionary for this specific agent instance.

    Returns:
        A dictionary containing the status, message, and output from Skyvern.
    """
    agent_name = agent_config.get("name", "Skyvern")
    logger.info(f"Attempting Skyvern execution for agent '{agent_name}' task: {task_description[:100]}...")

    skyvern_cmd_path = get_setting("code_tools.skyvern_command", "skyvern") # Add this setting
    # Potentially other settings for Skyvern (e.g., API keys, remote endpoints?)

    # Create a temporary directory for this run if needed
    run_workspace = None
    try:
        # Skyvern might operate on URLs directly or need a workspace for configs/outputs
        run_workspace = tempfile.mkdtemp(prefix=f"{agent_name.replace(' ', '_')}_", dir=SKYVERN_BASE_WORKSPACE)
        logger.info(f"Created temporary workspace for Skyvern run: {run_workspace}")

        # --- Environment & Command (Requires understanding Skyvern CLI/API) ---
        env = os.environ.copy()
        # Set environment variables needed by Skyvern (if any)
        # env["SKYVERN_API_KEY"] = get_setting("skyvern_api_key", None)

        # Command construction - Placeholder!
        # Needs to be determined from Skyvern documentation.
        # Example: skyvern run <workflow.yaml> --url <url> --data <data>
        # We need to figure out how to pass the task_description
        # Maybe generate a temporary workflow file?
        command = [
            skyvern_cmd_path,
            "run", # Example subcommand
            # How to pass the task? As an argument? Config file?
            "--task-description", task_description, # Purely hypothetical
            # Add other necessary flags based on Skyvern CLI
            f"--output-dir={run_workspace}" # Example: tell Skyvern where to save results
        ]
        logger.warning(f"Skyvern command structure is a placeholder: {' '.join(shlex.quote(c) for c in command)}")
        # --- End Placeholder Section ---

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_workspace, # Run relative to its workspace if needed
            env=env
        )

        stdout, stderr = await process.communicate()
        exit_code = process.returncode

        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')

        logger.info(f"Skyvern subprocess for '{agent_name}' finished with exit code: {exit_code}")

        full_output = f"--- Skyvern Run Log ---\nExit Code: {exit_code}\n"
        if stdout_str:
            full_output += f"\n--- STDOUT ---\n{stdout_str}"
        if stderr_str:
            full_output += f"\n--- STDERR ---\n{stderr_str}"
        
        # Potentially parse output from run_workspace if Skyvern saves files there
        # Example: Read a results.json file

        if exit_code == 0:
            return {"status": "success", "output": full_output}
        else:
            logger.error(f"Skyvern execution failed for '{agent_name}'. Exit Code: {exit_code}\nStderr: {stderr_str[:500]}...")
            return {"status": "error", "message": f"Skyvern failed with exit code {exit_code}.", "output": full_output}

    except FileNotFoundError:
        logger.error(f"'{skyvern_cmd_path}' command not found. Is Skyvern installed and in PATH?")
        return {"status": "error", "message": f"'{skyvern_cmd_path}' command not found."}
    except Exception as e:
        logger.error(f"Error during Skyvern subprocess execution for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"Skyvern execution failed: {e}"}
    finally:
        # Clean up the temporary workspace
        if run_workspace and os.path.exists(run_workspace):
            try:
                shutil.rmtree(run_workspace)
                logger.info(f"Cleaned up temporary Skyvern workspace: {run_workspace}")
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temporary Skyvern workspace '{run_workspace}': {cleanup_e}") 