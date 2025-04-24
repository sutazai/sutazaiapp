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

# Define the base workspace for AgentZero runs
AGENTZERO_BASE_WORKSPACE = os.path.join(get_setting("agent_workspace", "/opt/v3/workspace"), "agentzero_runs")
os.makedirs(AGENTZERO_BASE_WORKSPACE, exist_ok=True)

async def run_agentzero(
    agent_config: Dict[str, Any],
    user_input: str,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Executes an AgentZero task using the generic subprocess runner.

    Args:
        agent_config: Configuration specific to this agent instance.
        user_input: The input message or task for the agent.
        timeout: Optional timeout in seconds.

    Returns:
        Result dictionary from run_subprocess_with_workspace.
    """
    # AgentZero specific settings - adjust keys/defaults as needed
    command_key = "code_tools.agentzero_command"
    default_command = "agentzero run" # Placeholder
    workspace_key = "workspace.agentzero_dir" # Assume a dedicated dir setting
    default_workspace = "./workspace/agentzero_runs"

    # Example: AgentZero might take input via flags instead of file
    # Customize args as needed based on AgentZero's CLI
    extra_args = ["--prompt", user_input]

    return await run_subprocess_with_workspace(
        process_name="AgentZero",
        command_setting_key=command_key,
        default_command=default_command,
        workspace_setting_key=workspace_key,
        default_workspace_dir=default_workspace,
        agent_config=agent_config,
        user_input=user_input,
        timeout_seconds=timeout,
        extra_args=extra_args,
        pass_input=False, # Assuming input passed via args
        pass_config=True, # Assuming config passed via file
        run_in_workspace_cwd=True
    )

async def run_agentzero_old(
    task_description: str,
    agent_config: Dict[str, Any] # Pass agent config for specific settings
) -> Dict[str, Any]:
    """Runs AgentZero to perform a task using a subprocess.

    Args:
        task_description: The description of the task for AgentZero.
        agent_config: The configuration dictionary for this specific agent instance.

    Returns:
        A dictionary containing the status, message, and output from AgentZero.
    """
    agent_name = agent_config.get("name", "AgentZero")
    logger.info(f"Attempting AgentZero execution for agent '{agent_name}' task: {task_description[:100]}...")

    agentzero_cmd_path = get_setting("code_tools.agentzero_command", "agentzero") # Add this setting
    ollama_base_url = get_setting("ollama_base_url", "http://localhost:11434")
    llm_model = get_setting("default_llm_model", "llama3") 

    # Create a temporary directory for this run if needed
    run_workspace = None
    try:
        run_workspace = tempfile.mkdtemp(prefix=f"{agent_name.replace(' ', '_')}_", dir=AGENTZERO_BASE_WORKSPACE)
        logger.info(f"Created temporary workspace for AgentZero run: {run_workspace}")

        # --- Environment & Command (Requires understanding AgentZero CLI) ---
        env = os.environ.copy()
        # Set environment variables needed by AgentZero (e.g., for LLM endpoint/model)
        # Example (adjust based on AgentZero docs):
        # env["AGENTZERO_LLM_ENDPOINT"] = ollama_base_url 
        # env["AGENTZERO_LLM_MODEL"] = llm_model
        # env["AGENTZERO_WORKSPACE"] = run_workspace

        # Command construction - Placeholder!
        # Needs to be determined from AgentZero documentation.
        # How does it take the task? Config file? Arguments?
        command = [
            agentzero_cmd_path,
            "run", # Example subcommand
            "--task", task_description, # Example argument passing
            # Add other necessary flags based on AgentZero CLI
        ]
        logger.warning(f"AgentZero command structure is a placeholder: {' '.join(shlex.quote(c) for c in command)}")
        # --- End Placeholder Section ---

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_workspace, # Run in its dedicated workspace
            env=env
        )

        stdout, stderr = await process.communicate()
        exit_code = process.returncode

        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')

        logger.info(f"AgentZero subprocess for '{agent_name}' finished with exit code: {exit_code}")
        
        full_output = f"--- AgentZero Run Log ---\nExit Code: {exit_code}\n"
        if stdout_str:
            full_output += f"\n--- STDOUT ---\n{stdout_str}"
        if stderr_str:
            full_output += f"\n--- STDERR ---\n{stderr_str}"

        if exit_code == 0:
            return {"status": "success", "output": full_output}
        else:
            logger.error(f"AgentZero execution failed for '{agent_name}'. Exit Code: {exit_code}\nStderr: {stderr_str[:500]}...")
            return {"status": "error", "message": f"AgentZero failed with exit code {exit_code}.", "output": full_output}

    except FileNotFoundError:
        logger.error(f"'{agentzero_cmd_path}' command not found. Is AgentZero installed and in PATH?")
        return {"status": "error", "message": f"'{agentzero_cmd_path}' command not found."}
    except Exception as e:
        logger.error(f"Error during AgentZero subprocess execution for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"AgentZero execution failed: {e}"}
    finally:
        # Clean up the temporary workspace
        if run_workspace and os.path.exists(run_workspace):
            try:
                shutil.rmtree(run_workspace)
                logger.info(f"Cleaned up temporary AgentZero workspace: {run_workspace}")
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temporary AgentZero workspace '{run_workspace}': {cleanup_e}") 