import logging
from typing import Dict, Any, Callable
import asyncio
import os
import shlex
import tempfile
import shutil

# Import necessary components if AutoGPT is used as a library
# e.g., from autogpt.agent import Agent
# from autogpt.config import Config
# from autogpt.memory import get_memory
# ... other specific AutoGPT imports

# Import shared components
from sutazai_agi.core.config_loader import get_setting
from sutazai_agi.models.llm_interface import LLMInterface
from sutazai_agi.memory.vector_store import VectorStoreInterface
from sutazai_agi.core.ethical_verifier import EthicalVerifier

logger = logging.getLogger(__name__)

# Define the base workspace for AutoGPT runs
AUTOGPT_BASE_WORKSPACE = os.path.join(get_setting("agent_workspace", "/opt/v3/workspace"), "autogpt_runs")
os.makedirs(AUTOGPT_BASE_WORKSPACE, exist_ok=True)

async def execute_autogpt_task(
    agent_config: Dict[str, Any],
    goal: str,
    available_tools: Dict[str, Callable], # Tools are not directly passed to subprocess in this impl
) -> Dict[str, Any]:
    """Executes a task using the AutoGPT agent logic via a subprocess.

    Args:
        agent_config: Configuration specific to this AutoGPT agent instance.
        goal: The primary objective or task for the agent.
        available_tools: Dictionary of tools (currently unused in subprocess impl).

    Returns:
        A dictionary containing the status ('success' or 'error') and the result/
        error message (captured stdout/stderr).
    """
    agent_name = agent_config.get("name", "Unknown AutoGPT Agent")
    logger.info(f"Attempting AutoGPT subprocess execution for agent: {agent_name} with goal: {goal[:100]}...")
    
    autogpt_specific_config = agent_config.get("autogpt_config", {})
    ai_name = autogpt_specific_config.get("ai_name", "SutazAI-AutoGPT")
    ai_role = autogpt_specific_config.get("ai_role", "An autonomous AI assistant")
    # AutoGPT's iterations might be controlled internally or via args, not direct config here

    # --- Prepare Subprocess Environment & Command --- 
    autogpt_cmd_path = get_setting("code_tools.autogpt_command", "autogpt") # Add this setting
    ollama_base_url = get_setting("ollama_base_url", "http://localhost:11434")
    # Use the default coding model or a general one? Let's use default_llm_model for AutoGPT reasoning.
    llm_model = get_setting("default_llm_model", "llama3") 

    # Create a temporary workspace for this specific run
    run_workspace = None
    try:
        run_workspace = tempfile.mkdtemp(prefix=f"{agent_name.replace(' ', '_')}_", dir=AUTOGPT_BASE_WORKSPACE)
        logger.info(f"Created temporary workspace for AutoGPT run: {run_workspace}")

        # Environment variables for AutoGPT (adjust based on AutoGPT's actual config methods)
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = ollama_base_url # Common way to point to custom endpoint
        env["MODEL_NAME"] = llm_model         # Specify the model
        env["OPENAI_API_KEY"] = "NotNeeded"     # Ollama usually doesn't need a key
        # Disable telemetry/external calls if AutoGPT has flags/env vars for it
        env["ALLOWLISTED_PLUGINS"] = ""        # Disable plugins unless specifically needed/configured
        env["DENYLISTED_PLUGINS"] = "" 
        env["MEMORY_BACKEND"] = "json_file"   # Use local JSON memory
        env["WORKSPACE_DIRECTORY"] = run_workspace # Tell AutoGPT to use our temp workspace
        env["EXECUTE_LOCAL_COMMANDS"] = "False" # Security: Prevent arbitrary command execution by default
        # Add other relevant env vars: SKIP_REPROMPT, HEADLESS_BROWSER (if using internal browser tools), etc.
        # Check AutoGPT .env.template or docs for relevant variables
        env["SKIP_REPROMPT"] = "true" # Needed for non-interactive execution

        # Command construction (Highly dependent on AutoGPT version/arguments)
        # Option A: Pass goal via prompt file (safer for complex goals)
        prompt_file_path = os.path.join(run_workspace, "prompt.txt")
        with open(prompt_file_path, 'w') as f:
             f.write(goal)
        command = [
             autogpt_cmd_path,
             "run", # Assuming 'run' subcommand
             "--ai-settings", "ai_settings_generated.yaml", # Tell it to generate settings?
             "--prompt-settings", "prompt_settings_generated.yaml", # Or use a prompt file?
             "--prompt", prompt_file_path, # Pass goal via file
             "--continuous", # Run without user interaction
             f"--workspace-directory={run_workspace}"
             # Add --override-ai-name, --override-ai-role ?
        ]
        
        # Option B: Pass goal via command line (simpler but might fail with complex goals)
        # command = [
        #     autogpt_cmd_path, 
        #     "run", 
        #     "--continuous", 
        #     "--prompt", goal 
        # ]
        # We might need to create a minimal ai_settings.yaml in the workspace too.
        
        # TODO: Refine command based on testing AutoGPT CLI interaction with local models.
        # For now, proceeding with Option A concept (prompt file).

        logger.debug(f"Executing AutoGPT command: {' '.join(shlex.quote(c) for c in command)}")
        logger.debug(f"AutoGPT Environment: OPENAI_API_BASE={env.get('OPENAI_API_BASE')}, MODEL_NAME={env.get('MODEL_NAME')}")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_workspace, # Run within the dedicated workspace
            env=env
        )

        stdout, stderr = await process.communicate()
        exit_code = process.returncode

        stdout_str = stdout.decode('utf-8', errors='ignore')
        stderr_str = stderr.decode('utf-8', errors='ignore')

        logger.info(f"AutoGPT subprocess for '{agent_name}' finished with exit code: {exit_code}")
        
        # Combine stdout/stderr as the primary output for now
        full_output = f"--- AutoGPT Run Log ---\nExit Code: {exit_code}\n"
        if stdout_str:
            full_output += f"\n--- STDOUT ---\n{stdout_str}"
        if stderr_str:
            full_output += f"\n--- STDERR ---\n{stderr_str}"
        
        if exit_code == 0:
            return {"status": "success", "output": full_output}
        else:
            logger.error(f"AutoGPT execution failed for '{agent_name}'. Exit Code: {exit_code}\nStderr: {stderr_str[:500]}...")
            return {"status": "error", "message": f"AutoGPT failed with exit code {exit_code}.", "output": full_output}

    except FileNotFoundError:
        logger.error(f"'{autogpt_cmd_path}' command not found. Is AutoGPT installed and in PATH? Try `pip install autogpt`.")
        return {"status": "error", "message": f"'{autogpt_cmd_path}' command not found."}
    except Exception as e:
        logger.error(f"Error during AutoGPT subprocess execution for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"AutoGPT execution failed: {e}"}
    finally:
        # Clean up the temporary workspace
        if run_workspace and os.path.exists(run_workspace):
            try:
                shutil.rmtree(run_workspace)
                logger.info(f"Cleaned up temporary AutoGPT workspace: {run_workspace}")
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temporary AutoGPT workspace '{run_workspace}': {cleanup_e}") 