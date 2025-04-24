import asyncio
import os
import shlex
import json
import tempfile
import shutil
import logging
from typing import Dict, Any, Optional, Tuple

from sutazai_agi.core.config_loader import get_setting, PROJECT_ROOT

logger = logging.getLogger(__name__)

DEFAULT_SUBPROCESS_TIMEOUT = 300  # 5 minutes

async def run_subprocess_with_workspace(
    process_name: str,
    command_setting_key: str,
    default_command: str,
    workspace_setting_key: str,
    default_workspace_dir: str,
    agent_config: Dict[str, Any],
    user_input: str,
    timeout_seconds: Optional[int] = None,
    extra_args: Optional[list] = None,
    input_filename: str = "input.txt",
    config_filename: str = "agent_config.json",
    pass_config: bool = True,
    pass_input: bool = True,
    pass_workspace_arg: bool = True,
    run_in_workspace_cwd: bool = True,
) -> Dict[str, Any]:
    """Runs an external command as a subprocess within a temporary workspace.

    Args:
        process_name: Name of the process for logging (e.g., "AutoGPT", "LocalAGI").
        command_setting_key: Key in settings.yaml for the command base (e.g., "code_tools.autogpt_command").
        default_command: Default command if setting is not found.
        workspace_setting_key: Key in settings.yaml for the base workspace directory (e.g., "workspace.autogpt_dir").
        default_workspace_dir: Default base directory if setting is not found.
        agent_config: Configuration specific to this agent instance.
        user_input: The input message or task for the agent.
        timeout_seconds: Subprocess execution timeout. Defaults to DEFAULT_SUBPROCESS_TIMEOUT.
        extra_args: Additional command line arguments to pass to the subprocess.
        input_filename: Filename for the user input within the workspace.
        config_filename: Filename for the agent config within the workspace.
        pass_config: Whether to pass the agent config via a file argument (--config).
        pass_input: Whether to pass the user input via a file argument (--input).
        pass_workspace_arg: Whether to pass the workspace path via an argument (--workspace).
        run_in_workspace_cwd: Whether to run the subprocess with the workspace as the current directory.

    Returns:
        A dictionary containing the status ('success' or 'error'), output/message,
        and potentially stdout/stderr.
    """
    agent_name = agent_config.get("name", f"Unknown {process_name} Agent")
    timeout = timeout_seconds if timeout_seconds is not None else DEFAULT_SUBPROCESS_TIMEOUT
    logger.info(f"Executing {process_name} task for agent: {agent_name} via subprocess. Input: {user_input[:100]}...")

    command_base = get_setting(command_setting_key, default_command)
    cmd_base_parts = shlex.split(command_base)

    base_workspace = os.path.join(PROJECT_ROOT, get_setting(workspace_setting_key, default_workspace_dir))
    os.makedirs(base_workspace, exist_ok=True)

    run_workspace = None
    try:
        run_workspace = tempfile.mkdtemp(prefix=f"{process_name.lower()}_{agent_name}_", dir=base_workspace)
        logger.info(f"Created temporary workspace for {process_name} run: {run_workspace}")
    except Exception as e:
        logger.error(f"Failed to create workspace for {process_name}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to create workspace: {e}"}

    config_file_path = os.path.join(run_workspace, config_filename)
    input_file_path = os.path.join(run_workspace, input_filename)

    try:
        if pass_config:
            # Avoid passing tools dict directly if present
            config_to_write = agent_config.copy()
            config_to_write.pop('tools', None)
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_write, f)
        if pass_input:
            with open(input_file_path, 'w', encoding='utf-8') as f:
                f.write(user_input)
    except IOError as e:
        logger.error(f"Failed to write config/input files for {process_name}: {e}", exc_info=True)
        shutil.rmtree(run_workspace)
        return {"status": "error", "message": f"Failed to write input/config files: {e}"}

    cmd_parts = list(cmd_base_parts)
    if pass_config:
        cmd_parts.extend(["--config", config_file_path])
    if pass_input:
        cmd_parts.extend(["--input", input_file_path])
    if pass_workspace_arg:
         cmd_parts.extend([f"--workspace", run_workspace])
    if extra_args:
        cmd_parts.extend(extra_args)

    cmd_display = shlex.join(cmd_parts)
    logger.info(f"Executing {process_name} command: {cmd_display}")

    stdout_log = []
    stderr_log = []
    process = None

    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1" # Ensure logs aren't buffered

        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=run_workspace if run_in_workspace_cwd else None,
            env=env
        )

        async def log_output(stream, log_list, stream_name):
            if stream:
                async for line in stream:
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                    log_list.append(decoded_line)
                    logger.debug(f"[{process_name} {stream_name}] {decoded_line}")

        await asyncio.wait_for(
            asyncio.gather(
                log_output(process.stdout, stdout_log, "stdout"),
                log_output(process.stderr, stderr_log, "stderr"),
                process.wait() # Wait for the process to terminate
            ),
            timeout=timeout
        )

        return_code = process.returncode
        stdout_full = "\n".join(stdout_log)
        stderr_full = "\n".join(stderr_log)

        # Assume primary output is stdout, but return both for debugging
        output = stdout_full

        if return_code == 0:
            logger.info(f"{process_name} executed successfully for '{agent_name}'.")
            return {"status": "success", "output": output, "stdout": stdout_full, "stderr": stderr_full}
        else:
            logger.error(f"{process_name} failed for '{agent_name}' with exit code {return_code}.")
            error_message = f"{process_name} exited with code {return_code}. Stderr: {stderr_full[:500]}..."
            return {"status": "error", "message": error_message, "output": output, "stdout": stdout_full, "stderr": stderr_full}

    except asyncio.TimeoutError:
        logger.error(f"{process_name} process timed out after {timeout} seconds for '{agent_name}'. Terminating.")
        if process and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5) # Wait briefly for termination
            except asyncio.TimeoutError:
                logger.warning(f"Force killing {process_name} process {process.pid} after terminate timeout.")
                process.kill()
            except ProcessLookupError:
                 logger.warning(f"{process_name} process {process.pid} already terminated.")
            except Exception as term_err:
                 logger.error(f"Error terminating/killing {process_name} process {process.pid}: {term_err}")
        return {"status": "error", "message": f"{process_name} timed out ({timeout}s)"}
    except FileNotFoundError:
        logger.error(f"{process_name} command base '{command_base}' not found or not executable.")
        return {"status": "error", "message": f"Command base '{command_base}' not found."}
    except Exception as e:
        logger.error(f"Unexpected error running {process_name} for '{agent_name}': {e}", exc_info=True)
        return {"status": "error", "message": f"Unexpected error: {e}"}
    finally:
        if run_workspace and os.path.exists(run_workspace):
            try:
                shutil.rmtree(run_workspace)
                logger.debug(f"Cleaned up temporary {process_name} workspace: {run_workspace}")
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temporary {process_name} workspace '{run_workspace}': {cleanup_e}") 