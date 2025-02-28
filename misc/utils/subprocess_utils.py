#!/usr/bin/env python3.11
"""
SutazAI Subprocess Utilities

This module provides secure subprocess handling utilities for the SutazAI application.
"""

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def validate_path(path: Union[str, Path], must_exist: bool = True) -> bool:
    """
    Validate that a path is safe to use.
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        
    Returns:
        bool: True if path is safe, False otherwise
    """
    try:
        resolved_path = Path(path).resolve()
        path_str = str(resolved_path)

        # Ensure path is within project directory
        if not path_str.startswith("/opt/sutazaiapp/"):
            return False

        # Check existence if required
        if must_exist and not resolved_path.exists():
            return False

        return True
    except (TypeError, ValueError):
        return False


def validate_command(command: Union[str, List[str]]) -> List[str]:
    """
    Validate and sanitize command input.
    
    Args:
        command: Command as string or list of strings
        
    Returns:
        List of validated command parts
        
    Raises:
        ValueError: If command is invalid or potentially dangerous
    """
    if isinstance(command, str):
        command_parts = shlex.split(command)
    else:
        command_parts = command

    # Basic validation
    if not command_parts:
        raise ValueError("Empty command")

    # Validate each part
    for part in command_parts:
        if not isinstance(part, str):
            raise ValueError(f"Invalid command part: {part}")
        if ";" in part or "&&" in part or "||" in part:
            raise ValueError(f"Command contains potentially dangerous characters: {part}")

    return command_parts


def run_command(
    command: Union[str, List[str]],
    shell: bool = False,
    check: bool = True,
    capture_output: bool = True,
    timeout: Optional[int] = 30,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """
    Safely run a command with proper validation and error handling.
    
    Args:
        command: Command to run (string or list)
        shell: Whether to run command in shell
        check: Whether to raise CalledProcessError on non-zero exit
        capture_output: Whether to capture stdout/stderr
        timeout: Command timeout in seconds
        cwd: Working directory for command
        env: Environment variables for command
        
    Returns:
        CompletedProcess instance
        
    Raises:
        ValueError: If command validation fails
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out
    """
    # Validate command
    cmd_parts = validate_command(command)

    try:
        # Run command with safety checks
        result = subprocess.run(
            cmd_parts if not shell else " ".join(cmd_parts),
            shell=shell,
            check=check,
            capture_output=capture_output,
            timeout=timeout,
            cwd=cwd,
            env=env,
            text=True,
        )
        return result

    except subprocess.TimeoutExpired as e:
        raise subprocess.TimeoutExpired(
            cmd_parts,
            timeout,
            output=e.output,
            stderr=e.stderr,
        )
    except subprocess.CalledProcessError as e:
        if check:
            raise
        return e
    except Exception as e:
        raise RuntimeError(f"Command execution failed: {e}")


def run_python_module(
    module: str,
    args: List[str],
    python_cmd: str = "python3.11",
    **kwargs,
) -> subprocess.CompletedProcess:
    """
    Safely run a Python module as a command.
    
    Args:
        module: Python module name
        args: Module arguments
        python_cmd: Python executable to use
        **kwargs: Additional arguments for run_command
        
    Returns:
        CompletedProcess instance
    """
    cmd = [python_cmd, "-m", module] + args
    return run_command(cmd, **kwargs)


def get_command_output(
    command: Union[str, List[str]],
    **kwargs,
) -> Tuple[str, str]:
    """
    Get command output safely.
    
    Args:
        command: Command to run
        **kwargs: Additional arguments for run_command
        
    Returns:
        Tuple of (stdout, stderr)
    """
    result = run_command(command, capture_output=True, **kwargs)
    return result.stdout, result.stderr
