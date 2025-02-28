#!/usr/bin/env python3.11
import shlex
import subprocess
import sys


# Wrap all subprocess calls in a security-hardened helper
def safe_subprocess(command: list[str], timeout: int = 60) -> str:
    """Execute system commands with security constraints"""
    # Validate command input
    if not all(isinstance(arg, str) for arg in command):
    raise ValueError("Invalid command format")

    allowed_commands = ["apt-get", "pip", "python", "python3.11"]
    if command[0] not in allowed_commands:
    raise PermissionError(f"Command not allowed: {command[0]}")

    try:
        result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        )
    return result.stdout
    except subprocess.TimeoutExpired:
    raise RuntimeError(f"Command timed out: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
    raise RuntimeError(f"Command failed: {e.stderr}")

    # Update all system calls to use the safe wrapper
    def install_package(package: str):
        safe_subprocess([sys.executable, "-m", "pip", "install", package])

        def run_subprocess(command: str) -> subprocess.CompletedProcess:
            """
            Securely run a subprocess command.
            """
            # old code...
            #   result = subprocess.run(command, capture_output=True, shell=True)
            # updated for safety:
            result = subprocess.run(
            shlex.split(command),
            capture_output=True,
            text=True,
            check=False,
            )
        return result
