#!/usr/bin/env python3
"""
Code Sandbox Module

This module provides a secure environment for executing untrusted code
with proper isolation and resource limitations.
"""

import os
import sys
import subprocess
import tempfile
import json
import logging
import time
import resource
import shutil
from typing import Dict, Any

# Docker Python SDK
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/sandbox.log"), logging.StreamHandler()],
)
logger = logging.getLogger("CodeSandbox")


class CodeSandbox:
    """
    A secure sandbox for executing untrusted code with isolation.

    This class provides methods to execute code in a safe environment
    with limits on resources and access to the host system.
    """

    def __init__(self, config_path: str = "config/sandbox.json"):
        """
        Initialize the code sandbox.

        Args:
            config_path: Path to the sandbox configuration file
        """
        self.config_path = config_path

        # Create default config if it doesn't exist
        if not os.path.exists(config_path):
            self._create_default_config()

        # Load configuration
        self.load_config()

        # Setup sandbox environment
        self._setup_sandbox_environment()

        logger.info("Code sandbox initialized")

    def _create_default_config(self):
        """Create a default configuration file if none exists"""
        default_config = {
            "sandbox_type": "docker" if DOCKER_AVAILABLE else "process",
            "docker_image": "python:3.9-slim",
            "timeout_seconds": 30,
            "max_memory_mb": 1024,
            "max_processes": 10,
            "max_output_size_kb": 1024,
            "allowed_modules": [
                "math",
                "random",
                "datetime",
                "collections",
                "itertools",
                "functools",
                "re",
                "json",
                "csv",
                "numpy",
                "pandas",
            ],
            "blocked_modules": [
                "os",
                "sys",
                "subprocess",
                "socket",
                "requests",
                "urllib",
                "http",
                "ftplib",
                "telnetlib",
                "smtplib",
            ],
            "work_dir": "sandbox_workdir",
            "docker_options": {
                "network_disabled": True,
                "remove": True,
                "privileged": False,
                "auto_remove": True,
            },
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        # Write default config
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default sandbox configuration at {self.config_path}")

    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

                # Load configuration values
                self.sandbox_type = config.get("sandbox_type", "process")
                self.docker_image = config.get("docker_image", "python:3.9-slim")
                self.timeout_seconds = config.get("timeout_seconds", 30)
                self.max_memory_mb = config.get("max_memory_mb", 1024)
                self.max_processes = config.get("max_processes", 10)
                self.max_output_size_kb = config.get("max_output_size_kb", 1024)
                self.allowed_modules = config.get("allowed_modules", [])
                self.blocked_modules = config.get("blocked_modules", [])
                self.work_dir = config.get("work_dir", "sandbox_workdir")
                self.docker_options = config.get("docker_options", {})

                logger.info(f"Loaded sandbox configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading sandbox configuration: {str(e)}")
            raise

    def _setup_sandbox_environment(self):
        """Setup the sandbox environment"""
        # Ensure work directory exists
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir, exist_ok=True)
            logger.info(f"Created sandbox work directory: {self.work_dir}")

        # If using Docker, check Docker availability
        if self.sandbox_type == "docker":
            if DOCKER_AVAILABLE:
                try:
                    self.docker_client = docker.from_env()
                    self.docker_client.ping()
                    logger.info("Docker is available and will be used for sandboxing")
                except Exception as e:
                    logger.warning(f"Docker initialization failed: {str(e)}")
                    logger.warning("Falling back to process-based sandboxing")
                    self.sandbox_type = "process"
            else:
                logger.warning(
                    "Docker SDK not available, falling back to process-based sandboxing"
                )
                self.sandbox_type = "process"

    def _prepare_code(self, code: str) -> str:
        """
        Prepare code for execution by adding sandbox safeguards.

        Args:
            code: Raw code to prepare

        Returns:
            Modified code with safeguards
        """
        # Create a safer execution environment
        safe_imports = "\n".join(
            [f"import {module}" for module in self.allowed_modules]
        )

        # Create a wrapper to capture stdout and limit functionality
        wrapper = f"""
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Allowed imports
{safe_imports}

# Block dangerous modules
for module_name in {self.blocked_modules}:
    sys.modules[module_name] = None

# Capture stdout/stderr
output = io.StringIO()
error = io.StringIO()

def run_code():
    # Execute the provided code
    {code}

try:
    with redirect_stdout(output), redirect_stderr(error):
        run_code()
    result = {{"status": "success", "output": output.getvalue(), "error": error.getvalue()}}
except Exception as e:
    result = {{"status": "error", "error": str(e), "output": output.getvalue()}}

# Print result as JSON for parsing
import json
print(json.dumps(result))
"""
        return wrapper

    def _docker_execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in a Docker container.

        Args:
            code: Code to execute

        Returns:
            Execution results
        """
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, dir=self.work_dir
            ) as f:
                prepared_code = self._prepare_code(code)
                f.write(prepared_code.encode("utf-8"))
                temp_file = f.name

            # Run the code in a Docker container
            container = self.docker_client.containers.run(
                self.docker_image,
                command=f"python {os.path.basename(temp_file)}",
                volumes={
                    os.path.abspath(self.work_dir): {"bind": "/sandbox", "mode": "ro"}
                },
                working_dir="/sandbox",
                mem_limit=f"{self.max_memory_mb}m",
                cpu_quota=100000,  # 100% of one CPU
                network_disabled=self.docker_options.get("network_disabled", True),
                remove=self.docker_options.get("remove", True),
                privileged=self.docker_options.get("privileged", False),
                auto_remove=self.docker_options.get("auto_remove", True),
                detach=True,
            )

            try:
                # Wait for execution to complete with timeout
                container.wait(timeout=self.timeout_seconds)

                # Get output
                output = container.logs(stdout=True, stderr=False).decode("utf-8")

                # Parse the JSON result from the output
                try:
                    result_data = json.loads(output)
                    return result_data
                except json.JSONDecodeError:
                    return {
                        "status": "error",
                        "error": "Output format error",
                        "output": output,
                    }

            except Exception as e:
                # Handle timeout or other errors
                try:
                    container.kill()
                except Exception as kill_e:
                    logger.warning(f"Could not kill container {container.id}: {kill_e}")

                return {
                    "status": "error",
                    "error": f"Execution error: {str(e)}",
                    "output": "",
                }

            finally:
                # Clean up
                try:
                    container.remove(force=True)
                except Exception as remove_e:
                    logger.warning(f"Could not remove container {container.id}: {remove_e}")

                # Remove temporary file
                os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Docker execution error: {str(e)}")
            return {"status": "error", "error": f"Docker error: {str(e)}", "output": ""}

    def _process_execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in a separate Python process with resource limits.

        Args:
            code: Code to execute

        Returns:
            Execution results
        """
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, dir=self.work_dir
            ) as f:
                prepared_code = self._prepare_code(code)
                f.write(prepared_code.encode("utf-8"))
                temp_file = f.name

            # Prepare command
            cmd = [sys.executable, temp_file]

            # Run the code in a separate process
            try:
                # Start process
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=self._set_process_limits,
                    cwd=self.work_dir,
                )

                # Wait for process with timeout
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout_seconds)

                    # Limit output size
                    max_size = self.max_output_size_kb * 1024
                    stdout = stdout[:max_size]
                    stderr = stderr[:max_size]

                    # Parse output
                    try:
                        result_data = json.loads(stdout.decode("utf-8"))
                        return result_data
                    except json.JSONDecodeError:
                        return {
                            "status": "error",
                            "error": "Output format error",
                            "output": stdout.decode("utf-8"),
                        }

                except subprocess.TimeoutExpired:
                    # Kill process on timeout
                    process.kill()
                    process.wait()
                    return {
                        "status": "error",
                        "error": "Execution timeout",
                        "output": "",
                    }

            except Exception as e:
                logger.error(f"Process execution error: {str(e)}")
                return {
                    "status": "error",
                    "error": f"Process error: {str(e)}",
                    "output": "",
                }

            finally:
                # Remove temporary file
                os.unlink(temp_file)

        except Exception as e:
            logger.error(f"Process setup error: {str(e)}")
            return {
                "status": "error",
                "error": f"Process setup error: {str(e)}",
                "output": "",
            }

    def _set_process_limits(self):
        """Set resource limits for child process"""
        # Set CPU time limit
        resource.setrlimit(
            resource.RLIMIT_CPU, (self.timeout_seconds, self.timeout_seconds)
        )

        # Set memory limit
        memory_bytes = self.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        # Set process limit
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(
                resource.RLIMIT_NPROC, (self.max_processes, self.max_processes)
            )

        # Set file size limit
        file_size = self.max_output_size_kb * 2 * 1024  # Double the output size
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        Execute code in the sandbox.

        Args:
            code: Code to execute

        Returns:
            Execution results with status, output, and any errors
        """
        logger.info("Executing code in sandbox")

        start_time = time.time()

        if self.sandbox_type == "docker" and DOCKER_AVAILABLE:
            result = self._docker_execute(code)
        else:
            result = self._process_execute(code)

        execution_time = time.time() - start_time
        result["execution_time"] = execution_time

        logger.info(
            f"Code execution completed in {execution_time:.2f} seconds with status: {result['status']}"
        )

        return result

    def execute_file(self, file_path: str) -> Dict[str, Any]:
        """
        Execute a Python file in the sandbox.

        Args:
            file_path: Path to the Python file to execute

        Returns:
            Execution results with status, output, and any errors
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}",
                    "output": "",
                }

            # Read file content
            with open(file_path, "r") as f:
                code = f.read()

            # Execute the code
            return self.execute_code(code)

        except Exception as e:
            logger.error(f"Error executing file: {str(e)}")
            return {
                "status": "error",
                "error": f"File execution error: {str(e)}",
                "output": "",
            }

    def cleanup(self):
        """Clean up sandbox resources"""
        # Remove any temporary files in work directory
        for item in os.listdir(self.work_dir):
            item_path = os.path.join(self.work_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                logger.error(f"Error cleaning up {item_path}: {str(e)}")

        logger.info("Sandbox cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup when object is destroyed"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during sandbox cleanup in __del__: {e}")
            pass
