#!/usr/bin/env python3
"""
Performance Issue Resolver for SutazAI

This script identifies and resolves common performance issues in the SutazAI system.
Key features:
    - Identifies zombie processes and leftover bash processes
- Kills runaway bash processes with high CPU usage
- Ensures proper shutdown of backend and related processes
- Cleans up temporary files and caches
"""

import os
import sys
import time
import signal
import logging
import shutil
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/opt/sutazaiapp/logs/performance_fix.log"),
    ],
)
logger = logging.getLogger(__name__)

def kill_high_cpu_processes():
    """Find and kill processes with excessive CPU usage."""
    logger.info("Checking for high CPU usage processes...")

    try:
        # Find processes with high CPU usage - using absolute paths and fixed commands
        high_cpu_cmd = ["/usr/bin/ps", "-eo", "pid,pcpu,pmem,user,args", "--sort=-pcpu"]
        result = subprocess.run(high_cpu_cmd, capture_output=True, text=True, check=False)  # nosec B603 - Using fixed command with absolute path
        logger.info(f"Top CPU processes:\n{result.stdout}")

        # Look for bash processes with high CPU
        bash_cmd = ["/usr/bin/ps", "aux"]
        result = subprocess.run(bash_cmd, capture_output=True, text=True, check=False)  # nosec B603 - Using fixed command with absolute path

        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 11:
                continue

            # Check if this is a bash process
            if "bash" not in parts[10:]:
                continue

            pid = parts[1]
            try:
                cpu_usage = float(parts[2])
            except ValueError:
                continue

            # Kill bash processes with high CPU usage
            if cpu_usage > 80.0:
                logger.info(f"Killing high CPU bash process: {pid} (CPU: {cpu_usage}%)")
                try:
                    pid_num = int(pid)
                    os.kill(pid_num, signal.SIGKILL)
                except (ValueError, ProcessLookupError) as e:
                    logger.error(f"Failed to kill process {pid}: {e}")

    except Exception as e:
        logger.error(f"Error killing high CPU processes: {e}")

def restart_backend_service():
    """Stop and restart the backend service."""
    logger.info("Restarting backend service...")

    try:
        # Kill any existing backend processes
        pkill_cmd = ["/usr/bin/pkill", "-f", "python -m backend.backend_main"]
        subprocess.run(pkill_cmd, capture_output=True, text=True, check=False)  # nosec B603 - Using fixed command with absolute path
        time.sleep(2)  # Allow time for processes to terminate

        # Start backend service in background
        logger.info("Starting backend service...")
        # Use absolute path for python executable
        python_exec = sys.executable

        # Prepare the log file
        log_path = os.path.join("/opt/sutazaiapp/logs/backend_restart.log")
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

        # Use Popen with a list for command arguments
        with open(log_path, 'a') as log_file:
            process = subprocess.Popen(  # nosec B603 - Using fixed command arguments with python executable path
                [python_exec, "-m", "backend.backend_main"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd="/opt/sutazaiapp",
                start_new_session=True
            )

        logger.info(f"Started backend service with PID {process.pid}")

    except Exception as e:
        logger.error(f"Error restarting backend service: {e}")

def clean_caches():
    """Clean up temporary files and caches."""
    logger.info("Cleaning caches and temporary files...")

    try:
        cache_dirs = [
            "/opt/sutazaiapp/.mypy_cache",
            "/opt/sutazaiapp/.ruff_cache",
        ]

        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                logger.info(f"Removing cache directory: {cache_dir}")
                shutil.rmtree(cache_dir, ignore_errors=True)

        # Handle temporary files in a secure way
        temp_dir = tempfile.gettempdir()  # Use system's temp directory securely
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                if item.startswith("sutazai"):
                    item_path = os.path.join(temp_dir, item)
                    # Validate the path is safely within temp directory and not a symlink
                    if os.path.abspath(item_path).startswith(temp_dir) and not os.path.islink(item_path):
                        logger.info(f"Removing temporary item: {item_path}")
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        else:
                            os.remove(item_path)

        logger.info("Cache cleanup completed")

    except Exception as e:
        logger.error(f"Error cleaning caches: {e}")

def main():
    """Main entry point for the performance issue resolver."""
    logger.info("Starting performance issue resolver...")

    # Kill high CPU processes
    kill_high_cpu_processes()

    # Clean caches
    clean_caches()

    # Restart backend service
    restart_backend_service()

    logger.info("Performance issue resolution completed")

if __name__ == "__main__":
    main()
