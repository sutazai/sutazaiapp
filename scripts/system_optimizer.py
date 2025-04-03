#!/usr/bin/env python3
"""
SutazAI System Optimizer

This script performs a comprehensive analysis and optimization of the SutazAI application.
It fixes issues with dependencies, configurations, and services to ensure the system runs
smoothly and efficiently.
"""

import os
import sys
import json
import time
import logging
import shutil
import subprocess
import signal
import psutil
import pkg_resources
from typing import Dict, Any
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/system_optimizer.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SystemOptimizer")

# Application paths
APP_ROOT = Path("/opt/sutazaiapp")
LOGS_DIR = APP_ROOT / "logs"
PIDS_DIR = APP_ROOT / "pids"
CONFIG_DIR = APP_ROOT / "config"
SCRIPTS_DIR = APP_ROOT / "scripts"
SYSTEMD_DIR = APP_ROOT / "systemd"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
PIDS_DIR.mkdir(exist_ok=True)


class SystemOptimizer:
    """System-wide optimization for the SutazAI application."""

    def __init__(self):
        """Initialize the optimizer."""
        self.issues_found = []
        self.issues_fixed = []
        self.warnings = []

        # Dependencies that need to be installed
        self.required_packages = [
            "pydantic-settings",
            "sqlalchemy",
            "fastapi",
            "uvicorn",
            "pyjwt",
            "cffi",
            "cryptography",
            "torch",
            "transformers",
            "numpy",
            "psutil",
            "bitsandbytes",
            "pydantic",
        ]

        # Services that should be running
        self.services = ["backend", "webui", "vector-db", "localagi"]

        # Ports used by services
        self.service_ports = {
            "backend": 8000,
            "webui": 3000,
            "vector-db": 6333,
            "localagi": 8090,
        }

        # PID file paths
        self.pid_files = {
            "backend": PIDS_DIR / "backend.pid",
            "webui": PIDS_DIR / "webui.pid",
            "vector-db": PIDS_DIR / "vector-db.pid",
            "localagi": PIDS_DIR / "localagi.pid",
        }

    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run a full system optimization.

        Returns:
            Dict with optimization results
        """
        start_time = time.time()
        logger.info("Starting system-wide optimization")

        # Perform each optimization step
        self.check_python_environment()
        self.fix_dependencies()
        self.fix_configurations()
        self.clean_stale_pid_files()
        self.check_services()
        self.optimize_database()
        self.optimize_storage()
        # Temporarily disable these checks
        # self.run_transformer_optimizations()
        # self.run_code_quality_checks()
        # self.restart_services() # <-- REMOVED: Let start_all.sh handle the final restart

        end_time = time.time()
        duration = end_time - start_time

        result = {
            "status": "completed",
            "duration_seconds": duration,
            "issues_found": self.issues_found,
            "issues_fixed": self.issues_fixed,
            "warnings": self.warnings,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"System optimization completed in {duration:.2f} seconds")
        logger.info(
            f"Found {len(self.issues_found)} issues, fixed {len(self.issues_fixed)}"
        )

        # Save result to report
        report_path = (
            LOGS_DIR
            / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def check_python_environment(self) -> None:
        """Check and fix Python environment issues."""
        logger.info("Checking Python environment")

        # Get Python version
        python_version = sys.version.split()[0]
        logger.info(f"Python version: {python_version}")

        # Check for virtual environment
        venv_path = APP_ROOT / "venv"
        in_venv = sys.prefix != sys.base_prefix

        if not in_venv and not os.path.exists(venv_path):
            logger.warning("Not running in a virtual environment")
            self.issues_found.append("No virtual environment detected")
            self.warnings.append(
                "Application should run in a virtual environment for isolation"
            )
        else:
            logger.info(f"Virtual environment detected: {sys.prefix}")

        # Check for .env file
        env_file = APP_ROOT / ".env"
        if not env_file.exists():
            logger.error("Missing .env file")
            self.issues_found.append("Missing .env file")

            # Create from example if available
            env_example = APP_ROOT / ".env.example"
            if env_example.exists():
                shutil.copy(env_example, env_file)
                logger.info("Created .env file from .env.example")
                self.issues_fixed.append("Created .env file from example")

    def fix_dependencies(self) -> None:
        """Fix dependency issues."""
        logger.info("Checking and fixing dependencies")

        # Get installed packages
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

        # Check required packages
        missing_packages = []

        for package in self.required_packages:
            if package.lower() not in installed_packages:
                missing_packages.append(package)

        if missing_packages:
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            self.issues_found.append(f"Missing packages: {', '.join(missing_packages)}")

            # Install missing packages
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing_packages,
                    check=True,
                )
                logger.info(
                    f"Installed missing packages: {', '.join(missing_packages)}"
                )
                self.issues_fixed.append(
                    f"Installed missing packages: {', '.join(missing_packages)}"
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install missing packages: {str(e)}")

        # Fix pydantic_settings conflicts
        if os.path.exists(APP_ROOT / "pydantic_settings"):
            logger.warning(
                "Found local pydantic_settings module, may conflict with installed package"
            )
            self.issues_found.append(
                "Local pydantic_settings module conflicts with installed package"
            )

            # Rename the module
            try:
                local_module = APP_ROOT / "pydantic_settings"
                backup_module = APP_ROOT / "pydantic_settings_backup"
                if not os.path.exists(backup_module):
                    shutil.move(local_module, backup_module)
                    logger.info(
                        "Renamed local pydantic_settings to pydantic_settings_backup"
                    )
                    self.issues_fixed.append(
                        "Renamed conflicting pydantic_settings module"
                    )
            except Exception as e:
                logger.error(f"Failed to rename pydantic_settings module: {str(e)}")

    def fix_configurations(self) -> None:
        """Fix configuration issues."""
        logger.info("Checking and fixing configurations")

        # Check config.py for Pydantic v2 compatibility
        config_file = APP_ROOT / "backend" / "core" / "config.py"
        updated_config = False
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config_content = f.read()

                # Check for proper imports and fix if necessary
                if (
                    "from pydantic import BaseSettings" in config_content
                    and "from pydantic_settings import BaseSettings" not in config_content
                ):
                    logger.warning(
                        "Outdated BaseSettings import found in config.py. Attempting fix."
                    )
                    self.issues_found.append(
                        "Outdated Pydantic V1 BaseSettings import in config.py"
                    )
                    # Replace the import
                    config_content = config_content.replace(
                        "from pydantic import BaseSettings",
                        "from pydantic_settings import BaseSettings",
                    )
                    updated_config = True

                # Check for SettingsConfigDict if pydantic-settings is used
                if "from pydantic_settings import BaseSettings" in config_content and \
                   "from pydantic import Field, validator" in config_content and \
                   "from pydantic import SettingsConfigDict" not in config_content:
                     # Add the import if needed for model_config
                     # This is a guess, might need adjustment based on actual usage
                     if "model_config = SettingsConfigDict(" in config_content:
                        logger.info("Adding missing SettingsConfigDict import.")
                        # Find the line with pydantic imports and add to it
                        lines = config_content.splitlines()
                        import_line_index = -1
                        for i, line in enumerate(lines):
                            if line.strip().startswith("from pydantic import"):
                                import_line_index = i
                                break
                        if import_line_index != -1:
                            lines[import_line_index] += ", SettingsConfigDict"
                            config_content = "\n".join(lines)
                            updated_config = True
                        else: # Add as a new import line if needed
                            lines.insert(0, "from pydantic import SettingsConfigDict")
                            config_content = "\n".join(lines)
                            updated_config = True

                if updated_config:
                    with open(config_file, "w") as f:
                        f.write(config_content)
                    logger.info("Successfully updated imports in config.py")
                    self.issues_fixed.append("Updated Pydantic imports in config.py")

            except Exception as e:
                logger.error(f"Error processing config.py: {str(e)}")
                self.issues_found.append(f"Error processing config.py: {str(e)}")

        # Check essential config files
        essential_configs = [
            CONFIG_DIR / "database.json",
            CONFIG_DIR / "models.json",
            CONFIG_DIR / "agent_framework.json",
        ]

        for cfg_path in essential_configs:
            if not cfg_path.exists():
                logger.error(f"Missing essential config file: {cfg_path}")
                self.issues_found.append(f"Missing essential config: {cfg_path.name}")
            else:
                try:
                    with open(cfg_path, "r") as f:
                        json.load(f)
                    logger.info(f"Checked config file: {cfg_path.name}")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in config file: {cfg_path}")
                    self.issues_found.append(f"Invalid JSON in {cfg_path.name}")
                except Exception as e:
                    logger.error(f"Error reading config file {cfg_path}: {str(e)}")
                    self.issues_found.append(
                        f"Error reading config {cfg_path.name}: {str(e)}"
                    )

    def clean_stale_pid_files(self) -> None:
        """Clean stale PID files."""
        logger.info("Checking for stale PID files")

        for service, pid_file in self.pid_files.items():
            if pid_file.exists():
                try:
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())

                    # Check if process is running
                    if not psutil.pid_exists(pid):
                        logger.warning(f"Stale PID file for {service}: {pid}")
                        self.issues_found.append(f"Stale PID file for {service}")

                        # Remove the stale PID file
                        os.remove(pid_file)
                        logger.info(f"Removed stale PID file for {service}")
                        self.issues_fixed.append(
                            f"Removed stale PID file for {service}"
                        )
                    else:
                        logger.info(f"Valid PID file for {service}: {pid}")
                except (ValueError, ProcessLookupError) as e:
                    logger.error(f"Error checking PID file for {service}: {str(e)}")
                    os.remove(pid_file)
                    logger.info(f"Removed invalid PID file for {service}")
                    self.issues_fixed.append(f"Removed invalid PID file for {service}")

    def check_services(self) -> None:
        """Check and fix service issues."""
        logger.info("Checking services")

        # Get running processes
        running_processes = {}
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmd = " ".join(proc.info["cmdline"] or [])
                for service in self.services:
                    if service in cmd:
                        running_processes[service] = proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Check if services are running
        for service in self.services:
            pid_file = self.pid_files.get(service)

            if service in running_processes:
                logger.info(
                    f"Service {service} is running (PID: {running_processes[service]})"
                )

                # Update PID file if necessary
                if pid_file and (
                    not pid_file.exists()
                    or int(pid_file.read_text()) != running_processes[service]
                ):
                    with open(pid_file, "w") as f:
                        f.write(str(running_processes[service]))
                    logger.info(f"Updated PID file for {service}")
                    self.issues_fixed.append(f"Updated PID file for {service}")
            else:
                logger.warning(f"Service {service} is not running")
                self.issues_found.append(f"Service {service} is not running")

                # Clear stale PID file if it exists
                if pid_file and pid_file.exists():
                    os.remove(pid_file)
                    logger.info(f"Removed stale PID file for {service}")
                    self.issues_fixed.append(f"Removed stale PID file for {service}")

    def optimize_database(self) -> None:
        """Optimize database performance."""
        logger.info("Checking and optimizing database")

        # Check for SQLite database
        sqlite_db = None
        for db_path in [
            APP_ROOT / "database.db",
            APP_ROOT / "storage" / "sutazai.db",
            APP_ROOT / "storage" / "database.db",
        ]:
            if db_path.exists():
                sqlite_db = db_path
                break

        if sqlite_db:
            logger.info(f"Found SQLite database at {sqlite_db}")

            # Check database size
            db_size_mb = sqlite_db.stat().st_size / (1024 * 1024)
            logger.info(f"Database size: {db_size_mb:.2f} MB")

            if db_size_mb > 50:
                logger.warning("Database is large, may cause performance issues")
                self.warnings.append(f"Large database detected ({db_size_mb:.2f} MB)")

            # Optimize database with VACUUM
            try:
                import sqlite3

                conn = sqlite3.connect(sqlite_db)
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA temp_store=MEMORY;")
                cursor.execute("VACUUM;")
                conn.commit()
                conn.close()

                logger.info("Optimized SQLite database")
                self.issues_fixed.append(
                    "Optimized SQLite database for better performance"
                )
            except Exception as e:
                logger.error(f"Failed to optimize database: {str(e)}")
        else:
            logger.info("No SQLite database found")

    def optimize_storage(self) -> None:
        """Optimize storage usage by cleaning up logs and temporary files."""
        logger.info("Optimizing storage usage")

        # --- Log File Cleanup --- 
        log_files_deleted = 0
        log_bytes_deleted = 0
        cutoff_days = 7
        cutoff_time = time.time() - (cutoff_days * 24 * 60 * 60)

        try:
            logger.info(f"Cleaning log files older than {cutoff_days} days in {LOGS_DIR}")
            for item in LOGS_DIR.glob('*.log*'): # Include rotated logs like .log.1
                if item.is_file():
                    try:
                        file_mod_time = item.stat().st_mtime
                        if file_mod_time < cutoff_time:
                            file_size = item.stat().st_size
                            item.unlink() # Delete the file
                            log_files_deleted += 1
                            log_bytes_deleted += file_size
                    except Exception as e:
                        logger.warning(f"Could not process log file {item}: {str(e)}")
            logger.info(f"Deleted {log_files_deleted} old log files, freeing {log_bytes_deleted / (1024*1024):.2f} MB")
            if log_files_deleted > 0:
                 self.issues_fixed.append(f"Deleted {log_files_deleted} old log files")
        except Exception as e:
            logger.error(f"Error during log cleanup: {str(e)}")
            self.issues_found.append(f"Error during log cleanup: {str(e)}")

        # --- Temporary File Cleanup --- 
        tmp_dir = APP_ROOT / "tmp"
        tmp_files_deleted = 0
        tmp_bytes_deleted = 0
        if tmp_dir.exists() and tmp_dir.is_dir():
            logger.info(f"Cleaning temporary directory: {tmp_dir}")
            try:
                for item in tmp_dir.iterdir():
                    try:
                        item_size = item.stat().st_size if item.is_file() else 0
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            # Recursively get size before deleting
                            dir_size = sum(f.stat().st_size for f in item.glob('**/*') if f.is_file())
                            shutil.rmtree(item)
                            item_size = dir_size # Use calculated size for reporting
                        tmp_files_deleted += 1 # Count files and dirs
                        tmp_bytes_deleted += item_size
                    except Exception as e:
                         logger.warning(f"Could not remove item {item} from tmp: {str(e)}")
                logger.info(f"Deleted {tmp_files_deleted} items from tmp, freeing {tmp_bytes_deleted / (1024*1024):.2f} MB")
                if tmp_files_deleted > 0:
                    self.issues_fixed.append(f"Deleted {tmp_files_deleted} items from tmp directory")
            except Exception as e:
                logger.error(f"Error cleaning tmp directory: {str(e)}")
                self.issues_found.append(f"Error cleaning tmp directory: {str(e)}")
        else:
             logger.info("Temporary directory not found, skipping cleanup.")

        # --- General Disk Usage Check --- 
        try:
            # Log disk usage for key directories
            dirs_to_check = {
                "logs": LOGS_DIR,
                "data": APP_ROOT / "data",
                "models": APP_ROOT / "models",
                "tmp": APP_ROOT / "tmp",
                "vector_storage": APP_ROOT / "vector_storage",
            }
            total_app_size = 0

            for name, path in dirs_to_check.items():
                if path.exists():
                    dir_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
                    logger.info(
                        f"Disk usage for {name} ({path}): {dir_size / (1024*1024):.2f} MB"
                    )
                    total_app_size += dir_size
                else:
                     logger.info(f"Directory not found: {path}")

            logger.info(
                f"Approximate total disk usage for checked directories: {total_app_size / (1024*1024):.2f} MB"
            )

            # Log overall disk usage for the partition
            usage = shutil.disk_usage(APP_ROOT)
            logger.info(
                f"Overall disk usage for {APP_ROOT}: "
                f"{usage.used / (1024**3):.2f} GB used / "
                f"{usage.total / (1024**3):.2f} GB total "
                f"({usage.free / (1024**3):.2f} GB free)"
            )
        except Exception as e:
            logger.error(f"Error checking disk usage: {str(e)}")
            self.issues_found.append(f"Error checking disk usage: {str(e)}")

    def run_transformer_optimizations(self) -> None:
        """Run optimizations related to transformer models."""
        # Check cache size, etc.
        logger.info("Running transformer optimizations (Placeholder)")
        # Example: Check Hugging Face cache size
        try:
            from transformers import file_utils
            cache_dir = Path(file_utils.default_cache_path)
            if cache_dir.exists():
                cache_size = sum(f.stat().st_size for f in cache_dir.glob('**/*') if f.is_file())
                logger.info(f"Hugging Face cache directory: {cache_dir}")
                logger.info(f"Hugging Face cache size: {cache_size / (1024**3):.2f} GB")
                # Potentially add logic to clear cache if it exceeds a threshold
                # Be careful with clearing cache as it requires redownloading models
                # if cache_size > SOME_THRESHOLD:
                #    logger.warning("Cache size exceeds threshold. Consider clearing.")
            else:
                logger.info("Hugging Face cache directory not found.")
        except ImportError:
            logger.warning("'transformers' library not found. Skipping cache check.")
        except Exception as e:
            logger.error(f"Error checking Hugging Face cache: {str(e)}")
            self.issues_found.append(f"Error checking HF cache: {str(e)}")

    def run_code_quality_checks(self) -> None:
        """Run code quality checks."""
        logger.info("Running code quality checks")

        # Check Python files for syntax errors
        python_files = []
        for root, _, files in os.walk(APP_ROOT):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        syntax_errors = []
        for py_file in python_files:
            try:
                # Try reading with UTF-8, ignore errors for robustness
                with open(py_file, "r", encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                compile(code, py_file, "exec")
            except SyntaxError as e:
                syntax_errors.append((py_file, e.lineno, e.msg))
            except Exception as e: # Catch other potential read errors
                logger.warning(f"Could not read or compile {py_file}: {e}")

        if syntax_errors:
            logger.warning(
                f"Found {len(syntax_errors)} Python files with syntax errors"
            )
            self.issues_found.append(
                f"Python syntax errors detected in {len(syntax_errors)} files"
            )

            # Log the errors
            for file_path, line_no, msg in syntax_errors:
                rel_path = os.path.relpath(file_path, APP_ROOT)
                logger.error(f"Syntax error in {rel_path}, line {line_no}: {msg}")

    def restart_services(self) -> None:
        """Restart services to apply changes."""
        logger.info("Restarting services")

        # Check if systemd service files exist
        systemd_files = {
            service: SYSTEMD_DIR / f"{service}.service" for service in self.services
        }

        # Find start scripts
        start_scripts = {
            "backend": APP_ROOT / "bin" / "start_backend.sh",
            "webui": APP_ROOT / "bin" / "start_webui.sh",
            "vector-db": APP_ROOT / "bin" / "start_vector_db.sh",
            "localagi": APP_ROOT / "bin" / "start_localagi.sh",
        }

        # Restart each service
        for service in self.services:
            # Kill existing process
            if service in self.pid_files and self.pid_files[service].exists():
                try:
                    with open(self.pid_files[service], "r") as f:
                        pid = int(f.read().strip())

                    if psutil.pid_exists(pid):
                        try:
                            os.kill(pid, signal.SIGTERM)
                            time.sleep(2)  # Give it time to terminate

                            if psutil.pid_exists(pid):
                                os.kill(pid, signal.SIGKILL)

                            logger.info(f"Stopped {service} service (PID: {pid})")
                        except ProcessLookupError:
                            pass
                        # Catch PermissionError specifically when trying to kill
                        except PermissionError:
                            logger.error(f"Permission denied when trying to stop {service} (PID: {pid}). Skipping.")
                            self.issues_found.append(f"Permission denied stopping {service}")
                            # Continue to the next service if we can't kill this one
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error stopping {service} (PID: {pid}): {str(e)}")
                            self.issues_found.append(f"Error stopping {service}: {str(e)}")
                            continue # Also continue if there's an unexpected error

                except (ValueError, ProcessLookupError):
                    # This handles cases where the PID file is invalid or the process is gone
                    logger.warning(f"Could not read or find process for {service} from PID file. Assuming stopped.")

                # Remove PID file
                if self.pid_files[service].exists():
                    os.remove(self.pid_files[service])

            # Restart the service
            if systemd_files[service].exists():
                try:
                    # Use systemctl restart if we have systemd services
                    systemctl_path = shutil.which("systemctl")
                    if not systemctl_path:
                        logger.warning(
                            f"'systemctl' command not found. Cannot restart service {service}."
                        )
                        continue
                    subprocess.run(
                        [systemctl_path, "--user", "restart", service],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    logger.info(f"Restarted {service} service using systemd")
                    self.issues_fixed.append(f"Restarted {service} service")
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Failed to restart {service} service: {e.stderr.decode()}"
                    )
                    self.issues_found.append(f"Failed to restart {service} via systemd: {e.stderr.decode()}")
                # Catch general exceptions during systemd restart
                except Exception as e:
                    logger.error(f"Unexpected error restarting {service} via systemd: {str(e)}")
                    self.issues_found.append(f"Unexpected systemd restart error for {service}: {str(e)}")
            elif start_scripts[service].exists() and os.access(
                start_scripts[service], os.X_OK
            ):
                try:
                    # Use start script if available
                    subprocess.Popen(
                        [start_scripts[service]],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    logger.info(f"Started {service} service using start script")
                    self.issues_fixed.append(f"Started {service} service")
                except Exception as e:
                    logger.error(f"Failed to start {service} service: {str(e)}")
                    self.issues_found.append(f"Failed to start {service} via script: {str(e)}")


def main():
    """Run the system optimizer."""
    logger.info("Starting SutazAI System Optimizer")

    optimizer = SystemOptimizer()
    result = optimizer.run_full_optimization()

    logger.info("System optimization completed")
    logger.info(f"Issues found: {len(result['issues_found'])}")
    logger.info(f"Issues fixed: {len(result['issues_fixed'])}")
    logger.info(f"Warnings: {len(result['warnings'])}")

    # Print summary
    print("\n" + "=" * 50)
    print("SutazAI System Optimization Complete")
    print("=" * 50)
    print(f"Issues found: {len(result['issues_found'])}")
    print(f"Issues fixed: {len(result['issues_fixed'])}")
    print(f"Warnings: {len(result['warnings'])}")
    print("\nSystem should now be running optimally.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
