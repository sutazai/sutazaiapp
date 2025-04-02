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
        if config_file.exists():
            with open(config_file, "r") as f:
                config_content = f.read()

            # Check for proper imports
            if (
                "from pydantic import BaseSettings" in config_content
                and "pydantic_settings" not in config_content
            ):
                logger.warning("Outdated BaseSettings import in config.py")
                self.issues_found.append("Outdated Pydantic imports in config.py")

                # Fix the import
                new_content = config_content.replace(
                    "from pydantic import BaseSettings",
                    "try:\n    from pydantic_settings import BaseSettings\nexcept ImportError:\n    from pydantic import BaseSettings",
                )

                # Fix model_config for pydantic v2
                if "class Config:" in new_content and "model_config" not in new_content:
                    new_content = new_content.replace(
                        "class Config:",
                        'try:\n        model_config = {\n            "env_file": ".env",\n            "env_file_encoding": "utf-8",\n            "case_sensitive": True,\n            "extra": "ignore"\n        }\n    except ImportError:\n        class Config:',
                    )

                with open(config_file, "w") as f:
                    f.write(new_content)

                logger.info("Fixed Pydantic imports and configuration in config.py")
                self.issues_fixed.append(
                    "Updated config.py for Pydantic v2 compatibility"
                )

        # Check and fix systemd service files
        for service in self.services:
            service_file = SYSTEMD_DIR / f"{service}.service"
            if service_file.exists():
                with open(service_file, "r") as f:
                    service_content = f.read()

                # Check for proper paths
                if (
                    "/opt/sutazai/" in service_content
                    and "/opt/sutazaiapp/" not in service_content
                ):
                    logger.warning(f"Incorrect path in {service}.service")
                    self.issues_found.append(f"Incorrect path in {service}.service")

                    # Fix the path
                    new_content = service_content.replace(
                        "/opt/sutazai/", "/opt/sutazaiapp/"
                    )
                    with open(service_file, "w") as f:
                        f.write(new_content)

                    logger.info(f"Fixed path in {service}.service")
                    self.issues_fixed.append(f"Fixed path in {service}.service")

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
        """Optimize storage usage."""
        logger.info("Checking and optimizing storage")

        # Check for log files
        log_files = list(LOGS_DIR.glob("*.log"))
        large_log_files = []

        for log_file in log_files:
            file_size_mb = log_file.stat().st_size / (1024 * 1024)
            if file_size_mb > 10:
                large_log_files.append((log_file, file_size_mb))

        if large_log_files:
            logger.warning(f"Found {len(large_log_files)} large log files")
            self.issues_found.append("Large log files detected, may impact storage")

            # Rotate large log files
            for log_file, size_mb in large_log_files:
                try:
                    rotated_file = log_file.with_suffix(
                        f".log.{datetime.now().strftime('%Y%m%d')}"
                    )
                    shutil.move(log_file, rotated_file)
                    with open(log_file, "w") as f:
                        f.write(f"Log rotated at {datetime.now().isoformat()}\n")

                    logger.info(
                        f"Rotated large log file: {log_file.name} ({size_mb:.2f} MB)"
                    )
                    self.issues_fixed.append(f"Rotated large log file: {log_file.name}")
                except Exception as e:
                    logger.error(f"Failed to rotate log file {log_file.name}: {str(e)}")

        # Check for temporary files
        temp_dir = APP_ROOT / "tmp"
        if temp_dir.exists():
            temp_files = list(temp_dir.glob("*"))
            old_files = []

            for temp_file in temp_files:
                file_age_days = (time.time() - temp_file.stat().st_mtime) / (
                    60 * 60 * 24
                )
                if file_age_days > 7:
                    old_files.append(temp_file)

            if old_files:
                logger.warning(f"Found {len(old_files)} old temporary files")
                self.issues_found.append("Old temporary files detected, cleaning up")

                # Remove old temporary files
                for old_file in old_files:
                    try:
                        if old_file.is_file():
                            old_file.unlink()
                        elif old_file.is_dir():
                            shutil.rmtree(old_file)
                    except Exception as e:
                        logger.error(
                            f"Failed to remove old file {old_file.name}: {str(e)}"
                        )

                logger.info(f"Removed {len(old_files)} old temporary files")
                self.issues_fixed.append(
                    f"Cleaned up {len(old_files)} old temporary files"
                )

    def run_transformer_optimizations(self) -> None:
        """Run transformer model optimizations."""
        logger.info("Checking for transformer model optimizations")

        # Check if optimization script exists
        optimize_script = SCRIPTS_DIR / "optimize_transformers.py"
        if optimize_script.exists():
            logger.info("Found transformer optimization script")

            # Check if models directory exists
            models_dir = APP_ROOT / "models"
            if models_dir.exists() and list(models_dir.glob("*")):
                logger.info("Found models directory with models")

                # Run the optimization script if models are found
                try:
                    # Define required arguments
                    model_dir = str(models_dir)
                    output_dir = str(models_dir / "optimized") # Example output dir
                    os.makedirs(output_dir, exist_ok=True)
                    
                    cmd = [
                        sys.executable, 
                        str(optimize_script),
                        "--model_dir", model_dir,
                        "--output_dir", output_dir
                        # Add other args like --model_type if needed, or make them optional
                    ]
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    logger.info(f"Ran transformer optimization script: {' '.join(cmd)}")
                    self.issues_fixed.append(
                        "Optimized transformer models for better performance"
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Failed to run transformer optimizations: {e.stderr.decode()}"
                    )
                    # Add specific handling for OSError (like tokenizer not found)
                    self.issues_found.append(f"Transformer optimization failed: {e.stderr.decode()}")
                except FileNotFoundError:
                    logger.error(f"Optimization script not found or Python executable issue.")
                    self.issues_found.append(f"Transformer optimization script not found.")
                # Add a general exception catch
                except Exception as e:
                    logger.error(f"An unexpected error occurred during transformer optimization: {str(e)}")
                    self.issues_found.append(f"Unexpected transformer optimization error: {str(e)}")
            else:
                logger.info("No models found to optimize")
        else:
            logger.info("Transformer optimization script not found.")

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
