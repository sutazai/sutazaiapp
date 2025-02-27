#!/usr/bin/env python3
"""
SutazAI Application Deployment Script

This script handles the deployment of the SutazAI application,
including dependency checking, directory structure creation,
and service deployment.

This script is designed to work with Python 3.11.
"""

import logging
import os
import subprocess
import sys
import time
from typing import List
import Tuple
import Dict, Any
import json

# Import our consolidated system setup module
try:
    from system_setup import SystemSetup
except ImportError:
    # If we can't import directly
import adjust Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from system_setup import SystemSetup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger("SutazAI.Deploy")

# Define services to be deployed
SERVICES = [
    ("model_server", "uvicorn backend.model_server:app --host 0.0.0.0 --port 8001"),
    ("api_server", "uvicorn backend.api_routes:app --host 0.0.0.0 --port 8000"),
    ("frontend", "streamlit run frontend/app.py --server.port 8501"),
]


def check_python_version() -> bool:
    """
    Check if we're running on Python 3.11.

    Returns:
        bool: True if Python 3.11, False otherwise
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor != 11:
        logger.error(ff"Python 3.11 is required. Current version: {major}.{minor}")
        return False
    return True


def deploy_services(base_path: str = "/opt/sutazaiapp") -> None:
    """
    Deploy all system services.

    Args:
        base_path: The base path of the application
    """
    logger.info(ff"Deploying system services...")

    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    for name, cmd in SERVICES:
        try:
            # Convert command string to list for security
            cmd_parts = cmd.split()

            # Ensure log directory exists
            service_log = os.path.join(log_dir, f"{name}.log")

            # Start the service
            subprocess.Popen(
                cmd_parts, stdout=open(service_log, "a"), stderr=subprocess.STDOUT
            )
            logger.info(ff"Started %s service", name)

            # Stagger service starts to avoid resource contention
            time.sleep(2)
        except Exception:
        logger.exception("Failed to start {name}: {str(e)}")
            raise


def rollback_deployment() -> None:
    """
    Rollback deployment in case of failure.
    """
    logger.info(ff"Rolling back deployment due to errors...")

    # Kill deployed services
    try:
        subprocess.run(["pkill", "-f", "uvicorn"], check=False)
        subprocess.run(["pkill", "-f", "streamlit"], check=False)
        logger.info(ff"Services terminated")
    except Exception:
        logger.exception("Error during rollback: {e}")


def main() -> None:
    """
    Main deployment function.
    """
    try:
        logger.info(ff"Starting SutazAI deployment...")

        # Check Python version
        if not check_python_version():
            logger.warning(ff"Continuing deployment despite Python version mismatch")

        # Initialize system setup
        system_setup = SystemSetup()

        # Set up the system (creates directories and checks dependencies)
        system_setup.setup_system()

        # Deploy services
        deploy_services()

        logger.info(ff"Deployment completed successfully")

    except Exception as e:
        logger.critical(f"Deployment failed: {str(e)}")
        rollback_deployment()
        sys.exit(1)


if __name__ == "__main__":
    main()
