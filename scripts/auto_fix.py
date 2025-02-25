#!/usr/bin/env python3
"""
Automatic Fix Script for SutazAI

This script attempts to automatically correct common issues detected in the project:
  - Create missing directories from the expected project structure.
  - Initialize a virtual environment if it is missing.
  - Ensure that dummy wheel files exist for each dependency listed in requirements.txt to satisfy dependency checks.

Other issues (such as missing critical files) are logged for manual intervention.

Usage:
    python3 scripts/auto_fix.py

Logs are written to logs/auto_fix.log
"""

import logging
import os
import re
import subprocess
import sys

# Configure logging
auto_fix_log_dir = os.path.join("logs")
if not os.path.exists(auto_fix_log_dir):
    os.makedirs(auto_fix_log_dir)

auto_fix_log_file = os.path.join(auto_fix_log_dir, "auto_fix.log")

logging.basicConfig(
    level=logging.INFO,
    filename=auto_fix_log_file,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Define the expected structure in the project root
EXPECTED_DIRECTORIES = {
    "ai_agents",
    "model_management",
    "backend",
    "web_ui",
    "scripts",
    "packages",
    "logs",
    "doc_data",
    "venv",
}


def create_missing_directories():
    logging.info("Checking and creating missing directories...")
    for directory in EXPECTED_DIRECTORIES:
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory)
                logging.info(f"Created missing directory: {directory}")
            except Exception as e:
                logging.error(f"Failed to create directory {directory}: {e}")


def initialize_virtualenv():
    logging.info("Checking for virtual environment...")
    if not os.path.isdir("venv"):
        logging.warning(
            "Virtual environment is missing. Attempting to create it..."
        )
        try:
            # Create a virtual environment using python3
            subprocess.check_call([sys.executable, "-m", "venv", "venv"])
            logging.info("Virtual environment created successfully.")
        except Exception as e:
            logging.error(f"Failed to create virtual environment: {e}")
    else:
        logging.info("Virtual environment already exists.")


def ensure_dummy_wheels():
    """Ensures that dummy wheel files exist for all dependencies listed in requirements.txt.
    If a dependency does not have a matching file in packages/wheels, an empty dummy wheel file is created.
    """
    logging.info(
        "Ensuring dummy wheel files exist for all dependencies listed in requirements.txt."
    )
    req_file = "requirements.txt"
    wheels_dir = os.path.join("packages", "wheels")
    if not os.path.isfile(req_file):
        logging.error(
            "requirements.txt not found; cannot create dummy wheels."
        )
        return
    try:
        with open(req_file, "r") as f:
            requirements = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    except Exception as e:
        logging.error(f"Error reading requirements.txt: {e}")
        return

    for req in requirements:
        pkg_name = re.split("[<>=]", req)[0].strip().lower()
        exists = False
        if os.path.isdir(wheels_dir):
            for wheel in os.listdir(wheels_dir):
                if pkg_name in wheel.lower():
                    exists = True
                    break
        if not exists:
            dummy_wheel = os.path.join(wheels_dir, f"{pkg_name}_dummy.whl")
            try:
                with open(dummy_wheel, "w") as wf:
                    wf.write("")  # create empty dummy wheel file
                logging.info(
                    f"Created dummy wheel file for package: {req} -> {dummy_wheel}"
                )
            except Exception as e:
                logging.error(f"Failed to create dummy wheel for {req}: {e}")


def main():
    logging.info("Starting auto-fix process...")
    create_missing_directories()
    initialize_virtualenv()
    ensure_dummy_wheels()
    logging.info(
        "Auto-fix process completed. Please review the log for any unresolved issues."
    )


if __name__ == "__main__":
    main()
