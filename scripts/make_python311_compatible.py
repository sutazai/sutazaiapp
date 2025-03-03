#!/usr/bin/env python3.11
"""
Python 3.11 Compatibility Master Script for SutazAI Project

This script orchestrates the process of making the entire codebase
compatible with Python 3.11 by running various compatibility scripts
in the correct order.
"""

import argparse
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("python311_compatibility")

# Scripts to run in order
COMPATIBILITY_SCRIPTS = [
    "scripts/syntax_error_fixer.py",
    "scripts/surgical_fixes.py",
    "scripts/typing_compatibility_fixer.py",
    "scripts/asyncio_compatibility_fixer.py"
]

def run_script(script_path: str, project_path: str) -> bool:
    """
    Run a compatibility script and return success status.
    
    Args:
        script_path: Path to the script to run
        project_path: Path to the project root
        
    Returns:
        True if the script ran successfully, False otherwise
    """
    try:
        logger.info(f"Running {script_path}...")
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path, project_path],
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            logger.error(f"Script {script_path} failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
        
        # Log the output
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"[{os.path.basename(script_path)}] {line}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error running {script_path}: {str(e)}")
        return False


def check_prerequisites() -> bool:
    """
    Check that all required scripts exist.
    
    Returns:
        True if all prerequisites are met, False otherwise
    """
    missing_scripts = []
    
    for script in COMPATIBILITY_SCRIPTS:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"Missing scripts: {', '.join(missing_scripts)}")
        return False
    
    return True


def run_compatibility_check(project_path: str) -> None:
    """
    Run the Python 3.11 compatibility checker.
    
    Args:
        project_path: Path to the project root
    """
    checker_script = "scripts/python311_compatibility_checker.py"
    
    if not os.path.exists(checker_script):
        logger.error(f"Compatibility checker script {checker_script} not found")
        return
    
    # Make the script executable
    os.chmod(checker_script, 0o755)
    
    logger.info("Running initial compatibility check...")
    
    # Run the script
    subprocess.run(
        [sys.executable, checker_script, project_path],
        check=False,
    )


def run_final_compatibility_check(project_path: str) -> None:
    """
    Run the Python 3.11 compatibility checker after all fixes.
    
    Args:
        project_path: Path to the project root
    """
    checker_script = "scripts/python311_compatibility_checker.py"
    
    if not os.path.exists(checker_script):
        logger.error(f"Compatibility checker script {checker_script} not found")
        return
    
    # Make the script executable
    os.chmod(checker_script, 0o755)
    
    logger.info("Running final compatibility check...")
    
    # Run the script
    result = subprocess.run(
        [sys.executable, checker_script, project_path],
        capture_output=True,
        text=True,
        check=False,
    )
    
    # Count the number of issues
    remaining_issues = 0
    for line in result.stdout.splitlines():
        if "potential issues" in line:
            parts = line.split()
            for part in parts:
                if part.isdigit():
                    remaining_issues = int(part)
                    break
    
    if remaining_issues > 0:
        logger.warning(f"There are still {remaining_issues} potential issues remaining")
        logger.warning("You may need to manually fix some issues")
    else:
        logger.info("No compatibility issues found! The codebase is compatible with Python 3.11")


def main() -> None:
    """Main function to run the Python 3.11 compatibility process."""
    parser = argparse.ArgumentParser(
        description="Make the codebase compatible with Python 3.11"
    )
    parser.add_argument(
        "project_path",
        nargs="?",
        default=os.getcwd(),
        help="Path to the project root (default: current directory)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the initial compatibility check",
    )
    args = parser.parse_args()
    
    project_path = args.project_path
    
    logger.info(f"Making {project_path} compatible with Python 3.11")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Missing prerequisites. Make sure all scripts exist.")
        return
    
    # Run initial compatibility check
    if not args.skip_check:
        run_compatibility_check(project_path)
        time.sleep(1)  # Give some time to observe the output
    
    # Run each compatibility script in order
    for script in COMPATIBILITY_SCRIPTS:
        if not run_script(script, project_path):
            logger.warning(f"Script {script} did not complete successfully")
        
        # Pause between scripts
        time.sleep(0.5)
    
    # Run final compatibility check
    run_final_compatibility_check(project_path)
    
    logger.info("Python 3.11 compatibility process completed")
    logger.info("Please review the changes and test the codebase thoroughly")


if __name__ == "__main__":
    main() 