#!/usr/bin/env python3

"""
Main script to run all test fixes and verify the results.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fix_all.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_all_fixes")

# Absolute path to the project directory
BASE_DIR = Path("/opt/sutazaiapp")

def run_script(script_name, desc):
    """Run a Python script and log the result."""
    script_path = BASE_DIR / script_name
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    logger.info(f"Running {desc}...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully completed: {desc}")
            return True
        else:
            logger.error(f"Failed to run {script_name}: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Exception while running {script_name}: {e}")
        return False

def main():
    """Run all fix scripts in sequence."""
    logger.info("Starting comprehensive test fixes...")
    
    # List of scripts to run
    scripts = [
        ("fix_indentation_and_decorators.py", "indentation and decorator fixes"),
        ("fix_coroutine_warnings.py", "coroutine warning fixes"),
        ("fix_sync_exception_test.py", "sync exception test fix"),
        ("setup_pytest_config.py", "pytest configuration setup")
    ]
    
    # Run each script
    results = {}
    for script_name, desc in scripts:
        results[script_name] = run_script(script_name, desc)
    
    # Run verify script if it exists
    verify_script = "verify_fixes.py"
    if (BASE_DIR / verify_script).exists():
        logger.info("Running verification...")
        run_script(verify_script, "verification of fixes")
    
    # Log summary
    logger.info("Fix summary:")
    for script_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{status}: {script_name}")
    
    # Overall success
    if all(results.values()):
        logger.info("All fixes completed successfully!")
        return 0
    else:
        logger.error("Some fixes failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 