#!/opt/sutazaiapp/venv/bin/python3
import os
import sys
import logging
from typing import List, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/opt/sutazaiapp/logs/optimization_log.txt"),
    ],
)
logger = logging.getLogger(__name__)


def run_optimization_module(module_path: str, function_name: str = "main"):    """
    Dynamically run an optimization module with error handling.

    Args:    module_path (str): Full path to the Python script
    function_name (str, optional): Name of the function to call. Defaults to 'main'.
    """
    try:        # Import the module dynamically
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        spec.loader.exec_module(module)

        # Call the specified function
        if hasattr(module, function_name):            func = getattr(module, function_name)
            func()
            logger.info(f"Successfully ran optimization module: {module_path}")
        else:            logger.warning(
                f"No {function_name} function found in {module_path}")

    except Exception as e:        logger.error(f"Error running optimization module {module_path}: {e}")
        logger.exception(e)


def main():    """Run all optimization scripts in the scripts directory."""

    logger.info("🚀 Starting Comprehensive System Optimization 🚀")

    # List of optimization scripts to run
    optimization_scripts = [
        "/opt/sutazaiapp/scripts/comprehensive_cleanup.py",
        "/opt/sutazaiapp/scripts/process_optimizer.py",
        "/opt/sutazaiapp/scripts/system_maintenance_pro.py",
        "/opt/sutazaiapp/scripts/system_optimizer.py"
    ]

    # Run each optimization script
    for script_path in optimization_scripts:        run_optimization_module(script_path)

    logger.info("✅ Comprehensive System Optimization Completed Successfully ✅")


if __name__ == "__main__":    main()
