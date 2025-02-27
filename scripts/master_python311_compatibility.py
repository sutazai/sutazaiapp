#!/usr/bin/env python3.11
"""
Master Python 3.11 Compatibility Script

This script discovers and executes all Python 3.11 compatibility scripts
in the SutazAI project.
"""

import importlib
import importlib.util
import logging
import os
import sys
from typing import Any, Dict, List


def verify_python_version() -> None:
    """
    Verify that Python 3.11 is being used.

    Raises:
        RuntimeError: If Python version is not 3.11
    """
    major, minor = sys.version_info.major, sys.version_info.minor
    if major != 3 or minor != 11:
        raise RuntimeError(
            "This script requires Python 3.11. "
            f"Current version: {sys.version}"
        )


def setup_logging() -> logging.Logger:
    """
    Set up comprehensive logging.

    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(
                "/opt/sutazaiapp/logs/master_compatibility.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def find_compatibility_scripts(base_path: str) -> List[str]:
    """
    Find all Python scripts related to compatibility and optimization.

    Args:
        base_path: Base directory to search for scripts

    Returns:
        List of script paths
    """
    keywords = [
        "compatibility", "fix", "optimize", "cleanup",
        "validator", "syntax", "system_check", "repair",
    ]

    compatible_scripts = []

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if any(keyword in file.lower() for keyword in keywords):
                    compatible_scripts.append(file_path)

    return compatible_scripts


def execute_script(script_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute a Python script and capture its results.

    Args:
        script_path: Path to the Python script
        logger: Logger for reporting

    Returns:
        Dict with script execution results
    """
    try:
        logger.info(f"Executing script: {script_path}")

        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "main"):
            result = module.main()
            logger.info(f"Script {script_path} executed successfully")
            return {
                "script": script_path,
                "status": "success",
                "result": result,
            }
        logger.warning(f"No main() function found in {script_path}")
        return {
            "script": script_path,
            "status": "skipped",
            "reason": "No main() function",
        }

    except Exception as e:
        logger.exception(f"Error executing {script_path}: {e}")
        return {
            "script": script_path,
            "status": "failed",
            "error": str(e),
        }


def generate_compatibility_report(
    script_results: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """
    Generate a comprehensive compatibility report.

    Args:
        script_results: Results from script executions
        logger: Logger for reporting
    """
    report_path = "/opt/sutazaiapp/logs/compatibility_report.md"

    with open(report_path, "w") as report_file:
        report_file.write("# Python 3.11 Compatibility Report\n")
        report_file.write(
            f"**Generated:** {os.getenv('TIMESTAMP', 'Unknown')}\n\n"
        )

        report_file.write("## Script Execution Summary\n\n")

        success_count = sum(
            1 for result in script_results
            if result["status"] == "success"
        )
        failed_count = sum(
            1 for result in script_results
            if result["status"] == "failed"
        )
        skipped_count = sum(
            1 for result in script_results
            if result["status"] == "skipped"
        )

        report_file.write(f"- **Total Scripts:** {len(script_results)}\n")
        report_file.write(f"- **Successful Executions:** {success_count}\n")
        report_file.write(f"- **Failed Executions:** {failed_count}\n")
        report_file.write(f"- **Skipped Scripts:** {skipped_count}\n\n")

        report_file.write("## Detailed Results\n\n")

        for result in script_results:
            report_file.write(f"### {result['script']}\n")
            report_file.write(f"- **Status:** {result['status']}\n")

            if result["status"] == "success":
                report_file.write("- **Result:** Execution completed\n")
            elif result["status"] == "failed":
                report_file.write(
                    f"- **Error:** {result.get('error', 'Unknown error')}\n",
                )
            elif result["status"] == "skipped":
                report_file.write(
                    f"- **Reason:** {result.get('reason', 'No reason')}\n",
                )

            report_file.write("\n")

    logger.info(f"Compatibility report generated at {report_path}")


def main() -> None:
    """
    Main function to execute Python 3.11 compatibility scripts.
    """
    # Verify Python version
    verify_python_version()

    # Setup logging
    logger = setup_logging()

    # Find compatibility scripts
    scripts_path = "/opt/sutazaiapp/scripts"
    compatibility_scripts = find_compatibility_scripts(scripts_path)

    logger.info(f"Found {len(compatibility_scripts)} compatibility scripts")

    # Execute scripts and collect results
    script_results = []
    for script in compatibility_scripts:
        result = execute_script(script, logger)
        script_results.append(result)

    # Generate comprehensive report
    generate_compatibility_report(script_results, logger)

    logger.info("Python 3.11 compatibility check completed")


if __name__ == "__main__":
    main()
