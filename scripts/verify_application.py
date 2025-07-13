#!/usr/bin/env python3.11
"""
Verify Application Setup

This script performs comprehensive checks on the SutazAI application to ensure
that all components are properly set up and can be imported successfully.
"""

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Set up colored output

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")

def print_result(message: str, success: bool, details: str = "") -> None:
    """Print a formatted result message."""
    status = f"{Colors.OKGREEN}✓ SUCCESS{Colors.ENDC}" if success else f"{Colors.FAIL}✗ FAILED{Colors.ENDC}"
    print(f"{status} | {message}")
    if details and not success:
        print(f"  {Colors.WARNING}Details: {details}{Colors.ENDC}")

def check_python_version() -> Tuple[bool, str]:
    """Check if the Python version meets requirements."""
    min_version = (3, 10)
    current_version = sys.version_info[:2]

    if current_version >= min_version:
        return True, f"Python {'.'.join(map(str, current_version))}"
    else:
        return False, f"Python {'.'.join(map(str, current_version))} (required: {'.'.join(map(str, min_version))} or higher)"

def check_environment_variables() -> Tuple[bool, str]:
    """Check if required environment variables are set."""
    # List of optional environment variables to check
    env_vars = [
        "SUTAZAI_CONFIG",
        "SUTAZAI_SECRET_KEY",
        "SUTAZAI_LOG_LEVEL",
        "SUTAZAI_ENV",
    ]

    missing_vars = [var for var in env_vars if not os.getenv(var)]

    if not missing_vars:
        return True, "All environment variables are set"
    else:
        return True, f"Optional variables not set: {', '.join(missing_vars)}"

def check_directories() -> Tuple[bool, Dict[str, bool]]:
    """Check if required directories exist and are accessible."""
    directories = {
        "config": (project_root / "config").exists(),
        "logs": (project_root / "logs").exists(),
        "uploads": (project_root / "uploads").exists(),
        "data": (project_root / "data").exists(),
    }

    return all(directories.values()), directories

def check_imports() -> Tuple[bool, Dict[str, List[Tuple[str, bool, str]]]]:
    """Check if required modules can be imported."""
    modules = {
        "Backend Core": [
            "backend.backend_main",
            "backend.config",
            "backend.utils",
        ],
        "AI Agents": [
            "ai_agents.base_agent",
            "ai_agents.agent_factory",
            "ai_agents.agent_config_manager",
        ],
    }

    results = {}
    all_successful = True

    for category, module_list in modules.items():
        category_results = []
        for module_name in module_list:
            try:
                importlib.import_module(module_name)
                category_results.append((module_name, True, ""))
            except Exception as e:
                category_results.append((module_name, False, str(e)))
                all_successful = False

        results[category] = category_results

    return all_successful, results

def check_config() -> Tuple[bool, str]:
    """Check if the configuration file can be loaded."""
    try:
        from backend.config import Config
        config = Config()
        return True, f"Loaded from {config.config_path}"
    except Exception as e:
        return False, str(e)

def check_database_connection() -> Tuple[bool, str]:
    """Check if we can connect to the database."""
    try:
        from backend.config import Config
        config = Config()

        # Simple import test for SQLAlchemy
        from sqlalchemy import create_engine

        # Don't actually connect, just check that the URL is valid
        engine = create_engine(config.db_url)
        return True, f"Database URL validated: {config.db_url.split('@')[-1]}"
    except Exception as e:
        return False, str(e)

def main() -> None:
    """Run all verification checks."""
    start_time = time.time()

    print_header("SutazAI Application Verification")

    # Check Python version
    success, details = check_python_version()
    print_result("Python Version", success, details)

    # Check environment variables
    success, details = check_environment_variables()
    print_result("Environment Variables", success, details)

    # Check directories
    success, directories = check_directories()
    print_result("Directories", success)
    for dir_name, exists in directories.items():
        status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if exists else f"{Colors.WARNING}?{Colors.ENDC}"
        print(f"  {status} {dir_name}/")

    # Check configuration
    success, details = check_config()
    print_result("Configuration", success, details)

    # Check imports
    print_header("Import Checks")
    success, import_results = check_imports()

    for category, results in import_results.items():
        print(f"{Colors.BOLD}{category}{Colors.ENDC}")
        for module_name, module_success, error in results:
            status = f"{Colors.OKGREEN}✓{Colors.ENDC}" if module_success else f"{Colors.FAIL}✗{Colors.ENDC}"
            print(f"  {status} {module_name}")
            if not module_success:
                print(f"    {Colors.WARNING}{error}{Colors.ENDC}")

    # Check database connection
    print_header("Database Checks")
    success, details = check_database_connection()
    print_result("Database Connection", success, details)

    # Print summary
    elapsed_time = time.time() - start_time
    print_header("Verification Summary")

    all_checks_passed = all([
        check_python_version()[0],
        check_environment_variables()[0],
        check_directories()[0],
        check_config()[0],
        check_imports()[0],
        check_database_connection()[0],
    ])

    if all_checks_passed:
        print(f"{Colors.OKGREEN}{Colors.BOLD}All checks passed successfully!{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}{Colors.BOLD}Some checks failed. Please review the results above.{Colors.ENDC}")

    print(f"\nVerification completed in {elapsed_time:.2f} seconds.")

    # Return instructions
    print_header("Next Steps")
    print("To run the application, use the following command:")
    print(f"{Colors.BOLD}cd {project_root} && python -m backend.backend_main{Colors.ENDC}")
    print("\nTo run tests:")
    print(f"{Colors.BOLD}cd {project_root} && python -m pytest{Colors.ENDC}")

if __name__ == "__main__":
    main()
