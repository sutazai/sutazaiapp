#!/usr/bin/env python3
"""
Comprehensive Dependency Verification and Management Script

This script checks, verifies, and installs project dependencies,
ensuring compatibility and resolving potential import issues.
"""

import importlib
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional

import pkg_resources


def check_python_version() -> bool:
    """
    Verify Python version compatibility.

    Returns:
        bool: Whether the Python version is compatible
    """
    min_version = (3, 8)  # Minimum supported Python version
    current_version = sys.version_info

    if current_version < min_version:
        print(
            f"❌ Unsupported Python version. Minimum required: {'.'.join(map(str, min_version))}"
        )
        return False
    return True


def verify_package_installation(package_name: str) -> Dict[str, Any]:
    """
    Comprehensively check package installation and version.

    Args:
        package_name (str): Name of the package to verify

    Returns:
        Dict[str, Any]: Detailed package verification information
    """
    try:
        # Try importing the package
        module = importlib.import_module(package_name)

        # Get installed version
        version = pkg_resources.get_distribution(package_name).version

        return {
            "installed": True,
            "version": version,
            "module_path": module.__file__,
            "status": "✅ Successfully imported",
        }
    except ImportError:
        return {
            "installed": False,
            "version": None,
            "module_path": None,
            "status": "❌ Package not found",
        }
    except Exception as e:
        return {
            "installed": False,
            "version": None,
            "module_path": None,
            "status": f"❌ Error: {str(e)}",
        }


def install_missing_dependencies(packages: List[str]) -> List[str]:
    """
    Install missing dependencies using pip with error handling.

    Args:
        packages (List[str]): List of packages to install

    Returns:
        List[str]: List of successfully installed packages
    """
    successfully_installed = []
    for package in packages:
        try:
            print(f"🔄 Attempting to install {package}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package]
            )
            successfully_installed.append(package)
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")

    return successfully_installed


def verify_project_dependencies(verbose: bool = False) -> Dict[str, Any]:
    """
    Comprehensive dependency verification and management.

    Args:
        verbose (bool): Whether to print detailed information

    Returns:
        Dict[str, Any]: Detailed dependency verification results
    """
    # Comprehensive list of critical and optional dependencies
    dependencies = [
        # Core dependencies
        "pydantic",
        "pydantic-settings",
        "typing-extensions",
        # AI and NLP
        "spacy",
        "textstat",
        "schedule",
        # System and Performance
        "watchdog",
        "GPUtil",
        "ray",
        "psutil",
        # Documentation and Utilities
        "pdoc",
        "spellchecker",
        "python-dotenv",
        "loguru",
        # Development and Testing Tools
        "mypy",
        "pylint",
        "flake8",
        "black",
        "isort",
        "bandit",
    ]

    results = {
        "total_dependencies": len(dependencies),
        "verified_dependencies": {},
        "missing_dependencies": [],
    }

    for package in dependencies:
        result = verify_package_installation(package)
        results["verified_dependencies"][package] = result

        if not result["installed"]:
            results["missing_dependencies"].append(package)

        if verbose:
            print(f"{package}: {result['status']}")

    # Attempt to install missing dependencies
    if results["missing_dependencies"]:
        print("\n🛠️ Installing missing dependencies...")
        successfully_installed = install_missing_dependencies(
            results["missing_dependencies"]
        )
        results["successfully_installed"] = successfully_installed

    return results


def main():
    print("🔍 Starting Comprehensive Dependency Verification...")

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Perform detailed dependency verification
    verification_results = verify_project_dependencies(verbose=True)

    # Print summary
    print("\n📊 Dependency Verification Summary:")
    print(f"Total Dependencies: {verification_results['total_dependencies']}")
    print(
        f"Missing Dependencies: {len(verification_results.get('missing_dependencies', []))}"
    )

    if verification_results.get("successfully_installed"):
        print("Successfully Installed:")
        for pkg in verification_results["successfully_installed"]:
            print(f"  - {pkg}")

    print("\n✨ Dependency Verification Complete!")


if __name__ == "__main__":
    main()
