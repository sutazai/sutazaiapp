#!/usr/bin/env python3.11
"""
Fix Dependencies Script

This script helps resolve dependency conflicts in the SutazAI application.
It can update specific problematic dependencies, verify compatibility,
and restore the environment to a working state.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, and stderr."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd or PROJECT_ROOT),
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def print_header(message: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {message} ==={Colors.ENDC}\n")


def print_step(message: str) -> None:
    """Print a step message."""
    print(f"{Colors.OKBLUE}➤ {message}{Colors.ENDC}")


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def get_current_dependencies() -> Dict[str, str]:
    """Get the current installed dependencies and their versions."""
    print_step("Getting current dependencies...")
    
    returncode, stdout, stderr = run_command(["pip", "freeze"])
    if returncode != 0:
        print_error(f"Failed to get installed packages: {stderr}")
        return {}
    
    dependencies = {}
    for line in stdout.splitlines():
        if "==" in line:
            package, version = line.split("==", 1)
            dependencies[package.lower()] = version
    
    return dependencies


def get_dependency_conflicts() -> List[Dict[str, str]]:
    """Identify dependency conflicts using pip check."""
    print_step("Checking for dependency conflicts...")
    
    returncode, stdout, stderr = run_command(["pip", "check"])
    
    conflicts = []
    if returncode != 0:
        # Parse conflicts from pip check output
        # Example: "package X has requirement package Y>=1.0, but you have package Y 0.9"
        conflict_pattern = re.compile(
            r"(.+?) \d+\.\d+\.\d+ has requirement (.+?)[,\s]", re.IGNORECASE
        )
        
        for line in stdout.splitlines() + stderr.splitlines():
            match = conflict_pattern.search(line)
            if match:
                conflicts.append({
                    "package": match.group(1).strip(),
                    "requirement": match.group(2).strip()
                })
    
    return conflicts


def fix_known_conflicts() -> bool:
    """Fix known dependency conflicts."""
    print_header("Fixing Known Conflicts")
    
    # Define known problematic dependencies and their fixes
    known_fixes = [
        # Format: (package_name, target_version, reason)
        ("psutil", "6.0.0", "Required by safety 3.2.9"),
        ("transformers", "4.41.0", "Required by sentence-transformers 3.4.1"),
        ("tokenizers", "0.21.0", "Required by transformers"),
        ("pluggy", "1.5.0", "Required by pytest"),
    ]
    
    # Get current installed packages
    current_deps = get_current_dependencies()
    
    # Apply fixes where necessary
    fixed_something = False
    
    for package, target_version, reason in known_fixes:
        package_lower = package.lower()
        
        if package_lower in current_deps:
            current_version = current_deps[package_lower]
            if current_version != target_version:
                print_step(f"Fixing {package} (current: {current_version}, target: {target_version})")
                print_warning(f"Reason: {reason}")
                
                # Install the correct version
                returncode, stdout, stderr = run_command([
                    "pip", "install", "--upgrade", f"{package}=={target_version}"
                ])
                
                if returncode == 0:
                    print_success(f"Updated {package} to version {target_version}")
                    fixed_something = True
                else:
                    print_error(f"Failed to update {package}: {stderr}")
        else:
            print_warning(f"{package} is not installed, skipping")
    
    return fixed_something


def update_requirements_files() -> None:
    """Update requirements files to match the currently installed packages."""
    print_header("Updating Requirements Files")
    
    # Packages that need specific versions in requirements files
    critical_packages = {
        "psutil": "6.0.0",
        "transformers": ">=4.41.0,<5.0.0",
        "pluggy": "1.5.0",
        "tokenizers": "0.21.0",
    }
    
    requirements_files = [
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "packages" / "requirements.txt",
    ]
    
    for req_file in requirements_files:
        if not req_file.exists():
            print_warning(f"Requirements file not found: {req_file}")
            continue
        
        print_step(f"Updating {req_file.relative_to(PROJECT_ROOT)}")
        
        # Read the current content
        with open(req_file, "r") as f:
            content = f.read()
        
        # Update versions for critical packages
        for package, version in critical_packages.items():
            # Match both package==x.y.z and package>=x.y.z formats
            pattern = re.compile(
                rf"({package})[=<>~!]+[0-9][.0-9a-zA-Z,]*", 
                re.IGNORECASE
            )
            
            if version.startswith(">="):
                replacement = rf"\1{version}"
            else:
                replacement = rf"\1=={version}"
            
            content = pattern.sub(replacement, content)
        
        # Write the updated content
        with open(req_file, "w") as f:
            f.write(content)
        
        print_success(f"Updated {req_file.relative_to(PROJECT_ROOT)}")


def verify_fixed_dependencies() -> bool:
    """Verify that dependency conflicts are resolved."""
    print_header("Verifying Dependencies")
    
    print_step("Running final dependency check...")
    returncode, stdout, stderr = run_command(["pip", "check"])
    
    if returncode == 0:
        print_success("All dependencies are compatible!")
        return True
    else:
        print_warning("Some dependency conflicts still exist:")
        for line in stdout.splitlines():
            if line.strip():
                print(f"  {line.strip()}")
        return False


def test_imports() -> bool:
    """Test importing key modules to ensure they work correctly."""
    print_header("Testing Imports")
    
    import_tests = [
        ("backend.backend_main", "Backend main module"),
        ("backend.config", "Backend configuration"),
        ("ai_agents.base_agent", "AI agents base"),
    ]
    
    all_passed = True
    
    for module, description in import_tests:
        print_step(f"Testing import: {module} ({description})")
        
        test_code = f"import {module}; print('Successfully imported {module}')"
        returncode, stdout, stderr = run_command([sys.executable, "-c", test_code])
        
        if returncode == 0:
            print_success(f"Successfully imported {module}")
        else:
            print_error(f"Failed to import {module}")
            print(f"  Error: {stderr.strip()}")
            all_passed = False
    
    return all_passed


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fix dependency conflicts in the SutazAI application."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for conflicts without fixing them",
    )
    parser.add_argument(
        "--update-requirements", 
        action="store_true", 
        help="Update requirements files"
    )
    
    args = parser.parse_args()
    
    print_header("SutazAI Dependency Fixer")
    
    # Check current conflicts
    conflicts = get_dependency_conflicts()
    
    if not conflicts:
        print_success("No dependency conflicts found!")
        if args.check_only:
            return
    else:
        print_warning(f"Found {len(conflicts)} dependency conflicts:")
        for conflict in conflicts:
            print(f"  • {conflict['package']} requires {conflict['requirement']}")
        
        if args.check_only:
            return
    
    # Fix known conflicts
    fixed = fix_known_conflicts()
    
    # Update requirements files if requested
    if args.update_requirements or fixed:
        update_requirements_files()
    
    # Verify that all conflicts are resolved
    resolved = verify_fixed_dependencies()
    
    # Test imports
    imports_ok = test_imports()
    
    # Final message
    print_header("Summary")
    
    if resolved and imports_ok:
        print(f"{Colors.OKGREEN}{Colors.BOLD}All dependency issues have been resolved!{Colors.ENDC}")
        print(f"\nYou can now run the application with: {Colors.BOLD}python -m backend.backend_main{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}{Colors.BOLD}Some issues could not be automatically resolved.{Colors.ENDC}")
        print("\nPlease review the output above for details on remaining issues.")


if __name__ == "__main__":
    main() 