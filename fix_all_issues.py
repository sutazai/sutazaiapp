#!/usr/bin/env python3
"""
SutazAI Fix All Issues Script
----------------------------
This script orchestrates the execution of all fix scripts to address all issues in the project.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and print its output."""
    print(f"\n\033[1;34m=== {description} ===\033[0m")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\033[1;31mCommand failed with exit code {e.returncode}\033[0m")
        print(f"\033[1;31mError output:\033[0m")
        print(e.stderr)
        return False


def main():
    """Main entry point for fixing all issues."""
    project_root = Path(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = project_root / "scripts"
    
    print("\033[1;32m🔧 Starting comprehensive issue fixing process\033[0m")
    
    # Step 1: Run the system checkup to identify issues
    run_command(
        [sys.executable, os.path.join(project_root, "system_checkup.py")],
        "Running system checkup"
    )
    
    # Step 2: Fix syntax errors using syntax_fixer.py
    run_command(
        [sys.executable, os.path.join(scripts_dir, "syntax_fixer.py"), str(project_root)],
        "Fixing syntax errors"
    )
    
    # Step 3: Run the project optimizer
    if (scripts_dir / "project_optimizer.py").exists():
        run_command(
            [sys.executable, os.path.join(scripts_dir, "project_optimizer.py")],
            "Optimizing project structure"
        )
    
    # Step 4: Run system_comprehensive_audit.py
    if (scripts_dir / "system_comprehensive_audit.py").exists():
        run_command(
            [sys.executable, os.path.join(scripts_dir, "system_comprehensive_audit.py")],
            "Auditing system configuration"
        )
    
    # Step 5: Run system_health_check.py
    if (scripts_dir / "system_health_check.py").exists():
        run_command(
            [sys.executable, os.path.join(scripts_dir, "system_health_check.py")],
            "Checking system health"
        )
    
    # Step 6: Run dependency checks and fixes
    if (scripts_dir / "dependency_management.py").exists():
        run_command(
            [sys.executable, os.path.join(scripts_dir, "dependency_management.py")],
            "Managing dependencies"
        )
    
    # Step 7: Fix any import issues
    if (scripts_dir / "import_resolver.py").exists():
        run_command(
            [sys.executable, os.path.join(scripts_dir, "import_resolver.py")],
            "Resolving import issues"
        )
    
    # Step 8: Run final system checkup to verify fixes
    run_command(
        [sys.executable, os.path.join(project_root, "system_checkup.py")],
        "Running final system checkup"
    )
    
    print("\n\033[1;32m✅ All fix scripts have been executed!\033[0m")
    print("\033[1;33mNote: Some issues may require manual intervention.\033[0m")
    print("\033[1;33mPlease review the output above for any remaining issues.\033[0m")


if __name__ == "__main__":
    main() 