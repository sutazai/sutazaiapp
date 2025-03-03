#!/usr/bin/env python3.11
"""
Dependency Fixer for SutazAI

This script fixes common dependency issues in the SutazAI codebase:
1. Installs missing packages from requirements.txt
2. Fixes specific problematic packages with alternative installation methods
"""

import os
import subprocess
import sys
from pathlib import Path

def print_status(message):
    """Print a formatted status message."""
    print(f"\n{'='*80}\n{message}\n{'='*80}")

def run_command(command, check=True):
    """Run a command and handle potential errors."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return e

def fix_pluggy_package():
    """Fix the pluggy package which often has syntax errors."""
    print_status("Fixing pluggy package (fixing potential syntax error)")
    run_command("pip uninstall -y pluggy")
    run_command("pip install pluggy==1.2.0")  # Install a stable version

def fix_docx2txt_package():
    """Fix docx2txt which often fails to build."""
    print_status("Fixing docx2txt package")
    run_command("pip uninstall -y docx2txt")
    run_command("pip install git+https://github.com/ankushshah89/python-docx2txt.git")

def install_main_dependencies():
    """Install the main dependencies from requirements.txt"""
    print_status("Installing main dependencies")
    requirements_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "requirements.txt"
    run_command(f"pip install -r {requirements_path}")

def fix_import_issues():
    """Fix specific import issues that might not be fixed by the main requirements"""
    print_status("Installing specific dependencies for import issues")
    
    # Library dependencies
    dependencies = [
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.34.0",
        "pydantic>=2.10.0",
        "loguru>=0.7.0",
        "aiofiles>=23.2.0",
        "PyYAML>=6.0.0",
        "python-docx>=1.0.0",
        "opencv-python>=4.0.0",  # Provides cv2
        "numpy>=1.24.0",
        "transformers>=4.40.0",
        "semgrep>=1.58.0",
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}")

def check_python_path():
    """Check Python path and suggest fixes."""
    print_status("Checking Python path")
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Project root: {project_root}")
    
    # Set the PYTHONPATH environment variable for this session
    sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)
    
    print(f"Updated PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    print("To permanently set PYTHONPATH, add the following to your .bashrc or equivalent:")
    print(f"export PYTHONPATH={project_root}")

def fix_run_optimizations_script():
    """Fix the run_all_optimizations.py script which might have issues."""
    print_status("Fixing run_all_optimizations.py script")
    script_path = Path(os.path.dirname(os.path.abspath(__file__))) / "run_all_optimizations.py"
    
    # Specific fix for the reportOptionalMemberAccess issue
    with open(script_path, "r") as f:
        content = f.read()
    
    # Add a check before calling exec_module to prevent null access
    updated_content = content.replace(
        "spec.loader.exec_module(module)",
        "if spec and spec.loader:\n            spec.loader.exec_module(module)\n        else:\n            logger.error(f\"Could not load module {module_path}\")"
    )
    
    with open(script_path, "w") as f:
        f.write(updated_content)
    
    print(f"Updated {script_path}")

def main():
    """Main function to run all fixes."""
    print_status("Starting SutazAI dependency fixer")
    
    # Run all the fixes
    check_python_path()
    install_main_dependencies()
    fix_pluggy_package()
    fix_docx2txt_package()
    fix_import_issues()
    fix_run_optimizations_script()
    
    print_status("Dependency fixes completed")
    print("\nTo run your application with the correct Python path:")
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"PYTHONPATH={project_root} python your_script.py")

if __name__ == "__main__":
    main() 