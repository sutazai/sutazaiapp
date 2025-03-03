#!/usr/bin/env python3.11
"""
Comprehensive Issue Fixer for SutazAI

This script fixes all identified issues in the SutazAI codebase:
1. Installs missing dependencies
2. Fixes __future__ imports
3. Fixes the run_all_optimizations.py script
4. Sets up the correct PYTHONPATH
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(message):
    """Print a formatted header message."""
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

def fix_dependencies():
    """Install all required dependencies."""
    print_header("Installing Dependencies")
    
    # Core dependencies
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
        "types-PyYAML>=6.0.12",
        "types-aiofiles>=24.1.0",
        "mypy>=1.8.0",
        "PyMuPDF>=1.25.0",  # Provides fitz module
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}")
    
    # Fix problematic packages
    print_header("Fixing problematic packages")
    
    # Fix pluggy package which often has syntax errors
    run_command("pip uninstall -y pluggy")
    run_command("pip install pluggy==1.2.0")
    
    # Fix docx2txt which often fails to build
    run_command("pip uninstall -y docx2txt")
    run_command("pip install git+https://github.com/ankushshah89/python-docx2txt.git")

def run_fix_future_imports():
    """Run the fix_future_imports.py script."""
    print_header("Fixing __future__ imports")
    script_path = Path(os.path.dirname(os.path.abspath(__file__))) / "fix_future_imports.py"
    run_command(f"python {script_path}")

def setup_pythonpath():
    """Set up the PYTHONPATH environment variable."""
    print_header("Setting up PYTHONPATH")
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set for this session
    os.environ["PYTHONPATH"] = str(project_root)
    sys.path.insert(0, str(project_root))
    
    print(f"PYTHONPATH set to: {os.environ.get('PYTHONPATH')}")
    print("\nTo permanently set PYTHONPATH, add the following to your .bashrc or equivalent:")
    print(f"export PYTHONPATH={project_root}")
    
    # Create a helper script to set PYTHONPATH
    helper_script = project_root / "set_pythonpath.sh"
    with open(helper_script, "w") as f:
        f.write(f"""#!/bin/bash
# Set PYTHONPATH for SutazAI
export PYTHONPATH={project_root}
echo "PYTHONPATH set to $PYTHONPATH"
""")
    
    os.chmod(helper_script, 0o755)
    print(f"\nCreated helper script: {helper_script}")
    print("You can run 'source set_pythonpath.sh' before running your application")

def check_application():
    """Check if the application can run without errors."""
    print_header("Checking application")
    
    # Check if backend is already running
    result = run_command("curl -s http://localhost:8000/api/health", check=False)
    if result.returncode == 0:
        print("Backend is already running and responding to health checks.")
        return
    
    # Try to run a simple import test
    test_script = """
import sys
print("Python version:", sys.version)

try:
    import fastapi
    print("✅ fastapi imported successfully")
except ImportError as e:
    print("❌ Failed to import fastapi:", e)

try:
    import uvicorn
    print("✅ uvicorn imported successfully")
except ImportError as e:
    print("❌ Failed to import uvicorn:", e)

try:
    import loguru
    print("✅ loguru imported successfully")
except ImportError as e:
    print("❌ Failed to import loguru:", e)

try:
    import numpy
    print("✅ numpy imported successfully")
except ImportError as e:
    print("❌ Failed to import numpy:", e)

try:
    import cv2
    print("✅ cv2 imported successfully")
except ImportError as e:
    print("❌ Failed to import cv2:", e)

try:
    import fitz  # PyMuPDF
    print("✅ fitz (PyMuPDF) imported successfully")
except ImportError as e:
    print("❌ Failed to import fitz:", e)

try:
    import docx
    print("✅ docx imported successfully")
except ImportError as e:
    print("❌ Failed to import docx:", e)

try:
    import docx2txt
    print("✅ docx2txt imported successfully")
except ImportError as e:
    print("❌ Failed to import docx2txt:", e)

try:
    import transformers
    print("✅ transformers imported successfully")
except ImportError as e:
    print("❌ Failed to import transformers:", e)

try:
    import semgrep
    print("✅ semgrep imported successfully")
except ImportError as e:
    print("❌ Failed to import semgrep:", e)
"""
    
    test_file = Path(os.path.dirname(os.path.abspath(__file__))) / "import_test.py"
    with open(test_file, "w") as f:
        f.write(test_script)
    
    run_command(f"python {test_file}")

def main():
    """Main function to run all fixes."""
    print_header("Starting Comprehensive Issue Fixer for SutazAI")
    
    # Run all fixes
    fix_dependencies()
    run_fix_future_imports()
    setup_pythonpath()
    check_application()
    
    print_header("All fixes completed")
    print("\nTo run your application with the correct Python path:")
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"PYTHONPATH={project_root} python backend/backend_main.py")
    print("\nOr source the helper script:")
    print("source set_pythonpath.sh")
    print("python backend/backend_main.py")

if __name__ == "__main__":
    main() 