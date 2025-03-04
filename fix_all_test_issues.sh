#!/bin/bash

# Master script to fix all test issues and verify results

echo "Starting comprehensive test fix process..."

# Set resource limits to prevent high CPU and memory usage
ulimit -t 600  # CPU time limit (10 minutes)
ulimit -v 2000000  # Virtual memory limit (2GB)
ulimit -m 1500000  # Max memory size (1.5GB)

# Activate the virtual environment
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# 1. Fix pytest configuration
echo "1. Fixing pytest configuration..."
cat > pyproject.toml << EOF
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "asyncio: mark test as an asyncio test",
]
asyncio_default_fixture_loop_scope = "function"
EOF

# 2. Fix indentation in all test files
echo "2. Fixing indentation of @pytest.mark.asyncio decorators in all test files..."
chmod +x fix_asyncio_marker_indentation.sh
./fix_asyncio_marker_indentation.sh

# 3. Fix the unawaited coroutine warnings
echo "3. Fixing unawaited coroutine warnings..."
chmod +x fix_unawaited_coroutines.sh
./fix_unawaited_coroutines.sh

# 4. Create or update conftest.py
echo "4. Ensuring conftest.py is correctly configured..."
cat > tests/conftest.py << EOF
"""
Pytest configuration for the test suite.
"""
import os
import sys
import pytest

# Add the parent directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as an asyncio test")
EOF

# 5. Run the agent_manager_targeted.py tests to verify fixes
echo "5. Verifying fixes with test_agent_manager_targeted.py..."
python -m pytest tests/test_agent_manager_targeted.py -v

# 6. Run the sync_manager_complete_coverage.py tests to verify fixes
echo "6. Verifying fixes with test_sync_manager_complete_coverage.py..."
python -m pytest tests/test_sync_manager_complete_coverage.py -v

# 7. Generate a final test report
echo "7. Generating final test report..."
python3 -c '
import os
import re
import subprocess

print("\n=== TEST FIX VERIFICATION REPORT ===\n")

# Check if there are any remaining indentation issues
print("Checking for remaining indentation issues...")
issues_found = False
for root, dirs, files in os.walk("tests"):
    for file in files:
        if file.startswith("test_") and file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if "@pytest.mark.asyncio" in line and not line.strip().startswith("@") and not line.strip().startswith("def"):
                        print(f"  - Issue in {file_path}, line {i+1}: {line.strip()}")
                        issues_found = True

if not issues_found:
    print("  ✓ No indentation issues found")

# Check for duplicate decorators
print("\nChecking for duplicate decorators...")
issues_found = False
for root, dirs, files in os.walk("tests"):
    for file in files:
        if file.startswith("test_") and file.endswith(".py"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                content = f.read()
                if re.search(r"@pytest\.mark\.asyncio\s+@pytest\.mark\.asyncio", content):
                    print(f"  - Duplicate decorators found in {file_path}")
                    issues_found = True

if not issues_found:
    print("  ✓ No duplicate decorators found")

# Run all tests and capture the output
print("\nRunning all tests to generate summary...")
try:
    result = subprocess.run(["python", "-m", "pytest"], capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Count passes, failures, etc.
    match = re.search(r"(\d+) passed, (\d+) skipped, (\d+) failed", output)
    if match:
        passed, skipped, failed = match.groups()
        print(f"  Test Summary: {passed} passed, {skipped} skipped, {failed} failed")
    
    # Check for warning patterns
    coroutine_warnings = len(re.findall(r"coroutine .* was never awaited", output))
    if coroutine_warnings > 0:
        print(f"  - {coroutine_warnings} coroutine warnings remaining")
    else:
        print("  ✓ No coroutine warnings")
        
except Exception as e:
    print(f"  Error running tests: {e}")

print("\n=== END OF REPORT ===")
'

echo "Fix and verification process completed!" 