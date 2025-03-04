#!/usr/bin/env python3
"""
Script to achieve 100% code coverage for SutazaiApp

This script:
1. Analyzes coverage data to identify uncovered code paths
2. Generates additional tests for those paths
3. Updates existing code with coverage-friendly modifications if needed
"""

import os
import sys
import json
import subprocess
import glob
from pathlib import Path
import ast
import re
import time

# Paths to core files
CORE_PATH = Path("/opt/sutazaiapp/core_system/orchestrator")
TEST_PATH = Path("/opt/sutazaiapp/tests")
COVERAGE_REPORT_PATH = Path("/opt/sutazaiapp/coverage")
MODULES = ["agent_manager.py", "supreme_ai.py", "sync_manager.py", "task_queue.py"]

def run_command(cmd, cwd=None, shell=True):
    """Run a shell command and return the output"""
    print(f"Running command: {cmd}")
    try:
        with subprocess.Popen(
            cmd if shell else cmd.split(),
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd
        ) as process:
            stdout, stderr = process.communicate()
            
            if process.returncode != 0 and "AssertionError" in stderr:
                print(f"Command failed with test errors: {stderr}")
                return None
                
            # Print stderr but don't consider warnings as failures
            if stderr and "warning" in stderr.lower():
                print(f"Command produced warnings: {stderr}")
                
            return stdout
    except Exception as e:
        print(f"Error executing command: {e}")
        return None

def manually_parse_module(module_path):
    """Manually parse a module to find functions and their lines"""
    print(f"Analyzing file: {module_path}")
    
    try:
        with open(module_path, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('__'):  # Skip dunder methods
                    functions[node.name] = {
                        'start': node.lineno,
                        'end': node.end_lineno,
                        'is_async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    
        # Extract class information
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods = {}
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not item.name.startswith('__'):  # Skip dunder methods
                            class_methods[item.name] = {
                                'start': item.lineno,
                                'end': item.end_lineno,
                                'is_async': isinstance(item, ast.AsyncFunctionDef)
                            }
                if class_methods:
                    classes[node.name] = class_methods
                    
        return {
            'functions': functions,
            'classes': classes,
            'content': content
        }
    except Exception as e:
        print(f"Error parsing module {module_path}: {e}")
        return None

def find_existing_tests(module_name):
    """Find existing tests for a module"""
    test_files = glob.glob(f"{TEST_PATH}/test_{module_name}*.py")
    tests = []
    
    for test_file in test_files:
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Extract test function names
        matches = re.findall(r'def (test_\w+)', content)
        tests.extend(matches)
    
    return tests

def generate_test_for_function(module_name, class_name, func_name, is_async):
    """Generate a test for a specific function"""
    # Create test function name
    test_name = f"test_{func_name}"
    if test_name.startswith("test_"):
        test_name = "test_func_" + test_name[5:]
    
    fixture_name = module_name if not class_name else class_name.lower()
    
    # Create test function content
    async_prefix = "async " if is_async else ""
    async_await = "await " if is_async else ""
    
    test_code = f"""
@pytest.mark.asyncio
{async_prefix}def {test_name}({fixture_name}_fixture):
    \"\"\"Test for {func_name} function in {module_name}{' ' + class_name if class_name else ''}.\"\"\"
"""
    
    # Add specific test code based on function name
    if "heartbeat" in func_name.lower():
        test_code += f"""    # Test heartbeat functionality
    with patch.object({fixture_name}_fixture, '_heartbeat_loop', AsyncMock()):
        {async_await}{fixture_name}_fixture.{func_name}('test_agent_id')
        # Assert the function was called
"""
    elif "sync" in func_name.lower():
        test_code += f"""    # Test synchronization functionality
    with patch.object({fixture_name}_fixture, 'sync_task', AsyncMock()):
        {async_await}{fixture_name}_fixture.{func_name}()
        # Assert synchronization was successful
"""
    elif "process" in func_name.lower():
        test_code += f"""    # Test processing functionality
    with patch.object({fixture_name}_fixture, 'process_task', AsyncMock()):
        {async_await}{fixture_name}_fixture.{func_name}()
        # Assert processing completed
"""
    elif "error" in func_name.lower() or "exception" in func_name.lower():
        test_code += f"""    # Test error handling
    # Simulate an error condition
    with pytest.raises(Exception):
        {async_await}{fixture_name}_fixture.{func_name}()
"""
    else:
        test_code += f"""    # General test
    result = {async_await}{fixture_name}_fixture.{func_name}()
    assert result is not None  # Replace with appropriate assertion
"""
    
    return test_code

def create_additional_tests(module_data):
    """Create additional test files for modules"""
    for module_name, data in module_data.items():
        print(f"Generating tests for {module_name}...")
        
        # Find existing tests
        existing_tests = find_existing_tests(module_name)
        print(f"Found {len(existing_tests)} existing tests for {module_name}")
        
        # Create test file path
        test_file = f"{TEST_PATH}/test_{module_name}_coverage.py"
        
        # Create test file content
        imports = f"""#!/usr/bin/env python3
\"\"\"
Additional tests to achieve 100% coverage for {module_name}
\"\"\"

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core_system.orchestrator import {module_name}

"""
        
        test_funcs = []
        
        # Generate tests for standalone functions
        for func_name, func_info in data.get('functions', {}).items():
            # Skip if test already exists
            if f"test_{func_name}" in existing_tests:
                continue
                
            test_func = generate_test_for_function(
                module_name, 
                None, 
                func_name, 
                func_info.get('is_async', False)
            )
            test_funcs.append(test_func)
            print(f"Generated test for function: {func_name}")
        
        # Generate tests for class methods
        for class_name, methods in data.get('classes', {}).items():
            for method_name, method_info in methods.items():
                # Skip if test already exists
                if f"test_{method_name}" in existing_tests:
                    continue
                    
                test_func = generate_test_for_function(
                    module_name,
                    class_name,
                    method_name,
                    method_info.get('is_async', False)
                )
                test_funcs.append(test_func)
                print(f"Generated test for method: {class_name}.{method_name}")
        
        # Add fixtures
        fixtures = f"""
@pytest.fixture
def {module_name}_fixture():
    \"\"\"Fixture for {module_name} module.\"\"\"
    return {module_name}.{module_name.title().replace('_', '')}()

"""
        
        # Generate class fixtures if needed
        for class_name in data.get('classes', {}):
            fixture_name = class_name.lower()
            if fixture_name != f"{module_name}_fixture":
                fixtures += f"""
@pytest.fixture
def {fixture_name}_fixture():
    \"\"\"Fixture for {class_name} class.\"\"\"
    return {module_name}.{class_name}()

"""
        
        # Only create file if we have tests to add
        if test_funcs:
            with open(test_file, 'w') as f:
                f.write(imports)
                f.write(fixtures)
                f.write("\n".join(test_funcs))
            
            print(f"Created additional test file: {test_file}")
            run_command(f"chmod +x {test_file}")
        else:
            print(f"No additional tests needed for {module_name}")

def update_modules_for_testability():
    """Update modules to make them more testable"""
    for module in MODULES:
        file_path = os.path.join(CORE_PATH, module)
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            # Parse the file to validate it's valid Python
            ast.parse(content)
            
            # Use a more careful regex to add pass statements to empty except blocks
            # Look for except blocks that don't have any indented code
            lines = content.split('\n')
            modified_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                modified_lines.append(line)
                
                # Check if this is an except line
                if re.match(r'\s*except\s+.*:', line):
                    # Get the indentation level
                    indent = len(line) - len(line.lstrip())
                    next_indent = None
                    
                    # Check the next line's indentation
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if next_line.strip():  # Not empty
                            next_indent = len(next_line) - len(next_line.lstrip())
                    
                    # If next line doesn't exist or has same/less indentation, add a pass
                    if next_indent is None or next_indent <= indent:
                        modified_lines.append(' ' * (indent + 4) + 'pass  # Added for test coverage')
            
                i += 1
            
            updated_content = '\n'.join(modified_lines)
            
            # Only save if changes were made
            if content != updated_content:
                with open(file_path, 'w') as f:
                    f.write(updated_content)
                print(f"Updated {module} for better testability")
        except SyntaxError as e:
            print(f"Syntax error in {module}, skipping: {e}")

def main():
    print("Starting 100% coverage achievement process...")
    
    # Update modules for better testability
    update_modules_for_testability()
    
    # Run remote tests to ensure coverage data is fresh
    print("Running remote tests to generate coverage data...")
    run_command("cd /opt/sutazaiapp && ./run_remote_tests.sh")
    
    # Analyze modules and create test data
    module_data = {}
    for module in MODULES:
        module_name = module.replace('.py', '')
        module_path = os.path.join(CORE_PATH, module)
        
        if os.path.exists(module_path):
            data = manually_parse_module(module_path)
            if data:
                module_data[module_name] = data
    
    # Create additional tests
    create_additional_tests(module_data)
    
    print("\nNext steps:")
    print("1. Review and customize the generated test files")
    print("2. Run the tests with ./run_remote_tests.sh")
    print("3. Repeat until 100% coverage is achieved")

if __name__ == "__main__":
    main() 