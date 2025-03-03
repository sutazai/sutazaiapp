#!/usr/bin/env python3
"""
Surgical Fixes Script for Python Syntax Errors

This script directly modifies specific files with known syntax issues,
using surgical precision to fix each file according to its exact problem.
"""

import os
import sys

# File paths relative to the base directory
TARGET_FILES = {
    "ai_agents/document_processor/tests/test_document_processor.py": {
        "issue": "unterminated docstring",
        "line": 128
    },
    "ai_agents/document_processor/tests/conftest.py": {
        "issue": "unterminated docstring",
        "line": 119
    },
    "ai_agents/auto_gpt/src/__init__.py": {
        "issue": "unterminated docstring",
        "line": 95
    },
    "ai_agents/supreme_ai/orchestrator.py": {
        "issue": "duplicated docstring",
        "line": 140
    },
    "doc_data/core_system/system_setup.py": {
        "issue": "unterminated string",
        "line": 77
    },
    "scripts/system_manager.py": {
        "issue": "unterminated docstring",
        "line": 136
    },
    "scripts/python_compatibility_manager.py": {
        "issue": "unterminated docstring",
        "line": 69
    }
}

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"{Colors.HEADER}{message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}{message}{Colors.ENDC}")

def print_success(message):
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}{message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}{message}{Colors.ENDC}")

def check_syntax(file_path):
    """Check if a file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Use Python's built-in compile function to check for syntax errors
        compile(content, file_path, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def fix_test_document_processor(file_path):
    """Fix the syntax in test_document_processor.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any auto-generated docstrings at the end of the file
    if content.endswith('"""This is an auto-generated docstring."""'):
        content = content[:-len('"""This is an auto-generated docstring."""')]
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_conftest(file_path):
    """Fix the syntax in conftest.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any auto-generated docstrings at the end of the file
    if content.endswith('"""This is an auto-generated docstring."""'):
        content = content[:-len('"""This is an auto-generated docstring."""')]
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_init_py(file_path):
    """Fix the syntax in __init__.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix class docstring issues
    if 'class AutoGPTAgent(BaseAgent):' in content:
        content = content.replace('class AutoGPTAgent(BaseAgent):\n            """This is an auto-generated docstring."""', 
                                 'class AutoGPTAgent(BaseAgent):\n            """AutoGPT Agent implementation."""')
    
    # Remove any dangling docstrings
    if content.endswith('"""This is an auto-generated docstring."""'):
        content = content[:-len('"""This is an auto-generated docstring."""')]
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_orchestrator(file_path):
    """Fix the syntax in orchestrator.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix doubled docstring issue
    if '"""This is an auto-generated docstring."""This is an auto-generated docstring."""' in content:
        content = content.replace('"""This is an auto-generated docstring."""This is an auto-generated docstring."""', 
                                '"""This is an auto-generated docstring."""')
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_system_setup(file_path):
    """Fix the syntax in system_setup.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and fix the problematic f-string line
    for i, line in enumerate(lines):
        if 'f"Installing {len()}' in line:
            lines[i] = 'f"Installing {len(missing_packages)} missing packages..."\n'
    
    # Write the file back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return check_syntax(file_path)

def fix_system_manager(file_path):
    """Fix the syntax in system_manager.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any auto-generated docstrings at the end of the file
    if content.endswith('"""This is an auto-generated docstring."""'):
        content = content[:-len('"""This is an auto-generated docstring."""')]
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_python_compatibility_manager(file_path):
    """Fix the syntax in python_compatibility_manager.py"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any auto-generated docstrings at the end of the file
    if content.endswith('"""This is an auto-generated docstring."""'):
        content = content[:-len('"""This is an auto-generated docstring."""')]
    
    # Ensure the file has proper closure
    if not content.endswith('\n'):
        content += '\n'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return check_syntax(file_path)

def fix_all_files_direct_replacement(base_dir):
    """
    A more aggressive approach that just creates clean copies of the files.
    This is used as a last resort if other methods fail.
    """
    print_header("Applying direct replacements for all files")
    
    fixed_count = 0
    
    for rel_path, info in TARGET_FILES.items():
        full_path = os.path.join(base_dir, rel_path)
        print_info(f"Processing {full_path}")
        
        # Skip if file doesn't exist
        if not os.path.exists(full_path):
            print_error(f"File not found: {full_path}")
            continue
        
        # Skip if already fixed
        if check_syntax(full_path)[0]:
            print_info(f"No syntax errors in {full_path}")
            continue
        
        # Read the file content
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Remove any extra triple-quoted docstring from the end
        while lines and ('"""' in lines[-1] or "'''" in lines[-1]):
            lines.pop()
        
        # Write the file back without the problematic line
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        # Check if fix worked
        is_valid, error = check_syntax(full_path)
        if is_valid:
            print_success(f"✓ Successfully fixed {rel_path}")
            fixed_count += 1
        else:
            print_warning(f"! Could not fix {rel_path}: {error}")
    
    return fixed_count

def main():
    """Main function to perform surgical fixes."""
    print_header("Starting Surgical Syntax Fixes")
    
    # Get base directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = os.getcwd()
    
    # Map each file to its specific fix function
    fix_functions = {
        "ai_agents/document_processor/tests/test_document_processor.py": fix_test_document_processor,
        "ai_agents/document_processor/tests/conftest.py": fix_conftest,
        "ai_agents/auto_gpt/src/__init__.py": fix_init_py,
        "ai_agents/supreme_ai/orchestrator.py": fix_orchestrator,
        "doc_data/core_system/system_setup.py": fix_system_setup,
        "scripts/system_manager.py": fix_system_manager,
        "scripts/python_compatibility_manager.py": fix_python_compatibility_manager
    }
    
    fixed_count = 0
    
    # Try specialized fixes first
    for rel_path, fix_func in fix_functions.items():
        full_path = os.path.join(base_dir, rel_path)
        print_info(f"Processing {full_path}")
        
        # Skip if file doesn't exist
        if not os.path.exists(full_path):
            print_error(f"File not found: {full_path}")
            continue
        
        # Skip if already fixed
        if check_syntax(full_path)[0]:
            print_info(f"No syntax errors in {full_path}")
            continue
        
        # Apply the specific fix for this file
        is_valid, error = fix_func(full_path)
        if is_valid:
            print_success(f"✓ Successfully fixed {rel_path}")
            fixed_count += 1
        else:
            print_warning(f"! Specialized fix couldn't resolve all issues in {rel_path}: {error}")
    
    # If specialized fixes didn't work for all files, try direct replacement
    if fixed_count < len(fix_functions):
        print_warning("\nSome files couldn't be fixed with specialized approaches.")
        print_info("Attempting direct replacement as a last resort...")
        fixed_count = fix_all_files_direct_replacement(base_dir)
    
    print_header("\nSurgical Fix Summary")
    print_info(f"Fixed {fixed_count} out of {len(fix_functions)} target files")
    
    # Check which files still have issues
    remaining_errors = []
    for rel_path in fix_functions:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path) and not check_syntax(full_path)[0]:
            remaining_errors.append(rel_path)
    
    if remaining_errors:
        print_warning("\nFiles still containing syntax errors:")
        for file_path in remaining_errors:
            print_warning(f"- {file_path}")
        print_warning("\nThese files may require manual inspection.")
    else:
        print_success("\nAll target files have been successfully fixed!")

if __name__ == "__main__":
    main() 