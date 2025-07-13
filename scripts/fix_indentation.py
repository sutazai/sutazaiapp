#!/usr/bin/env python3
"""
Script to fix indentation errors in Python files.

This script scans Python files for indentation issues and attempts
to fix them by standardizing indentation.
"""

import os
import re
import sys
from typing import List, Tuple, Dict, Optional


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


def print_header(message: str) -> None:
    print(f"{Colors.HEADER}{message}{Colors.ENDC}")


def print_info(message: str) -> None:
    print(f"{Colors.BLUE}{message}{Colors.ENDC}")


def print_success(message: str) -> None:
    print(f"{Colors.GREEN}{message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    print(f"{Colors.YELLOW}{message}{Colors.ENDC}")


def print_error(message: str) -> None:
    print(f"{Colors.RED}{message}{Colors.ENDC}")


def get_python_files(directory: str) -> List[str]:
    """Get all Python files in the directory tree."""
    python_files = []
    exclude_dirs = ["venv", ".git", "__pycache__", "node_modules", ".pytest_cache"]

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files


def check_for_syntax(file_path: str) -> Tuple[bool, str]:
    """Check if a file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Use Python's built-in compile function to check for syntax errors
        compile(content, file_path, 'exec')
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except UnicodeDecodeError:
        return False, "File contains non-UTF-8 characters"


def fix_indentation(content: str) -> str:
    """
    Fix indentation issues in Python code by standardizing to 4 spaces.

    This function:
    1. Detects inconsistent indentation
    2. Converts tabs to spaces
    3. Standardizes indentation levels
    """
    lines = content.splitlines()
    result_lines = []

    # First, replace all tabs with 4 spaces
    for i, line in enumerate(lines):
        if '\t' in line:
            # Replace tab with 4 spaces
            lines[i] = line.replace('\t', '    ')

    # Track indentation levels
    indentation_stack = [0]  # Start with no indentation
    current_block_indentation = 0

    for i, line in enumerate(lines):
        stripped_line = line.lstrip()

        # Skip empty lines or comments
        if not stripped_line or stripped_line.startswith('#'):
            result_lines.append(line)
            continue

        # Calculate leading whitespace
        leading_spaces = len(line) - len(stripped_line)

        # Check for lines that should increase indentation
        if i > 0 and lines[i-1].rstrip().endswith((':',)):
            # This line should be indented relative to the previous line
            expected_indent = indentation_stack[-1] + 4
            if leading_spaces < expected_indent:
                # Line needs more indentation
                result_lines.append(' ' * expected_indent + stripped_line)
                current_block_indentation = expected_indent
                indentation_stack.append(expected_indent)
                continue

        # Check for dedent (less indentation than previous line)
        if leading_spaces < current_block_indentation:
            # Pop indentation levels until we find the matching one
            while indentation_stack and indentation_stack[-1] > leading_spaces:
                indentation_stack.pop()

            # If we didn't find a matching level, use the last one
            if not indentation_stack:
                indentation_stack = [0]

            current_block_indentation = indentation_stack[-1]

        # Check for indent (more indentation than previous line)
        elif leading_spaces > current_block_indentation:
            indentation_stack.append(leading_spaces)
            current_block_indentation = leading_spaces

        # Add the line with proper indentation
        result_lines.append(' ' * current_block_indentation + stripped_line)

    return '\n'.join(result_lines)


def fix_common_indent_errors(content: str) -> str:
    """
    Fix common indentation errors that the general algorithm might miss.
    """
    lines = content.splitlines()
    result_lines = []

    # Regular expression to detect common indentation patterns
    class_def_pattern = re.compile(r'^\s*class\s+\w+\s*\(.*\)\s*:')
    function_def_pattern = re.compile(r'^\s*def\s+\w+\s*\(.*\)\s*:')
    method_def_pattern = re.compile(r'^\s*def\s+\w+\s*\(self.*\)\s*:')

    # Track the indentation level for classes and functions
    in_class = False
    in_function = False
    class_indent = 0
    function_indent = 0

    for i, line in enumerate(lines):
        # Check if line defines a class
        if class_def_pattern.match(line):
            in_class = True
            in_function = False
            class_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue

        # Check if line defines a function or method
        if function_def_pattern.match(line):
            in_function = True
            if in_class and method_def_pattern.match(line):
                # It's a method, should be indented inside class
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces <= class_indent:
                    # Method needs more indentation
                    result_lines.append(' ' * (class_indent + 4) + line.lstrip())
                    function_indent = class_indent + 4
                    continue

            function_indent = len(line) - len(line.lstrip())
            result_lines.append(line)
            continue

        # Handle indentation for class/function body
        if in_function or in_class:
            stripped_line = line.lstrip()
            if not stripped_line:  # Empty line
                result_lines.append(line)
                continue

            leading_spaces = len(line) - len(stripped_line)

            # Inside function body - should be indented
            if in_function and leading_spaces <= function_indent and stripped_line and not stripped_line.startswith(('class ', 'def ')):
                # Function body needs more indentation
                result_lines.append(' ' * (function_indent + 4) + stripped_line)
                continue

            # Inside class body but outside method - should be indented
            if in_class and not in_function and leading_spaces <= class_indent and stripped_line and not stripped_line.startswith(('class ', 'def ')):
                # Class body needs more indentation
                result_lines.append(' ' * (class_indent + 4) + stripped_line)
                continue

        # If we get here, just append the line as is
        result_lines.append(line)

    return '\n'.join(result_lines)


def fix_indentation_issues_in_file(file_path: str) -> bool:
    """Process a file to fix indentation issues."""
    try:
        # Check if the file has syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        if is_valid:
            return False  # No changes needed

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try general indentation fixing
        fixed_content = fix_indentation(content)

        # Try specific indentation patterns
        if fixed_content == content or not check_for_syntax(file_path)[0]:
            fixed_content = fix_common_indent_errors(fixed_content)

        # If file was modified, write it back
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            # Check if the fix resolved the syntax error
            is_valid_now, _ = check_for_syntax(file_path)
            if is_valid_now:
                return True  # Fixed successfully
            else:
                print_warning(f"  Indentation fix didn't resolve all syntax errors in {file_path}")
                return True  # Still made changes

        return False  # No changes needed

    except Exception as e:
        print_error(f"  Error processing {file_path}: {str(e)}")
        return False


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print_error("Usage: python fix_indentation.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    print_header(f"Starting indentation fix on {directory}")

    python_files = get_python_files(directory)
    print_info(f"Found {len(python_files)} Python files")

    files_fixed = 0

    for file_path in python_files:
        # Check for syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)

        # Look for indentation errors
        if not is_valid and any(word in error_msg.lower() for word in
                               ['indent', 'expected', 'tab', 'space']):
            print_info(f"Attempting to fix {file_path}")
            if fix_indentation_issues_in_file(file_path):
                print_success(f"✓ Fixed: {file_path}")
                files_fixed += 1

    print_header("\nIndentation Fix Summary")
    print_info(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main()
