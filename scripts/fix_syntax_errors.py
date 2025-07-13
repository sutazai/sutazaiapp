#!/usr/bin/env python3.11
"""
Syntax Error Fixer

This script systematically finds and fixes common syntax errors in Python files:
    1. Unterminated triple-quoted strings
2. Unmatched brackets/parentheses
3. Indentation errors
4. Other common syntax issues

Usage:
    python scripts/fix_syntax_errors.py [directory]
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

# ANSI color codes for terminal output

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
    """Print a header message."""
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}{message}{Colors.ENDC}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}{message}{Colors.ENDC}")

def get_all_python_files(directory: str, exclude_dirs: List[str] = None) -> List[str]:
    """Get a list of all Python files in the given directory, recursively."""
    if exclude_dirs is None:
        exclude_dirs = ["venv", ".git", "__pycache__"]

    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    return python_files

def check_file_for_syntax(file_path: str) -> Tuple[bool, str | None]:
    """
    Check if a Python file has syntax errors.

    Returns:
        Tuple[bool, str | None]: (has_errors, error_message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        ast.parse(source)
        return False, None
    except SyntaxError as e:
        return True, f"{e.__class__.__name__}: {e}"
    except Exception as e:
        return True, f"{e.__class__.__name__}: {e}"

def fix_unmatched_brackets(content: str) -> str:
    """
    Try to fix unmatched brackets in the content.
    This is a basic implementation that might not catch all cases.
    """
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}

    # Count opening and closing brackets
    open_count = {'(': 0, '[': 0, '{': 0}
    close_count = {')': 0, ']': 0, '}': 0}

    for char in content:
        if char in '([{':
            open_count[char] += 1
        elif char in ')]}':
            close_count[char] += 1

    # Add missing closing brackets at the end
    for close_char, open_char in pairs.items():
        diff = open_count[open_char] - close_count[close_char]
        if diff > 0:
            content += close_char * diff

    return content

def fix_triple_quotes(content: str) -> str:
    """
    Try to fix unterminated triple-quoted strings.
    This is a basic implementation and might not catch all cases.
    """
    # Regular expression to find triple-quoted strings
    triple_quotes_pattern = re.compile(r'("""|\'\'\').*?(?:\1|$)', re.DOTALL)

    # Find all matches
    matches = list(triple_quotes_pattern.finditer(content))

    # For each match, check if it's terminated
    for match in matches:
        match_str = match.group(0)
        quote_type = match.group(1)  # """ or '''

        # If the string doesn't end with the same quote type, it's unterminated
        if not match_str.endswith(quote_type):
            # Add the missing quotes at the end of the match
            end_pos = match.end()
            content = content[:end_pos] + quote_type + content[end_pos:]

    return content

def fix_indentation(content: str) -> str:
    """
    Attempt to fix indentation issues.
    This is a simplistic approach and may not work for all cases.
    """
    lines = content.split('\n')
    fixed_lines = []
    indent_stack = [0]  # Stack to keep track of indentation levels

    for line in lines:
        stripped = line.lstrip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line)
            continue

        # Calculate current indentation
        current_indent = len(line) - len(stripped)

        # Check if this line should increase indentation (ends with ':')
        if stripped.endswith(':'):
            fixed_lines.append(line)
            indent_stack.append(current_indent + 4)  # Standard 4-space indent
            continue

        # Check if this line should be indented according to the stack
        expected_indent = indent_stack[-1]

        # If line starts with 'else', 'elif', 'except', 'finally', maintain same indentation
        if stripped.startswith(('else:', 'elif', 'except', 'finally')):
            if indent_stack[-1] > 0:
                expected_indent = indent_stack[-2]  # Same level as the parent 'if'

        # If too little indentation, add more
        if current_indent < expected_indent:
            fixed_lines.append(' ' * expected_indent + stripped)
        # If too much indentation, reduce it
        elif current_indent > expected_indent and len(indent_stack) > 1:
            # Pop indentation levels until we find the correct one
            while indent_stack and current_indent < indent_stack[-1]:
                indent_stack.pop()
            fixed_lines.append(' ' * indent_stack[-1] + stripped)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_file(file_path: str) -> bool:
    """
    Apply fixes to a Python file with syntax errors.

    Returns:
        bool: True if fixes were applied, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_unmatched_brackets(content)
        content = fix_triple_quotes(content)
        content = fix_indentation(content)

        # Check if content was modified
        if content != original_content:
            # Create a backup
            backup_path = f"{file_path}.bak"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        return False
    except Exception as e:
        print_error(f"Error fixing {file_path}: {e}")
        return False

def main() -> None:
    """Main entry point for the script."""
    print_header("SutazAI Syntax Error Fixer")

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.getcwd()

    print(f"Scanning directory: {directory}")

    # Get all Python files
    python_files = get_all_python_files(directory)
    print(f"Found {len(python_files)} Python files")

    # Check each file for syntax errors and fix if needed
    files_with_errors = 0
    files_fixed = 0

    for file_path in python_files:
        if file_path.endswith("fix_syntax_errors.py"):
            continue  # Skip this script

        has_errors, error_message = check_file_for_syntax(file_path)

        if has_errors:
            files_with_errors += 1
            print_warning(f"Syntax error in {file_path}: {error_message}")

            if fix_file(file_path):
                files_fixed += 1
                print_success(f"Applied fixes to {file_path}")

                # Check if fixes resolved the issue
                has_errors_after, error_message_after = check_file_for_syntax(file_path)
                if has_errors_after:
                    print_warning(f"File still has errors after fixing: {error_message_after}")
                else:
                    print_success(f"Successfully fixed syntax errors in {file_path}")
            else:
                print_warning(f"No fixes applied to {file_path}")

    # Print summary
    print_header("\nSummary:")
    print(f"Total Python files: {len(python_files)}")
    print(f"Files with syntax errors: {files_with_errors}")
    print(f"Files fixed: {files_fixed}")

    if files_with_errors > files_fixed:
        print_warning(f"{files_with_errors - files_fixed} files still have syntax errors")
        print("These files may require manual attention.")

    if files_fixed > 0:
        print_success("Fix process completed. Check the fixed files for correctness.")

if __name__ == "__main__":
    main()
