#!/usr/bin/env python3
"""
Script to fix unmatched brackets, parentheses, and braces in Python files.

This script scans Python files for unmatched brackets of different types
and attempts to fix them by adding the missing closing or opening brackets.
"""

import os
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


def fix_unmatched_brackets(content: str) -> str:
    """Fix unmatched brackets, parentheses, and braces."""
    bracket_pairs = {
        '(': ')',
        '[': ']',
        '{': '}'
    }

    # For each type of bracket, count and fix
    for open_bracket, close_bracket in bracket_pairs.items():
        # Count the occurrences
        open_count = content.count(open_bracket)
        close_count = content.count(close_bracket)

        # Fix missing closing brackets
        if open_count > close_count:
            # Add missing closing brackets at the end
            for _ in range(open_count - close_count):
                # Find the last line with code (not just whitespace)
                lines = content.splitlines()
                last_code_line_idx = len(lines) - 1
                while last_code_line_idx >= 0 and not lines[last_code_line_idx].strip():
                    last_code_line_idx -= 1

                if last_code_line_idx >= 0:
                    lines[last_code_line_idx] = lines[last_code_line_idx] + close_bracket
                    content = '\n'.join(lines)
                else:
                    content += close_bracket

        # Fix missing opening brackets
        elif close_count > open_count:
            # This is trickier - we'll try to find obvious cases
            # like a closing bracket at the beginning of the file
            if content.strip().startswith(close_bracket):
                content = open_bracket + content
            else:
                # For other cases, we'll insert at the beginning of the file with a comment
                content = f"{open_bracket} # Added by fix_brackets.py\n{content}"

    return content


def fix_unmatched_brackets_advanced(content: str) -> str:
    """Fix unmatched brackets using a more advanced stack-based approach."""
    lines = content.splitlines()
    result_lines = lines.copy()

    # First pass: analyze brackets on each line
    for i, line in enumerate(lines):
        stack = []
        for char in line:
            if char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack:
                    # Found closing without opening
                    if char == ')':
                        result_lines[i] = '(' + result_lines[i]
                    elif char == ']':
                        result_lines[i] = '[' + result_lines[i]
                    elif char == '}':
                        result_lines[i] = '{' + result_lines[i]
                else:
                    opening = stack.pop()
                    # Check if closing matches opening
                    if (opening == '(' and char != ')') or \
                       (opening == '[' and char != ']') or \
                       (opening == '{' and char != '}'):
                        # Mismatch, add appropriate closing
                        if opening == '(':
                            result_lines[i] = result_lines[i] + ')'
                        elif opening == '[':
                            result_lines[i] = result_lines[i] + ']'
                        elif opening == '{':
                            result_lines[i] = result_lines[i] + '}'

                        # Put back the current char to re-process
                        stack.append(char)

        # Add any missing closing brackets at the end of the line
        while stack:
            opening = stack.pop()
            if opening == '(':
                result_lines[i] = result_lines[i] + ')'
            elif opening == '[':
                result_lines[i] = result_lines[i] + ']'
            elif opening == '{':
                result_lines[i] = result_lines[i] + '}'

    return '\n'.join(result_lines)


def fix_bracket_issues_in_file(file_path: str) -> bool:
    """Process a file to fix unmatched brackets."""
    try:
        # Check if the file has syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        if is_valid:
            return False  # No changes needed

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try both bracket fixing approaches
        fixed_content = fix_unmatched_brackets(content)

        # If first approach didn't change anything, try advanced method
        if fixed_content == content:
            fixed_content = fix_unmatched_brackets_advanced(content)

        # If file was modified, write it back
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            # Check if the fix resolved the syntax error
            is_valid_now, _ = check_for_syntax(file_path)
            if is_valid_now:
                return True  # Fixed successfully
            else:
                print_warning(f"  Bracket fix didn't resolve all syntax errors in {file_path}")
                return True  # Still made changes

        return False  # No changes needed

    except Exception as e:
        print_error(f"  Error processing {file_path}: {str(e)}")
        return False


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print_error("Usage: python fix_brackets.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    print_header(f"Starting bracket fix on {directory}")

    python_files = get_python_files(directory)
    print_info(f"Found {len(python_files)} Python files")

    files_fixed = 0

    for file_path in python_files:
        # Check for syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)

        # Look for unmatched bracket errors
        if not is_valid and any(word in error_msg.lower() for word in
                               ['bracket', 'parenthes', 'brace', 'unexpected', 'unmatched']):
            print_info(f"Attempting to fix {file_path}")
            if fix_bracket_issues_in_file(file_path):
                print_success(f"✓ Fixed: {file_path}")
                files_fixed += 1

    print_header("\nBracket Fix Summary")
    print_info(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main()
