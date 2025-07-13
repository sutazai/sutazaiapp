#!/usr/bin/env python3
"""
Script to fix unterminated triple quotes in Python files.

This script scans Python files for unterminated triple quotes
and fixes them by properly closing the quotes.
"""

import os
import re
import sys
from typing import List, Tuple


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


def fix_unterminated_triple_quotes(content: str) -> str:
    """Fix unterminated triple-quoted strings."""
    # Find triple quote positions
    triple_quote_positions = []

    # Find all triple quotes (both """ and ''')
    for triple_quote in ['"""', "'''"]:
        positions = [match.start() for match in re.finditer(re.escape(triple_quote), content)]
        for pos in positions:
            triple_quote_positions.append((pos, triple_quote))

    # Sort by position
    triple_quote_positions.sort(key=lambda x: x[0])

    # If we have an odd number of triple quotes, we have an unterminated string
    if len(triple_quote_positions) % 2 == 1:
        # Get the last triple quote type
        last_pos, last_quote = triple_quote_positions[-1]

        # Add the missing closing quote at the end of the line or before the next code
        next_line_pos = content.find('\n', last_pos)
        if next_line_pos != -1:
            # Add quote at the end of the line
            return content[:next_line_pos] + last_quote + content[next_line_pos:]
        else:
            # No newline found, add at the end of the file
            return content + last_quote

    # Process each pair of triple quotes to find unterminated strings
    current_quote = None
    for i, (pos, quote_type) in enumerate(triple_quote_positions):
        if i % 2 == 0:  # Opening quote
            current_quote = quote_type
        elif current_quote is not None:  # Closing quote and we have an opening quote
            # Check if closing quote matches opening quote
            if quote_type != current_quote:
                # Mismatched quotes, insert the correct one
                return content[:pos] + current_quote + content[pos:]

    return content


def process_file(file_path: str) -> bool:
    """Process a single file to fix unterminated triple quotes."""
    try:
        # Check if the file has syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        if is_valid:
            return False  # No changes needed

        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix unterminated triple quotes
        fixed_content = fix_unterminated_triple_quotes(content)

        # If file was modified, write it back
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            # Check if the fix resolved the syntax error
            is_valid_now, _ = check_for_syntax(file_path)
            if is_valid_now:
                return True  # Fixed successfully
            else:
                print_warning(f"  Triple quote fix didn't resolve all syntax errors in {file_path}")
                return True  # Still made changes

        return False  # No changes needed

    except Exception as e:
        print_error(f"  Error processing {file_path}: {str(e)}")
        return False


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print_error("Usage: python fix_triple_quotes.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    print_header(f"Starting triple quote fix on {directory}")

    python_files = get_python_files(directory)
    print_info(f"Found {len(python_files)} Python files")

    files_fixed = 0

    for file_path in python_files:
        # Check for syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        if not is_valid and 'unterminated' in error_msg and 'string' in error_msg:
            print_info(f"Attempting to fix {file_path}")
            if process_file(file_path):
                print_success(f"✓ Fixed: {file_path}")
                files_fixed += 1

    print_header("\nTriple Quote Fix Summary")
    print_info(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main()
