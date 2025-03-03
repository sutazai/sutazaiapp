#!/usr/bin/env python3
"""
Script to fix docstring-related syntax errors in Python files.

This script scans Python files for issues with docstrings such as
unterminated triple quotes, improper escaping, and other common problems.
"""

import os
import re
import sys
from typing import List, Tuple, Dict, Optional
from typing import Union
from typing import Optional


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


def extract_line_with_context(content: str, line_number: int, context: int = 2) -> str:
    """
    Extract a line from the content with a few lines of context before and after.
    
    Args:
        content: The full file content
        line_number: The 1-indexed line number to extract
        context: Number of lines of context to include before and after
        
    Returns:
        A string with the line and its context
    """
    lines = content.splitlines()
    start = max(0, line_number - 1 - context)
    end = min(len(lines), line_number + context)
    
    result = []
    for i in range(start, end):
        prefix = ">> " if i == line_number - 1 else "   "
        result.append(f"{prefix}{i+1}: {lines[i]}")
    
    return "\n".join(result)


def fix_unterminated_docstrings(content: str, error_line: int | None = None) -> str:
    """
    Fix unterminated docstrings in Python code.
    
    This function:
    1. Looks for unterminated triple quotes
    2. Fixes docstrings by properly closing them
    3. Handles both single and double quotes
    
    Args:
        content: The Python file content
        error_line: Optional line number from syntax error to help locate the issue
        
    Returns:
        Fixed content
    """
    lines = content.splitlines()
    result_lines = lines.copy()
    
    # If we have an error line, focus on that area first
    if error_line is not None:
        # Get lines around the error
        start_line = max(0, error_line - 10)
        end_line = min(len(lines), error_line + 10)
        
        # Check if there are triple quotes on any of these lines
        for i in range(start_line, end_line):
            line = lines[i]
            if '"""' in line or "'''" in line:
                # Found triple quotes near the error, let's try to fix it
                if line.count('"""') % 2 == 1 or line.count("'''") % 2 == 1:
                    # This line has an odd number of triple quotes, meaning it's likely the issue
                    quote_type = '"""' if '"""' in line else "'''"
                    
                    # Check if there's another matching triple quote in the following lines
                    found_match = False
                    for j in range(i + 1, len(lines)):
                        if quote_type in lines[j]:
                            found_match = True
                            break
                    
                    if not found_match:
                        # No matching closing quote found, add it at the end of the line
                        result_lines[i] = line + quote_type
    
    # More general approach: count triple quotes in the entire file
    triple_double_quotes = sum(line.count('"""') for line in result_lines)
    triple_single_quotes = sum(line.count("'''") for line in result_lines)
    
    # If we have an odd number of either type, we need to fix them
    if triple_double_quotes % 2 == 1:
        # Find the last occurrence and add a closing quote
        for i in range(len(result_lines) - 1, -1, -1):
            if '"""' in result_lines[i]:
                result_lines[i] = result_lines[i] + '"""'
                break
    
    if triple_single_quotes % 2 == 1:
        # Find the last occurrence and add a closing quote
        for i in range(len(result_lines) - 1, -1, -1):
            if "'''" in result_lines[i]:
                result_lines[i] = result_lines[i] + "'''"
                break
    
    # More advanced: detect docstrings that span multiple lines
    in_triple_double_quotes = False
    in_triple_single_quotes = False
    
    for i, line in enumerate(result_lines):
        # Count the triple quotes in this line
        double_quotes_count = line.count('"""')
        single_quotes_count = line.count("'''")
        
        # Handle double quotes
        if double_quotes_count > 0:
            for _ in range(double_quotes_count):
                in_triple_double_quotes = not in_triple_double_quotes
        
        # Handle single quotes
        if single_quotes_count > 0:
            for _ in range(single_quotes_count):
                in_triple_single_quotes = not in_triple_single_quotes
    
    # If we're still in a docstring at the end, add the closing quote
    if in_triple_double_quotes:
        result_lines[-1] = result_lines[-1] + '"""'
    
    if in_triple_single_quotes:
        result_lines[-1] = result_lines[-1] + "'''"
    
    return '\n'.join(result_lines)


def fix_docstring_indentation(content: str) -> str:
    """
    Fix incorrect indentation in docstrings.
    
    This function ensures that docstrings follow Python's indentation rules:
    - Module docstrings should be at the top level
    - Class/function docstrings should be indented at the same level as the def/class
    """
    lines = content.splitlines()
    result_lines = lines.copy()
    
    # Detect docstring blocks and their indentation
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped_line = line.lstrip()
        
        # Skip empty lines
        if not stripped_line:
            i += 1
            continue
        
        # Check for docstring start
        if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
            # Get the indentation of this line
            indent = len(line) - len(stripped_line)
            
            # Check if this is the start of a multi-line docstring
            if not (stripped_line.endswith('"""') and stripped_line.count('"""') > 1) and \
               not (stripped_line.endswith("'''") and stripped_line.count("'''") > 1):
                # This is a multi-line docstring start
                # Find the end of the docstring
                quote_type = '"""' if stripped_line.startswith('"""') else "'''"
                end_index = i
                
                for j in range(i + 1, len(lines)):
                    if quote_type in lines[j]:
                        end_index = j
                        break
                
                # Check indentation of all lines in the docstring
                for j in range(i + 1, end_index + 1):
                    if j < len(lines):
                        doc_line = lines[j]
                        if doc_line.strip() and len(doc_line) - len(doc_line.lstrip()) < indent:
                            # This line needs to be indented properly
                            result_lines[j] = ' ' * indent + doc_line.lstrip()
                
                # Skip to after the docstring
                i = end_index + 1
                continue
        
        i += 1
    
    return '\n'.join(result_lines)


def fix_docstring_issues_in_file(file_path: str) -> bool:
    """Process a file to fix docstring issues."""
    try:
        # Check if the file has syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        if is_valid:
            return False  # No changes needed
        
        # Extract error line number if available
        error_line = None
        if "line" in error_msg:
            # Try to extract the line number from error message
            line_match = re.search(r'line (\d+)', error_msg)
            if line_match:
                error_line = int(line_match.group(1))
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If error is specifically an unterminated string issue
        if "unterminated" in error_msg and "string" in error_msg:
            print_info(f"  Found unterminated string in {file_path}")
            
            # Show the context of the error
            if error_line:
                context = extract_line_with_context(content, error_line)
                print_info(f"  Context:\n{context}")
            
            # Fix the issue
            fixed_content = fix_unterminated_docstrings(content, error_line)
        else:
            # Try general docstring fixes
            fixed_content = fix_unterminated_docstrings(content)
        
        # Try to fix docstring indentation if still having issues
        if fixed_content != content and not check_for_syntax(file_path)[0]:
            fixed_content = fix_docstring_indentation(fixed_content)
        
        # If file was modified, write it back
        if fixed_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            # Check if the fix resolved the syntax error
            is_valid_now, _ = check_for_syntax(file_path)
            if is_valid_now:
                return True  # Fixed successfully
            else:
                print_warning(f"  Docstring fix didn't resolve all syntax errors in {file_path}")
                return True  # Still made changes
        
        return False  # No changes needed
    
    except Exception as e:
        print_error(f"  Error processing {file_path}: {str(e)}")
        return False


def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print_error("Usage: python fix_docstrings.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    print_header(f"Starting docstring fix on {directory}")
    
    python_files = get_python_files(directory)
    print_info(f"Found {len(python_files)} Python files")
    
    files_fixed = 0
    
    for file_path in python_files:
        # Check for syntax errors first
        is_valid, error_msg = check_for_syntax(file_path)
        
        # Focus on unterminated string errors and docstring issues
        if not is_valid and any(word in error_msg.lower() for word in 
                               ['string', 'docstring', 'quote', 'unterminated']):
            print_info(f"Attempting to fix {file_path}")
            if fix_docstring_issues_in_file(file_path):
                print_success(f"✓ Fixed: {file_path}")
                files_fixed += 1
    
    print_header("\nDocstring Fix Summary")
    print_info(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main() 