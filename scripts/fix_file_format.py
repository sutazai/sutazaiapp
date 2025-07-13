#!/usr/bin/env python3.11
"""
Fix File Format Script

This script scans and fixes formatting issues in Python files, focusing on:
    1. Ensuring __future__ imports come at the beginning of the file (after docstrings)
2. Fixing spacing between imports and docstrings
3. Ensuring proper indentation
4. Fixing syntax errors like unmatched brackets and unterminated triple-quoted strings
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ANSI color codes for output

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
    print(f"{Colors.HEADER}{message}{Colors.ENDC}")

def print_success(message: str) -> None:
    """Print a success message."""
    print(f"{Colors.OKGREEN}{message}{Colors.ENDC}")

def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"{Colors.WARNING}{message}{Colors.ENDC}")

def print_error(message: str) -> None:
    """Print an error message."""
    print(f"{Colors.FAIL}{message}{Colors.ENDC}")

def get_all_python_files(directory: str, exclude_dirs: List[str] = []) -> List[str]:
    """Get a list of all Python files in the given directory, recursively."""
    if not exclude_dirs:
        exclude_dirs = ["venv", ".git", "__pycache__"]

    python_files = []

    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files

def check_file_for_syntax(file_path: str) -> Tuple[bool, str | None]:
    """Check if a file has syntax errors."""
    import py_compile
    import tempfile
    from pathlib import Path

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
        temp_path = temp.name

    try:
        # Try to compile the file
        py_compile.compile(file_path, cfile=temp_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        # Return the error message
        return False, str(e)
    except Exception as e:
        return False, str(e)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def fix_unmatched_brackets(content: str) -> str:
    """Fix unmatched brackets in Python code."""
    bracket_pairs = {')': '(', ']': '[', '}': '{'}

    stack = []
    problem_indices = []

    for i, char in enumerate(content):
        if char in bracket_pairs.values():
            stack.append((char, i))
        elif char in bracket_pairs:
            if not stack or stack[-1][0] != bracket_pairs[char]:
                problem_indices.append(i)
            elif stack:
                stack.pop()

    # Add remaining unmatched opening brackets
    for bracket, idx in stack:
        # Find matching closing bracket
        matching_char = next(key for key, value in bracket_pairs.items() if value == bracket)

        # Find appropriate position to insert
        lines = content.split('\n')
        pos = idx
        lineno = content[:pos].count('\n')

        if lineno < len(lines) - 1:
            # Add at the end of the current line
            line_end = pos + (len(lines[lineno]) - (pos - content[:pos].rfind('\n') - 1))
            content = content[:line_end] + matching_char + content[line_end:]
        else:
            # Add at the end of the file
            content += matching_char

    return content

def fix_triple_quotes(content: str) -> str:
    """Fix unterminated triple-quoted strings."""
    # Find all triple quote starts
    triple_single = re.finditer(r"'''(?!\s*''')", content)
    triple_double = re.finditer(r'"""(?!\s*""")', content)

    # For each opening triple quote without a close, add closing quotes
    for matches, quote_type in [(triple_single, "'''"), (triple_double, '"""')]:
        for match in matches:
            start_pos = match.start()

            # Check if there's a matching closing triple quote
            rest_of_content = content[start_pos + 3:]
            if quote_type not in rest_of_content:
                # No closing quote, add one at a reasonable position
                # Find end of line or file
                next_line_pos = content.find('\n', start_pos)
                if next_line_pos == -1:
                    # End of file
                    content = content + quote_type
                else:
                    # End of line
                    content = content[:next_line_pos] + quote_type + content[next_line_pos:]

    return content

def fix_indentation(content: str) -> str:
    """Fix indentation issues in Python code."""
    lines = content.split('\n')
    fixed_lines = []

    current_indent = 0
    for line in lines:
        stripped = line.lstrip()

        # Handle empty lines or comment-only lines
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line)
            continue

        # Handle lines that increase indentation
        if stripped.endswith(':'):
            fixed_lines.append(' ' * (4 * current_indent) + stripped)
            current_indent += 1
            continue

        # Handle lines that potentially decrease indentation
        if any(stripped.startswith(kw) for kw in ['return', 'break', 'continue', 'pass', 'raise']):
            current_indent = max(0, current_indent - 1)

        # Special case for closing brackets at the start of a line
        if stripped[0] in [')', ']', '}']:
            current_indent = max(0, current_indent - 1)

        fixed_lines.append(' ' * (4 * current_indent) + stripped)

    return '\n'.join(fixed_lines)

def fix_file_format(file_path):
    """Fix formatting issues in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            print(f"  - Warning: Could not read {file_path} (likely binary file)")
            return False, "Binary file"

    # Skip empty files
    if not content.strip():
        return False, "Empty file"

    original_content = content
    was_modified = False
    fix_reason = []

    # Fix common issues:

    # Check for syntax errors first
    is_valid, error_msg = check_file_for_syntax(file_path)
    if not is_valid:
        print_warning(f"  - Syntax error in {file_path}: {error_msg}")

        # Try to fix common syntax issues
        content = fix_unmatched_brackets(content)
        content = fix_triple_quotes(content)
        content = fix_indentation(content)

        # Check if fixes resolved the issues
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        is_valid_after, error_msg_after = check_file_for_syntax(file_path)
        if is_valid_after:
            print_success(f"  - Fixed syntax errors in {file_path}")
            was_modified = True
            fix_reason.append("Fixed syntax errors")
        else:
            print_error(f"  - Could not fix all syntax errors in {file_path}: {error_msg_after}")

        # Reread the content after potential syntax fixes
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    # Continue with existing formatting fixes

    # 1. Fix files where docstring and __future__ imports are adjacent without spacing
    pattern = r'(""".*?""")(\s*from\s+__future__\s+import)'
    replacement = r'\1\n\n\2'
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        content = new_content
        was_modified = True
        fix_reason.append("Fixed spacing between docstring and __future__ imports")

    # 2. Fix files where __future__ import comes before docstring
    future_match = re.search(r'from\s+__future__\s+import\s+\w+', content)
    docstring_match = re.search(r'""".*?"""', content, re.DOTALL)

    if future_match and docstring_match:
        future_pos = future_match.start()
        docstring_pos = docstring_match.start()

        if 0 < future_pos < docstring_pos:
            # __future__ import before docstring - swap them
            future_import = future_match.group(0)
            docstring = docstring_match.group(0)

            # Remove both
            content = content.replace(future_import, '', 1)
            content = content.replace(docstring, '', 1)

            # Add shebang line if it exists
            shebang = ""
            if content.startswith('#!'):
                shebang_end = content.find('\n')
                shebang = content[:shebang_end+1]
                content = content[shebang_end+1:]

            # Add them back in correct order
            content = shebang + docstring + '\n\n' + future_import + '\n\n' + content.lstrip()
            was_modified = True
            fix_reason.append("Reordered docstring and __future__ imports")

    # 3. Fix badly formatted auto_gpt/__init__.py file with no spacing
    if '/auto_gpt/src/__init__.py' in str(file_path):
        # Try to fix the specific syntax problem in this file
        shebang_match = re.match(r'(#!/usr/bin/env python3\.11)(.*?)"""(.*?)"""', content, re.DOTALL)
        if shebang_match:
            shebang = shebang_match.group(1)
            docstring_content = shebang_match.group(3)

            # Extract the imports and code sections
            rest_match = re.search(r'""".*?"""(.*)$', content, re.DOTALL)
            if rest_match:
                rest_content = rest_match.group(1)

                # Put it all together with proper spacing
                content = f"{shebang}\n\n\"\"\"{docstring_content}\"\"\"\n\n{rest_content.lstrip()}"
                was_modified = True
                fix_reason.append("Fixed auto_gpt/__init__.py formatting")

    # Apply the changes if any
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Return True if file was modified along with concatenated reasons
        if was_modified:
            return True, "; ".join(fix_reason)

    return False, "No issues found"

def scan_and_fix_directory(directory):
    """Scan a directory recursively and fix formatting issues in all Python files."""
    fixed_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                was_fixed, reason = fix_file_format(file_path)
                if was_fixed:
                    fixed_files.append((file_path, reason))

    return fixed_files

def main():
    """Main function."""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print_header(f"Scanning {project_root} for Python files with formatting and syntax issues...")

    directories_to_scan = [
        project_root / "ai_agents",
        project_root / "backend",
        project_root / "model_management",
        project_root / "scripts",
    ]

    total_fixed = 0
    syntax_errors_fixed = 0
    for directory in directories_to_scan:
        if directory.exists():
            print(f"\nScanning {directory}...")
            fixed_files = scan_and_fix_directory(directory)
            if fixed_files:
                print_success(f"Fixed issues in {len(fixed_files)} files in {directory}:")
                for file_path, reason in fixed_files:
                    if "syntax" in reason.lower():
                        print_success(f"  - {file_path} ({reason})")
                        syntax_errors_fixed += 1
                    else:
                        print(f"  - {file_path} ({reason})")
                total_fixed += len(fixed_files)

    if total_fixed > 0:
        print_header("\nSummary:")
        print_success(f"Successfully fixed {total_fixed} files in total.")
        if syntax_errors_fixed > 0:
            print_success(f"Fixed syntax errors in {syntax_errors_fixed} files.")
    else:
        print_success("\nNo files needed fixing. All files are correctly formatted and have no syntax errors.")

if __name__ == "__main__":
    main()
