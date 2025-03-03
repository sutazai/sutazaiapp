#!/usr/bin/env python3
"""
Advanced Syntax Repair Script

This script detects and repairs common syntax errors in Python files:
1. Unterminated triple-quoted strings
2. Unmatched parentheses, brackets, and braces
3. Indentation errors
4. Invalid docstrings
5. Other common syntax errors
"""

import os
import re
import sys
import ast
import tokenize
from io import StringIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# ANSI color codes for console output
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

def get_python_files(directory: str, exclude_dirs: Optional[List[str]] = None) -> List[str]:
    """Get all Python files in the given directory tree."""
    if exclude_dirs is None:
        exclude_dirs = ["venv", ".git", "__pycache__", "node_modules", ".pytest_cache"]
    
    python_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def check_syntax(file_path: str) -> Tuple[bool, Optional[str]]:
    """Check if a file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def fix_unterminated_triple_quotes(content: str) -> str:
    """Fix unterminated triple-quoted strings."""
    # Regular expressions to detect triple quotes that aren't terminated
    triple_single_regex = re.compile(r"(''')(?!.*?''')", re.DOTALL)
    triple_double_regex = re.compile(r'(""")(?!.*?""")', re.DOTALL)
    
    # Find and fix unterminated triple single quotes
    match = triple_single_regex.search(content)
    if match:
        content += "'''"
    
    # Find and fix unterminated triple double quotes
    match = triple_double_regex.search(content)
    if match:
        content += '"""'
    
    # Fix multi-line docstrings with excessive quotes
    content = re.sub(r'"{9,}', '"""', content)
    content = re.sub(r"'{9,}", "'''", content)
    
    return content

def fix_unmatched_brackets(content: str) -> str:
    """Fix unmatched brackets in the content."""
    # Enhanced version with better balance tracking
    brackets = {'(': ')', '[': ']', '{': '}'}
    reverse_brackets = {')': '(', ']': '[', '}': '{'}
    
    # Count opening and closing brackets
    stack = []
    positions = []  # Store positions for better fixing
    
    for i, char in enumerate(content):
        if char in brackets:
            stack.append(char)
            positions.append(i)
        elif char in reverse_brackets:
            if stack and stack[-1] == reverse_brackets[char]:
                stack.pop()
                positions.pop()
            else:
                # Found closing bracket without matching opening
                # Insert matching opening before this position
                opening = reverse_brackets[char]
                # Find appropriate position (start of line or after whitespace)
                line_start = content.rfind('\n', 0, i) + 1
                content = content[:line_start] + opening + " # AUTO-ADDED\n" + content[line_start:]
                # Adjust for the inserted content
                i += len(opening) + len(" # AUTO-ADDED\n")
    
    # Add missing closing brackets at the end
    if stack:
        # Add comment to indicate auto-fixing
        content += "\n# AUTO-ADDED closing brackets\n"
        for bracket in reversed(stack):
            content += brackets[bracket] + " # AUTO-ADDED\n"
    
    return content

def fix_indentation(content: str) -> str:
    """Fix common indentation errors."""
    lines = content.split('\n')
    fixed_lines = []
    
    # Keep track of expected indentation level
    indent_level = 0
    indent_size = 4  # Assuming 4 spaces per level
    
    # Keywords that increase indentation
    indent_keywords = ['if', 'for', 'while', 'def', 'class', 'with', 'try', 'except', 'finally', 'elif', 'else']
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Skip empty lines or comments
        if not stripped or stripped.startswith('#'):
            fixed_lines.append(line)
            continue
        
        # Check for keywords that should increase indentation for next line
        if any(stripped.startswith(keyword + ' ') or stripped == keyword for keyword in indent_keywords) and stripped.endswith(':'):
            fixed_lines.append(line)
            indent_level += 1
            continue
        
        # Check for dedent keywords
        if stripped.startswith(('else:', 'elif ', 'except:', 'except ', 'finally:')):
            indent_level = max(0, indent_level - 1)  # Dedent but never below 0
            # Add proper indentation
            spaces = ' ' * (indent_level * indent_size)
            fixed_lines.append(spaces + stripped)
            indent_level += 1  # These keywords also indent the next line
            continue
        
        # Regular lines should match expected indentation
        current_indent = len(line) - len(line.lstrip())
        expected_indent = indent_level * indent_size
        
        # Only fix if significantly off
        if abs(current_indent - expected_indent) > indent_size:
            spaces = ' ' * expected_indent
            fixed_lines.append(spaces + stripped)
        else:
            # Leave alone if close enough
            fixed_lines.append(line)
        
        # Check for line endings that might reduce indentation
        if stripped.endswith(('break', 'continue', 'return', 'pass', 'raise')):
            indent_level = max(0, indent_level - 1)
    
    return '\n'.join(fixed_lines)

def fix_invalid_docstrings(content: str) -> str:
    """Fix invalid docstrings, including ones with syntax errors."""
    # More advanced docstring handling
    
    # Fix docstrings with unmatched quotes inside them
    pattern = r'"""(.*?)(?:"""|\Z)'
    
    def replace_with_docstring(match):
        # Get docstring content
        docstring = match.group(1)
        
        # If it doesn't end with """, we need to find where it should end
        if not match.group(0).endswith('"""'):
            # Find first non-docstring-like line
            lines = docstring.split('\n')
            valid_lines = []
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Very simple heuristic: if line looks like code (has certain patterns)
                if (i > 0 and re.match(r'^\s*(def|class|if|for|while|try|except|finally|with|return)\b', stripped)
                    or stripped.startswith('import ') 
                    or stripped.startswith('from ')):
                    # This is likely code, not part of the docstring
                    break
                valid_lines.append(line)
            
            # Reconstruct valid part of the docstring
            docstring = '\n'.join(valid_lines)
            return f'"""{docstring}"""'
        
        return match.group(0)
    
    fixed_content = re.sub(pattern, replace_with_docstring, content, flags=re.DOTALL)
    
    # Fix docstrings inside functions that might be missing closing quotes
    func_pattern = r'(def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?\s*:)\s*(?:"""(.*?)(?:"""|$))'
    
    def fix_func_docstring(match):
        func_def = match.group(1)
        docstring = match.group(2) if match.group(2) else ""
        
        # Check if docstring is properly closed
        if not match.group(0).endswith('"""'):
            # Find where to end the docstring - first line that looks like code
            lines = docstring.split('\n')
            valid_lines = []
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Heuristic for code detection
                if (i > 0 and re.match(r'^\s*(def|class|if|for|while|try|except|finally|with|return)\b', stripped)
                    or stripped.startswith('import ') 
                    or stripped.startswith('from ')):
                    break
                valid_lines.append(line)
            
            docstring = '\n'.join(valid_lines)
            return f'{func_def}\n    """{docstring}"""'
        
        return match.group(0)
    
    fixed_content = re.sub(func_pattern, fix_func_docstring, fixed_content, flags=re.DOTALL)
    
    return fixed_content

def fix_file(file_path: str) -> Tuple[bool, List[str]]:
    """Fix syntax errors in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixed_items = []
        
        # First pass - basic fixes
        content = fix_unterminated_triple_quotes(content)
        if content != original_content:
            fixed_items.append("Fixed unterminated triple quotes")
        
        # Check if first pass fixed the issues
        is_valid, _ = check_syntax(content)
        if is_valid:
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixed_items
            return True, []
        
        # Second pass - fix brackets
        temp_content = content
        content = fix_unmatched_brackets(content)
        if content != temp_content:
            fixed_items.append("Fixed unmatched brackets")
        
        # Check again
        is_valid, _ = check_syntax(content)
        if is_valid:
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixed_items
        
        # Third pass - fix indentation
        temp_content = content
        content = fix_indentation(content)
        if content != temp_content:
            fixed_items.append("Fixed indentation issues")
        
        # Check again
        is_valid, _ = check_syntax(content)
        if is_valid:
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixed_items
        
        # Fourth pass - fix comment blocks
        temp_content = content
        content = fix_comment_blocks(content)
        if content != temp_content:
            fixed_items.append("Fixed comment blocks")
        
        # Check again
        is_valid, _ = check_syntax(content)
        if is_valid:
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True, fixed_items
        
        # Fifth pass - fix docstrings
        temp_content = content
        content = fix_invalid_docstrings(content)
        if content != temp_content:
            fixed_items.append("Fixed invalid docstrings")
        
        # Final check
        is_valid, error_msg = check_syntax(content)
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if is_valid:
                return True, fixed_items
            return False, fixed_items
        
        return is_valid, []
        
    except Exception as e:
        print_error(f"Error processing {file_path}: {str(e)}")
        return False, []

def fix_comment_blocks(content: str) -> str:
    """Fix incomplete comment blocks or malformed comments."""
    lines = content.split('\n')
    in_comment_block = False
    comment_indent = 0
    
    for i in range(len(lines)):
        stripped = lines[i].strip()
        
        # Handle block comments that might be causing issues
        if stripped.startswith('#'):
            indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Start of a comment block
            if not in_comment_block and i > 0 and not lines[i-1].strip().startswith('#'):
                in_comment_block = True
                comment_indent = indent
            
            # Continue comment block
            if in_comment_block:
                # Fix inconsistent indentation in comment blocks
                if indent != comment_indent and stripped != '#':
                    lines[i] = ' ' * comment_indent + stripped
        else:
            in_comment_block = False
    
    return '\n'.join(lines)

def main() -> None:
    """Main function."""
    if len(sys.argv) < 2:
        print_error("Usage: python advanced_syntax_repair.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    print_header(f"Starting advanced syntax repair on {directory}")
    
    python_files = get_python_files(directory)
    print_info(f"Found {len(python_files)} Python files")
    
    completely_fixed = 0
    partially_fixed = 0
    unfixable = 0
    
    fixed_files = []
    
    for file_path in python_files:
        print_info(f"Attempting to fix {file_path}")
        is_valid, _ = check_syntax(file_path)
        
        if is_valid:
            # Skip already valid files
            continue
        
        is_fixed, fixes = fix_file(file_path)
        
        if is_fixed and fixes:
            if is_valid:
                print_success(f"✓ Fixed: {file_path}")
                fixed_files.append((file_path, fixes, True))
                completely_fixed += 1
            else:
                print_warning(f"⚠ Partially fixed: {file_path}")
                fixed_files.append((file_path, fixes, False))
                partially_fixed += 1
                # Print the fixes that were applied
                for fix in fixes:
                    print(f"    Applied: {fix}")
        else:
            print_error(f"✗ Couldn't fix: {file_path}")
            unfixable += 1
    
    print_header("\nSyntax Repair Summary")
    print_info(f"Completely fixed: {completely_fixed} files")
    print_info(f"Partially fixed: {partially_fixed} files")
    print_info(f"Unable to fix: {unfixable} files")
    
    if fixed_files:
        print_header("\nList of files fixed:")
        for file_path, fixes, is_complete in fixed_files:
            status = "✓" if is_complete else "⚠"
            print(f"  {status} {file_path}: {', '.join(fixes)}")

if __name__ == "__main__":
    main() 