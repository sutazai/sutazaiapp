#!/usr/bin/env python3.11
"""Comprehensive Syntax Fixer for SutazAI Project"""

import ast
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

class SyntaxFixer:
    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """Initialize the syntax fixer."""
        self.base_path = base_path
        self.ignored_dirs = {
            ".git", ".venv", "venv", "__pycache__",
            "node_modules", "build", "dist"
        }

    def fix_syntax_errors(self, file_path: str) -> Optional[str]:
        """Fix syntax errors in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            # Apply fixes in sequence
            fixed_source = source

            # Fix unterminated triple quotes
            fixed_source = self._fix_unterminated_quotes(fixed_source)

            # Fix indentation
            fixed_source = self._fix_indentation(fixed_source)

            # Fix missing colons
            fixed_source = self._fix_missing_colons(fixed_source)

            # Fix unmatched brackets/parentheses/braces
            fixed_source = self._fix_unmatched_brackets(fixed_source)

            # Fix function definitions
            fixed_source = self._fix_function_definitions(fixed_source)

            return fixed_source

        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")
            return None

    def _fix_unterminated_quotes(self, source: str) -> str:
        """Fix unterminated triple quotes."""
        lines = source.split('\n')
        fixed_lines = []
        in_docstring = False
        docstring_start = None
        
        for i, line in enumerate(lines):
            if '"""' in line:
                count = line.count('"""')
                if count % 2 != 0:
                    if not in_docstring:
                        in_docstring = True
                        docstring_start = i
                    else:
                        in_docstring = False
                        
            fixed_lines.append(line)
            
        if in_docstring:
            # Add missing closing quotes
            if docstring_start is not None:
                fixed_lines[docstring_start] = fixed_lines[docstring_start].rstrip() + '"""'
            
        return '\n'.join(fixed_lines)

    def _fix_indentation(self, source: str) -> str:
        """Fix indentation issues."""
        lines = source.split('\n')
        fixed_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.lstrip()
            if not stripped:  # Empty line
                fixed_lines.append('')
                continue
                
            # Adjust indent level based on line content
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'else:', 'elif ', 'except', 'finally:')):
                indent = ' ' * (4 * indent_level)
                fixed_lines.append(indent + stripped)
                if stripped.endswith(':'):
                    indent_level += 1
            else:
                indent = ' ' * (4 * indent_level)
                fixed_lines.append(indent + stripped)
                
                # Check for block enders
                if stripped in ['return', 'break', 'continue', 'pass'] or stripped.startswith('return '):
                    indent_level = max(0, indent_level - 1)
                    
        return '\n'.join(fixed_lines)

    def _fix_missing_colons(self, source: str) -> str:
        """Fix missing colons in function/class definitions and control structures."""
        lines = source.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(stripped.startswith(x) for x in ['def ', 'class ', 'if ', 'else', 'elif ', 'try', 'except', 'finally']):
                if not stripped.endswith(':'):
                    lines[i] = line.rstrip() + ':'
        return '\n'.join(lines)

    def _fix_unmatched_brackets(self, source: str) -> str:
        """Fix unmatched brackets, parentheses, and braces."""
        pairs = {')': '(', ']': '[', '}': '{'}
        lines = source.split('\n')
        stack = []
        
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char in '([{':
                    stack.append((char, i, j))
                elif char in ')]}':
                    if not stack:
                        # Missing opening bracket
                        lines[i] = line[:j] + pairs[char] + line[j:]
                    else:
                        opening = stack.pop()
                        if opening[0] != pairs[char]:
                            # Mismatched brackets
                            lines[i] = line[:j] + pairs[char] + line[j+1:]
                            
        # Add missing closing brackets
        while stack:
            char, i, j = stack.pop()
            closing = {'(': ')', '[': ']', '{': '}'}[char]
            lines[i] = lines[i] + closing
            
        return '\n'.join(lines)

    def _fix_function_definitions(self, source: str) -> str:
        """Fix common function definition issues."""
        lines = source.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('def '):
                # Ensure proper function definition format
                if '(' not in stripped:
                    name = stripped[4:].strip()
                    lines[i] = line.replace(name, f"{name}()")
                elif ')' not in stripped:
                    lines[i] = line + ')'
                if not stripped.endswith(':'):
                    lines[i] = lines[i] + ':'
                    
        return '\n'.join(lines)

    def fix_project_syntax(self) -> None:
        """Fix syntax errors in all Python files in the project."""
        for root, dirs, files in os.walk(self.base_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    logger.info(f"Processing {file_path}")
                    
                    fixed_source = self.fix_syntax_errors(file_path)
                    if fixed_source is not None:
                        try:
                            # Try to parse the fixed source to validate it
                            ast.parse(fixed_source)
                            
                            # Write back the fixed source
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(fixed_source)
                            logger.info(f"Fixed syntax in {file_path}")
                        except SyntaxError as e:
                            logger.error(f"Failed to fix {file_path}: {e}")

def main():
    """Main entry point."""
    fixer = SyntaxFixer()
    fixer.fix_project_syntax()

if __name__ == "__main__":
    main()

