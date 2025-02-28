#!/usr/bin/env python3.11
"""
SutazAI Code Cleanup Script

This script performs automated code cleanup tasks:
1. Fixes import statements
2. Fixes indentation
3. Fixes line length issues
4. Removes unused imports
"""

import ast
import os
import re
from typing import List, Set, Dict, Any

def fix_imports(source: str) -> str:
    """Fix import statements and add missing imports."""
    # Common type imports that are often missing
    type_imports = {
    'List': 'typing.List',
    'Dict': 'typing.Dict',
    'Tuple': 'typing.Tuple',
    'Optional': 'typing.Optional',
    'Any': 'typing.Any',
    'Set': 'typing.Set'
    }

    # Parse the AST to find undefined names
    try:
        tree = ast.parse(source)
        except SyntaxError:
        return source

        # Find all undefined names that match our type imports
        undefined_types = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in type_imports and not any(
                    line.strip().startswith(f'from typing import {node.id}')
                    for line in source.split('\n')
                        ):
                        undefined_types.add(node.id)

                        # Add missing imports at the top of the file
                        if undefined_types:
                            import_line = 'from typing import ' + ', '.join(
                                sorted(undefined_types)) + '\n'
                            lines = source.split('\n')
                            # Find the best place to insert the import
                            insert_pos = 0
                            for i, line in enumerate(lines):
                                if line.startswith(
                                    'import ') or line.startswith('from '):
                                    insert_pos = i + 1
                                    elif line and not line.startswith(
                                        '#') and not line.startswith('"""'):
                                break

                                lines.insert(insert_pos, import_line)
                            return '\n'.join(lines)

                        return source

                        def fix_indentation(source: str) -> str:
                            """Fix indentation issues in the code."""
                            lines = source.split('\n')
                            fixed_lines = []
                            indent_level = 0

                            for line in lines:
                                stripped = line.lstrip()
                                if not stripped:  # Empty line
                                    fixed_lines.append('')
                                continue

                                # Detect indentation level changes
                                if stripped.startswith(
                                    ('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except')):
                                    fixed_lines.append(
                                        ' ' * (4 * indent_level) + stripped)
                                    indent_level += 1
                                    elif stripped.startswith(
                                        ('return', 'break', 'continue', 'pass', 'raise')):
                                    indent_level = max(0, indent_level - 1)
                                    fixed_lines.append(
                                        ' ' * (4 * indent_level) + stripped)
                                    else:
                                    fixed_lines.append(
                                        ' ' * (4 * indent_level) + stripped)

                                return '\n'.join(fixed_lines)

                                def fix_line_length(
                                    source: str,
                                    max_length: int = 79) -> str:
                                    """Fix lines that exceed the maximum length."""
                                    lines = source.split('\n')
                                    fixed_lines = []

                                    for line in lines:
                                        if len(line) <= max_length:
                                            fixed_lines.append(line)
                                        continue

                                        # Don't break comment lines
                                        if line.strip().startswith('#'):
                                            fixed_lines.append(line)
                                        continue

                                        # Handle function calls and parameters
                                        if '(' in line and ')' in line:
                                            fixed = break_function_call(
                                                line,
                                                max_length)
                                            fixed_lines.extend(fixed)
                                            else:
                                            # For other long lines, try to break at operators
                                            fixed = break_at_operators(
                                                line,
                                                max_length)
                                            fixed_lines.extend(fixed)

                                        return '\n'.join(fixed_lines)

                                        def break_function_call(
                                            line: str,
                                            max_length: int) -> List[str]:
                                            """Break a long function call into multiple lines."""
                                            indent = len(
                                                line) - len(line.lstrip())
                                            base_indent = ' ' * indent
                                            extra_indent = ' ' * (indent + 4)

                                            # Extract function name and arguments
                                            match = re.match(
                                                r'^(\s*)(.*?\()(.*)\)(.*)$',
                                                line)
                                            if not match:
                                            return [line]

                                            prefix, func, args, suffix = match.groups()
                                            if not args:
                                            return [line]

                                            # Split arguments
                                            args_list = split_args(args)
                                            if not args_list:
                                            return [line]

                                            # Format the broken line
                                            result = [prefix + func]
                                            for i, arg in enumerate(args_list):
                                                if i < len(args_list) - 1:
                                                    result.append(
                                                        extra_indent + arg + ',
                                                        ')
                                                    else:
                                                    result.append(
                                                        extra_indent + arg + ')' + suffix)

                                                return result

                                                def break_at_operators(
                                                    line: str,
                                                    max_length: int) -> List[str]:
                                                    """Break a long line at operators."""
                                                                                                        operators = [' + \
                                                        ', ' - ', ' * ', ' / ', ' and ', ' or ', ' in ', ' not in ', ' is ', ' is not ']
                                                    indent = len(
                                                        line) - len(line.lstrip())
                                                    base_indent = ' ' * indent
                                                    extra_indent = ' ' * (
                                                        indent + 4)

                                                    if len(line) <= max_length:
                                                    return [line]

                                                    for op in operators:
                                                        if op in line:
                                                            parts = line.split(
                                                                op)
                                                            if len(parts) > 1:
                                                                                                                                result = [base_indent + \
                                                                    parts[0] + \
                                                                    op + \
                                                                    '\\']
                                                                                                                                for part in \
                                                                    parts[1:-1]:
                                                                    result.append(
                                                                        extra_indent + part + op + '\\')
                                                                    result.append(
                                                                        extra_indent + parts[-1])
                                                                return result

                                                            return [line]

                                                            def split_args(
                                                                args_str: str) -> List[str]:
                                                                """Split function arguments intelligently."""
                                                                args = []
                                                                current_arg = ''
                                                                paren_level = 0
                                                                bracket_level = 0

                                                                                                                                for char in \
                                                                    args_str:
                                                                                                                                        if char == ',' and \
                                                                        paren_level == 0 and \
                                                                        bracket_level == 0:
                                                                        if current_arg:
                                                                            args.append(
                                                                                current_arg.strip())
                                                                            current_arg = ''
                                                                            else:
                                                                            if char == '(':
                                                                                paren_level += 1
                                                                                elif char == ')':
                                                                                paren_level -= 1
                                                                                elif char == '[':
                                                                                bracket_level += 1
                                                                                elif char == ']':
                                                                                bracket_level -= 1
                                                                                current_arg += char

                                                                                if current_arg:
                                                                                    args.append(
                                                                                        current_arg.strip())

                                                                                return args

                                                                                def remove_unused_imports(
                                                                                    source: str) -> str:
                                                                                    """Remove unused imports from the code."""
                                                                                    try:
                                                                                        tree = ast.parse(
                                                                                            source)
                                                                                        except SyntaxError:
                                                                                        return source

                                                                                        # Find all imported names
                                                                                        imports = {}
                                                                                        for node in ast.walk(
                                                                                            tree):
                                                                                            if isinstance(
                                                                                                node,
                                                                                                ast.Import):
                                                                                                                                                                                                for name in \
                                                                                                    node.names:
                                                                                                                                                                                                        imports[name.asname or \
                                                                                                        name.name] = name.name
                                                                                                    elif isinstance(
                                                                                                        node,
                                                                                                        ast.ImportFrom):
                                                                                                                                                                                                        for name in \
                                                                                                        node.names:
                                                                                                                                                                                                                imports[name.asname or \
                                                                                                            name.name] = f"{node.module}.{name.name}"

                                                                                                        # Find all used names
                                                                                                        used_names = set()
                                                                                                        for node in ast.walk(
                                                                                                            tree):
                                                                                                            if isinstance(
                                                                                                                node,
                                                                                                                ast.Name):
                                                                                                                used_names.add(
                                                                                                                    node.id)

                                                                                                                # Remove unused imports
                                                                                                                lines = source.split(
                                                                                                                    '\n')
                                                                                                                result = []
                                                                                                                                                                                                                                for line in \
                                                                                                                    lines:
                                                                                                                    if line.strip(
                                                                                                                        ).startswith(('import ', 'from ')):
                                                                                                                        # Check if this import is used
                                                                                                                        import_used = False
                                                                                                                                                                                                                                                for name in \
                                                                                                                            imports:
                                                                                                                                                                                                                                                        if name in used_names and \
                                                                                                                                name in line:
                                                                                                                                import_used = True
                                                                                                                            break
                                                                                                                            if import_used:
                                                                                                                                result.append(
                                                                                                                                    line)
                                                                                                                                else:
                                                                                                                                result.append(
                                                                                                                                    line)

                                                                                                                            return '\n'.join(
                                                                                                                                result)

                                                                                                                            def cleanup_file(
                                                                                                                                filepath: str) -> None:
                                                                                                                                """Clean up a single Python file."""
                                                                                                                                print(
                                                                                                                                    f"Cleaning up {filepath}...")

                                                                                                                                try:
                                                                                                                                    with open(
                                                                                                                                        filepath,
                                                                                                                                        'r',
                                                                                                                                        encoding='utf-8') as f:
                                                                                                                                    source = f.read()

                                                                                                                                    # Apply fixes in sequence
                                                                                                                                    source = fix_imports(
                                                                                                                                        source)
                                                                                                                                    source = fix_indentation(
                                                                                                                                        source)
                                                                                                                                    source = fix_line_length(
                                                                                                                                        source)
                                                                                                                                    source = remove_unused_imports(
                                                                                                                                        source)

                                                                                                                                    # Write the cleaned up code back to the file
                                                                                                                                    with open(
                                                                                                                                        filepath,
                                                                                                                                        'w',
                                                                                                                                        encoding='utf-8') as f:
                                                                                                                                    f.write(
                                                                                                                                        source)

                                                                                                                                    print(
                                                                                                                                        f"Successfully cleaned up {filepath}")

                                                                                                                                    except Exception as e:
                                                                                                                                        print(
                                                                                                                                            f"Error cleaning up {filepath}: {e}")

                                                                                                                                        def cleanup_directory(
                                                                                                                                            directory: str) -> None:
                                                                                                                                                                                                                                                                                        """Clean up all Python files in \
                                                                                                                                                a directory recursively."""
                                                                                                                                            for root, _, files in os.walk(
                                                                                                                                                directory):
                                                                                                                                                                                                                                                                                                for file in \
                                                                                                                                                    files:
                                                                                                                                                    if file.endswith(
                                                                                                                                                        '.py'):
                                                                                                                                                        filepath = os.path.join(
                                                                                                                                                            root,
                                                                                                                                                            file)
                                                                                                                                                        cleanup_file(
                                                                                                                                                            filepath)

                                                                                                                                                        def main():
                                                                                                                                                            """Main execution function."""
                                                                                                                                                            import sys

                                                                                                                                                            if len(
                                                                                                                                                                sys.argv) != 2:
                                                                                                                                                                print(
                                                                                                                                                                    "Usage: code_cleanup.py <directory>")
                                                                                                                                                                sys.exit(
                                                                                                                                                                    1)

                                                                                                                                                                directory = sys.argv[1]
                                                                                                                                                                if not os.path.exists(
                                                                                                                                                                    directory):
                                                                                                                                                                    print(
                                                                                                                                                                        f"Error: {directory} does not exist")
                                                                                                                                                                    sys.exit(
                                                                                                                                                                        1)

                                                                                                                                                                    if os.path.isfile(
                                                                                                                                                                        directory):
                                                                                                                                                                        cleanup_file(
                                                                                                                                                                            directory)
                                                                                                                                                                        else:
                                                                                                                                                                        cleanup_directory(
                                                                                                                                                                            directory)

                                                                                                                                                                        if __name__ == "__main__":
                                                                                                                                                                            main() 