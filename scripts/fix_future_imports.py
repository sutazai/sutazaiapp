#!/usr/bin/env python3.11
"""
Fix __future__ imports in Python files

This script scans Python files and ensures that __future__ imports are at the beginning of the file.
"""

import os
import re
from pathlib import Path

def fix_future_imports(file_path):
    """Fix __future__ imports in a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if there's a __future__ import that's not at the beginning
    future_import_pattern = r'from\s+__future__\s+import\s+[a-zA-Z_]+'
    future_imports = re.findall(future_import_pattern, content)

    if not future_imports:
        return False  # No __future__ imports to fix

    # Check if there's code before the first __future__ import
    first_future_pos = content.find(future_imports[0])
    code_before = content[:first_future_pos].strip()

    # If there's only comments or empty lines before, it's fine
    code_before_without_comments = re.sub(r'#.*$', '', code_before, flags=re.MULTILINE).strip()
    if not code_before_without_comments or code_before_without_comments.startswith('#!'):
        return False  # __future__ imports are already at the beginning

    # Extract all __future__ imports
    all_future_imports = []
    for future_import in future_imports:
        all_future_imports.append(future_import)

    # Remove all __future__ imports from the content
    for future_import in all_future_imports:
        content = content.replace(future_import, '')

    # Add shebang line if it exists
    shebang = ""
    if content.startswith('#!'):
        shebang_end = content.find('\n')
        shebang = content[:shebang_end+1]
        content = content[shebang_end+1:]

    # Add all __future__ imports at the beginning
    new_content = shebang + '\n'.join(all_future_imports) + '\n\n' + content.lstrip()

    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True

def scan_and_fix_directory(directory):
    """Scan a directory recursively and fix __future__ imports in all Python files."""
    fixed_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_future_imports(file_path):
                    fixed_files.append(file_path)

    return fixed_files

def main():
    """Main function."""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Scanning {project_root} for Python files with misplaced __future__ imports...")

    directories_to_scan = [
        project_root / "ai_agents",
        project_root / "backend",
    ]

    total_fixed = 0
    for directory in directories_to_scan:
        if directory.exists():
            fixed_files = scan_and_fix_directory(directory)
            if fixed_files:
                print(f"\nFixed __future__ imports in {len(fixed_files)} files in {directory}:")
                for file in fixed_files:
                    print(f"  - {file}")
                total_fixed += len(fixed_files)

    if total_fixed > 0:
        print(f"\nSuccessfully fixed __future__ imports in {total_fixed} files.")
    else:
        print("\nNo files needed fixing for __future__ imports.")

if __name__ == "__main__":
    main()
