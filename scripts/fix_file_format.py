#!/usr/bin/env python3.11
"""
Fix File Format Script

This script scans and fixes formatting issues in Python files, focusing on:
1. Ensuring __future__ imports come at the beginning of the file (after docstrings)
2. Fixing spacing between imports and docstrings
3. Ensuring proper indentation
"""

import os
import re
from pathlib import Path

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
    
    # Fix common issues:
    
    # 1. Fix files where docstring and __future__ imports are adjacent without spacing
    pattern = r'(""".*?""")(\s*from\s+__future__\s+import)'
    replacement = r'\1\n\n\2'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
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
                
                # Reorganize the file structure
                new_content = f"{shebang}\n"
                new_content += f'"""{docstring_content}"""\n\n'
                
                # Extract and add future imports
                future_match = re.search(r'from\s+__future__\s+import\s+\w+', rest_content)
                if future_match:
                    future_import = future_match.group(0)
                    new_content += f"{future_import}\n\n"
                    rest_content = rest_content.replace(future_import, '', 1)
                
                # Add remaining content with proper spacing
                imports = []
                class_defs = []
                other_lines = []
                
                lines = [line for line in rest_content.split('\n') if line.strip()]
                for line in lines:
                    if line.startswith(('import ', 'from ')):
                        imports.append(line)
                    elif line.startswith('class '):
                        class_defs.append(line)
                    else:
                        other_lines.append(line)
                
                # Add imports
                for imp in imports:
                    new_content += f"{imp}\n"
                
                if imports:
                    new_content += "\n"
                
                # Add other code
                for line in other_lines + class_defs:
                    new_content += f"{line}\n"
                
                content = new_content
    
    # If nothing changed, return
    if content == original_content:
        return False, "No changes needed"
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return True, "Fixed formatting issues"

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
    print(f"Scanning {project_root} for Python files with formatting issues...")
    
    directories_to_scan = [
        project_root / "ai_agents",
        project_root / "backend",
        project_root / "model_management",
        project_root / "scripts",
    ]
    
    total_fixed = 0
    for directory in directories_to_scan:
        if directory.exists():
            fixed_files = scan_and_fix_directory(directory)
            if fixed_files:
                print(f"\nFixed formatting issues in {len(fixed_files)} files in {directory}:")
                for file_path, reason in fixed_files:
                    print(f"  - {file_path} ({reason})")
                total_fixed += len(fixed_files)
    
    if total_fixed > 0:
        print(f"\nSuccessfully fixed formatting issues in {total_fixed} files.")
    else:
        print("\nNo files needed fixing for formatting issues.")

if __name__ == "__main__":
    main() 