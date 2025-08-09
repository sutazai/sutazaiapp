#!/usr/bin/env python3
"""
Check and enforce naming conventions
Part of CLAUDE.md hygiene enforcement
"""

import os
import re
import sys
from pathlib import Path

def check_filename(filepath):
    """Check if filename follows kebab-case convention"""
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Skip special files
    if filename in ['__init__.py', '__main__.py', 'setup.py', 'conftest.py']:
        return True
    
    # Check for kebab-case (lowercase with hyphens)
    if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name_without_ext):
        return False
    
    return True

def check_python_naming(filepath):
    """Check Python code for naming convention violations"""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for camelCase variables (should be snake_case)
        camel_case_vars = re.findall(r'\b([a-z]+[A-Z][a-zA-Z]*)\s*=', content)
        for var in camel_case_vars:
            if var not in ['className', 'innerHTML']:  # Common exceptions
                violations.append(f"Variable '{var}' should use snake_case")
        
        # Check for non-standard class names (should be PascalCase)
        class_names = re.findall(r'class\s+([a-z_][a-zA-Z0-9_]*)\s*[:\(]', content)
        for name in class_names:
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                violations.append(f"Class '{name}' should use PascalCase")
                
    except Exception:
        pass
    
    return violations

def main():
    """Check all files for naming convention violations"""
    file_violations = []
    code_violations = []
    
    # Check Python files
    IGNORE_PARTS = ['venv', 'node_modules', '.git', '__pycache__', 'opt/sutazaiapp/jarvis']
    for filepath in Path('.').glob('**/*.py'):
        # Skip virtual environments, dependencies, and archived duplicates
        if any(part in str(filepath) for part in IGNORE_PARTS):
            continue
        
        # Check filename
        if not check_filename(str(filepath)):
            file_violations.append(str(filepath))
        
        # Check code content
        violations = check_python_naming(str(filepath))
        if violations:
            code_violations.append((str(filepath), violations))
    
    # Check shell scripts
    for filepath in Path('.').glob('**/*.sh'):
        if any(part in str(filepath) for part in IGNORE_PARTS):
            continue
        if not check_filename(str(filepath)):
            file_violations.append(str(filepath))
    
    # Check agent files
    agent_dir = Path('.claude/agents')
    if agent_dir.exists():
        for filepath in agent_dir.glob('*.md'):
            if not check_filename(str(filepath)):
                file_violations.append(str(filepath))
    
    # Report violations
    if file_violations or code_violations:
        print("ERROR: Naming convention violations detected!")
        print("-" * 60)
        
        if file_violations:
            print("\nFilename violations (should use kebab-case):")
            for f in file_violations:
                print(f"  - {f}")
        
        if code_violations:
            print("\nCode naming violations:")
            for filepath, violations in code_violations:
                print(f"\n{filepath}:")
                for v in violations:
                    print(f"  - {v}")
        
        print("-" * 60)
        print("\nNaming conventions:")
        print("  - Files: kebab-case (e.g., my-script.py)")
        print("  - Variables: snake_case (e.g., my_variable)")
        print("  - Classes: PascalCase (e.g., MyClass)")
        print("  - Constants: UPPER_SNAKE_CASE (e.g., MY_CONSTANT)")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
