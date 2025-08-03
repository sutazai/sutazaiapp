#!/usr/bin/env python3
"""
Check Python Script Headers
Purpose: Verifies all Python scripts have proper headers per Rule 8
Usage: python check-python-headers.py [file1.py file2.py ...]
Requirements: Python 3.8+
"""

import sys
import re
from pathlib import Path

def check_python_header(file_path: Path) -> tuple[bool, str]:
    """Check if Python file has proper header"""
    try:
        content = file_path.read_text()
        lines = content.splitlines()
        
        if not lines:
            return False, "Empty file"
            
        # Check for docstring
        if not (lines[0].startswith('"""') or lines[0].startswith("'''")):
            # Check if it starts after shebang
            if len(lines) > 1 and lines[0].startswith("#!") and (lines[1].startswith('"""') or lines[1].startswith("'''")):
                docstring_start = 1
            else:
                return False, "Missing docstring header"
        else:
            docstring_start = 0
            
        # Look for required fields in docstring
        docstring_lines = []
        in_docstring = False
        for i, line in enumerate(lines[docstring_start:], docstring_start):
            if i == docstring_start:
                in_docstring = True
            if in_docstring:
                docstring_lines.append(line)
                if line.endswith('"""') or line.endswith("'''"):
                    break
                    
        docstring_text = "\n".join(docstring_lines)
        
        # Check for required fields
        required_fields = ["Purpose:", "Usage:"]
        missing_fields = []
        
        for field in required_fields:
            if field not in docstring_text:
                missing_fields.append(field)
                
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
            
        return True, "Header OK"
        
    except Exception as e:
        return False, f"Error reading file: {e}"

def main():
    # If files provided as arguments, check those
    # Otherwise, this is used by pre-commit which provides files via stdin
    files_to_check = []
    
    if len(sys.argv) > 1:
        files_to_check = [Path(f) for f in sys.argv[1:]]
    else:
        # Read from stdin for pre-commit
        for line in sys.stdin:
            files_to_check.append(Path(line.strip()))
            
    violations = []
    
    for file_path in files_to_check:
        if file_path.suffix == ".py":
            # Skip test files and __init__.py
            if "test" in file_path.name or file_path.name == "__init__.py":
                continue
                
            valid, reason = check_python_header(file_path)
            if not valid:
                violations.append(f"{file_path}: {reason}")
                
    if violations:
        print("Python header violations found (Rule 8):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())