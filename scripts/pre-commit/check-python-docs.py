#!/usr/bin/env python3
"""
Purpose: Verify Python scripts have proper documentation (Rule 8 enforcement)
Usage: python check-python-docs.py <file1> <file2> ...
Requirements: Python 3.8+
"""

import sys
import ast
import re
from pathlib import Path
from typing import List, Tuple, Optional

class PythonDocChecker(ast.NodeVisitor):
    """AST visitor to check Python documentation compliance."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.violations = []
        self.has_module_docstring = False
        self.has_main_guard = False
        
    def visit_Module(self, node: ast.Module) -> None:
        """Check module-level documentation."""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            docstring = node.body[0].value.s
            self.has_module_docstring = True
            
            # Check docstring format
            if not self._check_module_docstring_format(docstring):
                self.violations.append((
                    1,
                    "Module docstring missing required sections",
                    "Must include: Purpose, Usage, Requirements"
                ))
        else:
            self.violations.append((
                1,
                "Missing module-level docstring",
                'Add docstring with """Purpose: ..., Usage: ..., Requirements: ..."""'
            ))
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function documentation."""
        # Skip private functions and test functions
        if node.name.startswith('_') or node.name.startswith('test_'):
            return
            
        docstring = ast.get_docstring(node)
        
        if not docstring:
            self.violations.append((
                node.lineno,
                f"Function '{node.name}' missing docstring",
                "Add docstring describing purpose, parameters, and return value"
            ))
        elif len(node.args.args) > 1 and 'Args:' not in docstring and 'Parameters:' not in docstring:
            self.violations.append((
                node.lineno,
                f"Function '{node.name}' docstring missing parameter documentation",
                "Document parameters in docstring"
            ))
            
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check class documentation."""
        docstring = ast.get_docstring(node)
        
        if not docstring:
            self.violations.append((
                node.lineno,
                f"Class '{node.name}' missing docstring",
                "Add docstring describing class purpose and usage"
            ))
            
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If) -> None:
        """Check for main guard."""
        if isinstance(node.test, ast.Compare) and \
           isinstance(node.test.left, ast.Name) and \
           node.test.left.id == '__name__' and \
           len(node.test.comparators) == 1 and \
           isinstance(node.test.comparators[0], ast.Str) and \
           node.test.comparators[0].s == '__main__':
            self.has_main_guard = True
            
        self.generic_visit(node)
    
    def _check_module_docstring_format(self, docstring: str) -> bool:
        """Check if module docstring has required format."""
        required_sections = ['purpose:', 'usage:', 'requirements:']
        docstring_lower = docstring.lower()
        
        return all(section in docstring_lower for section in required_sections)

def check_file_header(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check for proper file header and imports."""
    violations = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return violations
            
        # Check shebang for executable scripts
        if filepath.stat().st_mode & 0o111:  # Check if executable
            if not lines[0].startswith('#!/usr/bin/env python'):
                violations.append((
                    1,
                    "Executable script missing proper shebang",
                    "Add #!/usr/bin/env python3 as first line"
                ))
        
        # Check for hardcoded paths
        for line_num, line in enumerate(lines, 1):
            if '/home/' in line or '/Users/' in line:
                violations.append((
                    line_num,
                    "Hardcoded user-specific path detected",
                    "Use relative paths or configuration"
                ))
            
        # Check for proper logging instead of print
        if 'print(' in ''.join(lines) and 'if __name__' in ''.join(lines):
            has_logging = 'import logging' in ''.join(lines[:20])
            if not has_logging:
                violations.append((
                    0,
                    "Using print() instead of proper logging",
                    "Import logging module and use logging.info() instead"
                ))
                
    except Exception as e:
        violations.append((
            0,
            f"Error reading file: {e}",
            "Ensure file is valid Python"
        ))
        
    return violations

def check_script_organization(filepath: Path) -> List[Tuple[int, str, str]]:
    """Check if script follows proper organization."""
    violations = []
    
    # Scripts should be in appropriate directories
    relative_path = filepath.relative_to(Path("/opt/sutazaiapp"))
    path_parts = relative_path.parts
    
    if len(path_parts) > 1 and path_parts[0] not in ['scripts', 'tests', 'backend', 'agents']:
        if filepath.name not in ['setup.py', 'manage.py']:
            violations.append((
                0,
                f"Python script in unexpected location: {relative_path}",
                "Move to scripts/, tests/, or appropriate module directory"
            ))
    
    return violations

def main():
    """Main function to check Python documentation."""
    if len(sys.argv) < 2:
        print("No files to check")
        return 0
    
    total_violations = 0
    
    for filepath_str in sys.argv[1:]:
        filepath = Path(filepath_str)
        
        # Skip non-Python files
        if filepath.suffix != '.py':
            continue
            
        # Skip certain files
        if any(skip in str(filepath) for skip in ['__pycache__', 'venv', '.venv', 'migrations']):
            continue
        
        violations = []
        
        # Check file header and organization
        violations.extend(check_file_header(filepath))
        violations.extend(check_script_organization(filepath))
        
        # Parse and check AST
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            checker = PythonDocChecker(filepath)
            checker.visit(tree)
            violations.extend(checker.violations)
            
            # Check for main guard in scripts
            if 'scripts' in str(filepath) and not checker.has_main_guard:
                violations.append((
                    0,
                    "Script missing if __name__ == '__main__' guard",
                    "Add main guard to make script importable"
                ))
                
        except SyntaxError as e:
            violations.append((
                e.lineno or 0,
                f"Syntax error: {e.msg}",
                "Fix Python syntax"
            ))
        except Exception as e:
            violations.append((
                0,
                f"Error parsing file: {e}",
                "Ensure file is valid Python"
            ))
        
        # Report violations for this file
        if violations:
            print(f"\n❌ Rule 8 violations in {filepath}:")
            for line_num, message, fix in violations:
                if line_num > 0:
                    print(f"  Line {line_num}: {message}")
                else:
                    print(f"  File: {message}")
                print(f"    Fix: {fix}")
            
            total_violations += len(violations)
    
    if total_violations > 0:
        print(f"\n❌ Rule 8 Violation: Found {total_violations} documentation issues")
        print("\n📋 Python documentation requirements:")
        print("  1. Module docstring with Purpose, Usage, Requirements")
        print("  2. Function/class docstrings for all public members")
        print("  3. Use argparse/click for command-line arguments")
        print("  4. Proper logging instead of print()")
        print("  5. Main guard for scripts")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())