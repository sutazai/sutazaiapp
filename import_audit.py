#!/usr/bin/env python3
"""
Comprehensive Python Import Audit Tool
Scans all Python files for import issues and violations
"""

import os
import ast
import sys
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ImportAuditor:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues = {
            'non_existent_modules': [],
            'circular_imports': [],
            'fantasy_modules': [],
            'relative_vs_absolute_conflicts': [],
            'missing_init_files': [],
            'import_errors': [],
            'unused_imports': [],
            'star_imports': [],
            'dynamic_import_failures': [],
            'missing_third_party': []
        }
        
        # Fantasy/fictional module patterns
        self.fantasy_patterns = [
            'advanced', 'agi', 'process', 'configurator', 'transfer', 'black_box',
            'superintelligence', 'neural_process', 'ai_process', 'advanced_ai',
            'agi_core', 'sentient', 'consciousness'
        ]
        
        # Standard library modules (partial list of common ones)
        self.stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'typing', 'collections',
            'datetime', 'time', 'logging', 'subprocess', 'threading', 're',
            'urllib', 'http', 'socket', 'ssl', 'hashlib', 'base64', 'uuid',
            'asyncio', 'concurrent', 'multiprocessing', 'queue', 'pickle',
            'io', 'csv', 'xml', 'html', 'email', 'sqlite3', 'zipfile',
            'tarfile', 'gzip', 'shutil', 'glob', 'fnmatch', 'tempfile',
            'configparser', 'argparse', 'getpass', 'platform', 'warnings',
            'traceback', 'inspect', 'copy', 'itertools', 'functools',
            'operator', 'random', 'math', 'statistics', 'decimal', 'fractions'
        }
        
        # Track all Python files and their module paths
        self.python_files = {}
        self.module_graph = defaultdict(set)
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Skip some directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)
                    # Map file to potential module name
                    rel_path = file_path.relative_to(self.root_path)
                    module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
                    if module_parts[-1] != '__init__':
                        module_name = '.'.join(module_parts)
                        self.python_files[module_name] = file_path
        
        return python_files
    
    def parse_file(self, file_path: Path) -> Tuple[ast.AST, List[str]]:
        """Parse a Python file and return AST and any syntax errors"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            return tree, []
        except SyntaxError as e:
            return None, [f"Syntax error: {e}"]
        except Exception as e:
            return None, [f"Parse error: {e}"]
    
    def extract_imports(self, tree: ast.AST, file_path: Path = None) -> Dict[str, List]:
        """Extract all import statements from AST"""
        imports = {
            'standard': [],      # import module
            'from': [],         # from module import name
            'star': [],         # from module import *
            'relative': [],     # from .module import name
            'dynamic': []       # importlib.import_module()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports['standard'].append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ImportFrom):
                if node.names and node.names[0].name == '*':
                    imports['star'].append({
                        'module': node.module or '',
                        'level': node.level,
                        'line': node.lineno
                    })
                else:
                    import_info = {
                        'module': node.module or '',
                        'level': node.level,
                        'names': [(n.name, n.asname) for n in (node.names or [])],
                        'line': node.lineno
                    }
                    
                    if node.level > 0:  # Relative import
                        imports['relative'].append(import_info)
                    else:
                        imports['from'].append(import_info)
            
            elif isinstance(node, ast.Call):
                # Check for dynamic imports
                if (isinstance(node.func, ast.Attribute) and 
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'importlib' and
                    node.func.attr == 'import_module'):
                    if node.args:
                        imports['dynamic'].append({
                            'module': 'dynamic_import',
                            'line': node.lineno
                        })
        
        return imports
    
    def is_fantasy_module(self, module_name: str) -> bool:
        """Check if module name contains fantasy elements"""
        module_lower = module_name.lower()
        return any(pattern in module_lower for pattern in self.fantasy_patterns)
    
    def check_module_exists(self, module_name: str, file_path: Path) -> bool:
        """Check if a module exists and can be imported"""
        if not module_name:
            return True
            
        # Check if it's a standard library module
        if module_name.split('.')[0] in self.stdlib_modules:
            return True
            
        # Check if it's a local module
        if module_name in self.python_files:
            return True
            
        # Try to find spec
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError):
            return False
    
    def check_missing_init_files(self) -> List[str]:
        """Check for missing __init__.py files in package directories"""
        missing_init = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            # If directory contains .py files but no __init__.py
            py_files = [f for f in files if f.endswith('.py')]
            if py_files and '__init__.py' not in files:
                # Check if this directory should be a package
                rel_path = Path(root).relative_to(self.root_path)
                if len(rel_path.parts) > 0:  # Not root directory
                    missing_init.append(str(rel_path))
        
        return missing_init
    
    def detect_circular_imports(self) -> List[List[str]]:
        """Detect circular import dependencies"""
        # This is a simplified version - full detection requires complex graph analysis
        circular = []
        visited = set()
        rec_stack = set()
        
        def has_cycle(module, path):
            if module in rec_stack:
                cycle_start = path.index(module)
                circular.append(path[cycle_start:] + [module])
                return True
            if module in visited:
                return False
                
            visited.add(module)
            rec_stack.add(module)
            
            for neighbor in self.module_graph.get(module, []):
                if has_cycle(neighbor, path + [module]):
                    return True
            
            rec_stack.remove(module)
            return False
        
        for module in self.python_files.keys():
            if module not in visited:
                has_cycle(module, [])
        
        return circular
    
    def audit_file(self, file_path: Path) -> Dict:
        """Audit a single Python file for import issues"""
        file_issues = {
            'file': str(file_path.relative_to(self.root_path)),
            'issues': []
        }
        
        tree, parse_errors = self.parse_file(file_path)
        if parse_errors:
            file_issues['issues'].extend([{'type': 'parse_error', 'message': err} for err in parse_errors])
            return file_issues
        
        if not tree:
            return file_issues
            
        imports = self.extract_imports(tree, file_path)
        
        # Check each import type
        for import_info in imports['standard']:
            module = import_info['module']
            line = import_info['line']
            
            if self.is_fantasy_module(module):
                file_issues['issues'].append({
                    'type': 'fantasy_module',
                    'module': module,
                    'line': line,
                    'message': f"Import of fantasy module '{module}'"
                })
            
            if not self.check_module_exists(module, file_path):
                file_issues['issues'].append({
                    'type': 'non_existent_module',
                    'module': module,
                    'line': line,
                    'message': f"Module '{module}' does not exist"
                })
        
        # Check from imports
        for import_info in imports['from']:
            module = import_info['module']
            line = import_info['line']
            
            if self.is_fantasy_module(module):
                file_issues['issues'].append({
                    'type': 'fantasy_module',
                    'module': module,
                    'line': line,
                    'message': f"Import from fantasy module '{module}'"
                })
            
            if module and not self.check_module_exists(module, file_path):
                file_issues['issues'].append({
                    'type': 'non_existent_module',
                    'module': module,
                    'line': line,
                    'message': f"Module '{module}' does not exist"
                })
        
        # Check star imports
        for import_info in imports['star']:
            file_issues['issues'].append({
                'type': 'star_import',
                'module': import_info['module'],
                'line': import_info['line'],
                'message': f"Star import from '{import_info['module']}' (discouraged)"
            })
        
        # Build module graph for circular detection
        current_module = str(file_path.relative_to(self.root_path)).replace('/', '.').replace('.py', '')
        for import_info in imports['standard'] + imports['from']:
            if import_info['module']:
                self.module_graph[current_module].add(import_info['module'])
        
        return file_issues
    
    def run_audit(self) -> Dict:
        """Run complete import audit"""
        print("Starting comprehensive Python import audit...")
        
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files")
        
        # Check for missing __init__.py files
        missing_init = self.check_missing_init_files()
        if missing_init:
            self.issues['missing_init_files'] = missing_init
        
        # Audit each file
        all_file_issues = []
        for i, file_path in enumerate(python_files, 1):
            if i % 100 == 0:
                print(f"Processed {i}/{len(python_files)} files...")
            
            file_issues = self.audit_file(file_path)
            if file_issues['issues']:
                all_file_issues.append(file_issues)
                
                # Categorize issues
                for issue in file_issues['issues']:
                    issue_type = issue['type']
                    if issue_type in self.issues:
                        self.issues[issue_type].append({
                            'file': file_issues['file'],
                            'line': issue.get('line'),
                            'module': issue.get('module'),
                            'message': issue['message']
                        })
        
        # Detect circular imports
        circular_imports = self.detect_circular_imports()
        if circular_imports:
            self.issues['circular_imports'] = circular_imports
        
        # Prepare final report
        report = {
            'summary': {
                'total_files': len(python_files),
                'files_with_issues': len(all_file_issues),
                'total_issues': sum(len(issues) for issues in self.issues.values() if isinstance(issues, list))
            },
            'issues_by_type': {k: len(v) if isinstance(v, list) else 0 for k, v in self.issues.items()},
            'detailed_issues': self.issues,
            'files_with_issues': all_file_issues
        }
        
        return report

def main():
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = '/opt/sutazaiapp'
    
    auditor = ImportAuditor(root_path)
    report = auditor.run_audit()
    
    # Print summary
    print("\n" + "="*80)
    print("IMPORT AUDIT SUMMARY")
    print("="*80)
    print(f"Total Python files scanned: {report['summary']['total_files']}")
    print(f"Files with issues: {report['summary']['files_with_issues']}")
    print(f"Total issues found: {report['summary']['total_issues']}")
    
    print("\nIssues by type:")
    for issue_type, count in report['issues_by_type'].items():
        if count > 0:
            print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    
    # Print detailed issues
    print("\n" + "="*80)
    print("DETAILED ISSUES")
    print("="*80)
    
    for issue_type, issues in report['detailed_issues'].items():
        if issues:
            print(f"\n{issue_type.replace('_', ' ').title()}:")
            print("-" * 40)
            
            if isinstance(issues, list) and issues:
                for issue in issues[:10]:  # Show first 10 of each type
                    if isinstance(issue, dict):
                        print(f"  File: {issue.get('file', 'unknown')}")
                        if 'line' in issue and issue['line']:
                            print(f"  Line: {issue['line']}")
                        if 'module' in issue and issue['module']:
                            print(f"  Module: {issue['module']}")
                        print(f"  Message: {issue.get('message', 'No message')}")
                        print()
                    else:
                        print(f"  {issue}")
                
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more")
    
    # Save detailed report
    report_file = '/opt/sutazaiapp/import_audit_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return 0 if report['summary']['total_issues'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())