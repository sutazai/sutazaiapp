#!/usr/bin/env python3
"""
Detailed Import Analysis Tool
Performs comprehensive analysis of import issues across the codebase
"""

import os
import ast
import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class DetailedImportAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues = {
            'missing_modules': [],
            'fantasy_imports': [],
            'circular_imports': [],
            'star_imports': [],
            'missing_third_party': [],
            'relative_import_issues': [],
            'syntax_errors': []
        }
        
        self.fantasy_patterns = [
            'quantum', 'agi', 'magic', 'wizard', 'teleport', 'black_box',
            'superintelligence', 'neural_magic', 'ai_magic', 'quantum_ai',
            'agi_core', 'sentient', 'consciousness'
        ]

    def analyze_imports_in_file(self, file_path: Path) -> Dict:
        """Analyze imports in a single Python file"""
        results = {
            'file': str(file_path.relative_to(self.root_path)),
            'imports': [],
            'issues': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                results['issues'].append({
                    'type': 'syntax_error',
                    'message': f"Syntax error at line {e.lineno}: {e.msg}",
                    'line': e.lineno
                })
                return results
            
            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = {
                            'type': 'standard',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        }
                        results['imports'].append(import_info)
                        
                        # Check for issues
                        self._check_import_issues(import_info, results, file_path)
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    
                    if node.names and node.names[0].name == '*':
                        import_info = {
                            'type': 'star',
                            'module': module,
                            'level': node.level,
                            'line': node.lineno
                        }
                        results['imports'].append(import_info)
                        results['issues'].append({
                            'type': 'star_import',
                            'message': f"Star import from '{module}' at line {node.lineno}",
                            'line': node.lineno,
                            'module': module
                        })
                    else:
                        names = [(n.name, n.asname) for n in (node.names or [])]
                        import_info = {
                            'type': 'from',
                            'module': module,
                            'names': names,
                            'level': node.level,
                            'line': node.lineno
                        }
                        results['imports'].append(import_info)
                        
                        # Check for issues
                        self._check_import_issues(import_info, results, file_path)
            
        except Exception as e:
            results['issues'].append({
                'type': 'parse_error',
                'message': f"Failed to parse file: {str(e)}",
                'line': 0
            })
        
        return results

    def _check_import_issues(self, import_info: Dict, results: Dict, file_path: Path):
        """Check specific import for issues"""
        module = import_info.get('module', '')
        line = import_info.get('line', 0)
        
        if not module:
            return
        
        # Check for fantasy modules
        if any(pattern in module.lower() for pattern in self.fantasy_patterns):
            results['issues'].append({
                'type': 'fantasy_import',
                'message': f"Import of fantasy/fictional module '{module}' at line {line}",
                'line': line,
                'module': module
            })
        
        # Check if module exists
        if not self._module_exists(module, file_path):
            results['issues'].append({
                'type': 'missing_module',
                'message': f"Module '{module}' not found at line {line}",
                'line': line,
                'module': module
            })

    def _module_exists(self, module_name: str, current_file: Path) -> bool:
        """Check if a module exists and can be imported"""
        if not module_name:
            return True
        
        # Standard library modules (common ones)
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'typing', 'collections',
            'datetime', 'time', 'logging', 'subprocess', 'threading', 're',
            'urllib', 'http', 'socket', 'ssl', 'hashlib', 'base64', 'uuid',
            'asyncio', 'concurrent', 'multiprocessing', 'queue', 'pickle',
            'io', 'csv', 'xml', 'html', 'email', 'sqlite3', 'zipfile',
            'functools', 'itertools', 'operator', 'random', 'math'
        }
        
        base_module = module_name.split('.')[0]
        if base_module in stdlib_modules:
            return True
        
        # Try to find spec
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
            pass
        
        # Check if it's a local module relative to current file
        try:
            if module_name.startswith('.'):
                # Relative import - more complex to resolve
                return True  # Skip for now
            
            # Check if it's in the project
            module_path = module_name.replace('.', '/')
            potential_paths = [
                self.root_path / f"{module_path}.py",
                self.root_path / module_path / "__init__.py",
                current_file.parent / f"{module_path}.py",
                current_file.parent / module_path / "__init__.py"
            ]
            
            for path in potential_paths:
                if path.exists():
                    return True
        except Exception:
            pass
        
        return False

    def analyze_all_files(self) -> Dict:
        """Analyze all Python files in the project"""
        print("Starting detailed import analysis...")
        
        python_files = list(self.root_path.rglob("*.py"))
        python_files = [f for f in python_files if '__pycache__' not in str(f)]
        
        print(f"Found {len(python_files)} Python files to analyze")
        
        all_results = []
        summary = {
            'total_files': len(python_files),
            'files_with_issues': 0,
            'total_imports': 0,
            'total_issues': 0,
            'issues_by_type': defaultdict(int)
        }
        
        for i, file_path in enumerate(python_files, 1):
            if i % 100 == 0:
                print(f"Processed {i}/{len(python_files)} files...")
            
            result = self.analyze_imports_in_file(file_path)
            
            if result['imports'] or result['issues']:
                all_results.append(result)
                summary['total_imports'] += len(result['imports'])
                
                if result['issues']:
                    summary['files_with_issues'] += 1
                    summary['total_issues'] += len(result['issues'])
                    
                    for issue in result['issues']:
                        summary['issues_by_type'][issue['type']] += 1
        
        # Compile detailed report
        report = {
            'summary': dict(summary),
            'file_details': all_results,
            'critical_issues': self._identify_critical_issues(all_results)
        }
        
        return report

    def _identify_critical_issues(self, results: List[Dict]) -> Dict:
        """Identify the most critical import issues"""
        critical = {
            'missing_core_modules': [],
            'fantasy_imports': [],
            'broken_imports': []
        }
        
        for file_result in results:
            file_path = file_result['file']
            
            for issue in file_result['issues']:
                issue_type = issue['type']
                module = issue.get('module', '')
                
                # Critical: Missing core application modules
                if (issue_type == 'missing_module' and 
                    any(pattern in module for pattern in ['app.', 'backend.', 'core.'])):
                    critical['missing_core_modules'].append({
                        'file': file_path,
                        'module': module,
                        'line': issue['line']
                    })
                
                # Critical: Fantasy/fictional imports
                elif issue_type == 'fantasy_import':
                    critical['fantasy_imports'].append({
                        'file': file_path,
                        'module': module,
                        'line': issue['line']
                    })
                
                # Critical: Any import that completely breaks file parsing
                elif issue_type in ['syntax_error', 'parse_error']:
                    critical['broken_imports'].append({
                        'file': file_path,
                        'message': issue['message'],
                        'line': issue.get('line', 0)
                    })
        
        return critical

def main():
    analyzer = DetailedImportAnalyzer('/opt/sutazaiapp')
    report = analyzer.analyze_all_files()
    
    # Print summary
    print("\n" + "="*80)
    print("DETAILED IMPORT ANALYSIS SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"Total Python files: {summary['total_files']}")
    print(f"Files with issues: {summary['files_with_issues']}")
    print(f"Total imports: {summary['total_imports']}")
    print(f"Total issues: {summary['total_issues']}")
    
    print(f"\nIssues by type:")
    for issue_type, count in summary['issues_by_type'].items():
        print(f"  {issue_type.replace('_', ' ').title()}: {count}")
    
    # Print critical issues
    critical = report['critical_issues']
    
    print(f"\n" + "="*80)
    print("CRITICAL ISSUES")
    print("="*80)
    
    if critical['missing_core_modules']:
        print(f"\nMissing Core Modules ({len(critical['missing_core_modules'])}):")
        for issue in critical['missing_core_modules'][:10]:
            print(f"  {issue['file']}:{issue['line']} - {issue['module']}")
        if len(critical['missing_core_modules']) > 10:
            print(f"  ... and {len(critical['missing_core_modules']) - 10} more")
    
    if critical['fantasy_imports']:
        print(f"\nFantasy/Fictional Imports ({len(critical['fantasy_imports'])}):")
        for issue in critical['fantasy_imports'][:10]:
            print(f"  {issue['file']}:{issue['line']} - {issue['module']}")
        if len(critical['fantasy_imports']) > 10:
            print(f"  ... and {len(critical['fantasy_imports']) - 10} more")
    
    if critical['broken_imports']:
        print(f"\nBroken Files ({len(critical['broken_imports'])}):")
        for issue in critical['broken_imports'][:10]:
            print(f"  {issue['file']}:{issue['line']} - {issue['message']}")
        if len(critical['broken_imports']) > 10:
            print(f"  ... and {len(critical['broken_imports']) - 10} more")
    
    # Save detailed report
    report_file = '/opt/sutazaiapp/detailed_import_analysis.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return 0 if summary['total_issues'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())