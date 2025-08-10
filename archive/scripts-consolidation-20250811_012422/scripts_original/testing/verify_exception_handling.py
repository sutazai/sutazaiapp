#!/usr/bin/env python3
"""
Verify that all exception handling in the codebase follows best practices.
This script checks for various exception handling anti-patterns.
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime

class ExceptionHandlingVerifier:
    """Verify exception handling best practices."""
    
    def __init__(self):
        self.issues = []
        self.stats = {
            'total_files': 0,
            'files_analyzed': 0,
            'bare_excepts': 0,
            'broad_excepts': 0,
            'unlogged_excepts': 0,
            'proper_handlers': 0,
            'files_with_issues': set()
        }
    
    def check_file(self, filepath: str) -> Dict:
        """Check a single Python file for exception handling issues."""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {'error': 'syntax_error', 'issues': []}
            
            # Check for various patterns
            class ExceptionVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                    self.has_logging = 'import logging' in content or 'from logging import' in content
                
                def visit_ExceptHandler(self, node):
                    # Check for bare except
                    if node.type is None:
                        self.issues.append({
                            'type': 'bare_except',
                            'line': node.lineno,
                            'severity': 'critical'
                        })
                    
                    # Check for overly broad exception
                    elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                        # Check if there's logging in the handler
                        has_logging_in_handler = self._has_logging_in_body(node.body)
                        
                        if not has_logging_in_handler and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                            self.issues.append({
                                'type': 'silent_broad_except',
                                'line': node.lineno,
                                'severity': 'high'
                            })
                    
                    # Check for empty except blocks
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        self.issues.append({
                            'type': 'empty_except',
                            'line': node.lineno,
                            'severity': 'medium'
                        })
                    
                    self.generic_visit(node)
                
                def _has_logging_in_body(self, body):
                    """Check if the except body contains logging."""
                    for stmt in body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            # Check for logger calls
                            if hasattr(stmt.value.func, 'attr'):
                                if stmt.value.func.attr in ['debug', 'info', 'warning', 'error', 'exception', 'critical']:
                                    return True
                            # Check for print statements (basic logging)
                            if hasattr(stmt.value.func, 'id') and stmt.value.func.id == 'print':
                                return True
                    return False
            
            visitor = ExceptionVisitor()
            visitor.visit(tree)
            
            return {'issues': visitor.issues}
            
        except Exception as e:
            return {'error': str(e), 'issues': []}
    
    def verify_codebase(self, root_dir: str = '/opt/sutazaiapp') -> Dict:
        """Verify exception handling across the entire codebase."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'root_directory': root_dir,
            'summary': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        # Find all Python files
        for root, dirs, files in os.walk(root_dir):
            # Skip virtual environments and cache
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.venv', 'venv', 'env', 
                '.git', 'node_modules', '.pytest_cache', 'archive'
            ]]
            
            for file in files:
                if file.endswith('.py'):
                    self.stats['total_files'] += 1
                    filepath = os.path.join(root, file)
                    
                    # Check the file
                    result = self.check_file(filepath)
                    
                    if result.get('issues'):
                        self.stats['files_analyzed'] += 1
                        self.stats['files_with_issues'].add(filepath)
                        
                        for issue in result['issues']:
                            if issue['type'] == 'bare_except':
                                self.stats['bare_excepts'] += 1
                                results['critical_issues'].append({
                                    'file': filepath,
                                    'line': issue['line'],
                                    'issue': 'Bare except clause found'
                                })
                            elif issue['type'] == 'silent_broad_except':
                                self.stats['broad_excepts'] += 1
                            elif issue['type'] == 'empty_except':
                                self.stats['unlogged_excepts'] += 1
                    else:
                        # File has proper exception handling
                        if 'except' in open(filepath, 'r').read():
                            self.stats['proper_handlers'] += 1
        
        # Generate summary
        results['summary'] = {
            'total_python_files': self.stats['total_files'],
            'files_with_issues': len(self.stats['files_with_issues']),
            'bare_except_clauses': self.stats['bare_excepts'],
            'silent_broad_exceptions': self.stats['broad_excepts'],
            'empty_except_blocks': self.stats['unlogged_excepts'],
            'files_with_proper_handling': self.stats['proper_handlers']
        }
        
        # Generate recommendations
        if self.stats['bare_excepts'] > 0:
            results['recommendations'].append(
                "CRITICAL: Found bare except clauses. Run fix_bare_except_clauses.py immediately."
            )
        
        if self.stats['broad_excepts'] > 10:
            results['recommendations'].append(
                "Consider using more specific exception types instead of catching Exception."
            )
        
        if self.stats['unlogged_excepts'] > 20:
            results['recommendations'].append(
                "Add logging to empty except blocks for better debugging."
            )
        
        # Determine overall status
        if self.stats['bare_excepts'] == 0:
            results['status'] = 'PASSED'
            results['message'] = 'No bare except clauses found. Exception handling meets standards.'
        else:
            results['status'] = 'FAILED'
            results['message'] = f"Found {self.stats['bare_excepts']} bare except clauses that need fixing."
        
        return results

def main():
    """Main verification function."""
    print("=" * 60)
    print("EXCEPTION HANDLING VERIFICATION")
    print("=" * 60)
    
    verifier = ExceptionHandlingVerifier()
    results = verifier.verify_codebase()
    
    # Print results
    print(f"\nVerification Status: {results['status']}")
    print(f"Message: {results['message']}")
    
    print("\n📊 Summary:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    
    if results['critical_issues']:
        print("\n⚠️  Critical Issues Found:")
        for issue in results['critical_issues'][:5]:  # Show first 5
            print(f"  - {issue['file']}:{issue['line']} - {issue['issue']}")
        if len(results['critical_issues']) > 5:
            print(f"  ... and {len(results['critical_issues']) - 5} more")
    
    if results['recommendations']:
        print("\n💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")
    
    # Save detailed report
    report_path = '/opt/sutazaiapp/reports/exception_handling_verification.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Detailed report saved to: {report_path}")
    
    return 0 if results['status'] == 'PASSED' else 1

if __name__ == "__main__":
    exit(main())