#!/usr/bin/env python3
"""
Comprehensive Code Audit for SutazAI Application
Performs exhaustive analysis of all Python, JavaScript, and TypeScript files
"""

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAuditor:
    """Comprehensive code auditor for security and quality issues"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues = defaultdict(list)
        self.file_stats = {}
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for different types of issues"""
        return {
            'misleading_functions': [
                r'def\s+(\w*save\w*|write\w*|create\w*|delete\w*|remove\w*)\s*\([^)]*\):\s*(?:\n\s*"""[^"]*""")?\s*(?:pass|return\s*None|return\s*False|return\s*{}|return\s*\[\])',
                r'def\s+(\w*process\w*|handle\w*|execute\w*|run\w*)\s*\([^)]*\):\s*(?:\n\s*"""[^"]*""")?\s*(?:pass|return\s*None|return\s*".*")',
                r'def\s+(\w*connect\w*|init\w*|setup\w*|start\w*)\s*\([^)]*\):\s*(?:\n\s*"""[^"]*""")?\s*(?:pass|return\s*True|return\s*None)',
            ],
            'empty_implementations': [
                r'def\s+\w+\s*\([^)]*\):\s*(?:\n\s*"""[^"]*""")?\s*pass',
                r'class\s+\w+[^:]*:\s*(?:\n\s*"""[^"]*""")?\s*pass',
                r'async\s+def\s+\w+\s*\([^)]*\):\s*(?:\n\s*"""[^"]*""")?\s*pass',
            ],
            'stub_code': [
                r'#\s*TODO:?\s*(implement|add|fix|complete)',
                r'#\s*FIXME:?\s*',
                r'#\s*HACK:?\s*',
                r'#\s*XXX:?\s*',
                r'raise\s+NotImplementedError',
                r'print\s*\(\s*["\'].*not.*implement.*["\']',
                r'return\s*["\'].*not.*implement.*["\']',
            ],
            'security_issues': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.call\s*\(',
                r'os\.system\s*\(',
                r'shell=True',
                r'pickle\.loads?\s*\(',
                r'input\s*\(\s*["\'][^"\']*["\']\s*\)',
                r'raw_input\s*\(',
                r'__import__\s*\(',
                r'open\s*\([^,)]*,\s*["\']w',
            ],
            'hardcoded_credentials': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'["\'](?:password|passwd|pwd)["\']:\s*["\'][^"\']+["\']',
            ],
            'dead_code': [
                r'#.*dead\s+code',
                r'#.*unused',
                r'#.*deprecated',
                r'if\s+False:',
                r'if\s+0:',
                r'while\s+False:',
            ],
            'commented_code': [
                r'#\s*def\s+\w+',
                r'#\s*class\s+\w+',
                r'#\s*import\s+\w+',
                r'#\s*from\s+\w+',
                r'#\s*if\s+\w+',
                r'#\s*for\s+\w+',
                r'#\s*while\s+\w+',
            ],
            'dangerous_imports': [
                r'import\s+os',
                r'import\s+subprocess',
                r'import\s+pickle',
                r'from\s+os\s+import',
                r'from\s+subprocess\s+import',
            ]
        }
    
    def audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Audit a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return {'error': f"Failed to read file: {e}"}
        
        file_issues = {
            'misleading_functions': [],
            'empty_implementations': [],
            'stub_code': [],
            'security_issues': [],
            'hardcoded_credentials': [],
            'dead_code': [],
            'commented_code': [],
            'dangerous_imports': [],
            'syntax_errors': [],
            'documentation_mismatch': [],
            'function_analysis': []
        }
        
        # Check patterns
        for issue_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    file_issues[issue_type].append({
                        'line': line_num,
                        'match': match.group(0).strip(),
                        'pattern': pattern
                    })
        
        # Python-specific analysis
        if file_path.suffix == '.py':
            try:
                tree = ast.parse(content)
                file_issues.update(self._analyze_python_ast(tree, content))
            except SyntaxError as e:
                file_issues['syntax_errors'].append({
                    'line': e.lineno,
                    'error': str(e)
                })
            except Exception as e:
                file_issues['syntax_errors'].append({
                    'line': 0,
                    'error': f"AST parsing failed: {e}"
                })
        
        # Calculate file statistics
        lines = content.splitlines()
        file_issues['file_stats'] = {
            'total_lines': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'comment_lines': sum(1 for line in lines if line.strip().startswith('#')),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'file_size': len(content),
            'functions_count': len(re.findall(r'def\s+\w+', content)),
            'classes_count': len(re.findall(r'class\s+\w+', content)),
        }
        
        return file_issues
    
    def _analyze_python_ast(self, tree: ast.AST, content: str) -> Dict[str, List[Dict]]:
        """Analyze Python AST for specific issues"""
        issues = {
            'function_analysis': [],
            'documentation_mismatch': []
        }
        
        lines = content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_analysis = self._analyze_function(node, lines)
                if func_analysis:
                    issues['function_analysis'].append(func_analysis)
        
        return issues
    
    def _analyze_function(self, func_node: ast.FunctionDef, lines: List[str]) -> Dict[str, Any]:
        """Analyze a function for issues"""
        analysis = {
            'name': func_node.name,
            'line': func_node.lineno,
            'issues': []
        }
        
        # Check if function is empty or stub
        if len(func_node.body) == 1:
            if isinstance(func_node.body[0], ast.Pass):
                analysis['issues'].append('empty_function')
            elif isinstance(func_node.body[0], ast.Raise) and isinstance(func_node.body[0].exc, ast.Call):
                if hasattr(func_node.body[0].exc.func, 'id') and func_node.body[0].exc.func.id == 'NotImplementedError':
                    analysis['issues'].append('not_implemented')
        
        # Check for misleading function name vs implementation
        func_name_lower = func_node.name.lower()
        if any(keyword in func_name_lower for keyword in ['save', 'write', 'create', 'delete', 'remove']):
            # Check if function actually does anything meaningful
            if len(func_node.body) <= 2:  # Allow for docstring + simple return
                if all(isinstance(stmt, (ast.Pass, ast.Return)) for stmt in func_node.body):
                    analysis['issues'].append('misleading_name_empty_implementation')
        
        # Check for docstring vs implementation mismatch
        docstring = ast.get_docstring(func_node)
        if docstring:
            doc_lower = docstring.lower()
            # Check if docstring promises more than implementation delivers
            if any(word in doc_lower for word in ['save', 'create', 'delete', 'process', 'handle']):
                if len(func_node.body) <= 2 and all(isinstance(stmt, (ast.Pass, ast.Return, ast.Expr)) for stmt in func_node.body):
                    analysis['issues'].append('documentation_implementation_mismatch')
        
        return analysis if analysis['issues'] else None
    
    def audit_directory(self, extensions: List[str] = ['.py', '.js', '.ts']) -> Dict[str, Any]:
        """Audit all files in directory with specified extensions"""
        results = {
            'files_audited': 0,
            'total_issues': 0,
            'issues_by_type': defaultdict(int),
            'files_with_issues': {},
            'summary': {},
            'critical_files': []
        }
        
        # Find all files to audit
        files_to_audit = []
        for ext in extensions:
            files_to_audit.extend(self.root_path.rglob(f'*{ext}'))
        
        # Filter out common directories to ignore
        ignore_dirs = {'__pycache__', '.git', 'node_modules', 'venv', '.pytest_cache', 'build', 'dist'}
        files_to_audit = [f for f in files_to_audit if not any(part in ignore_dirs for part in f.parts)]
        
        logger.info(f"Auditing {len(files_to_audit)} files...")
        
        for file_path in files_to_audit:
            logger.info(f"Auditing: {file_path}")
            file_issues = self.audit_file(file_path)
            
            if 'error' in file_issues:
                logger.warning(f"Error auditing {file_path}: {file_issues['error']}")
                continue
            
            results['files_audited'] += 1
            
            # Count issues
            file_issue_count = 0
            for issue_type, issues in file_issues.items():
                if issue_type != 'file_stats' and isinstance(issues, list):
                    issue_count = len(issues)
                    results['issues_by_type'][issue_type] += issue_count
                    file_issue_count += issue_count
            
            if file_issue_count > 0:
                results['files_with_issues'][str(file_path)] = file_issues
                results['total_issues'] += file_issue_count
            
            # Mark critical files (many issues or critical security issues)
            critical_indicators = ['security_issues', 'hardcoded_credentials', 'misleading_functions']
            if any(len(file_issues.get(indicator, [])) > 0 for indicator in critical_indicators) or file_issue_count > 10:
                results['critical_files'].append({
                    'file': str(file_path),
                    'issue_count': file_issue_count,
                    'critical_issues': {k: len(v) for k, v in file_issues.items() 
                                      if k in critical_indicators and isinstance(v, list) and len(v) > 0}
                })
        
        # Generate summary
        results['summary'] = {
            'files_audited': results['files_audited'],
            'files_with_issues': len(results['files_with_issues']),
            'total_issues_found': results['total_issues'],
            'most_common_issues': dict(sorted(results['issues_by_type'].items(), key=lambda x: x[1], reverse=True)[:10]),
            'critical_files_count': len(results['critical_files'])
        }
        
        return results

def main():
    """Main audit execution"""
    root_path = "/opt/sutazaiapp"
    
    print("🔍 Starting Comprehensive Code Audit...")
    print(f"Root path: {root_path}")
    print("=" * 80)
    
    auditor = CodeAuditor(root_path)
    results = auditor.audit_directory()
    
    # Save results
    output_file = f"{root_path}/comprehensive_audit_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n📊 AUDIT SUMMARY")
    print("=" * 80)
    print(f"Files Audited: {results['summary']['files_audited']}")
    print(f"Files with Issues: {results['summary']['files_with_issues']}")  
    print(f"Total Issues Found: {results['summary']['total_issues_found']}")
    print(f"Critical Files: {results['summary']['critical_files_count']}")
    
    print("\n🚨 MOST COMMON ISSUES")
    print("-" * 40)
    for issue_type, count in results['summary']['most_common_issues'].items():
        print(f"{issue_type.replace('_', ' ').title()}: {count}")
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    if results['critical_files']:
        print(f"\n⚠️  CRITICAL FILES REQUIRING IMMEDIATE ATTENTION:")
        print("-" * 60)
        for critical_file in results['critical_files'][:10]:  # Show top 10 critical files
            print(f"📁 {critical_file['file']} ({critical_file['issue_count']} issues)")
            for issue_type, count in critical_file['critical_issues'].items():
                print(f"   • {issue_type.replace('_', ' ').title()}: {count}")
    
    return results

if __name__ == "__main__":
    main()