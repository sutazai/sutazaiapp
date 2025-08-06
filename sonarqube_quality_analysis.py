#!/usr/bin/env python3
"""
SonarQube-style Code Quality Analysis for SutazAI
Performs comprehensive static analysis focusing on:
- Code smells and anti-patterns
- Technical debt indicators
- Duplicate code detection
- Security vulnerabilities
- Complexity metrics
- Coding standards compliance
"""

import os
import re
import ast
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict, Counter
import subprocess

class CodeQualityAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.issues = defaultdict(list)
        self.metrics = {
            'total_files': 0,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'duplicates': [],
            'complexity': [],
            'security_issues': [],
            'code_smells': [],
            'technical_debt_minutes': 0
        }
        self.file_hashes = defaultdict(list)
        self.function_signatures = defaultdict(list)
        
    def analyze(self):
        """Main analysis entry point"""
        print("🔍 Starting SonarQube-style Code Quality Analysis...")
        
        # Analyze Python files
        python_files = list(self.root_path.rglob("*.py"))
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            self._analyze_python_file(file_path)
            
        # Analyze Docker configurations
        docker_files = list(self.root_path.glob("docker-compose*.yml"))
        docker_files.extend(self.root_path.rglob("Dockerfile*"))
        for file_path in docker_files:
            self._analyze_docker_file(file_path)
            
        # Detect duplicates
        self._detect_duplicate_code()
        
        # Calculate technical debt
        self._calculate_technical_debt()
        
        return self._generate_report()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Skip certain directories and files"""
        skip_dirs = {'__pycache__', '.git', 'node_modules', 'venv', '.venv', 
                    'build', 'dist', '.pytest_cache', 'archive', 'backups'}
        return any(part in skip_dirs for part in file_path.parts)
    
    def _analyze_python_file(self, file_path: Path):
        """Analyze a Python file for quality issues"""
        self.metrics['total_files'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.splitlines()
                
            # Basic metrics
            self.metrics['total_lines'] += len(lines)
            
            # Count line types
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    self.metrics['blank_lines'] += 1
                elif stripped.startswith('#'):
                    self.metrics['comment_lines'] += 1
                else:
                    self.metrics['code_lines'] += 1
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path, content)
            except SyntaxError as e:
                self.issues['syntax_errors'].append({
                    'file': str(file_path.relative_to(self.root_path)),
                    'error': str(e),
                    'severity': 'BLOCKER'
                })
            
            # Check for security issues
            self._check_security_issues(content, file_path)
            
            # Check for code smells
            self._check_code_smells(content, file_path, lines)
            
            # Store file hash for duplicate detection
            file_hash = hashlib.md5(content.encode()).hexdigest()
            self.file_hashes[file_hash].append(str(file_path.relative_to(self.root_path)))
            
        except Exception as e:
            self.issues['file_errors'].append({
                'file': str(file_path.relative_to(self.root_path)),
                'error': str(e),
                'severity': 'MAJOR'
            })
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, content: str):
        """Analyze Abstract Syntax Tree for complexity and patterns"""
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self, analyzer, file_path):
                self.analyzer = analyzer
                self.file_path = file_path
                self.current_function = None
                self.complexity_stack = []
                
            def visit_FunctionDef(self, node):
                # Calculate cyclomatic complexity
                complexity = 1  # Base complexity
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                
                rel_path = str(self.file_path.relative_to(self.analyzer.root_path))
                
                if complexity > 10:
                    self.analyzer.issues['high_complexity'].append({
                        'file': rel_path,
                        'function': node.name,
                        'complexity': complexity,
                        'line': node.lineno,
                        'severity': 'MAJOR' if complexity > 15 else 'MINOR'
                    })
                
                # Check for too many parameters
                if len(node.args.args) > 5:
                    self.analyzer.issues['too_many_parameters'].append({
                        'file': rel_path,
                        'function': node.name,
                        'count': len(node.args.args),
                        'line': node.lineno,
                        'severity': 'MINOR'
                    })
                
                # Store function signature for duplicate detection
                sig = f"{node.name}({len(node.args.args)})"
                self.analyzer.function_signatures[sig].append({
                    'file': rel_path,
                    'line': node.lineno
                })
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for god classes (too many methods)
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:
                    rel_path = str(self.file_path.relative_to(self.analyzer.root_path))
                    self.analyzer.issues['god_class'].append({
                        'file': rel_path,
                        'class': node.name,
                        'method_count': len(methods),
                        'line': node.lineno,
                        'severity': 'MAJOR'
                    })
                self.generic_visit(node)
        
        visitor = ComplexityVisitor(self, file_path)
        visitor.visit(tree)
    
    def _check_security_issues(self, content: str, file_path: Path):
        """Check for common security vulnerabilities"""
        rel_path = str(file_path.relative_to(self.root_path))
        
        security_patterns = [
            (r'eval\s*\(', 'Use of eval() - potential code injection', 'CRITICAL'),
            (r'exec\s*\(', 'Use of exec() - potential code injection', 'CRITICAL'),
            (r'pickle\.loads?\s*\(', 'Use of pickle - potential arbitrary code execution', 'MAJOR'),
            (r'subprocess.*shell\s*=\s*True', 'Shell injection vulnerability', 'CRITICAL'),
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected', 'BLOCKER'),
            (r'api_key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected', 'BLOCKER'),
            (r'SECRET.*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected', 'BLOCKER'),
            (r'os\.system\s*\(', 'Use of os.system - potential command injection', 'MAJOR'),
            (r'\.format\s*\(.*request\.|f["\'].*request\.', 'Potential format string vulnerability', 'MAJOR'),
            (r'verify\s*=\s*False', 'SSL verification disabled', 'MAJOR'),
        ]
        
        for pattern, description, severity in security_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_no = content[:match.start()].count('\n') + 1
                self.issues['security'].append({
                    'file': rel_path,
                    'issue': description,
                    'line': line_no,
                    'severity': severity,
                    'pattern': pattern
                })
    
    def _check_code_smells(self, content: str, file_path: Path, lines: List[str]):
        """Detect common code smells and anti-patterns"""
        rel_path = str(file_path.relative_to(self.root_path))
        
        # Long file
        if len(lines) > 500:
            self.issues['long_file'].append({
                'file': rel_path,
                'lines': len(lines),
                'severity': 'MINOR'
            })
        
        # TODO/FIXME comments (technical debt markers)
        todo_pattern = r'#\s*(TODO|FIXME|XXX|HACK|BUG|DEPRECATED)'
        for i, line in enumerate(lines, 1):
            if re.search(todo_pattern, line, re.IGNORECASE):
                self.issues['technical_debt_markers'].append({
                    'file': rel_path,
                    'line': i,
                    'comment': line.strip(),
                    'severity': 'INFO'
                })
        
        # Empty except blocks
        empty_except = r'except.*:\s*pass'
        for match in re.finditer(empty_except, content):
            line_no = content[:match.start()].count('\n') + 1
            self.issues['empty_except'].append({
                'file': rel_path,
                'line': line_no,
                'severity': 'MAJOR'
            })
        
        # Long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                self.issues['long_lines'].append({
                    'file': rel_path,
                    'line': i,
                    'length': len(line),
                    'severity': 'INFO'
                })
        
        # Multiple imports on same line
        multi_import = r'^import\s+\w+\s*,\s*\w+'
        for i, line in enumerate(lines, 1):
            if re.match(multi_import, line.strip()):
                self.issues['multiple_imports'].append({
                    'file': rel_path,
                    'line': i,
                    'severity': 'MINOR'
                })
    
    def _analyze_docker_file(self, file_path: Path):
        """Analyze Docker configurations for issues"""
        rel_path = str(file_path.relative_to(self.root_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            if file_path.name.startswith('docker-compose'):
                self._analyze_docker_compose(content, rel_path)
            elif 'Dockerfile' in file_path.name:
                self._analyze_dockerfile(content, rel_path)
                
        except Exception as e:
            self.issues['docker_errors'].append({
                'file': rel_path,
                'error': str(e),
                'severity': 'MAJOR'
            })
    
    def _analyze_docker_compose(self, content: str, file_path: str):
        """Analyze docker-compose files"""
        # Check for version consistency
        if 'version:' in content:
            version_match = re.search(r'version:\s*["\']?(\d+\.?\d*)', content)
            if version_match:
                version = version_match.group(1)
                if float(version) < 3.0:
                    self.issues['docker_compose'].append({
                        'file': file_path,
                        'issue': f'Outdated compose version: {version}',
                        'severity': 'MINOR'
                    })
        
        # Check for missing health checks
        services = re.findall(r'^\s{2}(\w+):', content, re.MULTILINE)
        for service in services:
            if f'{service}:' in content and 'healthcheck:' not in content[content.index(f'{service}:'):]:
                self.issues['docker_compose'].append({
                    'file': file_path,
                    'service': service,
                    'issue': 'Missing healthcheck',
                    'severity': 'MINOR'
                })
    
    def _analyze_dockerfile(self, content: str, file_path: str):
        """Analyze Dockerfiles for best practices"""
        lines = content.splitlines()
        
        # Check for multiple RUN commands (layer optimization)
        run_count = sum(1 for line in lines if line.strip().startswith('RUN'))
        if run_count > 5:
            self.issues['dockerfile'].append({
                'file': file_path,
                'issue': f'Too many RUN commands ({run_count}) - consider combining',
                'severity': 'MINOR'
            })
        
        # Check for apt-get update without clean
        if 'apt-get update' in content and 'apt-get clean' not in content:
            self.issues['dockerfile'].append({
                'file': file_path,
                'issue': 'apt-get update without cleanup',
                'severity': 'MINOR'
            })
        
        # Check for running as root
        if 'USER' not in content:
            self.issues['dockerfile'].append({
                'file': file_path,
                'issue': 'Container runs as root - consider adding USER',
                'severity': 'MAJOR'
            })
    
    def _detect_duplicate_code(self):
        """Detect duplicate code blocks"""
        # File-level duplicates
        for file_hash, files in self.file_hashes.items():
            if len(files) > 1:
                self.issues['duplicate_files'].append({
                    'files': files,
                    'severity': 'MAJOR'
                })
        
        # Function signature duplicates
        for sig, locations in self.function_signatures.items():
            if len(locations) > 3:  # More than 3 similar signatures
                self.issues['duplicate_functions'].append({
                    'signature': sig,
                    'locations': locations[:10],  # Limit to 10 examples
                    'total_count': len(locations),
                    'severity': 'MINOR'
                })
    
    def _calculate_technical_debt(self):
        """Calculate technical debt in minutes"""
        debt_minutes = 0
        
        # Time estimates per issue type (in minutes)
        debt_map = {
            'syntax_errors': 30,
            'security': 60,
            'high_complexity': 45,
            'god_class': 120,
            'empty_except': 15,
            'duplicate_files': 60,
            'duplicate_functions': 30,
            'long_file': 30,
            'technical_debt_markers': 20,
            'docker_compose': 15,
            'dockerfile': 20
        }
        
        for issue_type, minutes in debt_map.items():
            if issue_type in self.issues:
                debt_minutes += len(self.issues[issue_type]) * minutes
        
        self.metrics['technical_debt_minutes'] = debt_minutes
        self.metrics['technical_debt_days'] = round(debt_minutes / 480, 1)  # 8 hour days
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive quality report"""
        # Count issues by severity
        severity_counts = Counter()
        for issue_list in self.issues.values():
            for issue in issue_list:
                if isinstance(issue, dict) and 'severity' in issue:
                    severity_counts[issue['severity']] += 1
        
        # Calculate quality ratings (A-E scale like SonarQube)
        total_issues = sum(severity_counts.values())
        
        # Maintainability Rating
        debt_ratio = self.metrics['technical_debt_minutes'] / max(self.metrics['code_lines'], 1) * 100
        if debt_ratio < 5:
            maintainability_rating = 'A'
        elif debt_ratio < 10:
            maintainability_rating = 'B'
        elif debt_ratio < 20:
            maintainability_rating = 'C'
        elif debt_ratio < 50:
            maintainability_rating = 'D'
        else:
            maintainability_rating = 'E'
        
        # Reliability Rating (based on bugs)
        bugs = severity_counts.get('BLOCKER', 0) + severity_counts.get('CRITICAL', 0)
        if bugs == 0:
            reliability_rating = 'A'
        elif bugs <= 1:
            reliability_rating = 'B'
        elif bugs <= 5:
            reliability_rating = 'C'
        elif bugs <= 10:
            reliability_rating = 'D'
        else:
            reliability_rating = 'E'
        
        # Security Rating
        security_issues = len(self.issues.get('security', []))
        if security_issues == 0:
            security_rating = 'A'
        elif security_issues <= 1:
            security_rating = 'B'
        elif security_issues <= 5:
            security_rating = 'C'
        elif security_issues <= 10:
            security_rating = 'D'
        else:
            security_rating = 'E'
        
        return {
            'summary': {
                'total_files_analyzed': self.metrics['total_files'],
                'total_lines': self.metrics['total_lines'],
                'code_lines': self.metrics['code_lines'],
                'comment_lines': self.metrics['comment_lines'],
                'blank_lines': self.metrics['blank_lines'],
                'comment_ratio': round(self.metrics['comment_lines'] / max(self.metrics['code_lines'], 1) * 100, 1),
                'technical_debt_days': self.metrics['technical_debt_days'],
                'total_issues': total_issues,
                'issues_by_severity': dict(severity_counts),
                'quality_gates': {
                    'maintainability_rating': maintainability_rating,
                    'reliability_rating': reliability_rating,
                    'security_rating': security_rating,
                    'overall_rating': max(maintainability_rating, reliability_rating, security_rating)
                }
            },
            'issues': dict(self.issues),
            'metrics': self.metrics
        }

def main():
    analyzer = CodeQualityAnalyzer('/opt/sutazaiapp')
    report = analyzer.analyze()
    
    # Save detailed report
    with open('/opt/sutazaiapp/sonarqube_quality_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 SONARQUBE-STYLE CODE QUALITY REPORT")
    print("="*80)
    
    summary = report['summary']
    print(f"\n📈 Overall Metrics:")
    print(f"  • Files Analyzed: {summary['total_files_analyzed']}")
    print(f"  • Total Lines: {summary['total_lines']:,}")
    print(f"  • Code Lines: {summary['code_lines']:,}")
    print(f"  • Comment Ratio: {summary['comment_ratio']}%")
    print(f"  • Technical Debt: {summary['technical_debt_days']} days")
    
    print(f"\n🎯 Quality Gates:")
    gates = summary['quality_gates']
    rating_emoji = {'A': '✅', 'B': '🟢', 'C': '🟡', 'D': '🟠', 'E': '🔴'}
    print(f"  • Maintainability: {rating_emoji[gates['maintainability_rating']]} {gates['maintainability_rating']}")
    print(f"  • Reliability: {rating_emoji[gates['reliability_rating']]} {gates['reliability_rating']}")
    print(f"  • Security: {rating_emoji[gates['security_rating']]} {gates['security_rating']}")
    print(f"  • Overall: {rating_emoji[gates['overall_rating']]} {gates['overall_rating']}")
    
    print(f"\n⚠️ Issues by Severity:")
    for severity in ['BLOCKER', 'CRITICAL', 'MAJOR', 'MINOR', 'INFO']:
        count = summary['issues_by_severity'].get(severity, 0)
        if count > 0:
            print(f"  • {severity}: {count}")
    
    print(f"\n🔍 Top Issues to Address:")
    
    # Print critical issues
    if 'security' in report['issues'] and report['issues']['security']:
        print(f"\n  🔒 Security Issues ({len(report['issues']['security'])} found):")
        for issue in report['issues']['security'][:3]:
            print(f"    - {issue['file']}: {issue['issue']} (line {issue['line']})")
    
    if 'syntax_errors' in report['issues'] and report['issues']['syntax_errors']:
        print(f"\n  ❌ Syntax Errors ({len(report['issues']['syntax_errors'])} found):")
        for issue in report['issues']['syntax_errors'][:3]:
            print(f"    - {issue['file']}: {issue['error']}")
    
    if 'high_complexity' in report['issues'] and report['issues']['high_complexity']:
        print(f"\n  🌀 High Complexity ({len(report['issues']['high_complexity'])} found):")
        for issue in report['issues']['high_complexity'][:3]:
            print(f"    - {issue['file']}:{issue['function']} (complexity: {issue['complexity']})")
    
    if 'duplicate_files' in report['issues'] and report['issues']['duplicate_files']:
        print(f"\n  📑 Duplicate Files ({len(report['issues']['duplicate_files'])} sets found):")
        for dup_set in report['issues']['duplicate_files'][:2]:
            print(f"    - Files: {', '.join(dup_set['files'][:3])}")
    
    print(f"\n💾 Full report saved to: sonarqube_quality_report.json")
    print("="*80)
    
    return report

if __name__ == "__main__":
    main()