#\!/usr/bin/env python3
"""
Critical Quality Assessment Tool (ANAL-001)
Comprehensive static analysis for SutazAI codebase
"""

import ast
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

@dataclass
class SecurityIssue:
    """Security vulnerability finding"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    type: str
    file: str
    line: int
    code: str
    description: str
    remediation: str

@dataclass
class CodeSmell:
    """Code quality issue"""
    severity: str
    type: str
    file: str
    line: int
    issue: str
    fix: str

@dataclass
class DuplicateBlock:
    """Duplicate code block"""
    files: List[str]
    lines: Dict[str, Tuple[int, int]]
    size: int
    hash: str

class CodeQualityAnalyzer:
    """Comprehensive code quality analyzer"""
    
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.issues = {
            'security': [],
            'code_smells': [],
            'duplicates': [],
            'complexity': [],
            'dead_code': [],
            'unused_imports': [],
            'type_errors': [],
            'performance': []
        }
        self.metrics = {
            'total_files': 0,
            'total_lines': 0,
            'python_files': 0,
            'javascript_files': 0,
            'yaml_files': 0,
            'docker_files': 0,
            'fantasy_elements': 0,
            'hardcoded_secrets': 0,
            'unorganized_scripts': 0
        }
        self.file_hashes = defaultdict(list)
        
    def analyze(self) -> Dict[str, Any]:
        """Run comprehensive analysis"""
        print("🔍 Starting Critical Quality Assessment...")
        
        # Scan all files
        for py_file in self.root_dir.rglob("*.py"):
            if not self._should_skip(py_file):
                self.analyze_python_file(py_file)
                
        for yaml_file in self.root_dir.rglob("*.y*ml"):
            if not self._should_skip(yaml_file):
                self.analyze_yaml_file(yaml_file)
                
        for dockerfile in self.root_dir.rglob("Dockerfile*"):
            if not self._should_skip(dockerfile):
                self.analyze_dockerfile(dockerfile)
        
        # Check for unorganized scripts
        self.check_script_organization()
        
        # Find duplicate code
        self.find_duplicates()
        
        # Calculate metrics
        self.calculate_metrics()
        
        return self.generate_report()
    
    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_dirs = {'__pycache__', 'node_modules', '.git', 'venv', '.venv', 'htmlcov', '.pytest_cache'}
        return any(part in skip_dirs for part in file_path.parts)
    
    def analyze_python_file(self, file_path: Path):
        """Analyze Python file for issues"""
        self.metrics['python_files'] += 1
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            self.metrics['total_lines'] += len(lines)
            
            # Check for hardcoded credentials
            self.check_hardcoded_secrets(file_path, content)
            
            # Check for fantasy elements
            self.check_fantasy_elements(file_path, content)
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self.analyze_ast(tree, file_path)
            except SyntaxError as e:
                self.issues['code_smells'].append(CodeSmell(
                    severity='HIGH',
                    type='SYNTAX_ERROR',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=e.lineno or 0,
                    issue=f"Syntax error: {e.msg}",
                    fix="Fix syntax error to enable proper analysis"
                ))
            
            # Check complexity
            self.check_complexity(file_path, content)
            
            # Check for code smells
            self.check_code_smells(file_path, lines)
            
            # Store file hash for duplicate detection
            file_hash = hash(content)
            self.file_hashes[file_hash].append(str(file_path.relative_to(self.root_dir)))
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def check_hardcoded_secrets(self, file_path: Path, content: str):
        """Check for hardcoded credentials and secrets"""
        patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\']([^"\']+)["\']', 'Hardcoded secret'),
            (r'token\s*=\s*["\']([^"\']+)["\']', 'Hardcoded token'),
            (r'aws_access_key_id\s*=\s*["\']([^"\']+)["\']', 'AWS credentials'),
            (r'private_key\s*=\s*["\']([^"\']+)["\']', 'Private key exposed'),
            (r'DATABASE_URL\s*=\s*["\']([^"\']+)["\']', 'Database URL hardcoded'),
            (r'POSTGRES_PASSWORD\s*=\s*["\']([^"\']+)["\']', 'Database password'),
        ]
        
        for line_num, line in enumerate(content.splitlines(), 1):
            for pattern, desc in patterns:
                if match := re.search(pattern, line, re.IGNORECASE):
                    value = match.group(1)
                    # Skip obvious placeholders
                    if value not in ['', 'xxx', 'placeholder', 'example', 'test', 'TODO']:
                        self.metrics['hardcoded_secrets'] += 1
                        self.issues['security'].append(SecurityIssue(
                            severity='CRITICAL',
                            type='HARDCODED_SECRET',
                            file=str(file_path.relative_to(self.root_dir)),
                            line=line_num,
                            code=line.strip()[:100],
                            description=f"{desc}: '{value[:20]}...'",
                            remediation="Use environment variables or secure secret management (e.g., HashiCorp Vault)"
                        ))
    
    def check_fantasy_elements(self, file_path: Path, content: str):
        """Check for fantasy/fictional elements"""
        fantasy_keywords = [
            'advanced', 'AGI', 'Advanced System', 'transfer', 'process', 'configurator',
            'consciousness', 'sentient', 'self-aware', 'singularity',
            'brain-computer', 'neural-link', 'mind-meld', 'telepathy'
        ]
        
        for keyword in fantasy_keywords:
            if keyword.lower() in content.lower():
                self.metrics['fantasy_elements'] += 1
                self.issues['code_smells'].append(CodeSmell(
                    severity='HIGH',
                    type='FANTASY_ELEMENT',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=0,
                    issue=f"Fantasy element detected: '{keyword}'",
                    fix="Remove speculative/fictional features and implement real functionality"
                ))
                break
    
    def analyze_ast(self, tree: ast.AST, file_path: Path):
        """Analyze Python AST for issues"""
        
        class Analyzer(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.imports = set()
                self.used_names = set()
                
            def visit_Import(self, node):
                for alias in node.names:
                    self.imports.add(alias.name)
                    
            def visit_ImportFrom(self, node):
                if node.module:
                    self.imports.add(node.module)
                    
            def visit_Name(self, node):
                self.used_names.add(node.id)
                
            def visit_FunctionDef(self, node):
                # Check function complexity
                complexity = self.calculate_complexity(node)
                if complexity > 10:
                    self.parent.issues['complexity'].append(CodeSmell(
                        severity='HIGH' if complexity > 15 else 'MEDIUM',
                        type='HIGH_COMPLEXITY',
                        file=str(file_path.relative_to(self.parent.root_dir)),
                        line=node.lineno,
                        issue=f"Function '{node.name}' has complexity {complexity}",
                        fix="Refactor into smaller functions with single responsibilities"
                    ))
                self.generic_visit(node)
                
            def calculate_complexity(self, node):
                """Calculate cyclomatic complexity"""
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                        complexity += 1
                    elif isinstance(child, ast.BoolOp):
                        complexity += len(child.values) - 1
                return complexity
        
        analyzer = Analyzer(self)
        analyzer.visit(tree)
        
        # Check for unused imports
        for imp in analyzer.imports:
            if imp not in str(tree):  # Simple check, could be improved
                self.issues['unused_imports'].append(CodeSmell(
                    severity='LOW',
                    type='UNUSED_IMPORT',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=1,
                    issue=f"Unused import: {imp}",
                    fix=f"Remove unused import: {imp}"
                ))
    
    def check_complexity(self, file_path: Path, content: str):
        """Check code complexity metrics"""
        lines = content.splitlines()
        
        # Check file length
        if len(lines) > 500:
            self.issues['code_smells'].append(CodeSmell(
                severity='MEDIUM',
                type='LARGE_FILE',
                file=str(file_path.relative_to(self.root_dir)),
                line=0,
                issue=f"File has {len(lines)} lines (>500)",
                fix="Split into smaller, focused modules"
            ))
        
        # Check for long functions
        in_function = False
        func_start = 0
        func_name = ""
        func_lines = 0
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                if in_function and func_lines > 50:
                    self.issues['complexity'].append(CodeSmell(
                        severity='MEDIUM',
                        type='LONG_FUNCTION',
                        file=str(file_path.relative_to(self.root_dir)),
                        line=func_start,
                        issue=f"Function '{func_name}' has {func_lines} lines (>50)",
                        fix="Break down into smaller functions"
                    ))
                in_function = True
                func_start = i
                func_name = line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                func_lines = 0
            elif in_function:
                if line and not line[0].isspace():
                    in_function = False
                else:
                    func_lines += 1
    
    def check_code_smells(self, file_path: Path, lines: List[str]):
        """Check for common code smells"""
        
        for i, line in enumerate(lines, 1):
            # Check for print statements (should use logging)
            if re.search(r'^\s*print\s*\(', line):
                self.issues['code_smells'].append(CodeSmell(
                    severity='LOW',
                    type='PRINT_STATEMENT',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=i,
                    issue="Using print() instead of logging",
                    fix="Replace with proper logging (logger.info/debug/error)"
                ))
            
            # Check for bare except
            if re.search(r'^\s*except\s*:', line):
                self.issues['code_smells'].append(CodeSmell(
                    severity='HIGH',
                    type='BARE_EXCEPT',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=i,
                    issue="Bare except clause catches all exceptions",
                    fix="Catch specific exceptions (e.g., except ValueError:)"
                ))
            
            # Check for TODO/FIXME comments
            if 'TODO' in line or 'FIXME' in line:
                self.issues['code_smells'].append(CodeSmell(
                    severity='LOW',
                    type='TODO_COMMENT',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=i,
                    issue=f"Unresolved TODO/FIXME: {line.strip()[:50]}",
                    fix="Complete the TODO or create a tracked issue"
                ))
            
            # Check for long lines
            if len(line) > 120:
                self.issues['code_smells'].append(CodeSmell(
                    severity='LOW',
                    type='LONG_LINE',
                    file=str(file_path.relative_to(self.root_dir)),
                    line=i,
                    issue=f"Line too long ({len(line)} > 120 characters)",
                    fix="Break line according to PEP 8 guidelines"
                ))
    
    def analyze_yaml_file(self, file_path: Path):
        """Analyze YAML files for issues"""
        self.metrics['yaml_files'] += 1
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Check for hardcoded secrets in YAML
            self.check_hardcoded_secrets(file_path, content)
            
            # Check docker-compose specific issues
            if 'docker-compose' in file_path.name:
                self.check_docker_compose_issues(file_path, content)
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def check_docker_compose_issues(self, file_path: Path, content: str):
        """Check docker-compose files for issues"""
        issues = []
        
        # Check for missing health checks
        if 'healthcheck:' not in content:
            issues.append("Missing health checks for containers")
        
        # Check for hardcoded ports
        if re.search(r'ports:\s*\n\s*-\s*"\d+:\d+"', content):
            issues.append("Hardcoded ports (use environment variables)")
        
        # Check for missing resource limits
        if 'mem_limit' not in content and 'deploy:' not in content:
            issues.append("Missing memory limits for containers")
        
        for issue in issues:
            self.issues['code_smells'].append(CodeSmell(
                severity='MEDIUM',
                type='DOCKER_COMPOSE_ISSUE',
                file=str(file_path.relative_to(self.root_dir)),
                line=0,
                issue=issue,
                fix=f"Add {issue.lower().replace('missing ', '')}"
            ))
    
    def analyze_dockerfile(self, file_path: Path):
        """Analyze Dockerfiles for issues"""
        self.metrics['docker_files'] += 1
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            
            # Check for security issues
            for i, line in enumerate(lines, 1):
                # Running as root
                if 'USER root' in line:
                    self.issues['security'].append(SecurityIssue(
                        severity='HIGH',
                        type='DOCKER_ROOT_USER',
                        file=str(file_path.relative_to(self.root_dir)),
                        line=i,
                        code=line.strip(),
                        description="Container running as root user",
                        remediation="Create and use a non-root user"
                    ))
                
                # Using latest tag
                if re.search(r'FROM\s+\S+:latest', line):
                    self.issues['security'].append(SecurityIssue(
                        severity='MEDIUM',
                        type='DOCKER_LATEST_TAG',
                        file=str(file_path.relative_to(self.root_dir)),
                        line=i,
                        code=line.strip(),
                        description="Using :latest tag (non-reproducible builds)",
                        remediation="Pin to specific version (e.g., python:3.11-slim)"
                    ))
                
                # apt-get without cleanup
                if 'apt-get install' in line and '&& rm -rf /var/lib/apt/lists/*' not in content:
                    self.issues['performance'].append(CodeSmell(
                        severity='MEDIUM',
                        type='DOCKER_APT_CACHE',
                        file=str(file_path.relative_to(self.root_dir)),
                        line=i,
                        issue="apt-get without cleaning cache",
                        fix="Add && rm -rf /var/lib/apt/lists/* to reduce image size"
                    ))
                    
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def check_script_organization(self):
        """Check if scripts are properly organized"""
        script_locations = defaultdict(list)
        
        # Find all Python scripts
        for py_file in self.root_dir.rglob("*.py"):
            if not self._should_skip(py_file):
                parts = py_file.parts
                if len(parts) > 2:  # Not in root
                    script_locations[parts[-2]].append(str(py_file.relative_to(self.root_dir)))
                else:
                    script_locations['root'].append(str(py_file.relative_to(self.root_dir)))
        
        # Check for unorganized scripts
        if 'root' in script_locations:
            for script in script_locations['root']:
                if script not in ['setup.py', 'manage.py']:  # Allowed root scripts
                    self.metrics['unorganized_scripts'] += 1
                    self.issues['code_smells'].append(CodeSmell(
                        severity='MEDIUM',
                        type='UNORGANIZED_SCRIPT',
                        file=script,
                        line=0,
                        issue="Script in root directory",
                        fix="Move to appropriate directory (scripts/, tools/, etc.)"
                    ))
    
    def find_duplicates(self):
        """Find duplicate code blocks"""
        # Simple file-level duplicate detection
        for file_hash, files in self.file_hashes.items():
            if len(files) > 1:
                self.issues['duplicates'].append(DuplicateBlock(
                    files=files,
                    lines={f: (1, -1) for f in files},
                    size=100,  # Placeholder
                    hash=str(file_hash)
                ))
    
    def calculate_metrics(self):
        """Calculate overall metrics"""
        self.metrics['total_files'] = sum([
            self.metrics['python_files'],
            self.metrics['yaml_files'],
            self.metrics['docker_files']
        ])
        
        # Calculate compliance score
        total_issues = sum(len(issues) for issues in self.issues.values())
        if self.metrics['total_files'] > 0:
            self.metrics['issues_per_file'] = total_issues / self.metrics['total_files']
        else:
            self.metrics['issues_per_file'] = 0
        
        # Rules compliance
        self.metrics['rules_compliance'] = self.calculate_rules_compliance()
    
    def calculate_rules_compliance(self) -> float:
        """Calculate percentage compliance with CLAUDE.md rules"""
        total_checks = 10
        passed_checks = 10
        
        # Check 1: No fantasy elements
        if self.metrics['fantasy_elements'] > 0:
            passed_checks -= 1
        
        # Check 2: No hardcoded secrets
        if self.metrics['hardcoded_secrets'] > 0:
            passed_checks -= 2  # Critical issue
        
        # Check 3: Organized scripts
        if self.metrics['unorganized_scripts'] > 10:
            passed_checks -= 1
        
        # Check 4: No duplicate code
        if len(self.issues['duplicates']) > 5:
            passed_checks -= 1
        
        # Check 5: Low complexity
        if len(self.issues['complexity']) > 20:
            passed_checks -= 1
        
        # Check 6: No bare excepts
        bare_excepts = sum(1 for issue in self.issues['code_smells'] 
                          if issue.type == 'BARE_EXCEPT')
        if bare_excepts > 5:
            passed_checks -= 1
        
        # Check 7: Proper logging
        print_statements = sum(1 for issue in self.issues['code_smells']
                              if issue.type == 'PRINT_STATEMENT')
        if print_statements > 20:
            passed_checks -= 1
        
        # Check 8: Docker security
        docker_issues = len([i for i in self.issues['security'] 
                           if 'DOCKER' in i.type])
        if docker_issues > 3:
            passed_checks -= 1
        
        # Check 9: No syntax errors
        syntax_errors = sum(1 for issue in self.issues['code_smells']
                          if issue.type == 'SYNTAX_ERROR')
        if syntax_errors > 0:
            passed_checks -= 1
        
        return (passed_checks / total_checks) * 100
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        
        # Count issues by severity
        severity_counts = defaultdict(int)
        for category_issues in self.issues.values():
            for issue in category_issues:
                if hasattr(issue, 'severity'):
                    severity_counts[issue.severity] += 1
        
        # Generate fix priorities
        priorities = self.generate_fix_priorities()
        
        report = {
            'summary': {
                'compliance_score': f"{self.metrics['rules_compliance']:.1f}%",
                'total_issues': sum(len(issues) for issues in self.issues.values()),
                'critical_issues': severity_counts['CRITICAL'],
                'high_issues': severity_counts['HIGH'],
                'medium_issues': severity_counts['MEDIUM'],
                'low_issues': severity_counts['LOW'],
                'files_analyzed': self.metrics['total_files'],
                'total_lines': self.metrics['total_lines']
            },
            'metrics': self.metrics,
            'issues': {
                'security': [asdict(i) for i in self.issues['security']],
                'code_smells': [asdict(i) for i in self.issues['code_smells']],
                'duplicates': [asdict(i) for i in self.issues['duplicates']],
                'complexity': [asdict(i) for i in self.issues['complexity']],
                'dead_code': [asdict(i) for i in self.issues['dead_code']],
                'unused_imports': [asdict(i) for i in self.issues['unused_imports']],
                'type_errors': [asdict(i) for i in self.issues['type_errors']],
                'performance': [asdict(i) for i in self.issues['performance']]
            },
            'fix_priorities': priorities
        }
        
        return report
    
    def generate_fix_priorities(self) -> List[Dict[str, Any]]:
        """Generate prioritized list of fixes"""
        priorities = []
        
        # Priority 1: Critical security issues
        for issue in self.issues['security']:
            if issue.severity == 'CRITICAL':
                priorities.append({
                    'priority': 1,
                    'category': 'SECURITY',
                    'issue': f"{issue.file}:{issue.line} - {issue.description}",
                    'fix': issue.remediation,
                    'effort': 'High'
                })
        
        # Priority 2: High severity issues
        for category_issues in self.issues.values():
            for issue in category_issues:
                if hasattr(issue, 'severity') and issue.severity == 'HIGH':
                    priorities.append({
                        'priority': 2,
                        'category': issue.type if hasattr(issue, 'type') else 'QUALITY',
                        'issue': f"{issue.file}:{issue.line if hasattr(issue, 'line') else 0} - {issue.issue if hasattr(issue, 'issue') else issue.description}",
                        'fix': issue.fix if hasattr(issue, 'fix') else issue.remediation if hasattr(issue, 'remediation') else 'Review and fix',
                        'effort': 'Medium'
                    })
        
        # Priority 3: Code organization
        if self.metrics['unorganized_scripts'] > 0:
            priorities.append({
                'priority': 3,
                'category': 'ORGANIZATION',
                'issue': f"{self.metrics['unorganized_scripts']} scripts in root directory",
                'fix': "Move to scripts/ directory and organize by function",
                'effort': 'Low'
            })
        
        # Priority 4: Duplicate code
        if len(self.issues['duplicates']) > 0:
            priorities.append({
                'priority': 4,
                'category': 'DUPLICATION',
                'issue': f"{len(self.issues['duplicates'])} duplicate code blocks found",
                'fix': "Refactor into shared modules/functions",
                'effort': 'Medium'
            })
        
        # Sort by priority
        priorities.sort(key=lambda x: x['priority'])
        
        return priorities[:20]  # Top 20 priorities

def main():
    """Main execution"""
    root_dir = '/opt/sutazaiapp'
    
    analyzer = CodeQualityAnalyzer(root_dir)
    report = analyzer.analyze()
    
    # Save report
    report_file = Path(root_dir) / 'code_quality_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 CODE QUALITY ASSESSMENT SUMMARY")
    print("="*80)
    
    summary = report['summary']
    print(f"\n🎯 Compliance Score: {summary['compliance_score']}")
    print(f"📁 Files Analyzed: {summary['files_analyzed']}")
    print(f"📝 Total Lines: {summary['total_lines']:,}")
    print(f"\n⚠️  Total Issues: {summary['total_issues']}")
    print(f"   🔴 CRITICAL: {summary['critical_issues']}")
    print(f"   🟠 HIGH: {summary['high_issues']}")
    print(f"   🟡 MEDIUM: {summary['medium_issues']}")
    print(f"   🟢 LOW: {summary['low_issues']}")
    
    print(f"\n🔒 Security Issues: {len(report['issues']['security'])}")
    print(f"   - Hardcoded Secrets: {report['metrics']['hardcoded_secrets']}")
    
    print(f"\n🧹 Code Quality:")
    print(f"   - Code Smells: {len(report['issues']['code_smells'])}")
    print(f"   - High Complexity: {len(report['issues']['complexity'])}")
    print(f"   - Duplicate Blocks: {len(report['issues']['duplicates'])}")
    print(f"   - Fantasy Elements: {report['metrics']['fantasy_elements']}")
    print(f"   - Unorganized Scripts: {report['metrics']['unorganized_scripts']}")
    
    print(f"\n📋 Top Fix Priorities:")
    for i, priority in enumerate(report['fix_priorities'][:5], 1):
        print(f"   {i}. [{priority['category']}] {priority['issue'][:80]}...")
        print(f"      Fix: {priority['fix'][:70]}...")
    
    print(f"\n✅ Report saved to: {report_file}")
    
    # Return exit code based on compliance
    if float(summary['compliance_score'].rstrip('%')) < 50:
        print("\n❌ FAILING: Compliance below 50%")
        return 1
    else:
        print("\n✅ Analysis complete")
        return 0

if __name__ == "__main__":
    sys.exit(main())
