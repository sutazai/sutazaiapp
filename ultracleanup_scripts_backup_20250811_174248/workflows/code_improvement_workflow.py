#!/usr/bin/env python3
"""
Code Improvement Workflow
Analyzes code quality and suggests improvements using multiple AI agents
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import ast
import re
from dataclasses import dataclass, field
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeIssue:
    """Represents a code quality issue"""
    file_path: str
    line_number: int
    issue_type: str  # 'performance', 'security', 'style', 'bug', 'complexity'
    severity: str    # 'critical', 'high', 'medium', 'low'
    description: str
    suggested_fix: Optional[str] = None
    agent: Optional[str] = None


@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int = 0
    complexity_score: float = 0.0
    test_coverage: float = 0.0
    code_duplication: float = 0.0
    security_issues: int = 0
    performance_issues: int = 0
    style_violations: int = 0
    documentation_score: float = 0.0


@dataclass
class ImprovementReport:
    """Complete improvement report"""
    directory: str
    timestamp: datetime
    metrics: CodeMetrics
    issues: List[CodeIssue]
    improvements: List[Dict[str, Any]]
    agent_analyses: Dict[str, Any] = field(default_factory=dict)


class CodeAnalyzer:
    """Base code analyzer with common functionality"""
    
    def __init__(self):
        self.file_cache = {}
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file with caching"""
        if file_path not in self.file_cache:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_cache[file_path] = f.read()
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return None
        return self.file_cache[file_path]
    
    def get_python_files(self, directory: str) -> List[str]:
        """Get all Python files in directory"""
        python_files = []
        for root, _, files in os.walk(directory):
            # Skip common directories
            if any(skip in root for skip in ['__pycache__', '.git', 'venv', 'env', 'node_modules']):
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    def analyze_complexity(self, code: str) -> float:
        """Analyze code complexity using AST"""
        try:
            tree = ast.parse(code)
            complexity = 0
            
            for node in ast.walk(tree):
                # Count control flow statements
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += 0.5
                elif isinstance(node, ast.ClassDef):
                    complexity += 2
            
            # Normalize by lines of code
            lines = len(code.splitlines())
            return complexity / max(lines, 1) * 100
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return 0.0
    
    def find_code_patterns(self, code: str, patterns: Dict[str, str]) -> List[Tuple[int, str, str]]:
        """Find code patterns using regex"""
        findings = []
        lines = code.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern_name, pattern_regex in patterns.items():
                if re.search(pattern_regex, line):
                    findings.append((line_num, pattern_name, line.strip()))
        
        return findings


class SeniorAIEngineerAnalyzer(CodeAnalyzer):
    """Analyzer for Senior AI Engineer perspective"""
    
    def analyze(self, directory: str) -> Dict[str, Any]:
        """Analyze code from AI/ML perspective"""
        issues = []
        metrics = CodeMetrics()
        
        # AI/ML specific patterns to check
        ml_patterns = {
            'missing_gpu_check': r'torch\.cuda|tensorflow\.gpu|jax\.device',
            'unchecked_model_load': r'load_model|torch\.load|tf\.keras\.models\.load',
            'missing_batch_size': r'DataLoader\([^)]*\)',
            'hardcoded_hyperparams': r'learning_rate\s*=\s*[\d\.]+|epochs\s*=\s*\d+',
            'missing_gradient_clip': r'optimizer\.step\(\)',
            'no_model_eval': r'model\.eval\(\)|model\.train\(\)',
        }
        
        python_files = self.get_python_files(directory)
        
        for file_path in python_files:
            content = self.read_file(file_path)
            if not content:
                continue
            
            # Check for ML-specific issues
            findings = self.find_code_patterns(content, ml_patterns)
            
            for line_num, pattern_name, line_content in findings:
                issue = CodeIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type='performance' if 'gpu' in pattern_name else 'bug',
                    severity='medium',
                    description=f"ML pattern issue: {pattern_name}",
                    suggested_fix=self._get_ml_fix(pattern_name),
                    agent='senior-ai-engineer'
                )
                issues.append(issue)
            
            # Check for missing docstrings in ML functions
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if any(ml_term in node.name.lower() for ml_term in ['train', 'model', 'predict', 'inference']):
                            if not ast.get_docstring(node):
                                issues.append(CodeIssue(
                                    file_path=file_path,
                                    line_number=node.lineno,
                                    issue_type='style',
                                    severity='low',
                                    description=f"ML function '{node.name}' missing docstring",
                                    suggested_fix="Add comprehensive docstring explaining model behavior",
                                    agent='senior-ai-engineer'
                                ))
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        return {
            'issues': issues,
            'metrics': metrics,
            'recommendations': [
                "Consider implementing model versioning",
                "Add GPU memory monitoring",
                "Implement experiment tracking (MLflow/W&B)",
                "Add model performance benchmarking",
                "Consider using mixed precision training"
            ]
        }
    
    def _get_ml_fix(self, pattern_name: str) -> str:
        """Get ML-specific fix suggestions"""
        fixes = {
            'missing_gpu_check': "Add device check: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            'unchecked_model_load': "Wrap model loading in try-except and verify model architecture",
            'missing_batch_size': "Specify batch_size parameter in DataLoader",
            'hardcoded_hyperparams': "Move hyperparameters to configuration file or use argparse",
            'missing_gradient_clip': "Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
            'no_model_eval': "Set model mode appropriately: model.eval() for inference, model.train() for training"
        }
        return fixes.get(pattern_name, "Review and fix ML implementation")


class TestingQAValidatorAnalyzer(CodeAnalyzer):
    """Analyzer for Testing/QA perspective"""
    
    def analyze(self, directory: str) -> Dict[str, Any]:
        """Analyze code from testing/QA perspective"""
        issues = []
        metrics = CodeMetrics()
        
        # Testing patterns to check
        test_patterns = {
            'missing_error_handling': r'except\s*:',
            'bare_except': r'except\s+Exception\s*:',
            'missing_input_validation': r'def\s+\w+\([^)]+\):(?!.*if.*is None)',
            'hardcoded_values': r'["\'](?:localhost|127\.0\.0\.1|8080|3000)["\']',
            'no_assertions': r'def\s+test_\w+.*(?!.*assert)',
            'print_debugging': r'print\s*\(',
        }
        
        python_files = self.get_python_files(directory)
        test_files = [f for f in python_files if 'test' in f.lower()]
        
        # Calculate test coverage estimate
        metrics.test_coverage = len(test_files) / max(len(python_files), 1) * 100
        
        for file_path in python_files:
            content = self.read_file(file_path)
            if not content:
                continue
            
            findings = self.find_code_patterns(content, test_patterns)
            
            for line_num, pattern_name, line_content in findings:
                severity = 'high' if 'error' in pattern_name else 'medium'
                issue = CodeIssue(
                    file_path=file_path,
                    line_number=line_num,
                    issue_type='bug' if 'error' in pattern_name else 'style',
                    severity=severity,
                    description=f"QA issue: {pattern_name}",
                    suggested_fix=self._get_qa_fix(pattern_name),
                    agent='testing-qa-validator'
                )
                issues.append(issue)
            
            # Check for missing tests
            if '/test' not in file_path and not file_path.endswith('_test.py'):
                module_name = os.path.basename(file_path)[:-3]
                test_file = file_path.replace(module_name + '.py', f'test_{module_name}.py')
                if not os.path.exists(test_file):
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=0,
                        issue_type='testing',
                        severity='medium',
                        description=f"Missing test file for {module_name}",
                        suggested_fix=f"Create test file: {test_file}",
                        agent='testing-qa-validator'
                    ))
        
        return {
            'issues': issues,
            'metrics': metrics,
            'recommendations': [
                "Implement comprehensive unit test suite",
                "Add integration tests for API endpoints",
                "Set up continuous integration pipeline",
                "Add property-based testing for complex functions",
                "Implement test coverage reporting"
            ]
        }
    
    def _get_qa_fix(self, pattern_name: str) -> str:
        """Get QA-specific fix suggestions"""
        fixes = {
            'missing_error_handling': "Add specific exception handling with proper error messages",
            'bare_except': "Catch specific exceptions instead of broad Exception",
            'missing_input_validation': "Add input validation at function entry",
            'hardcoded_values': "Move hardcoded values to configuration",
            'no_assertions': "Add assertions to verify test outcomes",
            'print_debugging': "Replace print with proper logging"
        }
        return fixes.get(pattern_name, "Review and fix QA issue")


class InfrastructureDevOpsAnalyzer(CodeAnalyzer):
    """Analyzer for Infrastructure/DevOps perspective"""
    
    def analyze(self, directory: str) -> Dict[str, Any]:
        """Analyze code from DevOps perspective"""
        issues = []
        metrics = CodeMetrics()
        
        # Check for Docker files
        docker_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file == 'Dockerfile' or file.endswith('.dockerfile'):
                    docker_files.append(os.path.join(root, file))
        
        # Analyze Dockerfiles
        for docker_file in docker_files:
            content = self.read_file(docker_file)
            if not content:
                continue
            
            # Check for Docker best practices
            if 'apt-get update' in content and 'apt-get install' not in content.replace('\n', ' '):
                issues.append(CodeIssue(
                    file_path=docker_file,
                    line_number=0,
                    issue_type='performance',
                    severity='medium',
                    description="Separate RUN commands for apt-get update and install",
                    suggested_fix="Combine apt-get update && apt-get install in single RUN",
                    agent='infrastructure-devops-manager'
                ))
            
            if 'COPY . .' in content or 'ADD . .' in content:
                issues.append(CodeIssue(
                    file_path=docker_file,
                    line_number=0,
                    issue_type='performance',
                    severity='high',
                    description="Copying entire directory invalidates Docker cache",
                    suggested_fix="Copy only necessary files and use .dockerignore",
                    agent='infrastructure-devops-manager'
                ))
        
        # Check for configuration issues
        config_patterns = {
            'hardcoded_secrets': r'(password|secret|api_key)\s*=\s*["\'][^"\']+["\']',
            'missing_env_vars': r'os\.environ\[',
            'no_health_checks': r'@app\.(get|post)\(["\']\/health',
        }
        
        python_files = self.get_python_files(directory)
        
        for file_path in python_files:
            content = self.read_file(file_path)
            if not content:
                continue
            
            findings = self.find_code_patterns(content, config_patterns)
            
            for line_num, pattern_name, line_content in findings:
                if pattern_name == 'hardcoded_secrets':
                    issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        issue_type='security',
                        severity='critical',
                        description="Hardcoded secrets detected",
                        suggested_fix="Use environment variables or secret management service",
                        agent='infrastructure-devops-manager'
                    ))
        
        return {
            'issues': issues,
            'metrics': metrics,
            'recommendations': [
                "Implement multi-stage Docker builds",
                "Add health check endpoints",
                "Use environment-specific configurations",
                "Implement proper logging and monitoring",
                "Add container orchestration configs (K8s/Docker Compose)"
            ]
        }


class CodeImprovementWorkflow:
    """Main workflow orchestrator for code improvement"""
    
    def __init__(self):
        self.analyzers = {
            'senior-ai-engineer': SeniorAIEngineerAnalyzer(),
            'testing-qa-validator': TestingQAValidatorAnalyzer(),
            'infrastructure-devops-manager': InfrastructureDevOpsAnalyzer()
        }
        self.coordinator = None
    
    async def initialize(self):
        """Initialize the workflow"""
        # Coordinator initialization is optional
        self.coordinator = None
    
    async def analyze_directory(self, directory: str) -> ImprovementReport:
        """Analyze a directory and generate improvement report"""
        logger.info(f"Starting code analysis for: {directory}")
        
        # Verify directory exists
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # Initialize report
        report = ImprovementReport(
            directory=directory,
            timestamp=datetime.now(),
            metrics=CodeMetrics(),
            issues=[],
            improvements=[]
        )
        
        # Run all analyzers
        for agent_name, analyzer in self.analyzers.items():
            logger.info(f"Running analysis with {agent_name}")
            
            try:
                analysis = analyzer.analyze(directory)
                
                # Collect issues
                report.issues.extend(analysis.get('issues', []))
                
                # Store agent-specific analysis
                report.agent_analyses[agent_name] = analysis
                
                # Add recommendations as improvements
                for recommendation in analysis.get('recommendations', []):
                    report.improvements.append({
                        'agent': agent_name,
                        'type': 'recommendation',
                        'description': recommendation,
                        'priority': 'medium'
                    })
                
            except Exception as e:
                logger.error(f"Error in {agent_name} analysis: {e}")
        
        # Calculate overall metrics
        report.metrics = self._calculate_metrics(directory, report.issues)
        
        # Sort issues by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        report.issues.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        # Generate specific improvements from issues
        report.improvements.extend(self._generate_improvements(report.issues))
        
        return report
    
    def _calculate_metrics(self, directory: str, issues: List[CodeIssue]) -> CodeMetrics:
        """Calculate overall code metrics"""
        metrics = CodeMetrics()
        
        # Count lines of code
        python_files = CodeAnalyzer().get_python_files(directory)
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    metrics.lines_of_code += len(f.readlines())
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        # Count issues by type
        for issue in issues:
            if issue.issue_type == 'security':
                metrics.security_issues += 1
            elif issue.issue_type == 'performance':
                metrics.performance_issues += 1
            elif issue.issue_type == 'style':
                metrics.style_violations += 1
        
        # Calculate complexity score (average)
        analyzer = CodeAnalyzer()
        complexities = []
        for file_path in python_files[:10]:  # Sample first 10 files
            content = analyzer.read_file(file_path)
            if content:
                complexity = analyzer.analyze_complexity(content)
                complexities.append(complexity)
        
        if complexities:
            metrics.complexity_score = sum(complexities) / len(complexities)
        
        return metrics
    
    def _generate_improvements(self, issues: List[CodeIssue]) -> List[Dict[str, Any]]:
        """Generate specific improvements from issues"""
        improvements = []
        
        # Group issues by file
        issues_by_file = defaultdict(list)
        for issue in issues:
            issues_by_file[issue.file_path].append(issue)
        
        # Generate file-specific improvements
        for file_path, file_issues in issues_by_file.items():
            if len(file_issues) > 5:
                improvements.append({
                    'type': 'refactor',
                    'file': file_path,
                    'description': f"Refactor {file_path} - {len(file_issues)} issues found",
                    'priority': 'high',
                    'issues': [issue.description for issue in file_issues[:5]]
                })
        
        # Add critical fixes
        critical_issues = [i for i in issues if i.severity == 'critical']
        for issue in critical_issues[:10]:  # Top 10 critical
            improvements.append({
                'type': 'fix',
                'file': issue.file_path,
                'line': issue.line_number,
                'description': issue.description,
                'suggested_fix': issue.suggested_fix,
                'priority': 'critical'
            })
        
        return improvements
    
    def generate_report_text(self, report: ImprovementReport) -> str:
        """Generate human-readable report"""
        lines = []
        
        lines.append("# Code Improvement Report")
        lines.append(f"\nDirectory: {report.directory}")
        lines.append(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Metrics section
        lines.append("\n## Code Metrics")
        lines.append(f"- Lines of Code: {report.metrics.lines_of_code:,}")
        lines.append(f"- Complexity Score: {report.metrics.complexity_score:.2f}")
        lines.append(f"- Security Issues: {report.metrics.security_issues}")
        lines.append(f"- Performance Issues: {report.metrics.performance_issues}")
        lines.append(f"- Style Violations: {report.metrics.style_violations}")
        
        # Issues summary
        lines.append("\n## Issues Summary")
        severity_counts = defaultdict(int)
        for issue in report.issues:
            severity_counts[issue.severity] += 1
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severity_counts:
                lines.append(f"- {severity.capitalize()}: {severity_counts[severity]}")
        
        # Top issues
        lines.append("\n## Top Issues")
        for i, issue in enumerate(report.issues[:10], 1):
            lines.append(f"\n### {i}. [{issue.severity.upper()}] {issue.description}")
            lines.append(f"   - File: {issue.file_path}:{issue.line_number}")
            lines.append(f"   - Type: {issue.issue_type}")
            if issue.suggested_fix:
                lines.append(f"   - Fix: {issue.suggested_fix}")
            lines.append(f"   - Found by: {issue.agent}")
        
        # Improvements
        lines.append("\n## Recommended Improvements")
        priority_improvements = defaultdict(list)
        for improvement in report.improvements:
            priority_improvements[improvement.get('priority', 'medium')].append(improvement)
        
        for priority in ['critical', 'high', 'medium', 'low']:
            if priority in priority_improvements:
                lines.append(f"\n### {priority.capitalize()} Priority")
                for imp in priority_improvements[priority][:5]:
                    lines.append(f"- {imp['description']}")
                    if 'file' in imp:
                        lines.append(f"  - File: {imp['file']}")
        
        # Agent recommendations
        lines.append("\n## Agent Recommendations")
        for agent, analysis in report.agent_analyses.items():
            if 'recommendations' in analysis:
                lines.append(f"\n### {agent}")
                for rec in analysis['recommendations'][:3]:
                    lines.append(f"- {rec}")
        
        return '\n'.join(lines)
    
    def save_report(self, report: ImprovementReport, output_file: str):
        """Save report to file"""
        # Save text report
        with open(output_file, 'w') as f:
            f.write(self.generate_report_text(report))
        
        # Save JSON report
        json_file = output_file.replace('.md', '.json')
        with open(json_file, 'w') as f:
            json.dump({
                'directory': report.directory,
                'timestamp': report.timestamp.isoformat(),
                'metrics': {
                    'lines_of_code': report.metrics.lines_of_code,
                    'complexity_score': report.metrics.complexity_score,
                    'security_issues': report.metrics.security_issues,
                    'performance_issues': report.metrics.performance_issues,
                    'style_violations': report.metrics.style_violations
                },
                'issues': [
                    {
                        'file': issue.file_path,
                        'line': issue.line_number,
                        'type': issue.issue_type,
                        'severity': issue.severity,
                        'description': issue.description,
                        'fix': issue.suggested_fix,
                        'agent': issue.agent
                    }
                    for issue in report.issues
                ],
                'improvements': report.improvements
            }, f, indent=2)
        
        logger.info(f"Report saved to {output_file} and {json_file}")


async def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze code and suggest improvements')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--output', default='code_improvement_report.md', help='Output report file')
    
    args = parser.parse_args()
    
    # Create workflow
    workflow = CodeImprovementWorkflow()
    await workflow.initialize()
    
    # Analyze directory
    report = await workflow.analyze_directory(args.directory)
    
    # Save report
    workflow.save_report(report, args.output)
    
    # Print summary
    print(f"\nAnalysis complete!")
    print(f"Total issues found: {len(report.issues)}")
    print(f"Critical issues: {len([i for i in report.issues if i.severity == 'critical'])}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())