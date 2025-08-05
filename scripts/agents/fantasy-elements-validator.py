#!/usr/bin/env python3
"""
Fantasy Elements Enforcement Agent

This agent scans all code for forbidden terms, validates dependencies are real,
checks for speculative/placeholder code, and integrates with pre-commit hooks.

Purpose: Enforce CLAUDE.md Rule 1 "No Fantasy Elements"
Usage: python fantasy-elements-validator.py [--fix] [--pre-commit] [--report-only]
Requirements: ripgrep, python packages from requirements.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class Violation:
    """Represents a fantasy elements violation"""
    file_path: str
    line_number: int
    line_content: str
    violation_type: str
    forbidden_term: str
    severity: str
    suggested_fix: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    total_files_scanned: int
    violations_found: int
    violations_by_type: Dict[str, int]
    violations_by_severity: Dict[str, int]
    violations: List[Violation]
    dependency_issues: List[str]
    placeholder_code_found: int
    recommendations: List[str]


class FantasyElementsValidator:
    """Main validator class for fantasy elements enforcement"""
    
    # Forbidden terms and their categories
    FORBIDDEN_TERMS = {
        'specific implementation name (e.g., emailSender, dataProcessor)': {
            'terms': ['specific implementation name (e.g., emailSender, dataProcessor)', 'automated, programmatic, or algorithmic', 'automatically, programmatically', 'specificHandler (e.g., authHandler, dataHandler)', 'specificService (e.g., emailService, userService)', 'specificFunction (e.g., validateInput, processData)'],
            'severity': 'high',
            'suggestions': {
                'specific implementation name (e.g., emailSender, dataProcessor)': 'specific implementation name (e.g., emailSender, dataProcessor)',
                'automated, programmatic, or algorithmic': 'automated, programmatic, or algorithmic',
                'automatically, programmatically': 'automatically, programmatically',
                'specificHandler (e.g., authHandler, dataHandler)': 'specificHandler (e.g., authHandler, dataHandler)',
                'specificService (e.g., emailService, userService)': 'specificService (e.g., emailService, userService)',
                'specificFunction (e.g., validateInput, processData)': 'specificFunction (e.g., validateInput, processData)'
            }
        },
        'assistant, helper, processor, manager': {
            'terms': ['assistant, helper, processor, manager', 'helperService, processingService', 'automation, processing, computation', 'processingHandler, assistantHandler'],
            'severity': 'high',
            'suggestions': {
                'assistant, helper, processor, manager': 'assistant, helper, processor, manager',
                'helperService, processingService': 'helperService, processingService',
                'automation, processing, computation': 'automation, processing, computation',
                'processingHandler, assistantHandler': 'processingHandler, assistantHandler'
            }
        },
        'transfer, send, transmit, copy': {
            'terms': ['transfer, send, transmit, copy', 'transferData, sendData, transmitData', 'data transfer, transmission, migration', 'transferring, sending, transmitting'],
            'severity': 'high',
            'suggestions': {
                'transfer, send, transmit, copy': 'transfer, send, transmit, copy',
                'transferData, sendData, transmitData': 'transferData, sendData, transmitData',
                'data transfer, transmission, migration': 'data transfer, transmission, migration',
                'transferring, sending, transmitting': 'transferring, sending, transmitting'
            }
        },
        'external_service, third_party_api': {
            'terms': ['external service, third-party API, opaque system', 'externalService, thirdPartyAPI', 'external_service, third_party_api', 'externalService, thirdPartyAPI'],
            'severity': 'medium',
            'suggestions': {
                'external service, third-party API, opaque system': 'external service, third-party API, opaque system',
                'externalService, thirdPartyAPI': 'externalService, thirdPartyAPI',
                'external_service, third_party_api': 'external_service, third_party_api',
                'externalService, thirdPartyAPI': 'externalService, thirdPartyAPI'
            }
        },
        'hypothetical': {
            'terms': ['specific future version or roadmap item', 'conditional logic or feature flag', 'tested implementation or proof of concept', 'validated approach or tested solution', 'documented specification or proven concept', 'concrete implementation or real example'],
            'severity': 'medium',
            'suggestions': {
                'specific future version or roadmap item': 'specific future version or roadmap item',
                'conditional logic or feature flag': 'conditional logic or feature flag',
                'tested implementation or proof of concept': 'tested implementation or proof of concept',
                'validated approach or tested solution': 'validated approach or tested solution',
                'documented specification or proven concept': 'documented specification or proven concept',
                'concrete implementation or real example': 'concrete implementation or real example'
            }
        }
    }
    
    # File patterns to scan
    SCAN_PATTERNS = [
        "**/*.py",
        "**/*.js",
        "**/*.ts",
        "**/*.jsx",
        "**/*.tsx",
        "**/*.go",
        "**/*.rs",
        "**/*.java",
        "**/*.cpp",
        "**/*.c",
        "**/*.h",
        "**/*.hpp",
        "**/*.md",
        "**/*.yml",
        "**/*.yaml",
        "**/*.json",
        "**/*.toml",
        "**/*.cfg",
        "**/*.ini",
        "**/Dockerfile*",
        "**/*.sh",
        "**/*.bash"
    ]
    
    # Files to exclude from scanning
    EXCLUDE_PATTERNS = [
        "*.git/*",
        "*/node_modules/*",
        "*/__pycache__/*",
        "*/venv/*",
        "*/env/*",
        "*/.venv/*",
        "*/.env/*",
        "*/build/*",
        "*/dist/*",
        "*/.pytest_cache/*",
        "*/logs/*",
        "*/data/*",
        "*/archive/*",
        "*/backup*"
    ]
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.console = Console()
        self.violations: List[Violation] = []
        self.dependency_issues: List[str] = []
        
    def scan_forbidden_terms(self) -> List[Violation]:
        """Scan for forbidden terms using ripgrep"""
        violations = []
        
        self.console.print("[bold blue]Scanning for forbidden terms...[/bold blue]")
        
        # Build ripgrep command for all forbidden terms
        all_terms = []
        term_to_category = {}
        
        for category, config in self.FORBIDDEN_TERMS.items():
            for term in config['terms']:
                all_terms.append(term)
                term_to_category[term] = category
        
        # Create regex pattern for all terms (word boundaries)
        pattern = r'\b(' + '|'.join(re.escape(term) for term in all_terms) + r')\b'
        
        try:
            # Run ripgrep with case-insensitive search
            cmd = [
                'rg',
                '--line-number',
                '--with-filename',
                '-i',  # case-insensitive
                '--json',
                pattern
            ]
            
            # Add exclude patterns
            for exclude in self.EXCLUDE_PATTERNS:
                cmd.extend(['--glob', f'!{exclude}'])
            
            cmd.append(str(self.root_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode not in [0, 1]:  # 0 = found, 1 = not found
                self.console.print(f"[red]Error running ripgrep: {result.stderr}[/red]")
                return violations
            
            # Parse JSON output
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                try:
                    match_data = json.loads(line)
                    if match_data.get('type') == 'match':
                        file_path = match_data['data']['path']['text']
                        line_num = match_data['data']['line_number']
                        line_content = match_data['data']['lines']['text'].strip()
                        
                        # Find which term was matched
                        matched_term = None
                        for term in all_terms:
                            if re.search(r'\b' + re.escape(term) + r'\b', line_content, re.IGNORECASE):
                                matched_term = term
                                break
                        
                        if matched_term:
                            category = term_to_category[matched_term]
                            config = self.FORBIDDEN_TERMS[category]
                            
                            violation = Violation(
                                file_path=file_path,
                                line_number=line_num,
                                line_content=line_content,
                                violation_type='forbidden_term',
                                forbidden_term=matched_term,
                                severity=config['severity'],
                                suggested_fix=config['suggestions'].get(matched_term),
                                context=f"Category: {category}"
                            )
                            violations.append(violation)
                            
                except json.JSONDecodeError:
                    continue
                    
        except FileNotFoundError:
            self.console.print("[red]Error: ripgrep (rg) not found. Please install ripgrep.[/red]")
            sys.exit(1)
        
        return violations
    
    def scan_placeholder_code(self) -> List[Violation]:
        """Scan for placeholder and speculative code patterns"""
        violations = []
        
        self.console.print("[bold blue]Scanning for placeholder code...[/bold blue]")
        
        # Patterns that indicate placeholder or speculative code
        placeholder_patterns = [
            (r'TODO.*specific implementation name (e.g., emailSender, dataProcessor)', 'TODO with specific implementation name (e.g., emailSender, dataProcessor) reference'),
            (r'TODO.*specific future version or roadmap item', 'TODO with specific future version or roadmap item reference'),
            (r'\/\/.*imagine', 'Comment with imagine'),
            (r'#.*imagine', 'Comment with imagine'),
            (r'\/\/.*TODO.*telekinesis', 'TODO with telekinesis'),
            (r'#.*TODO.*telekinesis', 'TODO with telekinesis'),
            (r'placeholder.*function', 'Placeholder function'),
            (r'stub.*implementation', 'Stub implementation'),
            (r'mock.*data.*\(', 'Mock data function calls'),
            (r'fake.*api', 'Fake API references'),
            (r'dummy.*service', 'Dummy service references'),
            (r'temp.*fix', 'Temporary fix references'),
            (r'hack.*for.*now', 'Hack for now'),
            (r'quick.*dirty', 'Quick and dirty'),
            (r'will.*implement.*later', 'Will implement later'),
            (r'TODO.*implement', 'TODO implement (often speculative)')
        ]
        
        for pattern, description in placeholder_patterns:
            try:
                cmd = [
                    'rg',
                    '--line-number',
                    '--with-filename',
                    '-i',  # case-insensitive
                    '--json',
                    pattern
                ]
                
                for exclude in self.EXCLUDE_PATTERNS:
                    cmd.extend(['--glob', f'!{exclude}'])
                
                cmd.append(str(self.root_path))
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if not line:
                            continue
                            
                        try:
                            match_data = json.loads(line)
                            if match_data.get('type') == 'match':
                                file_path = match_data['data']['path']['text']
                                line_num = match_data['data']['line_number']
                                line_content = match_data['data']['lines']['text'].strip()
                                
                                violation = Violation(
                                    file_path=file_path,
                                    line_number=line_num,
                                    line_content=line_content,
                                    violation_type='placeholder_code',
                                    forbidden_term=pattern,
                                    severity='medium',
                                    suggested_fix=f"Replace with concrete implementation: {description}",
                                    context=description
                                )
                                violations.append(violation)
                                
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                self.console.print(f"[yellow]Warning: Error scanning pattern {pattern}: {e}[/yellow]")
        
        return violations
    
    def validate_dependencies(self) -> List[str]:
        """Validate that all dependencies are real and verifiable"""
        issues = []
        
        self.console.print("[bold blue]Validating dependencies...[/bold blue]")
        
        # Check different dependency files
        dependency_files = [
            ('requirements.txt', self._validate_python_deps),
            ('pyproject.toml', self._validate_pyproject_deps),
            ('package.json', self._validate_npm_deps),
            ('Cargo.toml', self._validate_cargo_deps),
            ('go.mod', self._validate_go_deps)
        ]
        
        for dep_file, validator in dependency_files:
            dep_path = self.root_path / dep_file
            if dep_path.exists():
                try:
                    file_issues = validator(dep_path)
                    issues.extend(file_issues)
                except Exception as e:
                    issues.append(f"Error validating {dep_file}: {e}")
        
        return issues
    
    def _validate_python_deps(self, file_path: Path) -> List[str]:
        """Validate Python dependencies in requirements.txt"""
        issues = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extract package name (before any version specifiers)
            package_name = re.split(r'[>=<~!]', line)[0].strip()
            
            # Skip local/development packages
            if package_name.startswith('-e') or package_name.startswith('git+'):
                continue
            
            # Check if package exists on PyPI
            if not self._check_pypi_package_exists(package_name):
                issues.append(f"{file_path}:{line_num} - Package '{package_name}' not found on PyPI")
        
        return issues
    
    def _validate_pyproject_deps(self, file_path: Path) -> List[str]:
        """Validate Python dependencies in pyproject.toml"""
        issues = []
        
        try:
            import tomli
        except ImportError:
            try:
                import tomllib as tomli
            except ImportError:
                issues.append(f"Cannot validate {file_path}: tomli/tomllib not available")
                return issues
        
        try:
            with open(file_path, 'rb') as f:
                data = tomli.load(f)
            
            # Check dependencies in different sections
            deps_sections = [
                ('project.dependencies', data.get('project', {}).get('dependencies', [])),
                ('project.optional-dependencies', data.get('project', {}).get('optional-dependencies', {})),
                ('tool.poetry.dependencies', data.get('tool', {}).get('poetry', {}).get('dependencies', {}))
            ]
            
            for section_name, deps in deps_sections:
                if isinstance(deps, list):
                    for dep in deps:
                        package_name = re.split(r'[>=<~!]', dep)[0].strip()
                        if not self._check_pypi_package_exists(package_name):
                            issues.append(f"{file_path} [{section_name}] - Package '{package_name}' not found on PyPI")
                elif isinstance(deps, dict):
                    for package_name in deps.keys():
                        if not self._check_pypi_package_exists(package_name):
                            issues.append(f"{file_path} [{section_name}] - Package '{package_name}' not found on PyPI")
        
        except Exception as e:
            issues.append(f"Error parsing {file_path}: {e}")
        
        return issues
    
    def _validate_npm_deps(self, file_path: Path) -> List[str]:
        """Validate npm dependencies in package.json"""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check dependencies and devDependencies
            for dep_type in ['dependencies', 'devDependencies']:
                deps = data.get(dep_type, {})
                for package_name in deps.keys():
                    if not self._check_npm_package_exists(package_name):
                        issues.append(f"{file_path} [{dep_type}] - Package '{package_name}' not found on npm")
        
        except Exception as e:
            issues.append(f"Error parsing {file_path}: {e}")
        
        return issues
    
    def _validate_cargo_deps(self, file_path: Path) -> List[str]:
        """Validate Rust dependencies in Cargo.toml"""
        issues = []
        
        try:
            import tomli
        except ImportError:
            try:
                import tomllib as tomli
            except ImportError:
                return [f"Cannot validate {file_path}: tomli/tomllib not available"]
        
        try:
            with open(file_path, 'rb') as f:
                data = tomli.load(f)
            
            deps = data.get('dependencies', {})
            for package_name in deps.keys():
                if not self._check_crates_package_exists(package_name):
                    issues.append(f"{file_path} [dependencies] - Package '{package_name}' not found on crates.io")
        
        except Exception as e:
            issues.append(f"Error parsing {file_path}: {e}")
        
        return issues
    
    def _validate_go_deps(self, file_path: Path) -> List[str]:
        """Validate Go dependencies in go.mod"""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            in_require_block = False
            for line in lines:
                line = line.strip()
                
                if line.startswith('require ('):
                    in_require_block = True
                    continue
                elif line == ')' and in_require_block:
                    in_require_block = False
                    continue
                
                if in_require_block or line.startswith('require '):
                    # Extract module name
                    parts = line.split()
                    if len(parts) >= 2:
                        module_name = parts[0] if in_require_block else parts[1]
                        if not self._check_go_module_exists(module_name):
                            issues.append(f"{file_path} - Module '{module_name}' may not exist")
        
        except Exception as e:
            issues.append(f"Error parsing {file_path}: {e}")
        
        return issues
    
    def _check_pypi_package_exists(self, package_name: str) -> bool:
        """Check if a Python package exists on PyPI"""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package_name}/json", timeout=5)
            return response.status_code == 200
        except:
            return True  # Assume exists if we can't check (network issues, etc.)
    
    def _check_npm_package_exists(self, package_name: str) -> bool:
        """Check if an npm package exists"""
        try:
            response = requests.get(f"https://registry.npmjs.org/{package_name}", timeout=5)
            return response.status_code == 200
        except:
            return True  # Assume exists if we can't check
    
    def _check_crates_package_exists(self, package_name: str) -> bool:
        """Check if a Rust crate exists on crates.io"""
        try:
            response = requests.get(f"https://crates.io/api/v1/crates/{package_name}", timeout=5)
            return response.status_code == 200
        except:
            return True  # Assume exists if we can't check
    
    def _check_go_module_exists(self, module_name: str) -> bool:
        """Check if a Go module exists (basic validation)"""
        # Basic validation - Go modules should have valid domain-like names
        if '.' not in module_name:
            return False
        
        try:
            # Try to parse as URL to check if domain is valid
            parsed = urlparse(f"https://{module_name}")
            return bool(parsed.netloc)
        except:
            return True  # Assume exists if we can't parse
    
    def generate_auto_fixes(self, violations: List[Violation]) -> Dict[str, List[str]]:
        """Generate auto-fix suggestions for violations"""
        fixes = {}
        
        for violation in violations:
            if violation.file_path not in fixes:
                fixes[violation.file_path] = []
            
            if violation.suggested_fix:
                fix_description = f"Line {violation.line_number}: Replace '{violation.forbidden_term}' with '{violation.suggested_fix}'"
                fixes[violation.file_path].append(fix_description)
        
        return fixes
    
    def apply_auto_fixes(self, violations: List[Violation]) -> int:
        """Apply automatic fixes where possible"""
        fixes_applied = 0
        files_to_fix = {}
        
        # Group violations by file
        for violation in violations:
            if violation.violation_type == 'forbidden_term' and violation.suggested_fix:
                if violation.file_path not in files_to_fix:
                    files_to_fix[violation.file_path] = []
                files_to_fix[violation.file_path].append(violation)
        
        # Apply fixes file by file
        for file_path, file_violations in files_to_fix.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply fixes (start from highest line number to avoid offset issues)
                file_violations.sort(key=lambda v: v.line_number, reverse=True)
                
                for violation in file_violations:
                    # Simple regex replacement with word boundaries
                    pattern = r'\b' + re.escape(violation.forbidden_term) + r'\b'
                    content = re.sub(pattern, violation.suggested_fix, content, flags=re.IGNORECASE)
                
                # Only write if content changed
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes_applied += len(file_violations)
                    self.console.print(f"[green]Fixed {len(file_violations)} violations in {file_path}[/green]")
            
            except Exception as e:
                self.console.print(f"[red]Error fixing {file_path}: {e}[/red]")
        
        return fixes_applied
    
    def create_pre_commit_hook(self) -> bool:
        """Create or update pre-commit hook"""
        git_hooks_dir = self.root_path / '.git' / 'hooks'
        if not git_hooks_dir.exists():
            self.console.print("[yellow]No .git directory found. Skipping pre-commit hook creation.[/yellow]")
            return False
        
        pre_commit_hook = git_hooks_dir / 'pre-commit'
        
        hook_content = f"""#!/bin/bash
# Fantasy Elements Validator Pre-commit Hook
# Auto-generated by fantasy-elements-validator.py

echo "Running fantasy elements validation..."

python3 "{os.path.abspath(__file__)}" --pre-commit

if [ $? -ne 0 ]; then
    echo "Fantasy elements validation failed. Commit blocked."
    echo "Run 'python3 {os.path.abspath(__file__)} --fix' to auto-fix issues."
    exit 1
fi

echo "Fantasy elements validation passed."
"""
        
        try:
            with open(pre_commit_hook, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(pre_commit_hook, 0o755)
            
            self.console.print(f"[green]Created pre-commit hook at {pre_commit_hook}[/green]")
            return True
        
        except Exception as e:
            self.console.print(f"[red]Error creating pre-commit hook: {e}[/red]")
            return False
    
    def generate_report(self, violations: List[Violation], dependency_issues: List[str]) -> ValidationReport:
        """Generate comprehensive validation report"""
        violations_by_type = {}
        violations_by_severity = {'high': 0, 'medium': 0, 'low': 0}
        
        for violation in violations:
            violations_by_type[violation.violation_type] = violations_by_type.get(violation.violation_type, 0) + 1
            violations_by_severity[violation.severity] = violations_by_severity.get(violation.severity, 0) + 1
        
        placeholder_code_count = sum(1 for v in violations if v.violation_type == 'placeholder_code')
        
        recommendations = []
        if violations:
            recommendations.append("Run with --fix flag to apply automatic fixes")
            recommendations.append("Review all violations and replace fantasy terms with concrete implementations")
        
        if dependency_issues:
            recommendations.append("Verify all dependencies exist and are properly specified")
        
        if placeholder_code_count > 0:
            recommendations.append("Replace all placeholder code with concrete implementations")
        
        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_files_scanned=self._count_scannable_files(),
            violations_found=len(violations),
            violations_by_type=violations_by_type,
            violations_by_severity=violations_by_severity,
            violations=violations,
            dependency_issues=dependency_issues,
            placeholder_code_found=placeholder_code_count,
            recommendations=recommendations
        )
    
    def _count_scannable_files(self) -> int:
        """Count total scannable files"""
        count = 0
        for pattern in self.SCAN_PATTERNS:
            try:
                cmd = ['find', str(self.root_path), '-name', pattern.replace('**/', ''), '-type', 'f']
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                count += len([line for line in result.stdout.strip().split('\n') if line])
            except:
                pass
        return count
    
    def display_report(self, report: ValidationReport, show_details: bool = True):
        """Display validation report with rich formatting"""
        
        # Summary panel
        if report.violations_found == 0 and not report.dependency_issues:
            summary_text = "[green]✅ No fantasy elements found! Codebase is compliant.[/green]"
        else:
            summary_text = f"[red]❌ Found {report.violations_found} violations and {len(report.dependency_issues)} dependency issues[/red]"
        
        summary_panel = Panel(
            summary_text,
            title="Fantasy Elements Validation Report",
            title_align="center",
            border_style="blue"
        )
        self.console.print(summary_panel)
        
        # Statistics table
        if report.violations_found > 0 or report.dependency_issues:
            stats_table = Table(title="Violation Statistics")
            stats_table.add_column("Category", style="cyan", no_wrap=True)
            stats_table.add_column("Count", style="red", justify="right")
            
            for violation_type, count in report.violations_by_type.items():
                stats_table.add_row(violation_type.replace('_', ' ').title(), str(count))
            
            stats_table.add_row("Dependency Issues", str(len(report.dependency_issues)))
            stats_table.add_row("", "")  # Separator
            
            for severity, count in report.violations_by_severity.items():
                if count > 0:
                    stats_table.add_row(f"{severity.title()} Severity", str(count))
            
            self.console.print(stats_table)
        
        # Detailed violations
        if show_details and report.violations:
            self.console.print("\n[bold red]Detailed Violations:[/bold red]")
            
            for i, violation in enumerate(report.violations[:50], 1):  # Limit to first 50
                severity_color = {"high": "red", "medium": "yellow", "low": "blue"}[violation.severity]
                
                violation_text = Text()
                violation_text.append(f"{i}. ", style="dim")
                violation_text.append(f"[{violation.severity.upper()}] ", style=severity_color)
                violation_text.append(f"{violation.file_path}:{violation.line_number}\n", style="cyan")
                violation_text.append(f"   Term: '{violation.forbidden_term}' ", style="red")
                violation_text.append(f"in line: {violation.line_content[:100]}...\n", style="dim")
                if violation.suggested_fix:
                    violation_text.append(f"   Suggested fix: {violation.suggested_fix}\n", style="green")
                
                self.console.print(violation_text)
            
            if len(report.violations) > 50:
                self.console.print(f"\n[dim]... and {len(report.violations) - 50} more violations[/dim]")
        
        # Dependency issues
        if report.dependency_issues:
            self.console.print("\n[bold red]Dependency Issues:[/bold red]")
            for issue in report.dependency_issues[:20]:  # Limit to first 20
                self.console.print(f"  • {issue}")
            
            if len(report.dependency_issues) > 20:
                self.console.print(f"[dim]... and {len(report.dependency_issues) - 20} more issues[/dim]")
        
        # Recommendations
        if report.recommendations:
            self.console.print("\n[bold blue]Recommendations:[/bold blue]")
            for rec in report.recommendations:
                self.console.print(f"  • {rec}")
    
    def save_report(self, report: ValidationReport, output_path: str):
        """Save report to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            self.console.print(f"[green]Report saved to {output_path}[/green]")
        except Exception as e:
            self.console.print(f"[red]Error saving report: {e}[/red]")
    
    def run_validation(self, fix_violations: bool = False, pre_commit_mode: bool = False) -> ValidationReport:
        """Run complete validation"""
        
        if not pre_commit_mode:
            self.console.print("[bold green]Fantasy Elements Validator[/bold green]")
            self.console.print("Enforcing CLAUDE.md Rule 1: No Fantasy Elements\n")
        
        # Scan for violations
        forbidden_violations = self.scan_forbidden_terms()
        placeholder_violations = self.scan_placeholder_code()
        all_violations = forbidden_violations + placeholder_violations
        
        # Validate dependencies
        dependency_issues = self.validate_dependencies()
        
        # Apply fixes if requested
        if fix_violations and all_violations:
            fixes_applied = self.apply_auto_fixes(all_violations)
            if fixes_applied > 0:
                self.console.print(f"[green]Applied {fixes_applied} automatic fixes[/green]")
                # Re-scan after fixes
                forbidden_violations = self.scan_forbidden_terms()
                placeholder_violations = self.scan_placeholder_code()
                all_violations = forbidden_violations + placeholder_violations
        
        # Generate report
        report = self.generate_report(all_violations, dependency_issues)
        
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Fantasy Elements Validator - Enforce CLAUDE.md Rule 1",
        epilog="This tool scans for forbidden terms, validates dependencies, and checks for speculative code."
    )
    
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply automatic fixes where possible'
    )
    
    parser.add_argument(
        '--pre-commit',
        action='store_true',
        help='Run in pre-commit mode (minimal output, exit code for CI)'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report without scanning (requires existing report.json)'
    )
    
    parser.add_argument(
        '--output',
        default='fantasy-elements-report.json',
        help='Output file for detailed report (default: fantasy-elements-report.json)'
    )
    
    parser.add_argument(
        '--root-path',
        default='/opt/sutazaiapp',
        help='Root path to scan (default: /opt/sutazaiapp)'
    )
    
    parser.add_argument(
        '--create-hook',
        action='store_true',
        help='Create pre-commit hook'
    )
    
    args = parser.parse_args()
    
    validator = FantasyElementsValidator(args.root_path)
    
    # Create pre-commit hook if requested
    if args.create_hook:
        validator.create_pre_commit_hook()
        return
    
    # Run validation
    report = validator.run_validation(
        fix_violations=args.fix,
        pre_commit_mode=args.pre_commit
    )
    
    # Display results
    if not args.pre_commit:
        validator.display_report(report, show_details=True)
        
        # Save detailed report
        validator.save_report(report, args.output)
    else:
        # Pre-commit mode: minimal output
        if report.violations_found > 0 or report.dependency_issues:
            print(f"Fantasy elements validation failed: {report.violations_found} violations, {len(report.dependency_issues)} dependency issues")
            sys.exit(1)
        else:
            print("Fantasy elements validation passed")
    
    # Exit with appropriate code
    if report.violations_found > 0 or report.dependency_issues:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()