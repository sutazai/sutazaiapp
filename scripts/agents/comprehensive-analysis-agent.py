#!/usr/bin/env python3
"""
Purpose: Comprehensive Analysis Agent - Enforces Rule 3: Analyze Everything—Every Time
Usage: python comprehensive-analysis-agent.py [--report-dir REPORT_DIR] [--format FORMAT] [--fix]
Requirements: Python 3.8+, os, sys, json, yaml, subprocess, pathlib, logging, typing
"""

import os
import sys
import json
import yaml
import subprocess
import logging
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveAnalysisAgent:
    """Agent that performs systematic analysis of the entire codebase."""
    
    def __init__(self, project_root: Path, report_dir: Path = None):
        self.project_root = project_root
        self.report_dir = report_dir or project_root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # Analysis results storage
        self.issues = defaultdict(list)
        self.recommendations = defaultdict(list)
        self.metrics = defaultdict(dict)
        
        # Patterns and configurations
        self.load_configurations()
        
    def load_configurations(self):
        """Load project configurations and standards."""
        self.claude_md_path = self.project_root / "CLAUDE.md"
        self.claude_local_path = self.project_root / "CLAUDE.local.md"
        
        # File patterns
        self.ignore_patterns = {
            "__pycache__", ".git", ".pytest_cache", "node_modules",
            "venv", ".venv", "*.egg-info", ".tox", ".mypy_cache"
        }
        
        # Naming conventions
        self.naming_patterns = {
            "python": re.compile(r"^[a-z_][a-z0-9_]*\.py$"),
            "javascript": re.compile(r"^[a-z][a-zA-Z0-9]*\.(js|jsx|ts|tsx)$"),
            "markdown": re.compile(r"^[A-Z0-9_\-]+\.md$"),
            "yaml": re.compile(r"^[a-z\-]+\.(yml|yaml)$"),
            "json": re.compile(r"^[a-z\-]+\.json$"),
            "shell": re.compile(r"^[a-z\-]+\.sh$")
        }
        
        # Known directories and their purposes
        self.directory_purposes = {
            "agents": "AI agent implementations",
            "backend": "Backend services and APIs",
            "frontend": "Frontend application",
            "scripts": "Utility and automation scripts",
            "config": "Configuration files",
            "docs": "Documentation",
            "tests": "Test suites",
            "data": "Data storage",
            "logs": "Log files",
            "deployment": "Deployment configurations"
        }
        
    def analyze(self, fix_issues: bool = False) -> Dict[str, Any]:
        """Perform comprehensive analysis of the entire codebase."""
        logger.info("Starting comprehensive codebase analysis...")
        
        # Reset analysis state
        self.issues.clear()
        self.recommendations.clear()
        self.metrics.clear()
        
        # Run all analysis modules
        self.analyze_files()
        self.analyze_folders()
        self.analyze_scripts()
        self.analyze_code_logic()
        self.analyze_dependencies()
        self.analyze_apis()
        self.analyze_configuration()
        self.analyze_build_deploy()
        self.analyze_logs_monitoring()
        self.analyze_testing()
        
        # Generate report
        report = self.generate_report()
        
        # Optionally fix issues
        if fix_issues:
            self.fix_issues()
        
        return report
        
    def analyze_files(self):
        """Analyze file naming conventions, redundancy, and dependencies."""
        logger.info("Analyzing files...")
        
        files_by_type = defaultdict(list)
        duplicate_files = defaultdict(list)
        naming_violations = []
        
        for root, _, files in os.walk(self.project_root):
            root_path = Path(root)
            
            # Skip ignored directories
            if any(pattern in str(root_path) for pattern in self.ignore_patterns):
                continue
                
            for file in files:
                file_path = root_path / file
                relative_path = file_path.relative_to(self.project_root)
                
                # Categorize by type
                extension = file_path.suffix.lower()
                files_by_type[extension].append(relative_path)
                
                # Check naming conventions
                if not self._check_naming_convention(file, extension):
                    naming_violations.append(relative_path)
                    self.issues["files"].append({
                        "type": "naming_violation",
                        "path": str(relative_path),
                        "message": f"File '{file}' violates naming conventions"
                    })
                
                # Calculate file hash for duplicate detection
                if file_path.is_file() and file_path.stat().st_size < 10_000_000:  # Skip large files
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        duplicate_files[file_hash].append(relative_path)
                    except Exception as e:
                        logger.warning(f"Error hashing {file_path}: {e}")
        
        # Identify duplicates
        for file_hash, paths in duplicate_files.items():
            if len(paths) > 1:
                self.issues["files"].append({
                    "type": "duplicate",
                    "paths": [str(p) for p in paths],
                    "message": f"Duplicate files found: {', '.join(str(p) for p in paths)}"
                })
        
        # Store metrics
        self.metrics["files"] = {
            "total": sum(len(files) for files in files_by_type.values()),
            "by_type": {k: len(v) for k, v in files_by_type.items()},
            "naming_violations": len(naming_violations),
            "duplicate_sets": len([d for d in duplicate_files.values() if len(d) > 1])
        }
        
    def analyze_folders(self):
        """Analyze folder structure, duplication, and organization."""
        logger.info("Analyzing folders...")
        
        folder_purposes = {}
        empty_folders = []
        duplicate_structures = defaultdict(list)
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            relative_path = root_path.relative_to(self.project_root)
            
            # Skip ignored directories
            if any(pattern in str(root_path) for pattern in self.ignore_patterns):
                continue
            
            # Check if folder is empty
            if not dirs and not files:
                empty_folders.append(relative_path)
                self.issues["folders"].append({
                    "type": "empty",
                    "path": str(relative_path),
                    "message": f"Empty folder: {relative_path}"
                })
            
            # Analyze folder purpose
            folder_name = root_path.name
            if folder_name in self.directory_purposes:
                folder_purposes[str(relative_path)] = self.directory_purposes[folder_name]
            
            # Check for duplicate module structures
            if files:
                structure_hash = self._calculate_structure_hash(files)
                duplicate_structures[structure_hash].append(relative_path)
        
        # Identify duplicate structures
        for structure_hash, paths in duplicate_structures.items():
            if len(paths) > 1:
                self.recommendations["folders"].append({
                    "type": "duplicate_structure",
                    "paths": [str(p) for p in paths],
                    "message": f"Similar folder structures found: {', '.join(str(p) for p in paths[:3])}"
                })
        
        self.metrics["folders"] = {
            "total": len(list(self.project_root.rglob("**/"))),
            "empty": len(empty_folders),
            "with_known_purpose": len(folder_purposes)
        }
        
    def analyze_scripts(self):
        """Analyze script reusability, documentation, and execution paths."""
        logger.info("Analyzing scripts...")
        
        scripts = []
        documented_scripts = []
        executable_scripts = []
        duplicate_functionality = defaultdict(list)
        
        # Find all script files
        for extension in [".py", ".sh", ".js"]:
            for script_path in self.project_root.rglob(f"*{extension}"):
                if any(pattern in str(script_path) for pattern in self.ignore_patterns):
                    continue
                    
                scripts.append(script_path)
                
                # Check documentation
                if self._has_proper_documentation(script_path):
                    documented_scripts.append(script_path)
                else:
                    self.issues["scripts"].append({
                        "type": "missing_documentation",
                        "path": str(script_path.relative_to(self.project_root)),
                        "message": f"Script lacks proper documentation header"
                    })
                
                # Check executable permissions (for shell scripts)
                if extension == ".sh" and not os.access(script_path, os.X_OK):
                    self.issues["scripts"].append({
                        "type": "permission",
                        "path": str(script_path.relative_to(self.project_root)),
                        "message": f"Shell script is not executable"
                    })
                
                # Analyze functionality for duplicates
                functionality_hash = self._analyze_script_functionality(script_path)
                if functionality_hash:
                    duplicate_functionality[functionality_hash].append(script_path)
        
        # Report duplicate functionality
        for func_hash, paths in duplicate_functionality.items():
            if len(paths) > 1:
                self.issues["scripts"].append({
                    "type": "duplicate_functionality",
                    "paths": [str(p.relative_to(self.project_root)) for p in paths],
                    "message": f"Scripts with similar functionality detected"
                })
        
        self.metrics["scripts"] = {
            "total": len(scripts),
            "documented": len(documented_scripts),
            "documentation_rate": len(documented_scripts) / len(scripts) if scripts else 0,
            "duplicate_sets": len([d for d in duplicate_functionality.values() if len(d) > 1])
        }
        
    def analyze_code_logic(self):
        """Analyze code logic for efficiency, edge cases, and complexity."""
        logger.info("Analyzing code logic...")
        
        complexity_issues = []
        edge_case_issues = []
        efficiency_issues = []
        
        # Analyze Python files
        for py_file in self.project_root.rglob("*.py"):
            if any(pattern in str(py_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for common issues
                issues = self._analyze_python_code(content, py_file)
                
                if issues.get("complexity"):
                    complexity_issues.extend(issues["complexity"])
                if issues.get("edge_cases"):
                    edge_case_issues.extend(issues["edge_cases"])
                if issues.get("efficiency"):
                    efficiency_issues.extend(issues["efficiency"])
                    
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")
        
        # Store findings
        for issue in complexity_issues:
            self.issues["code_logic"].append(issue)
        for issue in edge_case_issues:
            self.issues["code_logic"].append(issue)
        for issue in efficiency_issues:
            self.recommendations["code_logic"].append(issue)
        
        self.metrics["code_logic"] = {
            "complexity_issues": len(complexity_issues),
            "edge_case_issues": len(edge_case_issues),
            "efficiency_issues": len(efficiency_issues)
        }
        
    def analyze_dependencies(self):
        """Validate dependencies usage, security, and updates."""
        logger.info("Analyzing dependencies...")
        
        requirements_files = []
        package_files = []
        outdated_deps = []
        unused_deps = []
        security_issues = []
        
        # Find dependency files
        for pattern in ["requirements*.txt", "package.json", "pyproject.toml", "Pipfile"]:
            requirements_files.extend(self.project_root.rglob(pattern))
        
        # Analyze each dependency file
        for req_file in requirements_files:
            if any(pattern in str(req_file) for pattern in self.ignore_patterns):
                continue
                
            deps = self._parse_dependencies(req_file)
            
            # Check for issues
            for dep, version in deps.items():
                # Check if dependency is used
                if not self._is_dependency_used(dep):
                    unused_deps.append({
                        "dependency": dep,
                        "file": str(req_file.relative_to(self.project_root))
                    })
                    
                # Check for security issues (simplified)
                if self._has_known_vulnerability(dep, version):
                    security_issues.append({
                        "dependency": dep,
                        "version": version,
                        "file": str(req_file.relative_to(self.project_root))
                    })
        
        # Report issues
        for unused in unused_deps:
            self.issues["dependencies"].append({
                "type": "unused",
                "dependency": unused["dependency"],
                "file": unused["file"],
                "message": f"Unused dependency: {unused['dependency']}"
            })
            
        for security in security_issues:
            self.issues["dependencies"].append({
                "type": "security",
                "dependency": security["dependency"],
                "version": security["version"],
                "file": security["file"],
                "message": f"Security vulnerability in {security['dependency']} {security['version']}"
            })
        
        self.metrics["dependencies"] = {
            "files": len(requirements_files),
            "unused": len(unused_deps),
            "security_issues": len(security_issues)
        }
        
    def analyze_apis(self):
        """Analyze APIs for stability, error handling, and rate limits."""
        logger.info("Analyzing APIs...")
        
        api_endpoints = []
        missing_error_handling = []
        missing_rate_limits = []
        
        # Find API definitions
        for py_file in self.project_root.rglob("*.py"):
            if any(pattern in str(py_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find API endpoints (simplified pattern matching)
                endpoints = self._find_api_endpoints(content)
                
                for endpoint in endpoints:
                    api_endpoints.append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "endpoint": endpoint
                    })
                    
                    # Check for error handling
                    if not self._has_error_handling(content, endpoint):
                        missing_error_handling.append(endpoint)
                        
                    # Check for rate limiting
                    if not self._has_rate_limiting(content, endpoint):
                        missing_rate_limits.append(endpoint)
                        
            except Exception as e:
                logger.warning(f"Error analyzing APIs in {py_file}: {e}")
        
        # Report issues
        for endpoint in missing_error_handling:
            self.issues["apis"].append({
                "type": "missing_error_handling",
                "endpoint": endpoint,
                "message": f"API endpoint lacks proper error handling: {endpoint}"
            })
            
        for endpoint in missing_rate_limits:
            self.recommendations["apis"].append({
                "type": "missing_rate_limit",
                "endpoint": endpoint,
                "message": f"Consider adding rate limiting to: {endpoint}"
            })
        
        self.metrics["apis"] = {
            "total_endpoints": len(api_endpoints),
            "missing_error_handling": len(missing_error_handling),
            "missing_rate_limits": len(missing_rate_limits)
        }
        
    def analyze_configuration(self):
        """Analyze configuration files for scoping, secrets, and parameters."""
        logger.info("Analyzing configuration...")
        
        config_files = []
        hardcoded_secrets = []
        misconfigurations = []
        
        # Find configuration files
        for pattern in ["*.yml", "*.yaml", "*.json", "*.env", "*.ini", "*.toml"]:
            config_files.extend(self.project_root.rglob(pattern))
        
        for config_file in config_files:
            if any(pattern in str(config_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for hardcoded secrets
                secrets = self._find_hardcoded_secrets(content)
                if secrets:
                    hardcoded_secrets.append({
                        "file": str(config_file.relative_to(self.project_root)),
                        "secrets": secrets
                    })
                    
                # Check for misconfigurations
                issues = self._check_configuration_issues(content, config_file)
                if issues:
                    misconfigurations.extend(issues)
                    
            except Exception as e:
                logger.warning(f"Error analyzing config {config_file}: {e}")
        
        # Report issues
        for secret_info in hardcoded_secrets:
            self.issues["configuration"].append({
                "type": "hardcoded_secret",
                "file": secret_info["file"],
                "message": f"Hardcoded secrets found in configuration"
            })
            
        for issue in misconfigurations:
            self.issues["configuration"].append(issue)
        
        self.metrics["configuration"] = {
            "files": len(config_files),
            "hardcoded_secrets": len(hardcoded_secrets),
            "misconfigurations": len(misconfigurations)
        }
        
    def analyze_build_deploy(self):
        """Analyze build/deployment pipelines for completeness and reliability."""
        logger.info("Analyzing build/deployment...")
        
        pipeline_files = []
        missing_tests = []
        missing_rollback = []
        
        # Find CI/CD files
        ci_patterns = [
            ".github/workflows/*.yml",
            ".gitlab-ci.yml",
            "Jenkinsfile",
            "docker-compose*.yml",
            "Dockerfile*",
            "deploy*.sh"
        ]
        
        for pattern in ci_patterns:
            pipeline_files.extend(self.project_root.rglob(pattern))
        
        for pipeline_file in pipeline_files:
            try:
                with open(pipeline_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for test steps
                if not self._has_test_step(content):
                    missing_tests.append(pipeline_file)
                    
                # Check for rollback mechanism
                if not self._has_rollback_mechanism(content):
                    missing_rollback.append(pipeline_file)
                    
            except Exception as e:
                logger.warning(f"Error analyzing pipeline {pipeline_file}: {e}")
        
        # Report issues
        for file in missing_tests:
            self.issues["build_deploy"].append({
                "type": "missing_tests",
                "file": str(file.relative_to(self.project_root)),
                "message": "Pipeline lacks test execution step"
            })
            
        for file in missing_rollback:
            self.recommendations["build_deploy"].append({
                "type": "missing_rollback",
                "file": str(file.relative_to(self.project_root)),
                "message": "Consider adding rollback mechanism"
            })
        
        self.metrics["build_deploy"] = {
            "pipeline_files": len(pipeline_files),
            "missing_tests": len(missing_tests),
            "missing_rollback": len(missing_rollback)
        }
        
    def analyze_logs_monitoring(self):
        """Analyze logging and monitoring configuration."""
        logger.info("Analyzing logs/monitoring...")
        
        log_statements = 0
        sensitive_logging = []
        monitoring_configs = []
        
        # Analyze Python files for logging
        for py_file in self.project_root.rglob("*.py"):
            if any(pattern in str(py_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count log statements
                log_count = len(re.findall(r'log(ger)?\.(debug|info|warning|error|critical)', content))
                log_statements += log_count
                
                # Check for sensitive data logging
                sensitive = self._find_sensitive_logging(content)
                if sensitive:
                    sensitive_logging.append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "instances": sensitive
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing logging in {py_file}: {e}")
        
        # Find monitoring configurations
        monitoring_patterns = ["prometheus*.yml", "grafana*.json", "alerting*.yml"]
        for pattern in monitoring_patterns:
            monitoring_configs.extend(self.project_root.rglob(pattern))
        
        # Report issues
        for sensitive in sensitive_logging:
            self.issues["logs_monitoring"].append({
                "type": "sensitive_logging",
                "file": sensitive["file"],
                "message": "Potential sensitive data in logs"
            })
        
        self.metrics["logs_monitoring"] = {
            "log_statements": log_statements,
            "sensitive_logging_files": len(sensitive_logging),
            "monitoring_configs": len(monitoring_configs)
        }
        
    def analyze_testing(self):
        """Analyze testing coverage, flakiness, and redundancy."""
        logger.info("Analyzing testing...")
        
        test_files = []
        test_functions = 0
        missing_assertions = []
        
        # Find test files
        test_patterns = ["test_*.py", "*_test.py", "tests.py"]
        for pattern in test_patterns:
            test_files.extend(self.project_root.rglob(pattern))
        
        for test_file in test_files:
            if any(pattern in str(test_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count test functions
                test_count = len(re.findall(r'def test_\w+', content))
                test_functions += test_count
                
                # Check for missing assertions
                tests_without_assertions = self._find_tests_without_assertions(content)
                if tests_without_assertions:
                    missing_assertions.append({
                        "file": str(test_file.relative_to(self.project_root)),
                        "tests": tests_without_assertions
                    })
                    
            except Exception as e:
                logger.warning(f"Error analyzing tests in {test_file}: {e}")
        
        # Report issues
        for missing in missing_assertions:
            self.issues["testing"].append({
                "type": "missing_assertions",
                "file": missing["file"],
                "tests": missing["tests"],
                "message": f"Tests without assertions found"
            })
        
        # Try to get coverage data if available
        coverage_data = self._get_coverage_data()
        
        self.metrics["testing"] = {
            "test_files": len(test_files),
            "test_functions": test_functions,
            "missing_assertions": len(missing_assertions),
            "coverage": coverage_data
        }
        
    # Helper methods
    def _check_naming_convention(self, filename: str, extension: str) -> bool:
        """Check if filename follows naming conventions."""
        ext = extension.lstrip('.')
        
        # Map extensions to types
        type_mapping = {
            "py": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "javascript",
            "tsx": "javascript",
            "md": "markdown",
            "yml": "yaml",
            "yaml": "yaml",
            "json": "json",
            "sh": "shell"
        }
        
        file_type = type_mapping.get(ext)
        if file_type and file_type in self.naming_patterns:
            return bool(self.naming_patterns[file_type].match(filename))
        
        return True  # Default to valid if no pattern defined
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def _calculate_structure_hash(self, files: List[str]) -> str:
        """Calculate hash of directory structure."""
        sorted_files = sorted(files)
        structure_str = "|".join(sorted_files)
        return hashlib.md5(structure_str.encode()).hexdigest()
        
    def _has_proper_documentation(self, script_path: Path) -> bool:
        """Check if script has proper documentation header."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Read first 1000 chars
                
            # Check for documentation patterns
            if script_path.suffix == ".py":
                return bool(re.search(r'"""[\s\S]*?Purpose:[\s\S]*?Usage:[\s\S]*?"""', content))
            elif script_path.suffix == ".sh":
                return bool(re.search(r'# Purpose:.*\n# Usage:', content))
            
            return False
        except:
            return False
            
    def _analyze_script_functionality(self, script_path: Path) -> Optional[str]:
        """Analyze script functionality and return hash."""
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract main functionality indicators
            functions = re.findall(r'def (\w+)', content)
            classes = re.findall(r'class (\w+)', content)
            imports = re.findall(r'import (\w+)', content)
            
            # Create functionality signature
            signature = f"{sorted(functions)}|{sorted(classes)}|{sorted(imports)}"
            return hashlib.md5(signature.encode()).hexdigest()
            
        except:
            return None
            
    def _analyze_python_code(self, content: str, file_path: Path) -> Dict[str, List]:
        """Analyze Python code for various issues."""
        issues = {"complexity": [], "edge_cases": [], "efficiency": []}
        relative_path = file_path.relative_to(self.project_root)
        
        # Check cyclomatic complexity (simplified)
        function_matches = re.finditer(r'def (\w+).*?:\n((?:\s{4,}.*\n)*)', content)
        for match in function_matches:
            func_name = match.group(1)
            func_body = match.group(2)
            
            # Count decision points
            complexity = len(re.findall(r'\b(if|elif|for|while|except)\b', func_body))
            if complexity > 10:
                issues["complexity"].append({
                    "type": "high_complexity",
                    "path": str(relative_path),
                    "function": func_name,
                    "complexity": complexity,
                    "message": f"High cyclomatic complexity ({complexity}) in {func_name}"
                })
        
        # Check for missing edge case handling
        if re.search(r'\/\s*0', content) and not re.search(r'ZeroDivisionError', content):
            issues["edge_cases"].append({
                "type": "missing_edge_case",
                "path": str(relative_path),
                "message": "Potential division by zero without error handling"
            })
            
        # Check for efficiency issues
        if re.search(r'for.*in.*for.*in', content):
            issues["efficiency"].append({
                "type": "nested_loops",
                "path": str(relative_path),
                "message": "Nested loops detected - consider optimization"
            })
            
        return issues
        
    def _parse_dependencies(self, req_file: Path) -> Dict[str, str]:
        """Parse dependencies from requirements file."""
        deps = {}
        
        try:
            if req_file.name == "package.json":
                with open(req_file, 'r') as f:
                    data = json.load(f)
                    deps.update(data.get("dependencies", {}))
                    deps.update(data.get("devDependencies", {}))
            elif req_file.suffix == ".txt":
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            parts = re.split(r'[<>=!]', line)
                            if parts:
                                deps[parts[0].strip()] = line
        except Exception as e:
            logger.warning(f"Error parsing {req_file}: {e}")
            
        return deps
        
    def _is_dependency_used(self, dep: str) -> bool:
        """Check if dependency is used in codebase."""
        # Simplified check - look for imports
        for py_file in self.project_root.rglob("*.py"):
            if any(pattern in str(py_file) for pattern in self.ignore_patterns):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if re.search(rf'import {dep}|from {dep}', content):
                        return True
            except:
                continue
                
        return False
        
    def _has_known_vulnerability(self, dep: str, version: str) -> bool:
        """Check for known vulnerabilities (simplified)."""
        # This is a simplified check - in production, use safety or similar tools
        vulnerable_packages = {
            "requests": ["< 2.20.0"],
            "django": ["< 2.2.24", "< 3.1.13"],
            "flask": ["< 1.0"],
            "pyyaml": ["< 5.4"]
        }
        
        return dep.lower() in vulnerable_packages
        
    def _find_api_endpoints(self, content: str) -> List[str]:
        """Find API endpoint definitions."""
        endpoints = []
        
        # Flask/FastAPI patterns
        patterns = [
            r'@app\.route\(["\']([^"\']+)',
            r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)',
            r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    endpoints.append(match[-1])
                else:
                    endpoints.append(match)
                    
        return endpoints
        
    def _has_error_handling(self, content: str, endpoint: str) -> bool:
        """Check if endpoint has error handling."""
        # Simplified check - look for try/except near endpoint
        endpoint_pos = content.find(endpoint)
        if endpoint_pos > -1:
            nearby_content = content[endpoint_pos:endpoint_pos + 500]
            return bool(re.search(r'try:|except\s+\w+:', nearby_content))
        return False
        
    def _has_rate_limiting(self, content: str, endpoint: str) -> bool:
        """Check if endpoint has rate limiting."""
        # Look for common rate limiting decorators/imports
        return bool(re.search(r'ratelimit|rate_limit|limiter', content, re.IGNORECASE))
        
    def _find_hardcoded_secrets(self, content: str) -> List[str]:
        """Find potential hardcoded secrets."""
        secrets = []
        
        # Common secret patterns
        patterns = [
            r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']([^"\']+)',
            r'(?i)(secret[_-]?key|password|passwd|pwd)\s*[:=]\s*["\']([^"\']+)',
            r'(?i)(access[_-]?token|auth[_-]?token)\s*[:=]\s*["\']([^"\']+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple) and len(match) > 1:
                    value = match[1]
                    # Skip obvious placeholders
                    if not re.match(r'^(\$\{.*\}|<.*>|xxx+|your[_-].*|example|placeholder)$', value, re.IGNORECASE):
                        secrets.append(match[0])
                        
        return secrets
        
    def _check_configuration_issues(self, content: str, config_file: Path) -> List[Dict]:
        """Check for configuration issues."""
        issues = []
        
        # Check for common misconfigurations
        if config_file.suffix in [".yml", ".yaml"]:
            # Check for missing quotes around values that need them
            if re.search(r':\s*\$', content):
                issues.append({
                    "type": "unquoted_variable",
                    "file": str(config_file.relative_to(self.project_root)),
                    "message": "Unquoted environment variable reference"
                })
                
        return issues
        
    def _has_test_step(self, content: str) -> bool:
        """Check if pipeline has test step."""
        test_indicators = ['test', 'pytest', 'unittest', 'jest', 'mocha', 'rspec']
        return any(indicator in content.lower() for indicator in test_indicators)
        
    def _has_rollback_mechanism(self, content: str) -> bool:
        """Check if pipeline has rollback mechanism."""
        rollback_indicators = ['rollback', 'revert', 'previous', 'restore']
        return any(indicator in content.lower() for indicator in rollback_indicators)
        
    def _find_sensitive_logging(self, content: str) -> List[str]:
        """Find potential sensitive data in logging statements."""
        sensitive = []
        
        # Look for logging of sensitive fields
        log_patterns = re.findall(r'log(?:ger)?\.\w+\([^)]+\)', content)
        sensitive_fields = ['password', 'token', 'secret', 'key', 'credential', 'ssn', 'credit_card']
        
        for log_stmt in log_patterns:
            for field in sensitive_fields:
                if field in log_stmt.lower():
                    sensitive.append(log_stmt)
                    break
                    
        return sensitive
        
    def _find_tests_without_assertions(self, content: str) -> List[str]:
        """Find test functions without assertions."""
        tests_without_assertions = []
        
        # Find all test functions
        test_pattern = re.compile(r'def (test_\w+).*?:\n((?:\s{4,}.*\n)*)', re.MULTILINE)
        
        for match in test_pattern.finditer(content):
            test_name = match.group(1)
            test_body = match.group(2)
            
            # Check for assertions
            assertion_patterns = ['assert', 'self.assert', 'expect', '.to', '.should']
            has_assertion = any(pattern in test_body for pattern in assertion_patterns)
            
            if not has_assertion:
                tests_without_assertions.append(test_name)
                
        return tests_without_assertions
        
    def _get_coverage_data(self) -> Optional[Dict]:
        """Try to get test coverage data."""
        coverage_file = self.project_root / ".coverage"
        if coverage_file.exists():
            try:
                # Simplified - in production use coverage.py API
                return {"status": "coverage_data_found"}
            except:
                pass
        return None
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        timestamp = datetime.now().isoformat()
        
        # Calculate summary statistics
        total_issues = sum(len(issues) for issues in self.issues.values())
        total_recommendations = sum(len(recs) for recs in self.recommendations.values())
        
        # Severity classification
        critical_issues = 0
        high_issues = 0
        medium_issues = 0
        low_issues = 0
        
        for category_issues in self.issues.values():
            for issue in category_issues:
                if issue.get("type") in ["hardcoded_secret", "security", "missing_error_handling"]:
                    critical_issues += 1
                elif issue.get("type") in ["duplicate", "high_complexity", "missing_tests"]:
                    high_issues += 1
                elif issue.get("type") in ["naming_violation", "missing_documentation"]:
                    medium_issues += 1
                else:
                    low_issues += 1
        
        report = {
            "analysis_timestamp": timestamp,
            "project_root": str(self.project_root),
            "summary": {
                "total_issues": total_issues,
                "critical": critical_issues,
                "high": high_issues,
                "medium": medium_issues,
                "low": low_issues,
                "total_recommendations": total_recommendations
            },
            "metrics": dict(self.metrics),
            "issues": dict(self.issues),
            "recommendations": dict(self.recommendations),
            "compliance_score": self._calculate_compliance_score()
        }
        
        # Save report
        report_path = self.report_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Also save markdown report
        self._save_markdown_report(report)
        
        logger.info(f"Analysis complete. Report saved to {report_path}")
        
        return report
        
    def _calculate_compliance_score(self) -> Dict[str, float]:
        """Calculate compliance scores for different aspects."""
        scores = {}
        
        # File compliance
        if self.metrics.get("files"):
            total_files = self.metrics["files"]["total"]
            violations = self.metrics["files"]["naming_violations"]
            scores["file_compliance"] = (total_files - violations) / total_files if total_files > 0 else 1.0
            
        # Script compliance
        if self.metrics.get("scripts"):
            total_scripts = self.metrics["scripts"]["total"]
            documented = self.metrics["scripts"]["documented"]
            scores["script_compliance"] = documented / total_scripts if total_scripts > 0 else 1.0
            
        # Overall score
        if scores:
            scores["overall"] = sum(scores.values()) / len(scores)
        else:
            scores["overall"] = 0.0
            
        return scores
        
    def _save_markdown_report(self, report: Dict[str, Any]):
        """Save human-readable markdown report."""
        md_path = self.report_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(md_path, 'w') as f:
            f.write("# Comprehensive Codebase Analysis Report\n\n")
            f.write(f"Generated: {report['analysis_timestamp']}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            summary = report['summary']
            f.write(f"- **Total Issues**: {summary['total_issues']}\n")
            f.write(f"  - Critical: {summary['critical']}\n")
            f.write(f"  - High: {summary['high']}\n")
            f.write(f"  - Medium: {summary['medium']}\n")
            f.write(f"  - Low: {summary['low']}\n")
            f.write(f"- **Total Recommendations**: {summary['total_recommendations']}\n\n")
            
            # Compliance Scores
            f.write("## Compliance Scores\n\n")
            for key, score in report['compliance_score'].items():
                f.write(f"- **{key.replace('_', ' ').title()}**: {score:.1%}\n")
            f.write("\n")
            
            # Issues by Category
            f.write("## Issues by Category\n\n")
            for category, issues in report['issues'].items():
                if issues:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    for issue in issues[:10]:  # Show first 10
                        f.write(f"- {issue['message']}\n")
                    if len(issues) > 10:
                        f.write(f"- ... and {len(issues) - 10} more\n")
                    f.write("\n")
                    
            # Recommendations
            f.write("## Recommendations\n\n")
            for category, recs in report['recommendations'].items():
                if recs:
                    f.write(f"### {category.replace('_', ' ').title()}\n\n")
                    for rec in recs[:5]:  # Show first 5
                        f.write(f"- {rec['message']}\n")
                    if len(recs) > 5:
                        f.write(f"- ... and {len(recs) - 5} more\n")
                    f.write("\n")
                    
            # Metrics
            f.write("## Detailed Metrics\n\n")
            for category, metrics in report['metrics'].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for key, value in metrics.items():
                    f.write(f"- **{key.replace('_', ' ').title()}**: {value}\n")
                f.write("\n")
                
    def fix_issues(self):
        """Attempt to automatically fix certain issues."""
        logger.info("Attempting to fix issues...")
        
        fixed_count = 0
        
        # Fix file permissions for shell scripts
        for issue in self.issues.get("scripts", []):
            if issue["type"] == "permission":
                script_path = self.project_root / issue["path"]
                try:
                    os.chmod(script_path, 0o755)
                    fixed_count += 1
                    logger.info(f"Fixed permissions for {script_path}")
                except Exception as e:
                    logger.error(f"Failed to fix permissions for {script_path}: {e}")
                    
        # Remove empty folders
        for issue in self.issues.get("folders", []):
            if issue["type"] == "empty":
                folder_path = self.project_root / issue["path"]
                try:
                    if folder_path.exists() and not any(folder_path.iterdir()):
                        folder_path.rmdir()
                        fixed_count += 1
                        logger.info(f"Removed empty folder {folder_path}")
                except Exception as e:
                    logger.error(f"Failed to remove empty folder {folder_path}: {e}")
                    
        logger.info(f"Fixed {fixed_count} issues")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis Agent - Enforces Rule 3: Analyze Everything—Every Time"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Directory to save reports (default: PROJECT_ROOT/reports)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Report format (default: both)"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to automatically fix certain issues"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ComprehensiveAnalysisAgent(
        project_root=project_root,
        report_dir=args.report_dir
    )
    
    # Run analysis
    report = agent.analyze(fix_issues=args.fix)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE CODEBASE ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTotal Issues Found: {report['summary']['total_issues']}")
    print(f"  - Critical: {report['summary']['critical']}")
    print(f"  - High: {report['summary']['high']}")
    print(f"  - Medium: {report['summary']['medium']}")
    print(f"  - Low: {report['summary']['low']}")
    print(f"\nTotal Recommendations: {report['summary']['total_recommendations']}")
    print(f"\nOverall Compliance Score: {report['compliance_score']['overall']:.1%}")
    print("\nReports saved to:", agent.report_dir)
    
    # Exit with appropriate code
    if report['summary']['critical'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()