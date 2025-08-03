#!/usr/bin/env python3
"""
Functionality Preservation Validator Agent

Purpose: Enforces Rule 2 - Do Not Break Existing Functionality
This agent analyzes code changes to prevent regressions and ensure backward compatibility.

Usage: python functionality-preservation-validator.py [command] [options]
Requirements: ast, git, pytest, json, networkx for dependency graphs

Commands:
- validate: Run full validation on current changes
- analyze: Analyze specific files for breaking changes
- test: Run test comparison before/after changes
- graph: Generate dependency graph
- report: Create impact analysis report

Author: Functionality Preservation Agent
Version: 1.0.0
"""

import ast
import json
import os
import sys
import subprocess
import argparse
import logging
import hashlib
import difflib
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import re


@dataclass
class FunctionSignature:
    """Represents a function signature for comparison."""
    name: str
    args: List[str]
    defaults: List[str]
    return_annotation: Optional[str]
    decorators: List[str]
    docstring: Optional[str]
    file_path: str
    line_number: int


@dataclass
class ClassDefinition:
    """Represents a class definition for comparison."""
    name: str
    bases: List[str]
    methods: List[str]
    attributes: List[str]
    decorators: List[str]
    docstring: Optional[str]
    file_path: str
    line_number: int


@dataclass
class APIEndpoint:
    """Represents an API endpoint definition."""
    method: str
    path: str
    function_name: str
    parameters: List[str]
    response_schema: Optional[Dict]
    file_path: str
    line_number: int


@dataclass
class ValidationResult:
    """Represents the result of a validation check."""
    status: str  # "pass", "fail", "warning"
    category: str
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    breaking_change: bool = False


class ASTAnalyzer:
    """Analyzes Python AST for function and class definitions."""
    
    def __init__(self):
        self.functions: Dict[str, FunctionSignature] = {}
        self.classes: Dict[str, ClassDefinition] = {}
        self.imports: Dict[str, Set[str]] = {}
        self.api_endpoints: List[APIEndpoint] = []
        
    def analyze_file(self, file_path: str) -> bool:
        """Analyze a Python file and extract definitions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            self._extract_definitions(tree, file_path)
            return True
            
        except (SyntaxError, UnicodeDecodeError) as e:
            logging.warning(f"Could not parse {file_path}: {e}")
            return False
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
            return False
    
    def _extract_definitions(self, tree: ast.AST, file_path: str):
        """Extract function and class definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                self._extract_function(node, file_path)
            elif isinstance(node, ast.ClassDef):
                self._extract_class(node, file_path)
            elif isinstance(node, ast.Import):
                self._extract_import(node, file_path)
            elif isinstance(node, ast.ImportFrom):
                self._extract_import_from(node, file_path)
    
    def _extract_function(self, node: ast.FunctionDef, file_path: str):
        """Extract function signature information."""
        args = []
        defaults = []
        
        # Extract arguments
        for arg in node.args.args:
            args.append(arg.arg)
        
        # Extract default values
        for default in node.args.defaults:
            if isinstance(default, ast.Constant):
                defaults.append(str(default.value))
            else:
                defaults.append(ast.unparse(default))
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            else:
                decorators.append(ast.unparse(decorator))
        
        # Extract return annotation
        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns)
        
        # Extract docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        signature = FunctionSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            return_annotation=return_annotation,
            decorators=decorators,
            docstring=docstring,
            file_path=file_path,
            line_number=node.lineno
        )
        
        key = f"{file_path}::{node.name}"
        self.functions[key] = signature
        
        # Check for API endpoints (Flask/FastAPI patterns)
        self._check_api_endpoint(node, file_path)
    
    def _extract_class(self, node: ast.ClassDef, file_path: str):
        """Extract class definition information."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            else:
                bases.append(ast.unparse(base))
        
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            else:
                decorators.append(ast.unparse(decorator))
        
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) 
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        class_def = ClassDefinition(
            name=node.name,
            bases=bases,
            methods=methods,
            attributes=attributes,
            decorators=decorators,
            docstring=docstring,
            file_path=file_path,
            line_number=node.lineno
        )
        
        key = f"{file_path}::{node.name}"
        self.classes[key] = class_def
    
    def _extract_import(self, node: ast.Import, file_path: str):
        """Extract import statements."""
        if file_path not in self.imports:
            self.imports[file_path] = set()
        
        for alias in node.names:
            self.imports[file_path].add(alias.name)
    
    def _extract_import_from(self, node: ast.ImportFrom, file_path: str):
        """Extract from-import statements."""
        if file_path not in self.imports:
            self.imports[file_path] = set()
        
        module = node.module or ""
        for alias in node.names:
            self.imports[file_path].add(f"{module}.{alias.name}")
    
    def _check_api_endpoint(self, node: ast.FunctionDef, file_path: str):
        """Check if function is an API endpoint."""
        for decorator in node.decorator_list:
            decorator_str = ast.unparse(decorator)
            
            # Flask patterns
            flask_match = re.search(r'@app\.route\(["\']([^"\']+)["\'].*method.*["\']([^"\']+)["\']', decorator_str)
            if flask_match:
                path, method = flask_match.groups()
                self._add_api_endpoint(method.upper(), path, node.name, node.args.args, file_path, node.lineno)
                continue
            
            # FastAPI patterns
            fastapi_methods = ['get', 'post', 'put', 'delete', 'patch']
            for method in fastapi_methods:
                if f'@app.{method}(' in decorator_str or f'@router.{method}(' in decorator_str:
                    path_match = re.search(r'["\']([^"\']+)["\']', decorator_str)
                    if path_match:
                        path = path_match.group(1)
                        self._add_api_endpoint(method.upper(), path, node.name, node.args.args, file_path, node.lineno)
    
    def _add_api_endpoint(self, method: str, path: str, function_name: str, args: List, file_path: str, line_number: int):
        """Add an API endpoint to the list."""
        parameters = [arg.arg for arg in args if arg.arg != 'self']
        
        endpoint = APIEndpoint(
            method=method,
            path=path,
            function_name=function_name,
            parameters=parameters,
            response_schema=None,  # Could be enhanced to extract from type hints
            file_path=file_path,
            line_number=line_number
        )
        
        self.api_endpoints.append(endpoint)


class GitAnalyzer:
    """Analyzes git changes and their impact."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
    
    def get_modified_files(self, base_branch: str = "main") -> List[str]:
        """Get list of modified files compared to base branch."""
        try:
            cmd = ["git", "diff", "--name-only", f"{base_branch}...HEAD"]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            else:
                logging.warning(f"Git diff failed: {result.stderr}")
                return []
        except Exception as e:
            logging.error(f"Error getting modified files: {e}")
            return []
    
    def get_file_diff(self, file_path: str, base_branch: str = "main") -> str:
        """Get diff for a specific file."""
        try:
            cmd = ["git", "diff", f"{base_branch}...HEAD", "--", file_path]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return ""
        except Exception as e:
            logging.error(f"Error getting file diff for {file_path}: {e}")
            return ""
    
    def get_deleted_files(self, base_branch: str = "main") -> List[str]:
        """Get list of deleted files."""
        try:
            cmd = ["git", "diff", "--name-only", "--diff-filter=D", f"{base_branch}...HEAD"]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            else:
                return []
        except Exception as e:
            logging.error(f"Error getting deleted files: {e}")
            return []
    
    def stash_changes(self) -> bool:
        """Stash current changes."""
        try:
            cmd = ["git", "stash", "push", "-m", "functionality-validator-backup"]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Error stashing changes: {e}")
            return False
    
    def pop_stash(self) -> bool:
        """Pop the last stash."""
        try:
            cmd = ["git", "stash", "pop"]
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logging.error(f"Error popping stash: {e}")
            return False


class TestRunner:
    """Runs tests and compares results."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
    
    def run_tests(self, test_path: str = "tests/", output_file: str = None) -> Dict[str, Any]:
        """Run tests and return results."""
        try:
            cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short", "--json-report"]
            if output_file:
                cmd.extend(["--json-report-file", output_file])
            
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            
            test_result = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "passed": result.returncode == 0
            }
            
            # Try to parse JSON report if available
            if output_file and os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        test_result["json_report"] = json.load(f)
                except Exception:
                    pass
            
            return test_result
            
        except Exception as e:
            logging.error(f"Error running tests: {e}")
            return {"returncode": -1, "passed": False, "error": str(e)}
    
    def compare_test_results(self, before: Dict[str, Any], after: Dict[str, Any]) -> List[ValidationResult]:
        """Compare test results and identify regressions."""
        results = []
        
        # Basic comparison
        if before["passed"] and not after["passed"]:
            results.append(ValidationResult(
                status="fail",
                category="test_regression",
                message="Tests that previously passed are now failing",
                breaking_change=True,
                suggestion="Review failing tests and fix regressions before committing"
            ))
        
        # Detailed comparison if JSON reports available
        if "json_report" in before and "json_report" in after:
            before_report = before["json_report"]
            after_report = after["json_report"]
            
            before_failed = {test["nodeid"] for test in before_report.get("tests", []) if test["outcome"] == "failed"}
            after_failed = {test["nodeid"] for test in after_report.get("tests", []) if test["outcome"] == "failed"}
            
            newly_failed = after_failed - before_failed
            newly_passed = before_failed - after_failed
            
            for test in newly_failed:
                results.append(ValidationResult(
                    status="fail",
                    category="test_regression",
                    message=f"Test {test} started failing",
                    breaking_change=True
                ))
            
            for test in newly_passed:
                results.append(ValidationResult(
                    status="pass",
                    category="test_improvement",
                    message=f"Test {test} is now passing"
                ))
        
        return results


class FunctionalityPreservationValidator:
    """Main validator class that orchestrates all checks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.git_analyzer = GitAnalyzer(repo_path)
        self.ast_analyzer = ASTAnalyzer()
        self.test_runner = TestRunner(repo_path)
        self.validation_results: List[ValidationResult] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def validate_changes(self, base_branch: str = "main") -> List[ValidationResult]:
        """Run full validation on current changes."""
        self.validation_results = []
        
        logging.info("Starting functionality preservation validation...")
        
        # Get modified files
        modified_files = self.git_analyzer.get_modified_files(base_branch)
        deleted_files = self.git_analyzer.get_deleted_files(base_branch)
        
        if not modified_files and not deleted_files:
            self.validation_results.append(ValidationResult(
                status="pass",
                category="no_changes",
                message="No files were modified"
            ))
            return self.validation_results
        
        logging.info(f"Found {len(modified_files)} modified files and {len(deleted_files)} deleted files")
        
        # Analyze current state
        current_state = self._analyze_codebase()
        
        # Stash changes and analyze previous state
        if self.git_analyzer.stash_changes():
            try:
                previous_state = self._analyze_codebase()
                self._compare_states(previous_state, current_state)
            finally:
                self.git_analyzer.pop_stash()
        else:
            logging.warning("Could not stash changes, skipping state comparison")
        
        # Check deleted files
        self._check_deleted_files(deleted_files)
        
        # Run test comparison
        self._run_test_comparison(base_branch)
        
        # Check API compatibility
        self._check_api_compatibility()
        
        # Generate dependency impact analysis
        self._analyze_dependency_impact(modified_files)
        
        return self.validation_results
    
    def _analyze_codebase(self) -> Dict[str, Any]:
        """Analyze current codebase state."""
        state = {
            "functions": {},
            "classes": {},
            "imports": {},
            "api_endpoints": []
        }
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        # Analyze each file
        analyzer = ASTAnalyzer()
        for file_path in python_files:
            analyzer.analyze_file(file_path)
        
        state["functions"] = analyzer.functions
        state["classes"] = analyzer.classes
        state["imports"] = analyzer.imports
        state["api_endpoints"] = analyzer.api_endpoints
        
        return state
    
    def _compare_states(self, previous: Dict[str, Any], current: Dict[str, Any]):
        """Compare two codebase states and identify breaking changes."""
        # Compare functions
        self._compare_functions(previous["functions"], current["functions"])
        
        # Compare classes
        self._compare_classes(previous["classes"], current["classes"])
        
        # Compare API endpoints
        self._compare_api_endpoints(previous["api_endpoints"], current["api_endpoints"])
        
        # Compare imports
        self._compare_imports(previous["imports"], current["imports"])
    
    def _compare_functions(self, previous: Dict[str, FunctionSignature], current: Dict[str, FunctionSignature]):
        """Compare function signatures for breaking changes."""
        # Check for removed functions
        removed_functions = set(previous.keys()) - set(current.keys())
        for func_key in removed_functions:
            func = previous[func_key]
            self.validation_results.append(ValidationResult(
                status="fail",
                category="function_removed",
                message=f"Function '{func.name}' was removed",
                file_path=func.file_path,
                line_number=func.line_number,
                breaking_change=True,
                suggestion="Consider deprecating instead of removing, or ensure no code depends on this function"
            ))
        
        # Check for modified function signatures
        common_functions = set(previous.keys()) & set(current.keys())
        for func_key in common_functions:
            prev_func = previous[func_key]
            curr_func = current[func_key]
            
            # Check argument changes
            if prev_func.args != curr_func.args:
                self.validation_results.append(ValidationResult(
                    status="fail",
                    category="function_signature_changed",
                    message=f"Function '{prev_func.name}' signature changed from {prev_func.args} to {curr_func.args}",
                    file_path=curr_func.file_path,
                    line_number=curr_func.line_number,
                    breaking_change=True,
                    suggestion="Maintain backward compatibility by keeping original parameters or using *args/**kwargs"
                ))
            
            # Check default value changes
            if prev_func.defaults != curr_func.defaults:
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="function_defaults_changed",
                    message=f"Function '{prev_func.name}' default values changed",
                    file_path=curr_func.file_path,
                    line_number=curr_func.line_number,
                    suggestion="Verify that default value changes don't break existing callers"
                ))
        
        # Check for new functions (informational)
        new_functions = set(current.keys()) - set(previous.keys())
        for func_key in new_functions:
            func = current[func_key]
            self.validation_results.append(ValidationResult(
                status="pass",
                category="function_added",
                message=f"New function '{func.name}' added",
                file_path=func.file_path,
                line_number=func.line_number
            ))
    
    def _compare_classes(self, previous: Dict[str, ClassDefinition], current: Dict[str, ClassDefinition]):
        """Compare class definitions for breaking changes."""
        # Check for removed classes
        removed_classes = set(previous.keys()) - set(current.keys())
        for class_key in removed_classes:
            cls = previous[class_key]
            self.validation_results.append(ValidationResult(
                status="fail",
                category="class_removed",
                message=f"Class '{cls.name}' was removed",
                file_path=cls.file_path,
                line_number=cls.line_number,
                breaking_change=True,
                suggestion="Consider deprecating instead of removing, or ensure no code depends on this class"
            ))
        
        # Check for modified classes
        common_classes = set(previous.keys()) & set(current.keys())
        for class_key in common_classes:
            prev_class = previous[class_key]
            curr_class = current[class_key]
            
            # Check for removed methods
            removed_methods = set(prev_class.methods) - set(curr_class.methods)
            for method in removed_methods:
                self.validation_results.append(ValidationResult(
                    status="fail",
                    category="method_removed",
                    message=f"Method '{method}' removed from class '{prev_class.name}'",
                    file_path=curr_class.file_path,
                    line_number=curr_class.line_number,
                    breaking_change=True,
                    suggestion="Maintain backward compatibility by keeping the method or marking it as deprecated"
                ))
            
            # Check for inheritance changes
            if prev_class.bases != curr_class.bases:
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="class_inheritance_changed",
                    message=f"Class '{prev_class.name}' inheritance changed from {prev_class.bases} to {curr_class.bases}",
                    file_path=curr_class.file_path,
                    line_number=curr_class.line_number,
                    suggestion="Verify that inheritance changes don't break polymorphism or method resolution"
                ))
    
    def _compare_api_endpoints(self, previous: List[APIEndpoint], current: List[APIEndpoint]):
        """Compare API endpoints for breaking changes."""
        prev_endpoints = {(ep.method, ep.path): ep for ep in previous}
        curr_endpoints = {(ep.method, ep.path): ep for ep in current}
        
        # Check for removed endpoints
        removed_endpoints = set(prev_endpoints.keys()) - set(curr_endpoints.keys())
        for method, path in removed_endpoints:
            ep = prev_endpoints[(method, path)]
            self.validation_results.append(ValidationResult(
                status="fail",
                category="api_endpoint_removed",
                message=f"API endpoint {method} {path} was removed",
                file_path=ep.file_path,
                line_number=ep.line_number,
                breaking_change=True,
                suggestion="Consider deprecating the endpoint with proper versioning instead of removing"
            ))
        
        # Check for parameter changes in existing endpoints
        common_endpoints = set(prev_endpoints.keys()) & set(curr_endpoints.keys())
        for method, path in common_endpoints:
            prev_ep = prev_endpoints[(method, path)]
            curr_ep = curr_endpoints[(method, path)]
            
            if prev_ep.parameters != curr_ep.parameters:
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="api_parameters_changed",
                    message=f"API endpoint {method} {path} parameters changed",
                    file_path=curr_ep.file_path,
                    line_number=curr_ep.line_number,
                    suggestion="Ensure parameter changes maintain backward compatibility"
                ))
    
    def _compare_imports(self, previous: Dict[str, Set[str]], current: Dict[str, Set[str]]):
        """Compare imports to detect potential breaking changes."""
        for file_path in previous:
            if file_path in current:
                removed_imports = previous[file_path] - current[file_path]
                for import_name in removed_imports:
                    self.validation_results.append(ValidationResult(
                        status="warning",
                        category="import_removed",
                        message=f"Import '{import_name}' removed from {file_path}",
                        file_path=file_path,
                        suggestion="Verify that removing this import doesn't break functionality"
                    ))
    
    def _check_deleted_files(self, deleted_files: List[str]):
        """Check impact of deleted files."""
        for file_path in deleted_files:
            # Check if deleted file is imported elsewhere
            if self._is_file_imported(file_path):
                self.validation_results.append(ValidationResult(
                    status="fail",
                    category="imported_file_deleted",
                    message=f"Deleted file '{file_path}' is imported by other modules",
                    file_path=file_path,
                    breaking_change=True,
                    suggestion="Update imports in dependent files before deleting this file"
                ))
    
    def _is_file_imported(self, file_path: str) -> bool:
        """Check if a file is imported by other modules."""
        # Convert file path to module name
        module_name = file_path.replace('/', '.').replace('.py', '')
        
        # Search for imports in all Python files
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__']]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    if full_path != file_path:  # Don't check the file itself
                        try:
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if module_name in content or file_path in content:
                                    return True
                        except Exception:
                            continue
        
        return False
    
    def _run_test_comparison(self, base_branch: str):
        """Run tests before and after changes for comparison."""
        logging.info("Running test comparison...")
        
        # Run tests on current state
        current_results = self.test_runner.run_tests()
        
        # Stash changes and run tests on previous state
        if self.git_analyzer.stash_changes():
            try:
                previous_results = self.test_runner.run_tests()
                test_comparison = self.test_runner.compare_test_results(previous_results, current_results)
                self.validation_results.extend(test_comparison)
            finally:
                self.git_analyzer.pop_stash()
        else:
            logging.warning("Could not compare test results - unable to stash changes")
    
    def _check_api_compatibility(self):
        """Check API compatibility and schema validation."""
        # This is a placeholder for more sophisticated API compatibility checks
        logging.info("Checking API compatibility...")
        
        # Look for configuration files that define API schemas
        config_files = ['openapi.yaml', 'swagger.yaml', 'api_schema.json']
        for config_file in config_files:
            if os.path.exists(os.path.join(self.repo_path, config_file)):
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="api_schema_check",
                    message=f"API schema file {config_file} found - manual review recommended",
                    suggestion="Validate that API changes are reflected in schema documentation"
                ))
    
    def _analyze_dependency_impact(self, modified_files: List[str]):
        """Analyze the impact of changes on dependencies."""
        logging.info("Analyzing dependency impact...")
        
        # Check if requirements files were modified
        requirements_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']
        for file_path in modified_files:
            if any(req_file in file_path for req_file in requirements_files):
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="dependencies_changed",
                    message=f"Dependency file {file_path} was modified",
                    file_path=file_path,
                    suggestion="Verify that dependency changes are compatible and test thoroughly"
                ))
        
        # Check configuration files
        config_files = ['.env', 'config.py', 'settings.py']
        for file_path in modified_files:
            if any(config_file in file_path for config_file in config_files):
                self.validation_results.append(ValidationResult(
                    status="warning",
                    category="configuration_changed",
                    message=f"Configuration file {file_path} was modified",
                    file_path=file_path,
                    suggestion="Ensure configuration changes are backward compatible"
                ))
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.validation_results),
                "passed": len([r for r in self.validation_results if r.status == "pass"]),
                "warnings": len([r for r in self.validation_results if r.status == "warning"]),
                "failures": len([r for r in self.validation_results if r.status == "fail"]),
                "breaking_changes": len([r for r in self.validation_results if r.breaking_change])
            },
            "results": [asdict(result) for result in self.validation_results],
            "recommendations": self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Report saved to {output_file}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        breaking_changes = [r for r in self.validation_results if r.breaking_change]
        if breaking_changes:
            recommendations.append("❌ CRITICAL: Breaking changes detected. Review all failures before committing.")
            recommendations.append("Consider using feature flags or deprecation warnings for gradual migration.")
        
        warnings = [r for r in self.validation_results if r.status == "warning"]
        if warnings:
            recommendations.append("⚠️  Warnings detected. Manual review recommended.")
        
        api_changes = [r for r in self.validation_results if "api" in r.category]
        if api_changes:
            recommendations.append("🔗 API changes detected. Update documentation and notify API consumers.")
        
        test_failures = [r for r in self.validation_results if "test" in r.category and r.status == "fail"]
        if test_failures:
            recommendations.append("🧪 Test regressions detected. Fix failing tests before deployment.")
        
        if not breaking_changes and not test_failures:
            recommendations.append("✅ No critical issues detected. Changes appear safe to merge.")
        
        return recommendations
    
    def should_block_commit(self) -> bool:
        """Determine if the commit should be blocked based on validation results."""
        breaking_changes = [r for r in self.validation_results if r.breaking_change]
        test_failures = [r for r in self.validation_results if "test" in r.category and r.status == "fail"]
        
        return len(breaking_changes) > 0 or len(test_failures) > 0


def setup_git_hooks():
    """Setup git hooks to run validation automatically."""
    hooks_dir = os.path.join(".git", "hooks")
    pre_commit_hook = os.path.join(hooks_dir, "pre-commit")
    
    hook_content = """#!/bin/bash
# Functionality Preservation Validator Git Hook
# Auto-generated by functionality-preservation-validator.py

echo "🔍 Running functionality preservation validation..."

python scripts/agents/functionality-preservation-validator.py validate --format=summary

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Commit blocked."
    echo "Run 'python scripts/agents/functionality-preservation-validator.py validate --format=detailed' for details."
    exit 1
fi

echo "✅ Validation passed. Proceeding with commit."
"""
    
    try:
        os.makedirs(hooks_dir, exist_ok=True)
        with open(pre_commit_hook, 'w') as f:
            f.write(hook_content)
        os.chmod(pre_commit_hook, 0o755)
        print(f"✅ Git pre-commit hook installed at {pre_commit_hook}")
        return True
    except Exception as e:
        print(f"❌ Failed to install git hook: {e}")
        return False


def main():
    """Main entry point for the functionality preservation validator."""
    parser = argparse.ArgumentParser(
        description="Functionality Preservation Validator - Enforces Rule 2: Do Not Break Existing Functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python functionality-preservation-validator.py validate
  python functionality-preservation-validator.py validate --base-branch develop
  python functionality-preservation-validator.py analyze --files src/api.py src/models.py
  python functionality-preservation-validator.py setup-hooks
  python functionality-preservation-validator.py report --output validation-report.json
        """
    )
    
    parser.add_argument(
        "command",
        choices=["validate", "analyze", "test", "report", "setup-hooks"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Base branch for comparison (default: main)"
    )
    
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to analyze"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for reports"
    )
    
    parser.add_argument(
        "--format",
        choices=["summary", "detailed", "json"],
        default="detailed",
        help="Output format (default: detailed)"
    )
    
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to repository (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = FunctionalityPreservationValidator(args.repo_path)
    
    try:
        if args.command == "setup-hooks":
            success = setup_git_hooks()
            sys.exit(0 if success else 1)
        
        elif args.command == "validate":
            results = validator.validate_changes(args.base_branch)
            
            if args.format == "json":
                report = validator.generate_report()
                print(json.dumps(report, indent=2))
            elif args.format == "summary":
                breaking_changes = len([r for r in results if r.breaking_change])
                warnings = len([r for r in results if r.status == "warning"])
                failures = len([r for r in results if r.status == "fail"])
                
                print(f"Validation Results: {len(results)} total checks")
                print(f"  ❌ Breaking Changes: {breaking_changes}")
                print(f"  ⚠️  Warnings: {warnings}")
                print(f"  🚫 Failures: {failures}")
                
                if validator.should_block_commit():
                    print("\n❌ COMMIT BLOCKED: Critical issues detected")
                    sys.exit(1)
                else:
                    print("\n✅ Validation passed")
            else:  # detailed
                for result in results:
                    status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[result.status]
                    print(f"{status_icon} [{result.category}] {result.message}")
                    if result.file_path:
                        print(f"   📁 {result.file_path}:{result.line_number or 'N/A'}")
                    if result.suggestion:
                        print(f"   💡 {result.suggestion}")
                    print()
                
                if validator.should_block_commit():
                    print("❌ CRITICAL ISSUES DETECTED - COMMIT SHOULD BE BLOCKED")
                    sys.exit(1)
        
        elif args.command == "report":
            results = validator.validate_changes(args.base_branch)
            report = validator.generate_report(args.output)
            
            if not args.output:
                print(json.dumps(report, indent=2))
        
        elif args.command == "analyze":
            if not args.files:
                print("Error: --files required for analyze command")
                sys.exit(1)
            
            # Analyze specific files
            for file_path in args.files:
                if validator.ast_analyzer.analyze_file(file_path):
                    print(f"✅ Analyzed {file_path}")
                else:
                    print(f"❌ Failed to analyze {file_path}")
        
        elif args.command == "test":
            # Run test comparison
            validator._run_test_comparison(args.base_branch)
            test_results = [r for r in validator.validation_results if "test" in r.category]
            
            for result in test_results:
                status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[result.status]
                print(f"{status_icon} {result.message}")
    
    except KeyboardInterrupt:
        print("\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()