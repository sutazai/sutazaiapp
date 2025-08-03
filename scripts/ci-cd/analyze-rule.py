#!/usr/bin/env python3
"""
Purpose: Analyzes a specific hygiene rule and generates violation report
Usage: python analyze-rule.py --rule 1 --project-root /path --output violations.json
Requirements: Python 3.8+, gitpython, pathlib
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import re
import ast


class RuleAnalyzer:
    """Analyzes specific CLAUDE.md hygiene rules"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.violations = []
    
    def analyze_rule(self, rule_number: int) -> List[Dict[str, Any]]:
        """Analyze a specific rule and return violations"""
        
        rule_methods = {
            1: self.analyze_rule_1_fantasy_elements,
            2: self.analyze_rule_2_no_breaking_changes,
            3: self.analyze_rule_3_analyze_everything,
            4: self.analyze_rule_4_reuse_before_creating,
            5: self.analyze_rule_5_professional_project,
            6: self.analyze_rule_6_documentation,
            7: self.analyze_rule_7_script_organization,
            8: self.analyze_rule_8_python_standards,
            9: self.analyze_rule_9_version_control,
            10: self.analyze_rule_10_functionality_first,
            11: self.analyze_rule_11_docker_structure,
            12: self.analyze_rule_12_deployment_script,
            13: self.analyze_rule_13_no_garbage,
            14: self.analyze_rule_14_correct_ai_agent,
            15: self.analyze_rule_15_documentation_dedup,
            16: self.analyze_rule_16_local_llms
        }
        
        analyzer = rule_methods.get(rule_number)
        if analyzer:
            return analyzer()
        else:
            return [{
                "rule": rule_number,
                "file": "unknown",
                "message": f"No analyzer implemented for rule {rule_number}",
                "severity": "info"
            }]
    
    def analyze_rule_1_fantasy_elements(self) -> List[Dict[str, Any]]:
        """Rule 1: No fantasy elements"""
        violations = []
        
        # Fantasy/magic keywords to check
        fantasy_patterns = [
            r'\bmagic\w*\b',
            r'\bwizard\w*\b',
            r'\bteleport\w*\b',
            r'\bblack[\s-]?box\b',
            r'\bsomeday\b',
            r'\bmagically\b',
            r'# TODO:.*magic',
            r'superIntuitive\w*',
            r'dreamAPI'
        ]
        
        # Check Python and JavaScript files
        for pattern in ["**/*.py", "**/*.js", "**/*.ts"]:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        for fantasy_pattern in fantasy_patterns:
                            if re.search(fantasy_pattern, line, re.IGNORECASE):
                                violations.append({
                                    "rule": 1,
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "line": i,
                                    "message": f"Fantasy element found: {fantasy_pattern}",
                                    "severity": "critical",
                                    "content": line.strip()
                                })
                except Exception as e:
                    pass
        
        return violations
    
    def analyze_rule_2_no_breaking_changes(self) -> List[Dict[str, Any]]:
        """Rule 2: Don't break existing functionality"""
        violations = []
        
        # This rule is harder to detect automatically
        # Check for common breaking change patterns
        
        patterns = [
            (r'def (\w+)\(.*\):', r'Function signature changed: {}'),
            (r'class (\w+)', r'Class definition changed: {}'),
            (r'DELETE FROM', r'Destructive database operation'),
            (r'DROP TABLE', r'Destructive database operation'),
            (r'rm -rf', r'Dangerous file removal command')
        ]
        
        # Check recent git changes
        try:
            # Get list of modified files
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                modified_files = result.stdout.strip().split('\n')
                
                for file_name in modified_files:
                    if not file_name:
                        continue
                        
                    file_path = self.project_root / file_name
                    if file_path.exists() and file_path.is_file():
                        # Check for removed functions/classes
                        diff_result = subprocess.run(
                            ['git', 'diff', 'HEAD~1', '--', file_name],
                            capture_output=True,
                            text=True,
                            cwd=self.project_root
                        )
                        
                        if diff_result.returncode == 0:
                            diff_lines = diff_result.stdout.split('\n')
                            for line in diff_lines:
                                if line.startswith('-') and not line.startswith('---'):
                                    # Check if removed line contains function/class definition
                                    for pattern, message in patterns[:2]:
                                        match = re.search(pattern, line[1:])
                                        if match:
                                            violations.append({
                                                "rule": 2,
                                                "file": file_name,
                                                "message": message.format(match.group(1)),
                                                "severity": "critical",
                                                "content": line[1:].strip()
                                            })
        except Exception:
            pass
        
        return violations
    
    def analyze_rule_3_analyze_everything(self) -> List[Dict[str, Any]]:
        """Rule 3: Analyze everything every time"""
        # This is more of a process rule, check for incomplete analysis
        violations = []
        
        # Check for TODO/FIXME comments indicating incomplete work
        for pattern in ["**/*.py", "**/*.js", "**/*.yml", "**/*.sh"]:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        if re.search(r'(TODO|FIXME|HACK|XXX):', line):
                            violations.append({
                                "rule": 3,
                                "file": str(file_path.relative_to(self.project_root)),
                                "line": i,
                                "message": "Incomplete analysis/implementation marker found",
                                "severity": "medium",
                                "content": line.strip()
                            })
                except Exception:
                    pass
        
        return violations
    
    def analyze_rule_6_documentation(self) -> List[Dict[str, Any]]:
        """Rule 6: Clear, centralized documentation"""
        violations = []
        
        # Check for scattered documentation
        doc_files = list(self.project_root.glob("**/*.md"))
        doc_locations = {}
        
        for doc_file in doc_files:
            if self._should_skip_file(doc_file):
                continue
            
            # Group by directory
            parent = doc_file.parent
            if parent not in doc_locations:
                doc_locations[parent] = []
            doc_locations[parent].append(doc_file)
        
        # Check for documentation outside /docs
        docs_dir = self.project_root / "docs"
        for location, files in doc_locations.items():
            if location != docs_dir and location != self.project_root:
                for file in files:
                    violations.append({
                        "rule": 6,
                        "file": str(file.relative_to(self.project_root)),
                        "message": "Documentation file outside /docs directory",
                        "severity": "medium"
                    })
        
        # Check for poor formatting
        for doc_file in doc_files:
            if self._should_skip_file(doc_file):
                continue
            
            try:
                content = doc_file.read_text(encoding='utf-8')
                
                # Check for missing headers
                if not content.strip().startswith('#'):
                    violations.append({
                        "rule": 6,
                        "file": str(doc_file.relative_to(self.project_root)),
                        "message": "Documentation file missing header",
                        "severity": "low"
                    })
                
                # Check for extremely long lines
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if len(line) > 120 and not line.startswith('http'):
                        violations.append({
                            "rule": 6,
                            "file": str(doc_file.relative_to(self.project_root)),
                            "line": i,
                            "message": f"Line too long ({len(line)} chars)",
                            "severity": "low"
                        })
            except Exception:
                pass
        
        return violations
    
    def analyze_rule_7_script_organization(self) -> List[Dict[str, Any]]:
        """Rule 7: Eliminate script chaos"""
        violations = []
        
        # Expected script structure
        expected_dirs = {
            "dev": ["reset", "setup", "init", "start", "install"],
            "deploy": ["deploy", "release", "rollback"],
            "data": ["seed", "migrate", "backup", "restore"],
            "utils": ["clean", "format", "validate", "check"],
            "test": ["test", "coverage", "benchmark"]
        }
        
        scripts_dir = self.project_root / "scripts"
        
        # Find all scripts
        script_patterns = ["**/*.sh", "**/*.py"]
        for pattern in script_patterns:
            for script in self.project_root.glob(pattern):
                if self._should_skip_file(script) or not self._is_script(script):
                    continue
                
                script_name = script.stem.lower()
                script_location = script.parent
                
                # Check if script is in scripts directory
                if not str(script).startswith(str(scripts_dir)):
                    violations.append({
                        "rule": 7,
                        "file": str(script.relative_to(self.project_root)),
                        "message": "Script not in /scripts directory",
                        "severity": "high"
                    })
                    continue
                
                # Check if script is in correct subdirectory
                correct_dir = None
                for dir_name, keywords in expected_dirs.items():
                    if any(keyword in script_name for keyword in keywords):
                        correct_dir = dir_name
                        break
                
                if correct_dir:
                    expected_location = scripts_dir / correct_dir
                    if script_location != expected_location:
                        violations.append({
                            "rule": 7,
                            "file": str(script.relative_to(self.project_root)),
                            "message": f"Script should be in scripts/{correct_dir}/",
                            "severity": "medium"
                        })
        
        # Check for duplicate scripts
        script_names = {}
        for script in scripts_dir.rglob("*"):
            if script.is_file() and self._is_script(script):
                name = script.stem
                if name in script_names:
                    violations.append({
                        "rule": 7,
                        "file": str(script.relative_to(self.project_root)),
                        "message": f"Duplicate script name: {name}",
                        "severity": "high",
                        "duplicate_of": str(script_names[name].relative_to(self.project_root))
                    })
                else:
                    script_names[name] = script
        
        return violations
    
    def analyze_rule_8_python_standards(self) -> List[Dict[str, Any]]:
        """Rule 8: Python script sanity"""
        violations = []
        
        for py_file in self.project_root.glob("**/*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                # Check for shebang
                if lines and not lines[0].startswith('#!/usr/bin/env python'):
                    violations.append({
                        "rule": 8,
                        "file": str(py_file.relative_to(self.project_root)),
                        "line": 1,
                        "message": "Missing shebang line",
                        "severity": "low"
                    })
                
                # Check for docstring
                has_docstring = False
                for i, line in enumerate(lines[:10]):  # Check first 10 lines
                    if line.strip().startswith('"""') or line.strip().startswith("'''"):
                        has_docstring = True
                        break
                
                if not has_docstring:
                    violations.append({
                        "rule": 8,
                        "file": str(py_file.relative_to(self.project_root)),
                        "message": "Missing module docstring",
                        "severity": "medium"
                    })
                
                # Check for hardcoded values
                hardcoded_patterns = [
                    (r'["\']\/home\/\w+', "Hardcoded home directory path"),
                    (r'["\']\/tmp\/\w+', "Hardcoded temp path"),
                    (r'localhost:\d{4}', "Hardcoded localhost URL"),
                    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key")
                ]
                
                for i, line in enumerate(lines, 1):
                    for pattern, message in hardcoded_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append({
                                "rule": 8,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i,
                                "message": message,
                                "severity": "high",
                                "content": line.strip()
                            })
                
                # Check for proper main guard
                if "__main__" in content and "if __name__" not in content:
                    violations.append({
                        "rule": 8,
                        "file": str(py_file.relative_to(self.project_root)),
                        "message": "Missing if __name__ == '__main__' guard",
                        "severity": "medium"
                    })
                
            except Exception:
                pass
        
        return violations
    
    def analyze_rule_11_docker_structure(self) -> List[Dict[str, Any]]:
        """Rule 11: Docker structure must be clean"""
        violations = []
        
        docker_dir = self.project_root / "docker"
        
        # Find all Dockerfiles
        dockerfiles = list(self.project_root.glob("**/Dockerfile*"))
        
        for dockerfile in dockerfiles:
            if self._should_skip_file(dockerfile):
                continue
            
            # Check if Dockerfile is in proper location
            if dockerfile.parent != docker_dir and dockerfile.name != "Dockerfile":
                expected_location = docker_dir / dockerfile.parent.name / "Dockerfile"
                violations.append({
                    "rule": 11,
                    "file": str(dockerfile.relative_to(self.project_root)),
                    "message": f"Dockerfile should be at {expected_location}",
                    "severity": "high"
                })
            
            # Check Dockerfile contents
            try:
                content = dockerfile.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                # Check for version pinning
                for i, line in enumerate(lines, 1):
                    if line.startswith('FROM'):
                        if ':latest' in line or (' ' in line and ':' not in line.split()[1]):
                            violations.append({
                                "rule": 11,
                                "file": str(dockerfile.relative_to(self.project_root)),
                                "line": i,
                                "message": "Docker base image not version-pinned",
                                "severity": "high",
                                "content": line.strip()
                            })
                
                # Check for COPY vs ADD
                for i, line in enumerate(lines, 1):
                    if line.strip().startswith('ADD') and not any(
                        ext in line for ext in ['.tar', '.gz', '.tgz', 'http://', 'https://']
                    ):
                        violations.append({
                            "rule": 11,
                            "file": str(dockerfile.relative_to(self.project_root)),
                            "line": i,
                            "message": "Use COPY instead of ADD for simple file copying",
                            "severity": "low",
                            "content": line.strip()
                        })
                
            except Exception:
                pass
        
        # Check for .dockerignore
        dockerignore = self.project_root / ".dockerignore"
        if not dockerignore.exists():
            violations.append({
                "rule": 11,
                "file": ".dockerignore",
                "message": "Missing .dockerignore file",
                "severity": "medium"
            })
        
        return violations
    
    def analyze_rule_12_deployment_script(self) -> List[Dict[str, Any]]:
        """Rule 12: One self-updating deployment script"""
        violations = []
        
        # Find all deployment-related scripts
        deploy_patterns = ["deploy*.sh", "*deploy*.py", "release*.sh", "rollback*.sh"]
        deploy_scripts = []
        
        for pattern in deploy_patterns:
            deploy_scripts.extend(self.project_root.glob(f"**/{pattern}"))
        
        # Remove duplicates and filter
        deploy_scripts = list(set(s for s in deploy_scripts if not self._should_skip_file(s)))
        
        if len(deploy_scripts) > 1:
            # Find the canonical one (usually deploy.sh in root or scripts/deploy/)
            canonical = None
            for script in deploy_scripts:
                if script.name == "deploy.sh" and script.parent == self.project_root:
                    canonical = script
                    break
            
            if not canonical:
                for script in deploy_scripts:
                    if script.name == "deploy.sh":
                        canonical = script
                        break
            
            for script in deploy_scripts:
                if script != canonical:
                    violations.append({
                        "rule": 12,
                        "file": str(script.relative_to(self.project_root)),
                        "message": "Multiple deployment scripts found - should consolidate",
                        "severity": "high",
                        "canonical": str(canonical.relative_to(self.project_root)) if canonical else "deploy.sh"
                    })
        
        elif len(deploy_scripts) == 1:
            # Check if the single script is comprehensive
            script = deploy_scripts[0]
            try:
                content = script.read_text(encoding='utf-8')
                
                # Check for required components
                required_components = [
                    ("docker", "Docker deployment"),
                    ("migrate", "Database migrations"),
                    ("test", "Test execution"),
                    ("rollback", "Rollback capability"),
                    ("health", "Health checks")
                ]
                
                for keyword, component in required_components:
                    if keyword not in content.lower():
                        violations.append({
                            "rule": 12,
                            "file": str(script.relative_to(self.project_root)),
                            "message": f"Deployment script missing: {component}",
                            "severity": "medium"
                        })
                
            except Exception:
                pass
        
        else:
            violations.append({
                "rule": 12,
                "file": "deploy.sh",
                "message": "No deployment script found",
                "severity": "critical"
            })
        
        return violations
    
    def analyze_rule_13_no_garbage(self) -> List[Dict[str, Any]]:
        """Rule 13: No garbage, no rot"""
        violations = []
        
        # Garbage file patterns
        garbage_patterns = [
            ("*.bak", "Backup file"),
            ("*.backup", "Backup file"),
            ("*.old", "Old file"),
            ("*.orig", "Original file (merge leftover)"),
            ("*.tmp", "Temporary file"),
            ("*.temp", "Temporary file"),
            ("*.swp", "Vim swap file"),
            ("*.swo", "Vim swap file"),
            ("*~", "Editor backup file"),
            (".DS_Store", "macOS metadata"),
            ("Thumbs.db", "Windows metadata"),
            ("desktop.ini", "Windows metadata"),
            ("*.pyc", "Python bytecode"),
            ("__pycache__", "Python cache directory"),
            ("*.log.1", "Rotated log file"),
            ("*.log.2", "Rotated log file"),
            ("test*.txt", "Test file"),
            ("debug*.log", "Debug log"),
            ("*_copy.*", "Copy file"),
            ("*_old.*", "Old version"),
            ("* copy.*", "Copy file"),
            ("temp_*", "Temporary file"),
            ("tmp_*", "Temporary file")
        ]
        
        for pattern, description in garbage_patterns:
            for file_path in self.project_root.rglob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                if file_path.is_file():
                    violations.append({
                        "rule": 13,
                        "file": str(file_path.relative_to(self.project_root)),
                        "message": f"{description} should be removed",
                        "severity": "high" if pattern.endswith(('.bak', '.tmp', '.old')) else "medium"
                    })
        
        # Check for commented-out code
        code_patterns = ["**/*.py", "**/*.js", "**/*.java", "**/*.cpp"]
        for pattern in code_patterns:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    consecutive_comments = 0
                    start_line = 0
                    
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if stripped.startswith(('#', '//', '/*', '*')):
                            if consecutive_comments == 0:
                                start_line = i
                            consecutive_comments += 1
                        else:
                            if consecutive_comments > 10:  # More than 10 consecutive comment lines
                                violations.append({
                                    "rule": 13,
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "line": start_line,
                                    "message": f"Large block of commented code ({consecutive_comments} lines)",
                                    "severity": "medium"
                                })
                            consecutive_comments = 0
                
                except Exception:
                    pass
        
        return violations
    
    def analyze_rule_15_documentation_dedup(self) -> List[Dict[str, Any]]:
        """Rule 15: Keep documentation deduplicated"""
        violations = []
        
        # Find all documentation files
        doc_files = {}
        for doc_file in self.project_root.glob("**/*.md"):
            if self._should_skip_file(doc_file):
                continue
            
            # Group by similar names
            base_name = doc_file.stem.lower()
            base_name = re.sub(r'[-_\s]+', '', base_name)  # Normalize name
            
            if base_name not in doc_files:
                doc_files[base_name] = []
            doc_files[base_name].append(doc_file)
        
        # Check for duplicates
        for base_name, files in doc_files.items():
            if len(files) > 1:
                # Sort by path depth (prefer root-level docs)
                files.sort(key=lambda x: len(x.parts))
                
                primary = files[0]
                for duplicate in files[1:]:
                    violations.append({
                        "rule": 15,
                        "file": str(duplicate.relative_to(self.project_root)),
                        "message": f"Duplicate documentation file",
                        "severity": "medium",
                        "primary_file": str(primary.relative_to(self.project_root))
                    })
        
        # Check for duplicate content
        content_hashes = {}
        for doc_file in self.project_root.glob("**/*.md"):
            if self._should_skip_file(doc_file):
                continue
            
            try:
                content = doc_file.read_text(encoding='utf-8')
                # Simple hash of normalized content
                normalized = re.sub(r'\s+', ' ', content.strip())
                content_hash = hash(normalized)
                
                if content_hash in content_hashes and len(normalized) > 100:
                    violations.append({
                        "rule": 15,
                        "file": str(doc_file.relative_to(self.project_root)),
                        "message": "Duplicate documentation content",
                        "severity": "high",
                        "duplicate_of": str(content_hashes[content_hash].relative_to(self.project_root))
                    })
                else:
                    content_hashes[content_hash] = doc_file
                    
            except Exception:
                pass
        
        return violations
    
    def analyze_rule_4_reuse_before_creating(self) -> List[Dict[str, Any]]:
        """Rule 4: Reuse before creating"""
        violations = []
        
        # Look for similar function/class names
        function_names = {}
        class_names = {}
        
        for py_file in self.project_root.glob("**/*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        name = node.name.lower()
                        if name in function_names:
                            violations.append({
                                "rule": 4,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": node.lineno,
                                "message": f"Duplicate function '{node.name}' - consider reusing",
                                "severity": "medium",
                                "duplicate_in": str(function_names[name].relative_to(self.project_root))
                            })
                        else:
                            function_names[name] = py_file
                    
                    elif isinstance(node, ast.ClassDef):
                        name = node.name.lower()
                        if name in class_names:
                            violations.append({
                                "rule": 4,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": node.lineno,
                                "message": f"Duplicate class '{node.name}' - consider reusing",
                                "severity": "medium",
                                "duplicate_in": str(class_names[name].relative_to(self.project_root))
                            })
                        else:
                            class_names[name] = py_file
                            
            except Exception:
                pass
        
        return violations
    
    def analyze_rule_5_professional_project(self) -> List[Dict[str, Any]]:
        """Rule 5: Treat as professional project"""
        violations = []
        
        # Check for unprofessional patterns
        unprofessional_patterns = [
            (r'\btest\d+\b', "Test file with number suffix"),
            (r'\btemp\b', "Temporary file/variable"),
            (r'\basdf\b', "Keyboard mash variable name"),
            (r'\bfoo\b|\bbar\b', "Placeholder variable name"),
            (r'print\s*\(.*debug', "Debug print statement"),
            (r'console\.log\s*\(.*test', "Test console log"),
            (r'TODO:?\s*remove', "TODO marked for removal"),
            (r'HACK:?\s*', "HACK comment")
        ]
        
        for pattern in ["**/*.py", "**/*.js", "**/*.java"]:
            for file_path in self.project_root.glob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        for pattern, message in unprofessional_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                violations.append({
                                    "rule": 5,
                                    "file": str(file_path.relative_to(self.project_root)),
                                    "line": i,
                                    "message": message,
                                    "severity": "low",
                                    "content": line.strip()
                                })
                                
                except Exception:
                    pass
        
        return violations
    
    def analyze_rule_9_version_control(self) -> List[Dict[str, Any]]:
        """Rule 9: Backend & Frontend version control"""
        violations = []
        
        # Check for duplicate backend/frontend directories
        backend_dirs = []
        frontend_dirs = []
        
        for path in self.project_root.rglob("*"):
            if path.is_dir():
                name = path.name.lower()
                if "backend" in name:
                    backend_dirs.append(path)
                elif "frontend" in name:
                    frontend_dirs.append(path)
        
        # Check for multiples
        if len(backend_dirs) > 1:
            primary = min(backend_dirs, key=lambda x: len(x.parts))
            for dup in backend_dirs:
                if dup != primary:
                    violations.append({
                        "rule": 9,
                        "file": str(dup.relative_to(self.project_root)),
                        "message": "Duplicate backend directory",
                        "severity": "high",
                        "primary": str(primary.relative_to(self.project_root))
                    })
        
        if len(frontend_dirs) > 1:
            primary = min(frontend_dirs, key=lambda x: len(x.parts))
            for dup in frontend_dirs:
                if dup != primary:
                    violations.append({
                        "rule": 9,
                        "file": str(dup.relative_to(self.project_root)),
                        "message": "Duplicate frontend directory",
                        "severity": "high",
                        "primary": str(primary.relative_to(self.project_root))
                    })
        
        return violations
    
    def analyze_rule_10_functionality_first(self) -> List[Dict[str, Any]]:
        """Rule 10: Functionality-first cleanup"""
        # This is more of a process rule
        violations = []
        
        # Check for recent deletions without proper archiving
        try:
            # Check git log for recent deletions
            result = subprocess.run(
                ['git', 'log', '--diff-filter=D', '--summary', '-n', '20'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'delete mode' in line:
                        file_name = line.split()[-1]
                        # Check if file was archived
                        archive_dir = self.project_root / "archive"
                        if archive_dir.exists():
                            archived = any(file_name in str(f) for f in archive_dir.rglob("*"))
                            if not archived:
                                violations.append({
                                    "rule": 10,
                                    "file": file_name,
                                    "message": "File deleted without archiving",
                                    "severity": "medium"
                                })
                                
        except Exception:
            pass
        
        return violations
    
    def analyze_rule_14_correct_ai_agent(self) -> List[Dict[str, Any]]:
        """Rule 14: Engage correct AI agent"""
        violations = []
        
        # Check agent configuration files
        agent_configs = list(self.project_root.glob("**/agent*.json"))
        agent_configs.extend(self.project_root.glob("**/agent*.yaml"))
        agent_configs.extend(self.project_root.glob("**/agent*.yml"))
        
        for config_file in agent_configs:
            if self._should_skip_file(config_file):
                continue
            
            try:
                if config_file.suffix == '.json':
                    import json
                    with open(config_file) as f:
                        config = json.load(f)
                else:
                    import yaml
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                
                # Check for agent misconfigurations
                if isinstance(config, dict):
                    if 'agent' in config and 'capabilities' not in config:
                        violations.append({
                            "rule": 14,
                            "file": str(config_file.relative_to(self.project_root)),
                            "message": "Agent configuration missing capabilities",
                            "severity": "medium"
                        })
                        
            except Exception:
                pass
        
        return violations
    
    def analyze_rule_16_local_llms(self) -> List[Dict[str, Any]]:
        """Rule 16: Use local LLMs via Ollama"""
        violations = []
        
        # Check for direct LLM usage without Ollama
        llm_patterns = [
            (r'openai\.ChatCompletion', "Direct OpenAI usage - should use Ollama"),
            (r'anthropic\.Claude', "Direct Anthropic usage - should use Ollama"),
            (r'transformers\.AutoModel', "Direct transformers usage - should use Ollama"),
            (r'llama_cpp', "Direct llama.cpp usage - should use Ollama")
        ]
        
        for py_file in self.project_root.glob("**/*.py"):
            if self._should_skip_file(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern, message in llm_patterns:
                        if re.search(pattern, line) and 'ollama' not in line.lower():
                            violations.append({
                                "rule": 16,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i,
                                "message": message,
                                "severity": "low",
                                "content": line.strip()
                            })
                            
            except Exception:
                pass
        
        # Check for Ollama configuration
        ollama_config = self.project_root / "config" / "ollama.yaml"
        if not ollama_config.exists():
            violations.append({
                "rule": 16,
                "file": "config/ollama.yaml",
                "message": "Missing Ollama configuration file",
                "severity": "medium"
            })
        
        return violations
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', 
            'env', '.env', 'build', 'dist', '.pytest_cache',
            'vendor', '.vscode', '.idea', 'coverage'
        }
        
        parts = file_path.parts
        return any(part in skip_dirs for part in parts)
    
    def _is_script(self, file_path: Path) -> bool:
        """Check if file is a script"""
        if file_path.suffix in ['.sh', '.bash', '.zsh']:
            return True
        
        if file_path.suffix == '.py':
            try:
                content = file_path.read_text(encoding='utf-8')
                # Check if it has main execution
                return '__main__' in content or 'if __name__' in content
            except Exception:
                return False
        
        return False


def main():
    parser = argparse.ArgumentParser(description="Analyze hygiene rule violations")
    parser.add_argument("--rule", type=int, required=True, help="Rule number to analyze")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    analyzer = RuleAnalyzer(args.project_root)
    violations = analyzer.analyze_rule(args.rule)
    
    # Create report
    report = {
        "rule": args.rule,
        "timestamp": datetime.now().isoformat(),
        "project_root": args.project_root,
        "violations": violations,
        "summary": {
            "total": len(violations),
            "by_severity": {}
        }
    }
    
    # Count by severity
    for severity in ["critical", "high", "medium", "low", "info"]:
        count = sum(1 for v in violations if v.get("severity") == severity)
        if count > 0:
            report["summary"]["by_severity"][severity] = count
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    if args.verbose:
        print(f"Rule {args.rule} analysis complete:")
        print(f"  Total violations: {len(violations)}")
        for severity, count in report["summary"]["by_severity"].items():
            print(f"  {severity.title()}: {count}")
    
    # Exit with error if critical violations found
    if report["summary"]["by_severity"].get("critical", 0) > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()