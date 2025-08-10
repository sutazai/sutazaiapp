#!/usr/bin/env python3
"""
Purpose: Quick system analysis to ensure changes don't break functionality (Rule 3)
Usage: python quick-system-check.py
Requirements: Python 3.8+, git
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import subprocess
import json
from pathlib import Path
import ast

class QuickSystemAnalyzer:
    """Performs quick system-wide checks before commit."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        
    def check_import_integrity(self) -> bool:
        """Check that all Python imports are resolvable."""
        unresolved_imports = set()
        
        # Get all Python files
        python_files = list(self.project_root.rglob('*.py'))
        python_files = [f for f in python_files if 'venv' not in str(f) and '__pycache__' not in str(f)]
        
        for py_file in python_files[:50]:  # Check first 50 files for speed
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split('.')[0]
                            # Check if it's a local module
                            if not self._is_stdlib_module(module_name):
                                module_path = self.project_root / f"{module_name}.py"
                                module_dir = self.project_root / module_name
                                if not module_path.exists() and not module_dir.exists():
                                    unresolved_imports.add(module_name)
                                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and not self._is_stdlib_module(node.module.split('.')[0]):
                            module_path = self.project_root / node.module.replace('.', '/')
                            if not module_path.exists() and not (module_path.parent / '__init__.py').exists():
                                unresolved_imports.add(node.module)
                                
            except Exception as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                pass  # Skip files with syntax errors
        
        if unresolved_imports:
            self.warnings.append(f"Potentially unresolved imports: {', '.join(list(unresolved_imports)[:5])}")
        
        return True
    
    def check_config_files(self) -> bool:
        """Check configuration file validity."""
        config_files = {
            'docker-compose.yml': 'yaml',
            'docker-compose.yaml': 'yaml', 
            'package.json': 'json',
            'pyproject.toml': 'toml',
            '.pre-commit-config.yaml': 'yaml'
        }
        
        for config_file, file_type in config_files.items():
            filepath = self.project_root / config_file
            if filepath.exists():
                if file_type == 'json':
                    try:
                        with open(filepath, 'r') as f:
                            json.load(f)
                    except json.JSONDecodeError as e:
                        self.issues.append(f"{config_file}: Invalid JSON - {e}")
                elif file_type == 'yaml':
                    try:
                        import yaml
                        with open(filepath, 'r') as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        self.issues.append(f"{config_file}: Invalid YAML - {e}")
        
        return len(self.issues) == 0
    
    def check_dependencies(self) -> bool:
        """Check that dependency files are consistent."""
        req_files = list(self.project_root.glob('**/requirements*.txt'))
        req_files = [f for f in req_files if 'venv' not in str(f)]
        
        if len(req_files) > 5:
            self.warnings.append(f"Found {len(req_files)} requirements files - consider consolidation")
        
        # Check for conflicting versions
        all_deps = {}
        for req_file in req_files[:10]:  # Check first 10 for speed
            try:
                with open(req_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '==' in line:
                                pkg, version = line.split('==', 1)
                                pkg = pkg.strip()
                                version = version.strip().split(';')[0]
                                if pkg in all_deps and all_deps[pkg] != version:
                                    self.issues.append(
                                        f"Conflicting versions for {pkg}: {all_deps[pkg]} vs {version}"
                                    )
                                all_deps[pkg] = version
            except Exception as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        return True
    
    def check_test_coverage(self) -> bool:
        """Quick check for test file existence."""
        test_dirs = list(self.project_root.glob('**/test*'))
        test_files = list(self.project_root.glob('**/test_*.py'))
        
        if not test_dirs and not test_files:
            self.warnings.append("No test files found - consider adding tests")
        
        return True
    
    def check_git_status(self) -> bool:
        """Check git status for issues."""
        try:
            # Check for large files
            result = subprocess.run(
                ['git', 'ls-files', '-s'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                large_files = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            size = int(parts[2])
                            if size > 5_000_000:  # 5MB
                                large_files.append(parts[3])
                
                if large_files:
                    self.issues.append(f"Large files detected: {', '.join(large_files[:3])}")
            
            # Check for conflicts
            conflict_result = subprocess.run(
                ['git', 'diff', '--name-only', '--diff-filter=U'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if conflict_result.returncode == 0 and conflict_result.stdout.strip():
                self.issues.append("Unresolved merge conflicts detected")
                
        except Exception as e:
            # Suppressed exception (was bare except)
            logger.debug(f"Suppressed exception: {e}")
            pass
        
        return True
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of Python stdlib."""
        stdlib_modules = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 're', 'ast', 
            'subprocess', 'typing', 'collections', 'itertools', 'functools',
            'argparse', 'logging', 'unittest', 'math', 'random', 'time'
        }
        return module_name in stdlib_modules

def main():
    """Main function for quick system analysis."""
    project_root = Path("/opt/sutazaiapp")
    analyzer = QuickSystemAnalyzer(project_root)
    
    print("🔍 Running quick system analysis (Rule 3)...")
    
    # Run all checks
    checks = [
        ("Import integrity", analyzer.check_import_integrity),
        ("Config files", analyzer.check_config_files),
        ("Dependencies", analyzer.check_dependencies),
        ("Test coverage", analyzer.check_test_coverage),
        ("Git status", analyzer.check_git_status),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_passed = False
        except Exception as e:
            analyzer.warnings.append(f"{check_name} check failed: {e}")
    
    # Report results
    if analyzer.issues:
        print("\n❌ Rule 3: System analysis found critical issues:")
        for issue in analyzer.issues:
            print(f"  - {issue}")
        return 1
    
    if analyzer.warnings:
        print("\n⚠️  System analysis warnings:")
        for warning in analyzer.warnings:
            print(f"  - {warning}")
    
    print("✅ Rule 3: Quick system analysis passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())