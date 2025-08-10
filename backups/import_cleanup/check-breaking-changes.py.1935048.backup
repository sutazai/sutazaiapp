#!/usr/bin/env python3
"""
Purpose: Detect potential breaking changes in commits (Rule 2 enforcement)
Usage: python check-breaking-changes.py
Requirements: Python 3.8+, git
"""

import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Set

class BreakingChangeDetector:
    """Detects potential breaking changes in staged files."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.breaking_changes = []
        self.warnings = []
        
    def get_staged_files(self) -> List[Path]:
        """Get list of files staged for commit."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            return [Path(f) for f in result.stdout.strip().split('\n') if f]
        except:
            return []
    
    def get_deleted_files(self) -> List[Path]:
        """Get list of deleted files."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=D'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            return [Path(f) for f in result.stdout.strip().split('\n') if f]
        except:
            return []
    
    def check_api_changes(self, filepath: Path) -> None:
        """Check for breaking API changes."""
        if not filepath.exists():
            return
            
        # Get the diff
        try:
            diff_result = subprocess.run(
                ['git', 'diff', '--cached', str(filepath)],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if diff_result.returncode != 0:
                return
                
            diff_content = diff_result.stdout
            
            # Check for removed functions/classes
            removed_functions = re.findall(r'^-\s*def\s+(\w+)', diff_content, re.MULTILINE)
            removed_classes = re.findall(r'^-\s*class\s+(\w+)', diff_content, re.MULTILINE)
            
            if removed_functions:
                self.breaking_changes.append(
                    f"{filepath}: Removed functions: {', '.join(removed_functions)}"
                )
            
            if removed_classes:
                self.breaking_changes.append(
                    f"{filepath}: Removed classes: {', '.join(removed_classes)}"
                )
            
            # Check for changed function signatures
            signature_changes = re.findall(
                r'^-\s*def\s+(\w+)\(([^)]*)\).*\n\+\s*def\s+\1\(([^)]*)\)',
                diff_content,
                re.MULTILINE
            )
            
            for func_name, old_params, new_params in signature_changes:
                if old_params != new_params:
                    self.breaking_changes.append(
                        f"{filepath}: Function signature changed: {func_name}"
                    )
            
            # Check for removed API endpoints
            if 'router' in str(filepath) or 'api' in str(filepath):
                removed_routes = re.findall(r'^-.*@.*\.(get|post|put|delete|patch)\(', diff_content, re.MULTILINE)
                if removed_routes:
                    self.breaking_changes.append(
                        f"{filepath}: Removed API endpoints detected"
                    )
                    
        except:
            pass
    
    def check_config_changes(self, filepath: Path) -> None:
        """Check for breaking configuration changes."""
        breaking_config_files = [
            'docker-compose.yml',
            'docker-compose.yaml',
            '.env.example',
            'config.json',
            'settings.py',
            'constants.py'
        ]
        
        if filepath.name in breaking_config_files:
            try:
                diff_result = subprocess.run(
                    ['git', 'diff', '--cached', str(filepath)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if diff_result.returncode == 0:
                    diff_content = diff_result.stdout
                    
                    # Check for removed environment variables
                    removed_env_vars = re.findall(r'^-\s*(\w+)=', diff_content, re.MULTILINE)
                    if removed_env_vars:
                        self.breaking_changes.append(
                            f"{filepath}: Removed config variables: {', '.join(removed_env_vars[:5])}"
                        )
                    
                    # Check for renamed services in docker-compose
                    if 'docker-compose' in filepath.name:
                        removed_services = re.findall(r'^-\s*(\w+):\s*$', diff_content, re.MULTILINE)
                        if removed_services:
                            self.breaking_changes.append(
                                f"{filepath}: Removed services: {', '.join(removed_services)}"
                            )
            except:
                pass
    
    def check_dependency_changes(self, filepath: Path) -> None:
        """Check for breaking dependency changes."""
        if 'requirements' in filepath.name or filepath.name == 'package.json':
            try:
                diff_result = subprocess.run(
                    ['git', 'diff', '--cached', str(filepath)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if diff_result.returncode == 0:
                    diff_content = diff_result.stdout
                    
                    # Check for removed dependencies
                    if 'requirements' in filepath.name:
                        removed_deps = re.findall(r'^-(\w+)==', diff_content, re.MULTILINE)
                    else:  # package.json
                        removed_deps = re.findall(r'^-\s*"(\w+)":', diff_content, re.MULTILINE)
                    
                    if removed_deps:
                        self.warnings.append(
                            f"{filepath}: Removed dependencies: {', '.join(removed_deps[:5])}"
                        )
                    
                    # Check for major version downgrades
                    version_changes = re.findall(r'^-.*==(\d+)\..*\n\+.*==(\d+)\.', diff_content, re.MULTILINE)
                    for old_major, new_major in version_changes:
                        if int(new_major) < int(old_major):
                            self.breaking_changes.append(
                                f"{filepath}: Major version downgrade detected"
                            )
            except:
                pass
    
    def check_file_deletions(self, deleted_files: List[Path]) -> None:
        """Check if deleted files might break functionality."""
        for filepath in deleted_files:
            # Check if file might be imported/used elsewhere
            filename = filepath.stem
            
            try:
                # Search for references to this file
                search_result = subprocess.run(
                    ['git', 'grep', '-l', filename],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if search_result.returncode == 0 and search_result.stdout.strip():
                    references = search_result.stdout.strip().split('\n')
                    self.breaking_changes.append(
                        f"Deleted file '{filepath}' is referenced in {len(references)} other files"
                    )
            except:
                pass

def main():
    """Main function to detect breaking changes."""
    project_root = Path("/opt/sutazaiapp")
    detector = BreakingChangeDetector(project_root)
    
    print("🔍 Checking for breaking changes (Rule 2)...")
    
    # Get staged files
    staged_files = detector.get_staged_files()
    deleted_files = detector.get_deleted_files()
    
    if not staged_files and not deleted_files:
        print("No staged changes to check")
        return 0
    
    # Check each staged file
    for filepath in staged_files:
        if filepath.suffix in ['.py', '.js', '.ts']:
            detector.check_api_changes(filepath)
        
        detector.check_config_changes(filepath)
        detector.check_dependency_changes(filepath)
    
    # Check deleted files
    if deleted_files:
        detector.check_file_deletions(deleted_files)
    
    # Report results
    if detector.breaking_changes:
        print("\n❌ Rule 2 Violation: Potential breaking changes detected")
        print("\n📋 Breaking changes that need attention:")
        
        for change in detector.breaking_changes:
            print(f"  - {change}")
        
        print("\n📋 How to proceed:")
        print("  1. Ensure backward compatibility or provide migration path")
        print("  2. Update all references to removed/changed functionality")
        print("  3. Document breaking changes in CHANGELOG.md")
        print("  4. Consider using feature flags for gradual rollout")
        print("  5. Update tests to cover both old and new behavior")
        print("\n  If these are intentional breaking changes:")
        print("  - Add 'BREAKING CHANGE:' to your commit message")
        print("  - Update version number according to semver")
        
        return 1
    
    if detector.warnings:
        print("\n⚠️  Potential issues detected:")
        for warning in detector.warnings:
            print(f"  - {warning}")
    
    print("✅ Rule 2: No breaking changes detected")
    return 0

if __name__ == "__main__":
    sys.exit(main())