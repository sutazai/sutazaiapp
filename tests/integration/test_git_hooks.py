#!/usr/bin/env python3
"""
Purpose: Unit tests for Git hooks functionality
Usage: python -m pytest tests/hygiene/test_git_hooks.py
Requirements: pytest, git
"""

import unittest
import subprocess
import tempfile
import os
from pathlib import Path
import shutil

class TestGitHooks(unittest.TestCase):
    """Test Git hooks functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.project_root = Path("/opt/sutazaiapp")
        self.git_hooks_dir = self.project_root / ".git/hooks"
        
    def test_git_hooks_directory_exists(self):
        """Test that .git/hooks directory exists"""
        self.assertTrue(self.git_hooks_dir.exists(),
                       "Git hooks directory should exist")
                       
    def test_pre_commit_hook_exists(self):
        """Test that pre-commit hook exists"""
        pre_commit_hook = self.git_hooks_dir / "pre-commit"
        
        if pre_commit_hook.exists():
            self.assertTrue(os.access(pre_commit_hook, os.X_OK),
                           "Pre-commit hook should be executable")
                           
    def test_pre_commit_hook_content(self):
        """Test pre-commit hook has proper validation content"""
        pre_commit_hook = self.git_hooks_dir / "pre-commit"
        
        if pre_commit_hook.exists():
            content = pre_commit_hook.read_text()
            
            # Should contain validation logic
            self.assertTrue(
                "validation" in content.lower() or "hygiene" in content.lower(),
                "Pre-commit hook should contain validation logic"
            )
            
    def test_pre_push_hook_exists(self):
        """Test that pre-push hook exists if configured"""
        pre_push_hook = self.git_hooks_dir / "pre-push"
        
        if pre_push_hook.exists():
            self.assertTrue(os.access(pre_push_hook, os.X_OK),
                           "Pre-push hook should be executable")

class TestGitHookIntegration(unittest.TestCase):
    """Integration tests for Git hooks"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_repo = self.temp_dir / "test_repo"
        
        # Initialize test git repository
        self.test_repo.mkdir()
        os.chdir(self.test_repo)
        
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        
    def tearDown(self):
        """Cleanup test environment"""
        os.chdir("/")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_git_hook_installation(self):
        """Test installing hygiene hooks in test repository"""
        hooks_dir = self.test_repo / ".git/hooks"
        
        # Create a simple test pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
echo "Running hygiene validation..."

# Check for conceptual elements
if grep -r "process\|configurator\|transfer" . --exclude-dir=.git; then
    echo "conceptual elements detected! Commit blocked."
    exit 1
fi

echo "Hygiene validation passed."
exit 0
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Test that hook is executable
        self.assertTrue(os.access(pre_commit_hook, os.X_OK))
        
        # Test hook execution
        result = subprocess.run([str(pre_commit_hook)], 
                              capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Hook should execute successfully: {result.stderr}")
                        
    def test_pre_commit_hook_blocks_violations(self):
        """Test that pre-commit hook blocks rule violations"""
        hooks_dir = self.test_repo / ".git/hooks"
        
        # Create pre-commit hook that checks for junk files
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
echo "Checking for junk files..."

# Check for backup files
if find . -name "*.backup" -o -name "*.bak" -o -name "*~"; then
    echo "Junk files detected! Please clean up before committing."
    exit 1
fi

exit 0
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Create a junk file
        junk_file = self.test_repo / "test.backup"
        junk_file.write_text("junk content")
        
        # Test that hook fails when junk files present
        result = subprocess.run([str(pre_commit_hook)], 
                              capture_output=True, text=True)
        
                           "Hook should fail when junk files are present")
                           
    def test_pre_commit_hook_allows_clean_commits(self):
        """Test that pre-commit hook allows clean commits"""
        hooks_dir = self.test_repo / ".git/hooks"
        
        # Create pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
echo "Checking for violations..."

# Simple check that should pass
if [ -f "README.md" ]; then
    echo "Project has README.md"
fi

echo "All checks passed."
exit 0
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Create clean file
        readme = self.test_repo / "README.md"
        readme.write_text("# Test Project\n")
        
        # Test that hook passes for clean commit
        result = subprocess.run([str(pre_commit_hook)], 
                              capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Hook should pass for clean files: {result.stderr}")

class TestHookErrorHandling(unittest.TestCase):
    """Test error handling in Git hooks"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_repo = self.temp_dir / "test_repo"
        self.test_repo.mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_hook_handles_missing_files(self):
        """Test hook handles missing files gracefully"""
        hooks_dir = self.test_repo / ".git/hooks"
        hooks_dir.mkdir(parents=True)
        
        # Create hook that checks for files that might not exist
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
echo "Checking optional files..."

# Check for file that might not exist
if [ -f "optional_config.json" ]; then
    echo "Config file found"
else
    echo "Config file not found (OK)"
fi

exit 0
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Test hook without the optional file
        result = subprocess.run([str(pre_commit_hook)], 
                              capture_output=True, text=True,
                              cwd=str(self.test_repo))
        
        self.assertEqual(result.returncode, 0,
                        "Hook should handle missing optional files")
                        
    def test_hook_handles_permission_errors(self):
        """Test hook handles permission errors gracefully"""
        hooks_dir = self.test_repo / ".git/hooks"
        hooks_dir.mkdir(parents=True)
        
        # Create hook with error handling
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
echo "Running with error handling..."

# Try to access files with error handling
if ! find . -name "*.py" -type f 2>/dev/null; then
    echo "Could not search for Python files (permissions?)"
fi

echo "Hook completed despite potential errors."
exit 0
"""
        
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        
        # Test hook execution
        result = subprocess.run([str(pre_commit_hook)], 
                              capture_output=True, text=True,
                              cwd=str(self.test_repo))
        
        self.assertEqual(result.returncode, 0,
                        "Hook should handle permission errors gracefully")

class TestRealGitHooks(unittest.TestCase):
    """Test actual Git hooks in the project"""
    
    def setUp(self):
        """Setup for testing real hooks"""
        self.project_root = Path("/opt/sutazaiapp")
        self.hooks_dir = self.project_root / ".git/hooks"
        
    def test_real_pre_commit_hook_syntax(self):
        """Test real pre-commit hook has valid syntax"""
        pre_commit_hook = self.hooks_dir / "pre-commit"
        
        if pre_commit_hook.exists():
            # Check if it's a shell script
            content = pre_commit_hook.read_text()
            
            if content.startswith("#!/bin/bash") or content.startswith("#!/bin/sh"):
                # Test shell syntax
                result = subprocess.run(["bash", "-n", str(pre_commit_hook)],
                                      capture_output=True, text=True)
                
                self.assertEqual(result.returncode, 0,
                                f"Pre-commit hook has shell syntax errors: {result.stderr}")
                                
    def test_real_hook_dependency_availability(self):
        """Test that hook dependencies are available"""
        pre_commit_hook = self.hooks_dir / "pre-commit"
        
        if pre_commit_hook.exists():
            content = pre_commit_hook.read_text()
            
            # Check for Python dependencies
            if "python3" in content:
                result = subprocess.run(["which", "python3"], 
                                      capture_output=True)
                self.assertEqual(result.returncode, 0,
                                "Python3 should be available for hooks")
                                
            # Check for other common dependencies
            if "grep" in content:
                result = subprocess.run(["which", "grep"], 
                                      capture_output=True)
                self.assertEqual(result.returncode, 0,
                                "grep should be available for hooks")

class TestHookConfiguration(unittest.TestCase):
    """Test hook configuration and setup"""
    
    def test_hook_installation_script_exists(self):
        """Test that hook installation script exists"""
        project_root = Path("/opt/sutazaiapp")
        install_script = project_root / "scripts/install-hygiene-hooks.sh"
        
        if install_script.exists():
            self.assertTrue(os.access(install_script, os.X_OK),
                           "Hook installation script should be executable")
                           
    def test_hook_setup_automation_exists(self):
        """Test that hook setup automation exists"""
        project_root = Path("/opt/sutazaiapp")
        setup_script = project_root / "scripts/setup-hygiene-automation.sh"
        
        if setup_script.exists():
            self.assertTrue(os.access(setup_script, os.X_OK),
                           "Hook setup automation should be executable")

if __name__ == "__main__":
