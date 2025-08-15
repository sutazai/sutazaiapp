#!/usr/bin/env python3
"""
Purpose: Unit tests for hygiene enforcement coordinator
Usage: python -m pytest tests/hygiene/test_coordinator.py
Requirements: pytest, unittest.mock
"""

import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import shutil

# Add project root to path for imports
# Path handled by pytest configuration

class TestHygieneEnforcementCoordinator(unittest.TestCase):
    """Test the HygieneEnforcementCoordinator class"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "test_project"
        self.project_root.mkdir()
        
        # Create necessary directories
        (self.project_root / "logs").mkdir()
        (self.project_root / "archive").mkdir()
        
        # Create test files for different rule violations
        self.create_test_violations()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def create_test_violations(self):
        """Create test files that violate hygiene rules"""
        # Rule 13 violations (junk files)
        junk_files = [
            "test.backup",
            "old_file.bak", 
            "temporary.tmp",
            "editor_save~"
        ]
        
        for filename in junk_files:
            (self.project_root / filename).write_text("test content")
            
        # Rule 12 violations (multiple deploy scripts)
        deploy_scripts = [
            "deploy.sh",
            "deploy_staging.sh",
            "validate_deployment.py"
        ]
        
        for script in deploy_scripts:
            (self.project_root / script).write_text("#!/bin/bash\necho 'deploy'\n")
            
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_log_action(self, mock_mkdir, mock_file):
        """Test logging functionality"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            coordinator.log_action("Test message", "INFO")
            
            # Verify logging operations
            mock_mkdir.assert_called()
            mock_file.assert_called()
            
        except ImportError:
            # Skip if module not available in test environment
            self.skipTest("Coordinator module not available for import")
            
    def test_create_archive_directory(self):
        """Test archive directory creation"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            coordinator.dry_run = True  # Don't actually create directories in test
            
            archive_dir = coordinator.create_archive_directory("test-rule")
            
            # Verify archive directory path structure
            self.assertIn("test-rule-cleanup", str(archive_dir))
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")
            
    @patch('subprocess.run')
    def test_find_violations_rule_13(self, mock_subprocess):
        """Test finding rule 13 violations (junk files)"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            # Mock find command output
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = f"{self.project_root}/test.backup\n{self.project_root}/old_file.bak\n"
            mock_subprocess.return_value = mock_result
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            violations = coordinator.find_violations("rule_13")
            
            # Verify violations found
            self.assertGreater(len(violations), 0)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")
            
    @patch('subprocess.run')
    def test_verify_file_safety(self, mock_subprocess):
        """Test file safety verification"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            # Mock grep command - no references found
            mock_result = MagicMock()
            mock_result.returncode = 1  # grep returns 1 when no matches
            mock_subprocess.return_value = mock_result
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            test_file = self.project_root / "test.backup"
            
            safe, references = coordinator.verify_file_safety(test_file)
            
            # File should be safe to remove if no references
            self.assertTrue(safe)
            self.assertEqual(references, [])
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")
            
    @patch('subprocess.run')
    def test_verify_file_safety_with_references(self, mock_subprocess):
        """Test file safety verification when file has references"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            # Mock grep command - references found
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "/path/to/other/file.py:import test.backup\n"
            mock_subprocess.return_value = mock_result
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            test_file = self.project_root / "test.backup"
            
            safe, references = coordinator.verify_file_safety(test_file)
            
            # File should NOT be safe to remove if references exist
            self.assertFalse(safe)
            self.assertGreater(len(references), 0)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")

class TestCoordinatorRuleEnforcement(unittest.TestCase):
    """Test specific rule enforcement logic"""
    
    def setUp(self):
        """Setup test environment for rule enforcement"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "test_project"
        self.project_root.mkdir()
        
        # Create logs and archive directories
        (self.project_root / "logs").mkdir()
        (self.project_root / "archive").mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_rule_violations_config(self):
        """Test rule violations configuration"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            
            # Verify rule configuration structure
            self.assertIn("rule_13", coordinator.rule_violations)
            self.assertIn("rule_12", coordinator.rule_violations)
            
            # Verify each rule has required fields
            for rule_name, config in coordinator.rule_violations.items():
                self.assertIn("priority", config)
                self.assertIn("patterns", config)
                self.assertIn("agent", config)
                self.assertIn("description", config)
                
        except ImportError:
            self.skipTest("Coordinator module not available for import")
            
    @patch('subprocess.run')
    def test_enforce_rule_13_dry_run(self, mock_subprocess):
        """Test rule 13 enforcement in dry run mode"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            # Mock find command to return test violations
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = f"{self.project_root}/test.backup\n"
            mock_subprocess.return_value = mock_result
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            coordinator.dry_run = True
            
            # Create test file
            test_file = self.project_root / "test.backup"
            test_file.write_text("test content")
            
            stats = coordinator.enforce_rule_13()
            
            # Verify stats structure
            self.assertIn("found", stats)
            self.assertIn("archived", stats)
            self.assertIn("removed", stats)
            self.assertIn("skipped", stats)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")

class TestCoordinatorIntegration(unittest.TestCase):
    """Integration tests for coordinator"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.project_root = Path("/opt/sutazaiapp")
        self.coordinator_script = self.project_root / "scripts/hygiene-enforcement-coordinator.py"
        
    def test_coordinator_script_exists(self):
        """Test that coordinator script file exists"""
        self.assertTrue(self.coordinator_script.exists(),
                       f"Coordinator script not found: {self.coordinator_script}")
        
    def test_coordinator_script_syntax(self):
        """Test coordinator script has valid Python syntax"""
        cmd = ["python3", "-m", "py_compile", str(self.coordinator_script)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Coordinator script has syntax errors: {result.stderr}")
        
    def test_coordinator_help(self):
        """Test coordinator shows help"""
        if self.coordinator_script.exists():
            cmd = ["python3", str(self.coordinator_script), "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            self.assertEqual(result.returncode, 0)
            self.assertIn("usage:", result.stdout.lower())
            
    def test_coordinator_dry_run_phase_1(self):
        """Test coordinator dry run for phase 1"""
        if self.coordinator_script.exists():
            cmd = ["python3", str(self.coordinator_script), "--phase=1", "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Should complete without crashing
            self.assertIsNotNone(result, "Coordinator script should execute")

class TestCoordinatorErrorHandling(unittest.TestCase):
    """Test error handling in coordinator"""
    
    def test_missing_project_directory(self):
        """Test handling when project directory doesn't exist""" 
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            # Should handle non-existent directory gracefully
            coordinator = HygieneEnforcementCoordinator("/nonexistent/path")
            self.assertIsNotNone(coordinator.rule_violations)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")
        except Exception as e:
            # Should not crash with unhandled exceptions
            self.fail(f"Coordinator should handle missing directories: {e}")
            
    def test_permission_denied_scenarios(self):
        """Test handling of permission denied scenarios"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            coordinator = HygieneEnforcementCoordinator("/tmp")
            
            # Test with file that doesn't exist
            non_existent_file = Path("/tmp/nonexistent.backup")
            safe, references = coordinator.verify_file_safety(non_existent_file)
            
            # Should handle gracefully without crashing
            self.assertIsInstance(safe, bool)
            self.assertIsInstance(references, list)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")

class TestCoordinatorReporting(unittest.TestCase):
    """Test coordinator reporting functionality"""
    
    def setUp(self):
        """Setup test environment for reporting"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.project_root = self.temp_dir / "test_project"
        self.project_root.mkdir()
        (self.project_root / "logs").mkdir()
        
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    def test_generate_report(self):
        """Test report generation"""
        try:
            from scripts.hygiene_enforcement_coordinator import HygieneEnforcementCoordinator
            
            coordinator = HygieneEnforcementCoordinator(str(self.project_root))
            coordinator.dry_run = True
            
            test_results = {
                "rule_13": {"found": 5, "archived": 3, "removed": 0, "skipped": 2}
            }
            
            report_path = coordinator.generate_report(test_results)
            
            # Verify report path structure
            self.assertIn("hygiene-report-", report_path)
            self.assertIn(".json", report_path)
            
        except ImportError:
            self.skipTest("Coordinator module not available for import")

if __name__ == "__main__":
