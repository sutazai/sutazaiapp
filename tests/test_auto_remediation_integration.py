from sutazai.security_scanner import SecurityScanResult
from sutazai.config_manager import ConfigurationManager, SutazAIConfig
from sutazai.auto_remediation import (
    AutoRemediationManager,
    CustomRemediationScriptLoader,
    RemediationAction,
)
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestAutoRemediationIntegration(unittest.TestCase):
    def setUp(self):
        """
        Set up test environment for integration tests.
        """
        # Create a temporary project directory
        self.test_project_dir = tempfile.mkdtemp()

        # Create a test configuration with aggressive remediation
        self.test_config = SutazAIConfig(
            project_root=self.test_project_dir,
            auto_remediation_enabled=True,
            auto_remediate_high_severity=True,
            auto_remediate_medium_severity=True,
            auto_remediate_low_severity=False,
            max_remediation_attempts=3,
            remediation_mode="aggressive",
        )

        # Initialize the remediation manager
        self.remediation_manager = AutoRemediationManager(
            project_root=self.test_project_dir, config=self.test_config
        )

    def test_custom_remediation_script_integration(self):
        """
        Test integration of custom remediation scripts.
        """
        # Create a temporary custom remediation script
        custom_script_path = os.path.join(
            self.test_project_dir, "test_remediation_script.py"
        )
        with open(custom_script_path, "w") as f:
            f.write(
                '''
def remediate_test_vulnerability(context=None):
    """
    Test remediation function for integration testing.

    Args:
        context (dict): Optional context for remediation

    Returns:
        bool: Whether remediation was successful
    """
    # Simulate some remediation logic
    if context and context.get('simulate_failure', False):
        return False

    # Perform some mock remediation action
    print("Custom remediation script executed successfully")
    return True
'''
            )

        # Create a mock security report with a custom script vulnerability
        mock_security_report = SecurityScanResult(
            vulnerability_count=1,
            custom_vulnerabilities=[
                {
                    "custom_script": True,
                    "description": "Test custom script vulnerability",
                    "script_module": custom_script_path,
                    "script_function": "remediate_test_vulnerability",
                    "context": {},
                }
            ],
        )

        # Generate and execute remediation actions
        remediation_actions = self.remediation_manager.analyze_security_report(
            mock_security_report
        )
        results = self.remediation_manager.execute_remediation_actions(
            remediation_actions
        )

        # Verify results
        self.assertIn("successful", results)
        self.assertEqual(len(results["successful"]), 1)
        self.assertEqual(results["successful"][0].status, "successful")

    def test_custom_remediation_script_failure(self):
        """
        Test custom remediation script failure handling.
        """
        # Create a temporary custom remediation script
        custom_script_path = os.path.join(
            self.test_project_dir, "test_failure_script.py"
        )
        with open(custom_script_path, "w") as f:
            f.write(
                '''
def remediate_test_vulnerability(context=None):
    """
    Test remediation function that simulates failure.

    Args:
        context (dict): Optional context for remediation

    Returns:
        bool: Whether remediation was successful
    """
    # Always simulate failure
    return False
'''
            )

        # Create a mock security report with a custom script vulnerability
        mock_security_report = SecurityScanResult(
            vulnerability_count=1,
            custom_vulnerabilities=[
                {
                    "custom_script": True,
                    "description": "Test custom script vulnerability",
                    "script_module": custom_script_path,
                    "script_function": "remediate_test_vulnerability",
                    "context": {"simulate_failure": True},
                }
            ],
        )

        # Generate and execute remediation actions
        remediation_actions = self.remediation_manager.analyze_security_report(
            mock_security_report
        )
        results = self.remediation_manager.execute_remediation_actions(
            remediation_actions
        )

        # Verify results
        self.assertIn("failed", results)
        self.assertEqual(len(results["failed"]), 1)
        self.assertEqual(results["failed"][0].status, "failed")

    def test_advanced_remediation_strategies(self):
        """
        Test advanced remediation strategies.
        """
        from sutazai.auto_remediation import AdvancedRemediationStrategies

        # Test dependency tree update
        result = AdvancedRemediationStrategies.dependency_tree_update("requests")
        self.assertTrue(result)

        # Create a temporary configuration file for rollback test
        config_path = os.path.join(self.test_project_dir, "test_config.json")
        backup_path = os.path.join(self.test_project_dir, "test_config.json.bak")

        with open(config_path, "w") as f:
            json.dump({"test_key": "original_value"}, f)

        with open(backup_path, "w") as f:
            json.dump({"test_key": "backup_value"}, f)

        # Test configuration rollback
        result = AdvancedRemediationStrategies.rollback_configuration(
            config_path, backup_path
        )
        self.assertTrue(result)

        # Verify rollback
        with open(config_path, "r") as f:
            rolled_back_config = json.load(f)

        self.assertEqual(rolled_back_config["test_key"], "backup_value")

    def test_remediation_report_generation(self):
        """
        Test comprehensive remediation report generation.
        """
        # Create a mock security report
        mock_security_report = SecurityScanResult(
            vulnerability_count=3,
            high_vulnerabilities=[
                {
                    "file": "test.py",
                    "line": 10,
                    "description": "Test high vulnerability",
                }
            ],
            medium_vulnerabilities=[{"configuration": "Test config vulnerability"}],
        )

        # Generate and execute remediation actions
        remediation_actions = self.remediation_manager.analyze_security_report(
            mock_security_report
        )
        results = self.remediation_manager.execute_remediation_actions(
            remediation_actions
        )

        # Generate report
        self.remediation_manager.generate_remediation_report(results)

        # Check if report was generated
        report_files = [
            f
            for f in os.listdir(self.test_project_dir)
            if f.startswith("remediation_report_") and f.endswith(".json")
        ]
        self.assertTrue(len(report_files) > 0)

        # Verify report contents
        with open(os.path.join(self.test_project_dir, report_files[0]), "r") as f:
            report_data = json.load(f)

        self.assertIn("successful", report_data)
        self.assertTrue(len(report_data["successful"]) > 0)

    def tearDown(self):
        """
        Clean up test environment after each test.
        """
        # Remove temporary directory
        import shutil

        shutil.rmtree(self.test_project_dir, ignore_errors=True)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
