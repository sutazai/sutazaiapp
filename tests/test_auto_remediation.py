from sutazai.config_manager import ConfigurationManager, SutazAIConfig
from sutazai.auto_remediation import (
    AutoRemediationManager,
    RemediationAction,
)
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


class TestAutoRemediationManager(unittest.TestCase):
    def setUp(self):
        """
        Set up test environment for each test case.
        """
        # Create a temporary project directory
        self.test_project_dir = tempfile.mkdtemp()

        # Create a test configuration
        self.test_config = SutazAIConfig(
            project_root=self.test_project_dir,
            auto_remediation_enabled=True,
            auto_remediate_high_severity=True,
            auto_remediate_medium_severity=True,
            auto_remediate_low_severity=False,
            max_remediation_attempts=3,
        )

        # Initialize the remediation manager
        self.remediation_manager = AutoRemediationManager(
            project_root=self.test_project_dir, config=self.test_config
        )

    def test_should_remediate_action(self):
        """
        Test the _should_remediate_action method with different configurations.
        """
        # High severity action
        high_severity_action = RemediationAction(
            type="test_high_severity",
            description="High severity test action",
            severity="high",
        )
        self.assertTrue(
            self.remediation_manager._should_remediate_action(
                high_severity_action
            )
        )

        # Medium severity action
        medium_severity_action = RemediationAction(
            type="test_medium_severity",
            description="Medium severity test action",
            severity="medium",
        )
        self.assertTrue(
            self.remediation_manager._should_remediate_action(
                medium_severity_action
            )
        )

        # Low severity action (should be skipped)
        low_severity_action = RemediationAction(
            type="test_low_severity",
            description="Low severity test action",
            severity="low",
        )
        self.assertFalse(
            self.remediation_manager._should_remediate_action(
                low_severity_action
            )
        )

    def test_generate_remediation_actions(self):
        """
        """
            vulnerability_count=3,
            high_vulnerabilities=[
                {
                    "file": "test.py",
                    "line": 10,
                    "description": "Test high vulnerability",
                }
            ],
            medium_vulnerabilities=[
                {"configuration": "Test config vulnerability"}
            ],
            low_vulnerabilities=[
                {"path": "/tmp/test.txt", "permissions": "777"}
            ],
        )

        # Generate remediation actions
        )

        # Verify actions
        self.assertTrue(len(actions) > 0)

        # Check action types and severities
        action_types = [action.type for action in actions]
        action_severities = [action.severity for action in actions]

        self.assertIn("code_vulnerability", action_types)
        self.assertIn("configuration_vulnerability", action_types)
        self.assertIn("permission_vulnerability", action_types)

        self.assertIn("high", action_severities)
        self.assertIn("medium", action_severities)

    @patch("subprocess.run")
    def test_execute_remediation_actions(self, mock_subprocess_run):
        """
        Test executing remediation actions.
        """
        # Mock successful subprocess run
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # Create test actions
        test_actions = [
            RemediationAction(
                type="test_action",
                description="Test remediation action",
                command='echo "test"',
                severity="high",
            )
        ]

        # Execute remediation actions
        results = self.remediation_manager.execute_remediation_actions(
            test_actions
        )

        # Verify results
        self.assertIn("successful", results)
        self.assertEqual(len(results["successful"]), 1)
        self.assertEqual(results["successful"][0].status, "successful")

    def test_generate_remediation_report(self):
        """
        Test generating a remediation report.
        """
        # Prepare test results
        test_results = {
            "successful": [
                RemediationAction(
                    type="test_action",
                    description="Successful test action",
                    status="successful",
                )
            ],
            "failed": [],
            "skipped": [],
        }

        # Generate report
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            self.remediation_manager.generate_remediation_report(test_results)

            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_cleanup_old_reports(self):
        """
        Test cleaning up old remediation reports.
        """
        # Create some mock old report files
        for i in range(5):
            with open(
                os.path.join(
                    self.test_project_dir, f"remediation_report_{i}.json"
                ),
                "w",
            ) as f:
                f.write("{}")

        # Patch time to simulate old files
        with patch(
            "os.path.getmtime", return_value=0
        ):  # Make all files seem very old
            self.remediation_manager._cleanup_old_reports()

        # Verify reports are cleaned up
        remaining_reports = [
            f
            for f in os.listdir(self.test_project_dir)
            if f.startswith("remediation_report_") and f.endswith(".json")
        ]
        self.assertEqual(len(remaining_reports), 0)

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
