#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI Inventory Management System

Provides thorough testing for hardcoded item detection and documentation checks
"""

import ast
import json
import os
import sys
import unittest

from core_system.inventory_management_system import (
    DocumentationCheck,
    InventoryItem,
    InventoryManagementSystem,
)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInventoryManagementSystem(unittest.TestCase):
    """
    Comprehensive test suite for the Inventory Management System
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment and initialize inventory management system
        """
        cls.base_dir = "/opt/SutazAI"
        cls.inventory_manager = InventoryManagementSystem(
            base_dir=cls.base_dir
        )

    def test_inventory_manager_initialization(self):
        """
        Test inventory management system initialization
        """
        self.assertIsNotNone(self.inventory_manager)
        self.assertTrue(hasattr(self.inventory_manager, "base_dir"))
        self.assertTrue(hasattr(self.inventory_manager, "log_dir"))
        self.assertTrue(
            hasattr(self.inventory_manager, "hardcoded_items_inventory")
        )
        self.assertTrue(
            hasattr(self.inventory_manager, "documentation_checks")
        )

    def test_scan_project_for_hardcoded_items(self):
        """
        Test comprehensive hardcoded item scanning
        """
        hardcoded_items = (
            self.inventory_manager.scan_project_for_hardcoded_items()
        )

        # Validate hardcoded items structure
        self.assertIsInstance(hardcoded_items, list)

        # Check hardcoded item attributes
        for item in hardcoded_items:
            self.assertIsInstance(item, InventoryItem)
            self.assertIn(
                item.risk_level, ["Critical", "High", "Medium", "Low"]
            )
            self.assertIn(
                item.type,
                [
                    "Credential",
                    "Connection String",
                    "URL",
                    "File Path",
                    "Numeric Constant",
                    "Unknown",
                ],
            )

    def test_hardcoded_item_risk_assessment(self):
        """
        Test hardcoded item risk assessment logic
        """
        test_cases = [
            ("password123", "Critical"),
            ("secret_key", "Critical"),
            ("https://example.com", "High"),
            ("/path/to/file", "Medium"),
            ("12345", "Low"),
        ]

        for item, expected_risk in test_cases:
            risk_level = self.inventory_manager._assess_hardcoded_item_risk(
                item
            )
            self.assertEqual(risk_level, expected_risk)

    def test_hardcoded_item_type_determination(self):
        """
        Test hardcoded item type determination logic
        """
        test_cases = [
            ("password=secret123", "Credential"),
            ("mysql://user:pass@localhost", "Connection String"),
            ("https://example.com", "URL"),
            ("/path/to/file", "File Path"),
            ("12345", "Numeric Constant"),
        ]

        for item, expected_type in test_cases:
            item_type = self.inventory_manager._determine_item_type(item)
            self.assertEqual(item_type, expected_type)

    def test_perform_documentation_checks(self):
        """
        Test comprehensive documentation checks
        """
        documentation_checks = (
            self.inventory_manager.perform_documentation_checks()
        )

        # Validate documentation checks structure
        self.assertIsInstance(documentation_checks, list)

        # Check documentation check attributes
        for check in documentation_checks:
            self.assertIsInstance(check, DocumentationCheck)
            self.assertIn(
                check.check_type,
                [
                    "Module Documentation",
                    "Class Documentation",
                    "Function Documentation",
                ],
            )
            self.assertIn(check.status, ["Missing", "Unreviewed"])

    def test_module_documentation_check(self):
        """
        Test module-level documentation check
        """
        # Create a sample AST for testing
        sample_module = ast.parse("# No docstring")
        file_path = "/test/sample_module.py"

        module_doc_check = self.inventory_manager._check_module_documentation(
            sample_module, file_path
        )

        self.assertIsNotNone(module_doc_check)
        self.assertEqual(module_doc_check.check_type, "Module Documentation")
        self.assertEqual(module_doc_check.status, "Missing")

    def test_class_documentation_check(self):
        """
        Test class-level documentation check
        """
        # Create a sample AST for testing
        sample_module = ast.parse(
            """
class SampleClass:
    def sample_method(self):
        pass
"""
        )
        file_path = "/test/sample_module.py"

        class_doc_checks = self.inventory_manager._check_class_documentation(
            sample_module, file_path
        )

        self.assertIsInstance(class_doc_checks, list)
        self.assertTrue(len(class_doc_checks) > 0)

        for check in class_doc_checks:
            self.assertEqual(check.check_type, "Class Documentation")
            self.assertEqual(check.status, "Missing")

    def test_function_documentation_check(self):
        """
        Test function-level documentation check
        """
        # Create a sample AST for testing
        sample_module = ast.parse(
            """
def sample_function(arg1, arg2):
    pass
"""
        )
        file_path = "/test/sample_module.py"

        function_doc_checks = (
            self.inventory_manager._check_function_documentation(
                sample_module, file_path
            )
        )

        self.assertIsInstance(function_doc_checks, list)
        self.assertTrue(len(function_doc_checks) > 0)

        for check in function_doc_checks:
            self.assertEqual(check.check_type, "Function Documentation")
            self.assertEqual(check.status, "Missing")

    def test_generate_comprehensive_inventory_report(self):
        """
        Test comprehensive inventory report generation
        """
        inventory_report = (
            self.inventory_manager.generate_comprehensive_inventory_report()
        )

        # Validate report structure
        self.assertIn("timestamp", inventory_report)
        self.assertIn("hardcoded_items", inventory_report)
        self.assertIn("documentation_checks", inventory_report)
        self.assertIn("summary", inventory_report)

        # Check summary details
        summary = inventory_report["summary"]
        self.assertIn("total_hardcoded_items", summary)
        self.assertIn("hardcoded_items_by_risk", summary)
        self.assertIn("total_documentation_checks", summary)
        self.assertIn("documentation_status", summary)

    def test_inventory_report_persistence(self):
        """
        Test inventory report persistence
        """
        # Generate inventory report
        inventory_report = (
            self.inventory_manager.generate_comprehensive_inventory_report()
        )

        # Check log directory exists
        self.assertTrue(os.path.exists(self.inventory_manager.log_dir))

        # Check for recent inventory report file
        log_files = [
            f
            for f in os.listdir(self.inventory_manager.log_dir)
            if f.startswith("inventory_report_") and f.endswith(".json")
        ]
        self.assertTrue(len(log_files) > 0)

        # Validate most recent log file
        most_recent_log = max(
            [
                os.path.join(self.inventory_manager.log_dir, f)
                for f in log_files
            ],
            key=os.path.getctime,
        )

        with open(most_recent_log, "r") as f:
            persisted_report = json.load(f)

        # Compare persisted report with generated report
        self.assertEqual(
            set(inventory_report.keys()), set(persisted_report.keys())
        )


def main():
    """
    Run comprehensive inventory management system tests
    """
    unittest.main()


if __name__ == "__main__":
    main()
