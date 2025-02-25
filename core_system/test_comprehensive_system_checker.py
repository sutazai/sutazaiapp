#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI Comprehensive System Checker

Provides thorough testing for the system checking and optimization mechanism
"""

import json
import os
import sys
import unittest

from core_system.comprehensive_system_checker import ComprehensiveSystemChecker

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestComprehensiveSystemChecker(unittest.TestCase):
    """
    Comprehensive test suite for the system checker
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment and initialize system checker
        """
        cls.base_dir = "/opt/SutazAI"
        cls.system_checker = ComprehensiveSystemChecker(base_dir=cls.base_dir)

    def test_system_check_initialization(self):
        """
        Test system checker initialization
        """
        self.assertIsNotNone(self.system_checker)
        self.assertTrue(hasattr(self.system_checker, "base_dir"))
        self.assertTrue(hasattr(self.system_checker, "log_dir"))

    def test_perform_comprehensive_system_check(self):
        """
        Test comprehensive system check method
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )

        # Validate result structure
        self.assertIsInstance(system_check_results, dict)

        # Check key components
        self.assertIn("timestamp", system_check_results)
        self.assertIn("dependency_analysis", system_check_results)
        self.assertIn("code_structure_analysis", system_check_results)
        self.assertIn("potential_issues", system_check_results)
        self.assertIn("optimization_recommendations", system_check_results)

    def test_dependency_analysis(self):
        """
        Test dependency analysis functionality
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )
        dependency_analysis = system_check_results.get(
            "dependency_analysis", {}
        )

        self.assertIn("module_dependencies", dependency_analysis)
        self.assertIn("import_graph", dependency_analysis)
        self.assertIn("circular_dependencies", dependency_analysis)
        self.assertIn("cross_module_interactions", dependency_analysis)

    def test_code_structure_analysis(self):
        """
        Test code structure analysis functionality
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )
        code_structure = system_check_results.get(
            "code_structure_analysis", {}
        )

        self.assertIn("files", code_structure)
        self.assertIn("complexity_metrics", code_structure)
        self.assertIn("architectural_patterns", code_structure)

    def test_potential_issues_detection(self):
        """
        Test potential issues detection
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )
        potential_issues = system_check_results.get("potential_issues", [])

        # Validate issues structure
        for issue in potential_issues:
            self.assertIn("type", issue)
            self.assertIn("severity", issue)

    def test_optimization_recommendations(self):
        """
        Test optimization recommendations generation
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )
        recommendations = system_check_results.get(
            "optimization_recommendations", []
        )

        # Recommendations should be a list of strings
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)

    def test_analysis_results_persistence(self):
        """
        Test persistence of analysis results
        """
        system_check_results = (
            self.system_checker.perform_comprehensive_system_check()
        )

        # Check log directory exists
        self.assertTrue(os.path.exists(self.system_checker.log_dir))

        # Check for recent analysis results file
        log_files = [
            f
            for f in os.listdir(self.system_checker.log_dir)
            if f.startswith("comprehensive_system_check_")
            and f.endswith(".json")
        ]
        self.assertTrue(len(log_files) > 0)

        # Validate most recent log file
        most_recent_log = max(
            [os.path.join(self.system_checker.log_dir, f) for f in log_files],
            key=os.path.getctime,
        )

        with open(most_recent_log, "r") as f:
            persisted_results = json.load(f)

        # Compare persisted results with generated results
        self.assertEqual(
            set(system_check_results.keys()), set(persisted_results.keys())
        )

    def test_continuous_system_checking(self):
        """
        Test continuous system checking mechanism
        """
        # Start continuous system checking
        self.system_checker.start_continuous_system_checking(interval=1)

        # Wait briefly to allow thread to start
        import time

        time.sleep(2)

        # Check if logging is working
        log_files = [
            f
            for f in os.listdir(self.system_checker.log_dir)
            if f.startswith("comprehensive_system_check_")
            and f.endswith(".json")
        ]
        self.assertTrue(
            len(log_files) > 1
        )  # At least two logs (initial + continuous)


def main():
    """
    Run comprehensive system checker tests
    """
    unittest.main()


if __name__ == "__main__":
    main()
