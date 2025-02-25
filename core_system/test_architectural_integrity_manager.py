#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI Architectural Integrity Manager

Provides thorough testing for the architectural analysis and cross-referencing mechanism
"""

import json
import os
import sys
import unittest

from core_system.architectural_integrity_manager import (
    ArchitecturalIntegrityManager,
    ArchitecturalIntegrityReport,
)

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestArchitecturalIntegrityManager(unittest.TestCase):
    """
    Comprehensive test suite for the architectural integrity manager
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment and initialize architectural integrity manager
        """
        cls.base_dir = "/opt/SutazAI"
        cls.architectural_manager = ArchitecturalIntegrityManager(base_dir=cls.base_dir)

    def test_architectural_manager_initialization(self):
        """
        Test architectural integrity manager initialization
        """
        self.assertIsNotNone(self.architectural_manager)
        self.assertTrue(hasattr(self.architectural_manager, "base_dir"))
        self.assertTrue(hasattr(self.architectural_manager, "log_dir"))
        self.assertTrue(hasattr(self.architectural_manager, "architectural_graph"))

    def test_perform_architectural_integrity_analysis(self):
        """
        Test comprehensive architectural integrity analysis
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )

        # Validate report structure
        self.assertIsInstance(architectural_report, ArchitecturalIntegrityReport)

        # Check key components
        self.assertIsNotNone(architectural_report.timestamp)
        self.assertIsInstance(architectural_report.structural_analysis, dict)
        self.assertIsInstance(architectural_report.code_quality_metrics, dict)
        self.assertIsInstance(architectural_report.architectural_patterns, dict)
        self.assertIsInstance(architectural_report.integrity_issues, list)
        self.assertIsInstance(architectural_report.optimization_recommendations, list)
        self.assertIsInstance(architectural_report.cross_reference_map, dict)

    def test_structural_analysis(self):
        """
        Test project structure analysis
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        structural_analysis = architectural_report.structural_analysis

        # Validate structural analysis
        self.assertIn("directories", structural_analysis)
        self.assertIn("module_hierarchy", structural_analysis)

        # Check directory structure
        for dir_path, dir_info in structural_analysis["directories"].items():
            self.assertIn("total_subdirectories", dir_info)
            self.assertIn("total_files", dir_info)
            self.assertIn("file_types", dir_info)

    def test_code_quality_metrics(self):
        """
        Test code quality metrics calculation
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        code_quality_metrics = architectural_report.code_quality_metrics

        # Validate code quality metrics
        self.assertIn("total_modules", code_quality_metrics)
        self.assertIn("complexity_distribution", code_quality_metrics)
        self.assertIn("documentation_coverage", code_quality_metrics)
        self.assertIn("type_hint_usage", code_quality_metrics)

        # Check complexity distribution
        for module, complexity in code_quality_metrics[
            "complexity_distribution"
        ].items():
            self.assertIn("cyclomatic_complexity", complexity)
            self.assertIn("function_count", complexity)
            self.assertIn("class_count", complexity)

    def test_architectural_patterns(self):
        """
        Test architectural pattern identification
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        architectural_patterns = architectural_report.architectural_patterns

        # Validate architectural patterns
        self.assertIn("module_categories", architectural_patterns)
        self.assertIn("design_patterns", architectural_patterns)

        # Check module categories
        module_categories = architectural_patterns["module_categories"]
        expected_categories = [
            "core_system",
            "workers",
            "services",
            "utils",
            "external",
        ]
        for category in expected_categories:
            self.assertIn(category, module_categories)

        # Check design patterns
        design_patterns = architectural_patterns["design_patterns"]
        expected_patterns = ["singleton", "factory", "strategy", "decorator"]
        for pattern in expected_patterns:
            self.assertIn(pattern, design_patterns)

    def test_integrity_issues_detection(self):
        """
        Test architectural integrity issues detection
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        integrity_issues = architectural_report.integrity_issues

        # Validate integrity issues structure
        for issue in integrity_issues:
            self.assertIn("type", issue)
            self.assertIn("severity", issue)

            if issue["type"] == "circular_dependency":
                self.assertIn("modules", issue)

            if issue["type"] == "high_coupling":
                self.assertIn("module", issue)
                self.assertIn("fan_in", issue)
                self.assertIn("fan_out", issue)

    def test_cross_reference_map(self):
        """
        Test cross-reference map generation
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        cross_reference_map = architectural_report.cross_reference_map

        # Validate cross-reference map structure
        self.assertIn("module_imports", cross_reference_map)
        self.assertIn("inheritance_relationships", cross_reference_map)
        self.assertIn("function_calls", cross_reference_map)

        # Check module imports
        for module, imports in cross_reference_map["module_imports"].items():
            self.assertIsInstance(imports, list)

        # Check inheritance relationships
        for module, relationships in cross_reference_map[
            "inheritance_relationships"
        ].items():
            for relationship in relationships:
                self.assertIn("class", relationship)
                self.assertIn("base_classes", relationship)

        # Check function calls
        for module, calls in cross_reference_map["function_calls"].items():
            for call in calls:
                self.assertIn("function", call)
                self.assertIn("line", call)

    def test_optimization_recommendations(self):
        """
        Test optimization recommendations generation
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )
        recommendations = architectural_report.optimization_recommendations

        # Recommendations should be a list of strings
        for recommendation in recommendations:
            self.assertIsInstance(recommendation, str)

    def test_architectural_report_persistence(self):
        """
        Test persistence of architectural integrity report
        """
        architectural_report = (
            self.architectural_manager.perform_architectural_integrity_analysis()
        )

        # Check log directory exists
        self.assertTrue(os.path.exists(self.architectural_manager.log_dir))

        # Check for recent architectural report file
        log_files = [
            f
            for f in os.listdir(self.architectural_manager.log_dir)
            if f.startswith("architectural_integrity_") and f.endswith(".json")
        ]
        self.assertTrue(len(log_files) > 0)

        # Validate most recent log file
        most_recent_log = max(
            [os.path.join(self.architectural_manager.log_dir, f) for f in log_files],
            key=os.path.getctime,
        )

        with open(most_recent_log, "r") as f:
            persisted_report = json.load(f)

        # Compare persisted report with generated report
        self.assertEqual(
            set(asdict(architectural_report).keys()),
            set(persisted_report.keys()),
        )

    def test_architectural_graph_visualization(self):
        """
        Test architectural graph visualization
        """
        import tempfile

        # Perform architectural integrity analysis
        self.architectural_manager.perform_architectural_integrity_analysis()

        # Create temporary output path
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "architectural_graph.png")

            # Generate visualization
            self.architectural_manager.visualize_architectural_graph(output_path)

            # Verify visualization was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


def main():
    """
    Run comprehensive architectural integrity manager tests
    """
    unittest.main()


if __name__ == "__main__":
    main()
