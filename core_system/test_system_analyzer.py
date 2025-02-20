#!/usr/bin/env python3
"""
SutazAI Advanced System Analysis Test Suite

Comprehensive test suite for validating system analysis capabilities,
including project structure, dependency tracking, code quality assessment,
and performance profiling.

Test Categories:
- Project Structure Analysis
- Dependency Graph Generation
- Code Quality Assessment
- Performance Profiling
- Optimization Recommendations
"""

import os
import sys
import pytest
import json
from typing import Dict, Any

# Import system analysis components
from advanced_system_analysis.system_analyzer import SystemAnalyzer

class TestSystemAnalyzer:
    """
    Comprehensive system analysis test framework
    
    Validates the integrity and functionality of system analysis tools
    """
    
    @pytest.fixture
    def system_analyzer(self):
        """
        Fixture providing a SystemAnalyzer instance for testing
        
        Returns:
            Configured SystemAnalyzer instance
        """
        return SystemAnalyzer()
    
    def test_project_structure_analysis(self, system_analyzer):
        """
        Test project structure analysis capabilities
        
        Validates:
        - Successful project structure discovery
        - Correct directory and file tracking
        - Meaningful file type categorization
        """
        project_structure = system_analyzer.analyze_project_structure()
        
        # Basic structure validation
        assert 'directories' in project_structure, "Missing directories in project structure"
        assert 'file_types' in project_structure, "Missing file types in project structure"
        assert 'total_files' in project_structure, "Missing total files count"
        
        # Validate file count and types
        assert project_structure['total_files'] > 0, "No files discovered in project"
        assert len(project_structure['file_types']) > 0, "No file types categorized"
        
        # Validate directory structure
        for path, details in project_structure['directories'].items():
            assert 'subdirectories' in details, f"Missing subdirectories for {path}"
            assert 'files' in details, f"Missing files for {path}"
    
    def test_dependency_graph_generation(self, system_analyzer):
        """
        Test dependency graph generation
        
        Validates:
        - Successful dependency graph creation
        - Meaningful import tracking
        - Comprehensive module coverage
        """
        dependency_graph = system_analyzer.generate_dependency_graph()
        
        assert len(dependency_graph) > 0, "No dependencies discovered"
        
        # Validate dependency graph structure
        for module, dependencies in dependency_graph.items():
            assert isinstance(dependencies, list), f"Invalid dependencies for {module}"
    
    def test_code_quality_assessment(self, system_analyzer):
        """
        Test comprehensive code quality assessment
        
        Validates:
        - Successful code complexity analysis
        - Security vulnerability scanning
        - Meaningful metrics generation
        """
        code_quality = system_analyzer.assess_code_quality()
        
        # Validate complexity analysis
        assert 'complexity_analysis' in code_quality, "Missing complexity analysis"
        
        # Validate security vulnerability assessment
        assert 'security_vulnerabilities' in code_quality, "Missing security vulnerability assessment"
    
    def test_performance_profiling(self, system_analyzer):
        """
        Test system-wide performance profiling
        
        Validates:
        - Successful performance metrics collection
        - Meaningful performance data
        - Comprehensive resource tracking
        """
        performance_metrics = system_analyzer.profile_system_performance()
        
        # Validate performance metrics structure
        assert 'cpu_usage' in performance_metrics, "Missing CPU usage metrics"
        assert 'memory_usage' in performance_metrics, "Missing memory usage metrics"
        assert 'disk_io' in performance_metrics, "Missing disk I/O metrics"
        assert 'network_io' in performance_metrics, "Missing network I/O metrics"
        
        # Validate metric types and ranges
        assert isinstance(performance_metrics['cpu_usage'], list), "CPU usage must be a list"
        assert len(performance_metrics['cpu_usage']) > 0, "No CPU usage data collected"
    
    def test_optimization_recommendations(self, system_analyzer):
        """
        Test generation of system optimization recommendations
        
        Validates:
        - Successful recommendation generation
        - Meaningful and actionable recommendations
        - Comprehensive analysis coverage
        """
        # Simulate input data for recommendation generation
        project_structure = system_analyzer.analyze_project_structure()
        dependency_graph = system_analyzer.generate_dependency_graph()
        code_quality = system_analyzer.assess_code_quality()
        performance_metrics = system_analyzer.profile_system_performance()
        
        recommendations = system_analyzer.generate_optimization_recommendations(
            project_structure,
            dependency_graph,
            code_quality,
            performance_metrics
        )
        
        assert isinstance(recommendations, list), "Recommendations must be a list"
        assert len(recommendations) > 0, "No optimization recommendations generated"
    
    def test_comprehensive_analysis_report(self, system_analyzer):
        """
        Test generation of comprehensive system analysis report
        
        Validates:
        - Successful report generation
        - Comprehensive report structure
        - Meaningful insights and recommendations
        """
        analysis_report = system_analyzer.generate_comprehensive_analysis_report()
        
        # Validate report structure
        assert hasattr(analysis_report, 'timestamp'), "Missing timestamp"
        assert hasattr(analysis_report, 'architectural_insights'), "Missing architectural insights"
        assert hasattr(analysis_report, 'dependency_graph'), "Missing dependency graph"
        assert hasattr(analysis_report, 'code_quality_metrics'), "Missing code quality metrics"
        assert hasattr(analysis_report, 'performance_analysis'), "Missing performance analysis"
        assert hasattr(analysis_report, 'security_assessment'), "Missing security assessment"
        assert hasattr(analysis_report, 'optimization_recommendations'), "Missing optimization recommendations"
    
    def test_error_handling(self, system_analyzer):
        """
        Test error handling and recovery mechanisms
        
        Validates:
        - Graceful error handling
        - Logging of error events
        - Fallback mechanisms
        """
        # Simulate error scenarios for various methods
        try:
            # Intentionally pass invalid parameters
            system_analyzer.generate_optimization_recommendations(
                {}, {}, {}, {}
            )
        except Exception as e:
            pytest.fail(f"Unexpected error in recommendation generation: {e}")

def main():
    """
    Run system analysis tests with comprehensive reporting
    """
    pytest.main([
        '-v',  # Verbose output
        '--tb=short',  # Shorter traceback format
        '--capture=no',  # Show print statements
        '--doctest-modules',  # Run doctests
        '--cov=advanced_system_analysis',  # Coverage for system analysis
        '--cov-report=html',  # HTML coverage report
        __file__
    ])

if __name__ == '__main__':
    main() 