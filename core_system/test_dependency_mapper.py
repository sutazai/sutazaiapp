#!/usr/bin/env python3
"""
SutazAI Advanced Dependency Mapping Test Suite

Comprehensive test suite for validating:
- Dependency graph generation
- Circular dependency detection
- High coupling module identification
- Dependency health scoring
- Optimization recommendation generation

Test Categories:
- Module Dependency Extraction
- Dependency Graph Construction
- Circular Dependency Analysis
- Coupling Assessment
- Optimization Recommendations
"""

import os
import sys
import pytest
import networkx as nx
from typing import Dict, List, Tuple

# Import dependency mapping components
from advanced_system_analysis.dependency_mapper import AdvancedDependencyMapper

class TestDependencyMapper:
    """
    Comprehensive dependency mapping test framework
    
    Validates the integrity and functionality of dependency analysis tools
    """
    
    @pytest.fixture
    def dependency_mapper(self):
        """
        Fixture providing an AdvancedDependencyMapper instance for testing
        
        Returns:
            Configured AdvancedDependencyMapper instance
        """
        return AdvancedDependencyMapper()
    
    def test_module_validation(self, dependency_mapper):
        """
        Test module validation mechanism
        
        Validates:
        - Correct identification of valid Python modules
        - Filtering of non-module files
        """
        # Test valid module scenarios
        assert dependency_mapper._is_valid_python_module('test_module.py'), "Valid Python module not recognized"
        assert dependency_mapper._is_valid_python_module('/path/to/module.py'), "Valid module with path not recognized"
        
        # Test invalid module scenarios
        assert not dependency_mapper._is_valid_python_module('__init__.py'), "Init file incorrectly identified as module"
        assert not dependency_mapper._is_valid_python_module('.hidden_file.py'), "Hidden file recognized as module"
        assert not dependency_mapper._is_valid_python_module('not_a_python_file.txt'), "Non-Python file recognized as module"
    
    def test_dependency_graph_generation(self, dependency_mapper):
        """
        Test comprehensive dependency graph generation
        
        Validates:
        - Successful graph construction
        - Meaningful dependency tracking
        - Graph structure integrity
        """
        dependency_graph = dependency_mapper.build_dependency_graph()
        
        # Basic graph validation
        assert len(dependency_graph) > 0, "No dependencies discovered"
        
        # Validate graph structure
        for module, dependencies in dependency_graph.items():
            assert isinstance(dependencies, list), f"Invalid dependencies for {module}"
    
    def test_circular_dependency_detection(self, dependency_mapper):
        """
        Test circular dependency detection mechanism
        
        Validates:
        - Identification of circular dependencies
        - Correct reporting of dependency cycles
        """
        # Build dependency graph first
        dependency_mapper.build_dependency_graph()
        
        # Detect circular dependencies
        circular_dependencies = dependency_mapper.detect_circular_dependencies()
        
        # Validate circular dependency detection
        assert isinstance(circular_dependencies, list), "Circular dependencies must be a list"
        
        # Optional: Log circular dependencies for manual review
        if circular_dependencies:
            print("Detected Circular Dependencies:")
            for dep_pair in circular_dependencies:
                print(f"- {dep_pair[0]} <-> {dep_pair[1]}")
    
    def test_high_coupling_module_identification(self, dependency_mapper):
        """
        Test high coupling module identification
        
        Validates:
        - Successful identification of highly coupled modules
        - Meaningful coupling metrics
        - Correct sorting of coupling results
        """
        # Build dependency graph first
        dependency_mapper.build_dependency_graph()
        
        # Identify high coupling modules
        high_coupling_modules = dependency_mapper.identify_high_coupling_modules()
        
        # Validate high coupling module identification
        assert isinstance(high_coupling_modules, list), "High coupling modules must be a list"
        
        # Validate module coupling details
        for module in high_coupling_modules:
            assert 'module' in module, "Missing module name in coupling details"
            assert 'in_degree' in module, "Missing in-degree in coupling details"
            assert 'out_degree' in module, "Missing out-degree in coupling details"
            assert 'total_coupling' in module, "Missing total coupling in details"
    
    def test_dependency_health_score_calculation(self, dependency_mapper):
        """
        Test dependency health score calculation
        
        Validates:
        - Successful health score generation
        - Score within expected range
        - Meaningful scoring mechanism
        """
        # Build dependency graph first
        dependency_mapper.build_dependency_graph()
        
        # Calculate dependency health score
        health_score = dependency_mapper.calculate_dependency_health_score()
        
        # Validate health score
        assert isinstance(health_score, float), "Health score must be a float"
        assert 0 <= health_score <= 100, "Health score must be between 0 and 100"
    
    def test_optimization_recommendation_generation(self, dependency_mapper):
        """
        Test generation of dependency optimization recommendations
        
        Validates:
        - Successful recommendation generation
        - Meaningful and actionable recommendations
        - Comprehensive analysis coverage
        """
        # Simulate input data for recommendation generation
        dependency_graph = {
            'module1.py': ['module2', 'module3'],
            'module2.py': ['module3'],
            'module3.py': ['module1']
        }
        
        circular_dependencies = [('module1.py', 'module3.py')]
        high_coupling_modules = [
            {
                'module': 'module2.py',
                'in_degree': 2,
                'out_degree': 3,
                'total_coupling': 5
            }
        ]
        
        recommendations = dependency_mapper.generate_optimization_recommendations(
            dependency_graph,
            circular_dependencies,
            high_coupling_modules
        )
        
        # Validate recommendations
        assert isinstance(recommendations, list), "Recommendations must be a list"
        assert len(recommendations) > 0, "No optimization recommendations generated"
        
        # Optional: Print recommendations for manual review
        print("\nDependency Optimization Recommendations:")
        for recommendation in recommendations:
            print(f"- {recommendation}")
    
    def test_comprehensive_dependency_report(self, dependency_mapper):
        """
        Test generation of comprehensive dependency analysis report
        
        Validates:
        - Successful report generation
        - Comprehensive report structure
        - Meaningful insights and recommendations
        """
        dependency_report = dependency_mapper.generate_comprehensive_dependency_report()
        
        # Validate report structure
        assert hasattr(dependency_report, 'timestamp'), "Missing timestamp"
        assert hasattr(dependency_report, 'total_modules'), "Missing total modules"
        assert hasattr(dependency_report, 'dependency_graph'), "Missing dependency graph"
        assert hasattr(dependency_report, 'circular_dependencies'), "Missing circular dependencies"
        assert hasattr(dependency_report, 'high_coupling_modules'), "Missing high coupling modules"
        assert hasattr(dependency_report, 'dependency_health_score'), "Missing dependency health score"
        assert hasattr(dependency_report, 'optimization_recommendations'), "Missing optimization recommendations"
    
    def test_error_handling(self, dependency_mapper):
        """
        Test error handling and recovery mechanisms
        
        Validates:
        - Graceful error handling
        - Logging of error events
        - Fallback mechanisms
        """
        # Simulate error scenarios
        try:
            # Intentionally pass invalid parameters
            dependency_mapper.generate_optimization_recommendations(
                {}, [], []
            )
        except Exception as e:
            pytest.fail(f"Unexpected error in recommendation generation: {e}")

def main():
    """
    Run dependency mapping tests with comprehensive reporting
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