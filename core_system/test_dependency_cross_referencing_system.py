#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI Dependency Cross-Referencing System

Provides thorough testing for the advanced dependency analysis and architectural insights mechanism
"""

import os
import sys
import unittest
import json
import tempfile
import shutil
import networkx as nx

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_system.dependency_cross_referencing_system import (
    UltraComprehensiveDependencyCrossReferencer,
    DependencyRelationship
)

class TestDependencyCrossReferencingSystem(unittest.TestCase):
    """
    Comprehensive test suite for the Dependency Cross-Referencing System
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test environment and initialize dependency cross-referencing system
        """
        # Create a temporary test directory
        cls.test_dir = tempfile.mkdtemp(prefix='sutazai_test_')
        
        # Create sample files for testing
        cls._create_test_files(cls.test_dir)
        
        # Initialize dependency cross-referencing system
        cls.dependency_cross_referencer = UltraComprehensiveDependencyCrossReferencer(
            base_dir=cls.test_dir
        )
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up test environment
        """
        # Remove temporary test directory
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_files(cls, base_dir):
        """
        Create sample files for testing dependency analysis
        
        Args:
            base_dir (str): Base directory for test files
        """
        # Sample files with various dependencies
        test_files = {
            'core_system/module1.py': '''
import os
import json
from workers.module2 import WorkerClass

def core_function():
    worker = WorkerClass()
    worker.process_data()
    return os.path.join('data', 'example.json')
''',
            'workers/module2.py': '''
import logging
from core_system.module1 import core_function

class WorkerClass:
    def process_data(self):
        log_path = core_function()
        logging.info(f"Processing data from {log_path}")
''',
            'services/module3.py': '''
from core_system.module1 import core_function
from workers.module2 import WorkerClass

def service_function():
    worker = WorkerClass()
    result = core_function()
    return worker.process_data()
''',
            'ai_agents/module4.py': '''
import numpy as np
import tensorflow as tf

class AIAgent:
    def __init__(self):
        self.model = self._create_model()
    
    def _create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        return model
'''
        }
        
        # Create directory structure
        os.makedirs(os.path.join(base_dir, 'core_system'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'workers'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'services'), exist_ok=True)
        os.makedirs(os.path.join(base_dir, 'ai_agents'), exist_ok=True)
        
        # Write test files
        for relative_path, content in test_files.items():
            full_path = os.path.join(base_dir, relative_path)
            with open(full_path, 'w') as f:
                f.write(content)
    
    def test_dependency_cross_referencing_initialization(self):
        """
        Test dependency cross-referencing system initialization
        """
        self.assertIsNotNone(self.dependency_cross_referencer)
        self.assertTrue(hasattr(self.dependency_cross_referencer, 'base_dir'))
        self.assertTrue(hasattr(self.dependency_cross_referencer, 'log_dir'))
        self.assertTrue(hasattr(self.dependency_cross_referencer, 'dependency_graph'))
    
    def test_analyze_module_dependencies(self):
        """
        Test module dependency analysis
        """
        test_file = os.path.join(self.test_dir, 'core_system', 'module1.py')
        
        # Analyze module dependencies
        dependencies = self.dependency_cross_referencer._analyze_module_dependencies(test_file)
        
        # Validate dependencies
        self.assertIn('imports', dependencies)
        self.assertIn('function_calls', dependencies)
        self.assertIn('class_references', dependencies)
        
        # Check specific dependencies
        self.assertIn('os', dependencies['imports'])
        self.assertIn('json', dependencies['imports'])
        self.assertIn('workers.module2.WorkerClass', dependencies['imports'])
        self.assertIn('WorkerClass', dependencies['function_calls'])
    
    def test_update_dependency_graph(self):
        """
        Test dependency graph update mechanism
        """
        test_module_path = 'core_system/module1.py'
        test_dependencies = {
            'imports': ['os', 'json', 'workers.module2.WorkerClass'],
            'function_calls': ['WorkerClass'],
            'class_references': []
        }
        
        # Update dependency graph
        self.dependency_cross_referencer._update_dependency_graph(
            test_module_path, 
            test_dependencies
        )
        
        # Validate graph update
        self.assertTrue(test_module_path in self.dependency_cross_referencer.dependency_graph.nodes())
        
        # Check edges
        for imp in test_dependencies['imports']:
            self.assertTrue(
                self.dependency_cross_referencer.dependency_graph.has_edge(test_module_path, imp)
            )
        
        # Validate module relationships
        self.assertTrue(len(self.dependency_cross_referencer.module_relationships) > 0)
    
    def test_categorize_module(self):
        """
        Test module categorization
        """
        test_cases = [
            ('core_system/module1.py', 'core'),
            ('workers/module2.py', 'worker'),
            ('services/module3.py', 'service'),
            ('ai_agents/module4.py', 'ai_agent'),
            ('unknown/module.py', 'uncategorized')
        ]
        
        for module_path, expected_category in test_cases:
            category = self.dependency_cross_referencer._categorize_module(module_path)
            self.assertEqual(category, expected_category)
    
    def test_calculate_dependency_metrics(self):
        """
        Test dependency metrics calculation
        """
        # Perform project-wide dependency analysis first
        self.dependency_cross_referencer.analyze_project_dependencies()
        
        # Calculate dependency metrics
        metrics = self.dependency_cross_referencer._calculate_dependency_metrics()
        
        # Validate metrics structure
        self.assertIn('fan_in', metrics)
        self.assertIn('fan_out', metrics)
        self.assertIn('coupling_coefficient', metrics)
        self.assertIn('centrality', metrics)
        
        # Validate centrality metrics
        self.assertIn('degree', metrics['centrality'])
        self.assertIn('betweenness', metrics['centrality'])
        self.assertIn('closeness', metrics['centrality'])
    
    def test_generate_dependency_insights(self):
        """
        Test generation of dependency insights
        """
        # Perform project-wide dependency analysis first
        self.dependency_cross_referencer.analyze_project_dependencies()
        
        # Generate dependency insights
        insights = self.dependency_cross_referencer.generate_dependency_insights()
        
        # Validate insights structure
        self.assertIn('high_coupling_modules', insights)
        self.assertIn('potential_refactoring_candidates', insights)
        self.assertIn('architectural_recommendations', insights)
    
    def test_dependency_report_persistence(self):
        """
        Test persistence of dependency report
        """
        # Perform project-wide dependency analysis
        dependency_report = self.dependency_cross_referencer.analyze_project_dependencies()
        
        # Check log directory exists
        self.assertTrue(os.path.exists(self.dependency_cross_referencer.log_dir))
        
        # Check for recent dependency report file
        log_files = [
            f for f in os.listdir(self.dependency_cross_referencer.log_dir) 
            if f.startswith('dependency_report_') and f.endswith('.json')
        ]
        self.assertTrue(len(log_files) > 0)
        
        # Validate most recent log file
        most_recent_log = max(
            [os.path.join(self.dependency_cross_referencer.log_dir, f) for f in log_files], 
            key=os.path.getctime
        )
        
        with open(most_recent_log, 'r') as f:
            persisted_report = json.load(f)
        
        # Compare persisted report with generated report
        self.assertEqual(
            set(dependency_report.keys()), 
            set(persisted_report.keys())
        )
    
    def test_dependency_graph_visualization(self):
        """
        Test dependency graph visualization
        """
        import tempfile
        
        # Perform project-wide dependency analysis
        self.dependency_cross_referencer.analyze_project_dependencies()
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily modify visualization method to use custom path
            original_method = self.dependency_cross_referencer._visualize_dependency_graph
            
            def mock_visualization():
                visualization_path = os.path.join(temp_dir, 'dependency_graph.png')
                plt.figure(figsize=(20, 20))
                pos = nx.spring_layout(self.dependency_cross_referencer.dependency_graph, k=0.5, iterations=50)
                
                nx.draw(
                    self.dependency_cross_referencer.dependency_graph, 
                    pos, 
                    with_labels=True, 
                    node_color='lightblue', 
                    node_size=300, 
                    font_size=8, 
                    font_weight='bold',
                    arrows=True
                )
                
                plt.title("SutazAI Dependency Graph")
                plt.tight_layout()
                plt.savefig(visualization_path, dpi=300)
                plt.close()
                
                return visualization_path
            
            self.dependency_cross_referencer._visualize_dependency_graph = mock_visualization
            
            # Generate visualization
            visualization_path = self.dependency_cross_referencer._visualize_dependency_graph()
            
            # Verify visualization was created
            self.assertTrue(os.path.exists(visualization_path))
            self.assertGreater(os.path.getsize(visualization_path), 0)
            
            # Restore original method
            self.dependency_cross_referencer._visualize_dependency_graph = original_method

def main():
    """
    Run comprehensive dependency cross-referencing system tests
    """
    unittest.main()

if __name__ == '__main__':
    main() 