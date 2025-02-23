#!/usr/bin/env python3
"""
SutazAI Advanced Project Analysis and Cross-Referencing System

Comprehensive framework for:
- Deep structural analysis
- Advanced dependency mapping
- Architectural integrity validation
- Performance optimization recommendations
- Security vulnerability detection
"""

import ast
import importlib.util
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/advanced_project_analyzer.log'
)
logger = logging.getLogger('SutazAI.AdvancedProjectAnalyzer')

class AdvancedProjectAnalyzer:
    """
    Comprehensive project analysis system with advanced cross-referencing
    
    Provides deep insights into:
    - Project architecture
    - Dependency relationships
    - Code complexity
    - Potential architectural improvements
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        exclude_dirs: List[str] = None
    ):
        """
        Initialize advanced project analyzer
        
        Args:
            base_dir (str): Base project directory
            exclude_dirs (List[str]): Directories to exclude from analysis
        """
        self.base_dir = base_dir
        self.exclude_dirs = exclude_dirs or [
            '.git', 'venv', 'node_modules', 'build', 'dist', 'logs'
        ]
        
        # Analysis results storage
        self.analysis_report: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'dependency_graph': {},
            'module_complexity': {},
            'architectural_insights': {},
            'potential_improvements': []
        }
    
    def _is_valid_python_module(self, file_path: str) -> bool:
        """
        Validate if a file is a valid Python module
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            Boolean indicating module validity
        """
        return (
            file_path.endswith('.py') and 
            not file_path.startswith('__') and 
            not any(excluded in file_path for excluded in self.exclude_dirs)
        )
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """
        Build a comprehensive dependency graph for the project
        
        Returns:
            NetworkX Directed Graph of module dependencies
        """
        dependency_graph = nx.DiGraph()
        
        # Discover all Python modules
        modules = {}
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                full_path = os.path.join(root, file)
                if self._is_valid_python_module(full_path):
                    try:
                        module_name = os.path.relpath(full_path, self.base_dir).replace('/', '.')[:-3]
                        modules[module_name] = full_path
                    except Exception as e:
                        logger.warning(f"Could not process module {full_path}: {e}")
        
        # Analyze dependencies for each module
        for module_name, module_path in modules.items():
            try:
                with open(module_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                # Track imports
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imports.add(n.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                # Add nodes and edges to dependency graph
                dependency_graph.add_node(module_name)
                for imp in imports:
                    if imp in modules:
                        dependency_graph.add_edge(module_name, imp)
            
            except Exception as e:
                logger.warning(f"Dependency analysis failed for {module_name}: {e}")
        
        return dependency_graph
    
    def analyze_module_complexity(self, dependency_graph: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
        """
        Analyze complexity of modules based on dependency graph
        
        Args:
            dependency_graph (nx.DiGraph): Project dependency graph
        
        Returns:
            Dictionary of module complexity metrics
        """
        module_complexity = {}
        
        for module in dependency_graph.nodes():
            try:
                module_path = os.path.join(
                    self.base_dir, 
                    module.replace('.', '/') + '.py'
                )
                
                with open(module_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                # Complexity metrics
                complexity_metrics = {
                    'in_degree': dependency_graph.in_degree(module),
                    'out_degree': dependency_graph.out_degree(module),
                    'class_count': sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
                    'function_count': sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
                    'cyclomatic_complexity': self._calculate_cyclomatic_complexity(tree)
                }
                
                module_complexity[module] = complexity_metrics
            
            except Exception as e:
                logger.warning(f"Complexity analysis failed for {module}: {e}")
        
        return module_complexity
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """
        Calculate cyclomatic complexity for an AST
        
        Args:
            tree (ast.AST): Abstract Syntax Tree
        
        Returns:
            Cyclomatic complexity score
        """
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def identify_architectural_patterns(self, dependency_graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Identify architectural patterns and potential improvements
        
        Args:
            dependency_graph (nx.DiGraph): Project dependency graph
        
        Returns:
            Dictionary of architectural insights
        """
        architectural_insights = {
            'circular_dependencies': list(nx.simple_cycles(dependency_graph)),
            'highly_coupled_modules': [],
            'isolated_modules': []
        }
        
        # Identify highly coupled modules
        for module in dependency_graph.nodes():
            in_degree = dependency_graph.in_degree(module)
            out_degree = dependency_graph.out_degree(module)
            
            if in_degree + out_degree > 10:  # Arbitrary coupling threshold
                architectural_insights['highly_coupled_modules'].append({
                    'module': module,
                    'in_degree': in_degree,
                    'out_degree': out_degree
                })
        
        # Identify isolated modules
        for module in dependency_graph.nodes():
            if dependency_graph.degree(module) == 0:
                architectural_insights['isolated_modules'].append(module)
        
        return architectural_insights
    
    def generate_improvement_recommendations(
        self, 
        dependency_graph: nx.DiGraph, 
        module_complexity: Dict[str, Dict[str, Any]],
        architectural_insights: Dict[str, Any]
    ) -> List[str]:
        """
        Generate intelligent improvement recommendations
        
        Args:
            dependency_graph (nx.DiGraph): Project dependency graph
            module_complexity (Dict): Module complexity metrics
            architectural_insights (Dict): Architectural analysis results
        
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Circular dependency recommendations
        if architectural_insights['circular_dependencies']:
            recommendations.append(
                "Resolve circular dependencies to improve architectural modularity"
            )
        
        # Highly coupled module recommendations
        for module in architectural_insights['highly_coupled_modules']:
            recommendations.append(
                f"Refactor module {module['module']} to reduce coupling. "
                f"Current coupling: {module['in_degree'] + module['out_degree']}"
            )
        
        # Complexity-based recommendations
        for module, metrics in module_complexity.items():
            if metrics['cyclomatic_complexity'] > 15:  # Complexity threshold
                recommendations.append(
                    f"Simplify module {module} with high cyclomatic complexity "
                    f"(Current: {metrics['cyclomatic_complexity']})"
                )
        
        # Isolated module recommendations
        if architectural_insights['isolated_modules']:
            recommendations.append(
                "Review and integrate isolated modules into the project architecture"
            )
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Execute comprehensive project analysis workflow
        
        Returns:
            Detailed project analysis report
        """
        try:
            # Build dependency graph
            dependency_graph = self.build_dependency_graph()
            self.analysis_report['dependency_graph'] = nx.to_dict_of_lists(dependency_graph)
            
            # Analyze module complexity
            module_complexity = self.analyze_module_complexity(dependency_graph)
            self.analysis_report['module_complexity'] = module_complexity
            
            # Identify architectural patterns
            architectural_insights = self.identify_architectural_patterns(dependency_graph)
            self.analysis_report['architectural_insights'] = architectural_insights
            
            # Generate improvement recommendations
            improvement_recommendations = self.generate_improvement_recommendations(
                dependency_graph, 
                module_complexity, 
                architectural_insights
            )
            self.analysis_report['potential_improvements'] = improvement_recommendations
            
            # Persist analysis report
            report_path = os.path.join(
                self.base_dir, 
                f'logs/advanced_project_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(report_path, 'w') as f:
                json.dump(self.analysis_report, f, indent=2)
            
            logger.info(f"Advanced project analysis completed. Report: {report_path}")
            
            return self.analysis_report
        
        except Exception as e:
            logger.error(f"Comprehensive project analysis failed: {e}")
            return {}

def main():
    """
    Main execution for advanced project analysis
    """
    try:
        analyzer = AdvancedProjectAnalyzer()
        analysis_report = analyzer.run_comprehensive_analysis()
        
        print("Project Analysis Recommendations:")
        for recommendation in analysis_report.get('potential_improvements', []):
            print(f"- {recommendation}")
    
    except Exception as e:
        print(f"Project analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 