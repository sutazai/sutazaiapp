#!/usr/bin/env python3
"""
Advanced Dependency Mapping and Visualization Module

Provides comprehensive analysis of system dependencies, 
architectural relationships, and component interactions.
"""

import os
import sys
import importlib
import inspect
import ast
from typing import Dict, List, Any, Set, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime

class AdvancedDependencyMapper:
    """
    Ultra-Comprehensive Dependency Analysis and Visualization System
    
    Capabilities:
    - Multi-dimensional dependency tracking
    - Semantic relationship mapping
    - Architectural visualization
    - Dependency conflict detection
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        output_dir: str = None
    ):
        """
        Initialize Advanced Dependency Mapper
        
        Args:
            base_dir (str): Root directory of the project
            output_dir (str, optional): Directory for output artifacts
        """
        self.base_dir = base_dir
        self.output_dir = output_dir or os.path.join(base_dir, 'system_analysis', 'dependency_maps')
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SutazAI.DependencyMapper')
    
    def map_system_dependencies(
        self, 
        search_paths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive system dependency mapping
        
        Args:
            search_paths (List[str], optional): Specific paths to search
        
        Returns:
            Comprehensive dependency mapping
        """
        search_paths = search_paths or [
            'core_system', 
            'workers', 
            'ai_agents', 
            'services', 
            'utils'
        ]
        
        dependency_map = {
            'modules': {},
            'dependencies': {},
            'circular_dependencies': [],
            'metrics': {
                'total_modules': 0,
                'total_dependencies': 0
            }
        }
        
        # Create dependency graph
        dependency_graph = nx.DiGraph()
        
        for path in search_paths:
            full_path = os.path.join(self.base_dir, path)
            
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        module_path = os.path.join(root, file)
                        relative_path = os.path.relpath(module_path, self.base_dir)
                        module_name = relative_path.replace('/', '.').replace('.py', '')
                        
                        try:
                            module = importlib.import_module(module_name)
                            module_dependencies = self._analyze_module_dependencies(module)
                            
                            # Update dependency map
                            dependency_map['modules'][module_name] = {
                                'path': relative_path,
                                'dependencies': module_dependencies
                            }
                            
                            # Build dependency graph
                            dependency_graph.add_node(module_name)
                            for dep in module_dependencies:
                                dependency_graph.add_edge(module_name, dep)
                            
                            dependency_map['metrics']['total_modules'] += 1
                            dependency_map['metrics']['total_dependencies'] += len(module_dependencies)
                        
                        except Exception as e:
                            self.logger.warning(f"Failed to analyze module {module_name}: {e}")
        
        # Detect circular dependencies
        dependency_map['circular_dependencies'] = list(nx.simple_cycles(dependency_graph))
        
        # Visualize and persist results
        self._visualize_dependency_graph(dependency_graph)
        self._persist_dependency_map(dependency_map)
        
        return dependency_map
    
    def _analyze_module_dependencies(self, module: Any) -> List[str]:
        """
        Perform advanced module dependency analysis
        
        Args:
            module (Any): Module to analyze
        
        Returns:
            List of detected dependencies
        """
        dependencies = set()
        
        try:
            # Get source code
            source_lines, _ = inspect.getsourcelines(module)
            source_code = ''.join(source_lines)
            
            # Parse AST to detect imports
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        # Extract module name, handling relative and absolute imports
                        if isinstance(node, ast.ImportFrom):
                            base_module = node.module or ''
                            full_module = f"{base_module}.{alias.name}" if base_module else alias.name
                        else:
                            full_module = alias.name
                        
                        # Filter out standard library and third-party imports
                        if not full_module.startswith(('python', 'site-packages')):
                            dependencies.add(full_module)
        
        except Exception as e:
            self.logger.warning(f"Dependency analysis failed: {e}")
        
        return list(dependencies)
    
    def _visualize_dependency_graph(self, graph: nx.DiGraph):
        """
        Create visual representation of dependency graph
        
        Args:
            graph (nx.DiGraph): Dependency graph to visualize
        """
        try:
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(graph, k=0.5, iterations=50)
            nx.draw(
                graph, 
                pos, 
                with_labels=True, 
                node_color='lightblue', 
                node_size=300, 
                font_size=8, 
                font_weight='bold',
                arrows=True
            )
            
            # Save visualization
            plt.title("SutazAI System Dependency Graph")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, 'system_dependency_graph.png'), 
                dpi=300
            )
            plt.close()
            
            self.logger.info("Dependency graph visualization created")
        
        except Exception as e:
            self.logger.error(f"Dependency graph visualization failed: {e}")
    
    def _persist_dependency_map(self, dependency_map: Dict[str, Any]):
        """
        Persist dependency mapping to JSON
        
        Args:
            dependency_map (Dict): Comprehensive dependency mapping
        """
        try:
            output_file = os.path.join(
                self.output_dir, 
                f'system_dependency_map_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(output_file, 'w') as f:
                json.dump(dependency_map, f, indent=2)
            
            self.logger.info(f"Dependency map persisted: {output_file}")
        
        except Exception as e:
            self.logger.error(f"Dependency map persistence failed: {e}")

def main():
    """
    Execute comprehensive dependency mapping
    """
    mapper = AdvancedDependencyMapper()
    dependency_map = mapper.map_system_dependencies()
    
    print("Dependency Mapping Summary:")
    print(f"Total Modules: {dependency_map['metrics']['total_modules']}")
    print(f"Total Dependencies: {dependency_map['metrics']['total_dependencies']}")
    print("\nCircular Dependencies:")
    for cycle in dependency_map['circular_dependencies']:
        print(" -> ".join(cycle))

if __name__ == '__main__':
    main() 