#!/usr/bin/env python3
"""
SutazAI Advanced System Integration and Cross-Referencing Framework

Comprehensive system analysis and architectural mapping utility
providing deep insights into project structure, dependencies, 
and inter-component relationships.

Key Capabilities:
- Holistic system component discovery
- Dependency graph generation
- Architectural relationship tracking
- Performance and security cross-referencing
- Autonomous system topology documentation
"""

import os
import sys
import json
import logging
import importlib
import inspect
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core_system.utils.safe_import import safe_import

# Safely import external libraries
networkx = safe_import('networkx')
matplotlib_pyplot = safe_import('matplotlib.pyplot')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazai_project/SutazAI/logs/system_integration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('SutazAI.SystemIntegrator')

class AdvancedSystemIntegrator:
    """
    Ultra-Comprehensive System Integration and Analysis Framework
    
    Provides intelligent system component mapping, 
    architectural relationship tracking, and 
    autonomous documentation generation
    """
    
    CRITICAL_PATHS = [
        '/opt/sutazai_project/SutazAI/core_system',
        '/opt/sutazai_project/SutazAI/scripts',
        '/opt/sutazai_project/SutazAI/ai_agents',
        '/opt/sutazai_project/SutazAI/advanced_system_tools'
    ]
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        output_dir: str = '/opt/sutazai_project/SutazAI/system_topology'
    ):
        """
        Initialize advanced system integrator
        
        Args:
            base_dir (str): Base project directory
            output_dir (str): Directory for storing system topology
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize system topology graph
        self.system_topology = networkx.DiGraph() if networkx else None
        
        # Comprehensive component registry
        self.component_registry = {
            'modules': {},
            'classes': {},
            'functions': {},
            'dependencies': {}
        }
    
    def discover_system_components(self) -> Dict[str, Any]:
        """
        Comprehensively discover and catalog system components
        
        Returns:
            Detailed system component catalog
        """
        # Reset component registry
        self.component_registry = {
            'modules': {},
            'classes': {},
            'functions': {},
            'dependencies': {}
        }
        
        # Discover components in critical paths
        for base_path in self.CRITICAL_PATHS:
            self._discover_path_components(base_path)
        
        return self.component_registry
    
    def _discover_path_components(self, path: str):
        """
        Discover components in a specific path
        
        Args:
            path (str): Path to discover components in
        """
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.base_dir)
                    module_name = relative_path.replace('/', '.').replace('.py', '')
                    
                    try:
                        # Dynamically import module
                        module = importlib.import_module(module_name)
                        
                        # Catalog module details
                        self.component_registry['modules'][module_name] = {
                            'path': full_path,
                            'classes': [],
                            'functions': []
                        }
                        
                        # Discover classes and functions
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj):
                                class_details = {
                                    'module': module_name,
                                    'methods': [
                                        method for method in dir(obj) 
                                        if callable(getattr(obj, method)) and not method.startswith('__')
                                    ],
                                    'docstring': obj.__doc__
                                }
                                self.component_registry['classes'][f"{module_name}.{name}"] = class_details
                                self.component_registry['modules'][module_name]['classes'].append(name)
                            
                            if inspect.isfunction(obj):
                                func_details = {
                                    'module': module_name,
                                    'signature': str(inspect.signature(obj)),
                                    'docstring': obj.__doc__
                                }
                                self.component_registry['functions'][f"{module_name}.{name}"] = func_details
                                self.component_registry['modules'][module_name]['functions'].append(name)
                    
                    except Exception as e:
                        logger.warning(f"Could not process module {module_name}: {e}")
    
    def generate_dependency_graph(self) -> networkx.DiGraph:
        """
        Generate a comprehensive dependency graph
        
        Returns:
            NetworkX Directed Graph of system dependencies
        """
        dependency_graph = networkx.DiGraph()
        
        # Add modules as nodes
        for module_name, module_details in self.component_registry['modules'].items():
            dependency_graph.add_node(module_name, type='module')
            
            # Add classes as nodes
            for class_name in module_details['classes']:
                full_class_name = f"{module_name}.{class_name}"
                dependency_graph.add_node(full_class_name, type='class')
                dependency_graph.add_edge(module_name, full_class_name)
            
            # Add functions as nodes
            for func_name in module_details['functions']:
                full_func_name = f"{module_name}.{func_name}"
                dependency_graph.add_node(full_func_name, type='function')
                dependency_graph.add_edge(module_name, full_func_name)
        
        return dependency_graph
    
    def analyze_system_architecture(self) -> Dict[str, Any]:
        """
        Perform comprehensive system architecture analysis
        
        Returns:
            Detailed system architecture report
        """
        # Discover system components
        self.discover_system_components()
        
        # Generate dependency graph
        self.system_topology = self.generate_dependency_graph()
        
        # Create architecture report
        architecture_report = {
            'timestamp': datetime.now().isoformat(),
            'component_registry': self.component_registry,
            'dependency_graph': {
                'nodes': list(self.system_topology.nodes(data=True)),
                'edges': list(self.system_topology.edges())
            },
            'metrics': {
                'total_modules': len(self.component_registry['modules']),
                'total_classes': len(self.component_registry['classes']),
                'total_functions': len(self.component_registry['functions'])
            }
        }
        
        # Persist and visualize architecture
        self._persist_architecture_report(architecture_report)
        self._generate_topology_visualization()
        
        return architecture_report
    
    def _persist_architecture_report(self, report: Dict[str, Any]):
        """
        Persist system architecture report
        
        Args:
            report (Dict): System architecture report
        """
        report_filename = f'system_architecture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path = os.path.join(self.output_dir, report_filename)
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"System Architecture Report Generated: {report_path}")
        
        except Exception as e:
            logger.error(f"Could not persist architecture report: {e}")
    
    def _generate_topology_visualization(self):
        """
        Generate visual representation of system topology
        """
        try:
            plt.figure(figsize=(20, 20))
            pos = networkx.spring_layout(self.system_topology, k=0.5, iterations=50)
            
            # Node colors based on type
            node_colors = [
                'lightblue' if data.get('type') == 'module' else 
                'lightgreen' if data.get('type') == 'class' else 
                'lightsalmon' 
                for _, data in self.system_topology.nodes(data=True)
            ]
            
            # Draw nodes
            networkx.draw_networkx_nodes(
                self.system_topology, 
                pos, 
                node_color=node_colors, 
                node_size=100, 
                alpha=0.8
            )
            
            # Draw edges
            networkx.draw_networkx_edges(
                self.system_topology, 
                pos, 
                edge_color='gray', 
                arrows=True, 
                alpha=0.5
            )
            
            # Draw labels
            networkx.draw_networkx_labels(
                self.system_topology, 
                pos, 
                font_size=6, 
                font_weight='bold'
            )
            
            plt.title('SutazAI System Topology')
            plt.axis('off')
            
            # Save visualization
            visualization_path = os.path.join(
                self.output_dir, 
                f'system_topology_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            )
            plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
            
            logger.info(f"System Topology Visualization Generated: {visualization_path}")
        
        except ImportError:
            logger.warning("Matplotlib not available. Skipping topology visualization.")
        except Exception as e:
            logger.error(f"Topology visualization failed: {e}")
    
    def generate_markdown_documentation(self) -> str:
        """
        Generate comprehensive markdown documentation
        
        Returns:
            Markdown formatted system documentation
        """
        markdown_doc = "# SutazAI System Architecture\n\n"
        
        # Module overview
        markdown_doc += "## Modules Overview\n\n"
        for module_name, module_details in self.component_registry['modules'].items():
            markdown_doc += f"### {module_name}\n"
            markdown_doc += f"- **Path**: `{module_details['path']}`\n"
            markdown_doc += f"- **Classes**: {len(module_details['classes'])}\n"
            markdown_doc += f"- **Functions**: {len(module_details['functions'])}\n\n"
        
        # Detailed class documentation
        markdown_doc += "## Classes Catalog\n\n"
        for class_name, class_details in self.component_registry['classes'].items():
            markdown_doc += f"### {class_name}\n"
            markdown_doc += f"- **Module**: {class_details['module']}\n"
            markdown_doc += f"- **Methods**: {', '.join(class_details['methods'])}\n"
            markdown_doc += f"- **Docstring**:\n```\n{class_details['docstring'] or 'No docstring available'}\n```\n\n"
        
        # Function catalog
        markdown_doc += "## Functions Catalog\n\n"
        for func_name, func_details in self.component_registry['functions'].items():
            markdown_doc += f"### {func_name}\n"
            markdown_doc += f"- **Module**: {func_details['module']}\n"
            markdown_doc += f"- **Signature**: `{func_details['signature']}`\n"
            markdown_doc += f"- **Docstring**:\n```\n{func_details['docstring'] or 'No docstring available'}\n```\n\n"
        
        # Persist markdown documentation
        doc_path = os.path.join(
            self.output_dir, 
            f'system_architecture_docs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'
        )
        
        try:
            with open(doc_path, 'w') as f:
                f.write(markdown_doc)
            
            logger.info(f"System Architecture Documentation Generated: {doc_path}")
        except Exception as e:
            logger.error(f"Could not generate documentation: {e}")
        
        return markdown_doc

def main():
    """
    Main execution for system integration
    """
    try:
        system_integrator = AdvancedSystemIntegrator()
        
        # Analyze system architecture
        architecture_report = system_integrator.analyze_system_architecture()
        
        # Generate markdown documentation
        system_integrator.generate_markdown_documentation()
        
        print("System Integration Analysis Completed Successfully")
        print(f"Total Modules: {architecture_report['metrics']['total_modules']}")
        print(f"Total Classes: {architecture_report['metrics']['total_classes']}")
        print(f"Total Functions: {architecture_report['metrics']['total_functions']}")
    
    except Exception as e:
        logger.error(f"System Integration Analysis Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 