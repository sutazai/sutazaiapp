#!/usr/bin/env python3
"""
SutazAI System Integration and Cross-Referencing Framework

Provides advanced capabilities for:
- Comprehensive system component mapping
- Dependency graph generation
- Inter-module relationship tracking
- Architectural integrity validation
- Automated system topology documentation
"""

import importlib
import inspect
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type

import networkx as nx

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
    
    Provides intelligent system component mapping and 
    architectural relationship tracking
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        output_dir: str = '/opt/sutazai_project/SutazAI/system_topology'
    ):
        """
        Initialize system integrator
        
        Args:
            base_dir (str): Base project directory
            output_dir (str): Directory for storing system topology
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize system topology graph
        self.system_topology = nx.DiGraph()
    
    def discover_system_components(self) -> Dict[str, Any]:
        """
        Discover and catalog system components
        
        Returns:
            Comprehensive system component catalog
        """
        system_components = {
            'modules': {},
            'classes': {},
            'functions': {}
        }
        
        # Discover Python modules
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, self.base_dir)
                    
                    try:
                        # Import module dynamically
                        module_name = relative_path.replace('/', '.').replace('.py', '')
                        module = importlib.import_module(module_name)
                        
                        # Catalog module details
                        system_components['modules'][module_name] = {
                            'path': full_path,
                            'classes': [],
                            'functions': []
                        }
                        
                        # Discover classes and functions
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj):
                                system_components['classes'][f"{module_name}.{name}"] = {
                                    'module': module_name,
                                    'methods': [
                                        method for method in dir(obj) 
                                        if callable(getattr(obj, method)) and not method.startswith('__')
                                    ]
                                }
                                system_components['modules'][module_name]['classes'].append(name)
                            
                            if inspect.isfunction(obj):
                                system_components['functions'][f"{module_name}.{name}"] = {
                                    'module': module_name,
                                    'signature': str(inspect.signature(obj))
                                }
                                system_components['modules'][module_name]['functions'].append(name)
                    
                    except Exception as e:
                        logger.warning(f"Could not process module {relative_path}: {e}")
        
        return system_components
    
    def generate_dependency_graph(self, system_components: Dict[str, Any]) -> nx.DiGraph:
        """
        Generate a comprehensive dependency graph
        
        Args:
            system_components (Dict): Discovered system components
        
        Returns:
            NetworkX Directed Graph of system dependencies
        """
        dependency_graph = nx.DiGraph()
        
        # Add modules as nodes
        for module_name in system_components['modules']:
            dependency_graph.add_node(module_name, type='module')
        
        # Add classes as nodes
        for class_name, class_details in system_components['classes'].items():
            dependency_graph.add_node(class_name, type='class', module=class_details['module'])
            dependency_graph.add_edge(class_details['module'], class_name)
        
        # Add functions as nodes
        for func_name, func_details in system_components['functions'].items():
            dependency_graph.add_node(func_name, type='function', module=func_details['module'])
            dependency_graph.add_edge(func_details['module'], func_name)
        
        return dependency_graph
    
    def analyze_system_architecture(self) -> Dict[str, Any]:
        """
        Perform comprehensive system architecture analysis
        
        Returns:
            Detailed system architecture report
        """
        architecture_report = {
            'timestamp': datetime.now().isoformat(),
            'system_components': self.discover_system_components(),
            'dependency_graph': {}
        }
        
        # Generate dependency graph
        self.system_topology = self.generate_dependency_graph(architecture_report['system_components'])
        
        # Convert graph to serializable format
        architecture_report['dependency_graph'] = {
            'nodes': list(self.system_topology.nodes(data=True)),
            'edges': list(self.system_topology.edges())
        }
        
        # Persist architecture report
        self._persist_architecture_report(architecture_report)
        
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
    
    def generate_system_topology_visualization(self):
        """
        Generate a visual representation of system topology
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            plt.figure(figsize=(20, 20))
            pos = nx.spring_layout(self.system_topology, k=0.5, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.system_topology, 
                pos, 
                node_color='lightblue', 
                node_size=100, 
                alpha=0.8
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                self.system_topology, 
                pos, 
                edge_color='gray', 
                arrows=True, 
                alpha=0.5
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                self.system_topology, 
                pos, 
                font_size=8, 
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

def main():
    """
    Main execution for system integration
    """
    try:
        system_integrator = AdvancedSystemIntegrator()
        
        # Analyze system architecture
        architecture_report = system_integrator.analyze_system_architecture()
        
        # Generate system topology visualization
        system_integrator.generate_system_topology_visualization()
        
        print("System Integration Analysis Completed Successfully")
        print(f"Total Modules: {len(architecture_report['system_components']['modules'])}")
        print(f"Total Classes: {len(architecture_report['system_components']['classes'])}")
        print(f"Total Functions: {len(architecture_report['system_components']['functions'])}")
    
    except Exception as e:
        logger.error(f"System Integration Analysis Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 