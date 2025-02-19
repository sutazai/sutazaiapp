#!/usr/bin/env python3
"""
SutazAI Advanced Documentation Analysis and Cross-Referencing System

Comprehensive framework for:
- Deep semantic documentation analysis
- Advanced cross-component referencing
- Intelligent documentation insights
- Performance and complexity tracking
- Architectural pattern recognition
"""

import os
import sys
import json
import logging
import ast
import re
import networkx as nx
import importlib.util
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime

# Advanced NLP and semantic analysis
import spacy
import textstat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/advanced_documentation_analyzer.log'
)
logger = logging.getLogger('SutazAI.AdvancedDocumentationAnalyzer')

class AdvancedDocumentationAnalyzer:
    """
    Comprehensive documentation analysis system with advanced semantic insights
    
    Provides deep capabilities for:
    - Semantic documentation analysis
    - Cross-component dependency tracking
    - Documentation complexity assessment
    - Architectural pattern recognition
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        exclude_dirs: List[str] = None
    ):
        """
        Initialize advanced documentation analyzer
        
        Args:
            base_dir (str): Base project directory
            exclude_dirs (List[str]): Directories to exclude from analysis
        """
        self.base_dir = base_dir
        self.exclude_dirs = exclude_dirs or [
            '.git', 'venv', 'node_modules', 'build', 'dist', 'logs'
        ]
        
        # Load NLP model for semantic analysis
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("SpaCy English model not found. Downloading...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Analysis results storage
        self.analysis_report: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'semantic_insights': {},
            'documentation_complexity': {},
            'cross_references': {},
            'potential_improvements': []
        }
    
    def _is_valid_python_module(self, file_path: str) -> bool:
        """
        Validate if a file is a valid Python module for analysis
        
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
    
    def semantic_documentation_analysis(self, file_path: str) -> Dict[str, Any]:
        """
        Perform advanced semantic analysis of documentation
        
        Args:
            file_path (str): Path to the Python module
        
        Returns:
            Dictionary of semantic insights
        """
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                tree = ast.parse(content)
            
            # Extract docstrings
            module_doc = ast.get_docstring(tree) or ""
            
            # Semantic analysis using SpaCy
            doc = self.nlp(module_doc)
            
            # Semantic insights
            semantic_insights = {
                'named_entities': [
                    {
                        'text': ent.text,
                        'label': ent.label_
                    } for ent in doc.ents
                ],
                'key_concepts': [
                    token.lemma_ for token in doc 
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ']
                ],
                'readability_metrics': {
                    'flesch_reading_ease': textstat.flesch_reading_ease(module_doc),
                    'flesch_kincaid_grade': textstat.flesch_kincaid_grade(module_doc),
                    'complexity_score': textstat.text_standard(module_doc)
                }
            }
            
            return semantic_insights
        
        except Exception as e:
            logger.warning(f"Semantic analysis failed for {file_path}: {e}")
            return {}
    
    def build_cross_reference_graph(self) -> nx.DiGraph:
        """
        Build a comprehensive cross-reference graph for documentation
        
        Returns:
            NetworkX Directed Graph of documentation cross-references
        """
        cross_reference_graph = nx.DiGraph()
        
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
        
        # Analyze cross-references for each module
        for module_name, module_path in modules.items():
            try:
                with open(module_path, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Track cross-references
                cross_references = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            cross_references.add(n.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            cross_references.add(node.module.split('.')[0])
                
                # Add nodes and edges to cross-reference graph
                cross_reference_graph.add_node(module_name)
                for ref in cross_references:
                    if ref in modules:
                        cross_reference_graph.add_edge(module_name, ref)
            
            except Exception as e:
                logger.warning(f"Cross-reference analysis failed for {module_name}: {e}")
        
        return cross_reference_graph
    
    def analyze_documentation_complexity(self, cross_reference_graph: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
        """
        Analyze documentation complexity based on cross-reference graph
        
        Args:
            cross_reference_graph (nx.DiGraph): Cross-reference dependency graph
        
        Returns:
            Dictionary of documentation complexity metrics
        """
        documentation_complexity = {}
        
        for module in cross_reference_graph.nodes():
            try:
                module_path = os.path.join(
                    self.base_dir, 
                    module.replace('.', '/') + '.py'
                )
                
                with open(module_path, 'r') as f:
                    tree = ast.parse(f.read())
                
                # Complexity metrics
                complexity_metrics = {
                    'in_degree': cross_reference_graph.in_degree(module),
                    'out_degree': cross_reference_graph.out_degree(module),
                    'class_count': sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef)),
                    'function_count': sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
                    'docstring_complexity': self._calculate_docstring_complexity(module_path)
                }
                
                documentation_complexity[module] = complexity_metrics
            
            except Exception as e:
                logger.warning(f"Documentation complexity analysis failed for {module}: {e}")
        
        return documentation_complexity
    
    def _calculate_docstring_complexity(self, module_path: str) -> Dict[str, Any]:
        """
        Calculate complexity of docstrings in a module
        
        Args:
            module_path (str): Path to the Python module
        
        Returns:
            Dictionary of docstring complexity metrics
        """
        try:
            with open(module_path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Module-level docstring
            module_doc = ast.get_docstring(tree) or ""
            
            # Class and function docstrings
            class_docstrings = [
                ast.get_docstring(node) or "" 
                for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            
            function_docstrings = [
                ast.get_docstring(node) or "" 
                for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            ]
            
            return {
                'module_docstring_length': len(module_doc),
                'module_docstring_readability': textstat.flesch_reading_ease(module_doc),
                'class_docstrings_count': len(class_docstrings),
                'function_docstrings_count': len(function_docstrings),
                'avg_class_docstring_length': sum(len(doc) for doc in class_docstrings) / len(class_docstrings) if class_docstrings else 0,
                'avg_function_docstring_length': sum(len(doc) for doc in function_docstrings) / len(function_docstrings) if function_docstrings else 0
            }
        
        except Exception as e:
            logger.warning(f"Docstring complexity calculation failed for {module_path}: {e}")
            return {}
    
    def generate_documentation_recommendations(
        self, 
        cross_reference_graph: nx.DiGraph, 
        documentation_complexity: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Generate intelligent documentation improvement recommendations
        
        Args:
            cross_reference_graph (nx.DiGraph): Cross-reference dependency graph
            documentation_complexity (Dict): Documentation complexity metrics
        
        Returns:
            List of documentation improvement recommendations
        """
        recommendations = []
        
        # Cross-reference recommendations
        for module, metrics in documentation_complexity.items():
            # Docstring complexity recommendations
            if metrics.get('module_docstring_length', 0) < 50:
                recommendations.append(
                    f"Enhance module {module} documentation. Current docstring is too brief."
                )
            
            if metrics.get('module_docstring_readability', 0) < 30:
                recommendations.append(
                    f"Improve readability of {module} documentation. Current readability is low."
                )
        
        # Highly coupled module recommendations
        for module in cross_reference_graph.nodes():
            in_degree = cross_reference_graph.in_degree(module)
            out_degree = cross_reference_graph.out_degree(module)
            
            if in_degree + out_degree > 10:
                recommendations.append(
                    f"Refactor documentation for highly coupled module {module}. "
                    f"Current coupling: {in_degree + out_degree}"
                )
        
        return recommendations
    
    def run_comprehensive_documentation_analysis(self) -> Dict[str, Any]:
        """
        Execute comprehensive documentation analysis workflow
        
        Returns:
            Detailed documentation analysis report
        """
        try:
            # Perform semantic documentation analysis
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    if self._is_valid_python_module(full_path):
                        semantic_insights = self.semantic_documentation_analysis(full_path)
                        self.analysis_report['semantic_insights'][full_path] = semantic_insights
            
            # Build cross-reference graph
            cross_reference_graph = self.build_cross_reference_graph()
            self.analysis_report['cross_references'] = nx.to_dict_of_lists(cross_reference_graph)
            
            # Analyze documentation complexity
            documentation_complexity = self.analyze_documentation_complexity(cross_reference_graph)
            self.analysis_report['documentation_complexity'] = documentation_complexity
            
            # Generate documentation recommendations
            documentation_recommendations = self.generate_documentation_recommendations(
                cross_reference_graph, 
                documentation_complexity
            )
            self.analysis_report['potential_improvements'] = documentation_recommendations
            
            # Persist analysis report
            report_path = os.path.join(
                self.base_dir, 
                f'logs/advanced_documentation_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(report_path, 'w') as f:
                json.dump(self.analysis_report, f, indent=2)
            
            logger.info(f"Advanced documentation analysis completed. Report: {report_path}")
            
            return self.analysis_report
        
        except Exception as e:
            logger.error(f"Comprehensive documentation analysis failed: {e}")
            return {}

def main():
    """
    Main execution for advanced documentation analysis
    """
    try:
        analyzer = AdvancedDocumentationAnalyzer()
        analysis_report = analyzer.run_comprehensive_documentation_analysis()
        
        print("Documentation Analysis Recommendations:")
        for recommendation in analysis_report.get('potential_improvements', []):
            print(f"- {recommendation}")
    
    except Exception as e:
        print(f"Documentation analysis failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 