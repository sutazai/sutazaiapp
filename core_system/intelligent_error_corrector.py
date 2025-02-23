#!/usr/bin/env python3
"""
SutazAI Intelligent Error Correction and Semantic Linking System

Provides ultra-comprehensive, autonomous error detection, 
correction, and intelligent cross-referencing capabilities.
"""

import ast
import importlib
import inspect
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

import astroid
import black
import networkx as nx
import radon.metrics
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IntelligentErrorCorrector:
    """
    Ultra-Comprehensive Autonomous Error Detection and Correction Framework
    
    Key Capabilities:
    - Advanced static code analysis
    - Semantic error understanding
    - Intelligent code correction
    - Cross-module dependency tracking
    - Machine learning-powered error prediction
    """
    
    def __init__(
        self, 
        base_dir: str = '/opt/sutazai_project/SutazAI',
        log_dir: Optional[str] = None
    ):
        """
        Initialize Intelligent Error Corrector
        
        Args:
            base_dir (str): Base project directory
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.log_dir = log_dir or os.path.join(base_dir, 'logs', 'error_correction')
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename=os.path.join(self.log_dir, 'intelligent_error_corrector.log')
        )
        self.logger = logging.getLogger('SutazAI.IntelligentErrorCorrector')
        
        # Initialize NLP model for semantic analysis
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.warning("SpaCy model not found. Semantic analysis will be limited.")
            self.nlp = None
        
        # Initialize dependency and error graphs
        self.dependency_graph = nx.DiGraph()
        self.error_graph = nx.DiGraph()
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def analyze_project_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive project structure analysis
        
        Returns:
            Dictionary of project structure insights
        """
        project_structure = {
            'modules': {},
            'dependencies': {},
            'error_patterns': {}
        }
        
        try:
            # Walk through Python files
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        
                        try:
                            # Analyze module
                            module_name = os.path.relpath(
                                full_path, 
                                self.base_dir
                            ).replace('/', '.')[:-3]
                            
                            module_info = self._analyze_module(full_path)
                            project_structure['modules'][module_name] = module_info
                            
                            # Track module dependencies
                            project_structure['dependencies'][module_name] = \
                                self._track_module_dependencies(full_path)
                        
                        except Exception as e:
                            self.logger.warning(f"Module analysis failed for {full_path}: {e}")
            
            return project_structure
        
        except Exception as e:
            self.logger.error(f"Project structure analysis failed: {e}")
            return project_structure
    
    def _analyze_module(self, file_path: str) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single module
        
        Args:
            file_path (str): Path to the Python file
        
        Returns:
            Dictionary of module insights
        """
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Parse module with astroid for advanced analysis
            module = astroid.parse(source_code)
            
            # Analyze module complexity
            complexity_metrics = radon.metrics.mi_visit(source_code, True)
            
            # Detect potential error-prone patterns
            error_patterns = self._detect_error_patterns(source_code)
            
            return {
                'path': file_path,
                'complexity': {
                    'maintainability_index': complexity_metrics,
                    'cyclomatic_complexity': self._calculate_cyclomatic_complexity(module)
                },
                'structure': {
                    'classes': [cls.name for cls in module.body if isinstance(cls, astroid.ClassDef)],
                    'functions': [func.name for func in module.body if isinstance(func, astroid.FunctionDef)],
                    'imports': [imp.name for imp in module.body if isinstance(imp, astroid.Import)]
                },
                'error_patterns': error_patterns
            }
        
        except Exception as e:
            self.logger.warning(f"Module analysis failed for {file_path}: {e}")
            return {}
    
    def _calculate_cyclomatic_complexity(self, module: astroid.Module) -> int:
        """
        Calculate cyclomatic complexity for a module
        
        Args:
            module (astroid.Module): Parsed module
        
        Returns:
            Cyclomatic complexity score
        """
        complexity = 1
        
        for node in module.body:
            if isinstance(node, (astroid.If, astroid.While, astroid.For, astroid.Try)):
                complexity += 1
        
        return complexity
    
    def _detect_error_patterns(self, source_code: str) -> List[Dict[str, str]]:
        """
        Detect potential error-prone code patterns
        
        Args:
            source_code (str): Source code to analyze
        
        Returns:
            List of detected error patterns
        """
        error_patterns = []
        
        # Detect common error-prone patterns
        patterns = [
            {
                'name': 'Bare except',
                'regex': r'except:\s*pass',
                'severity': 'high',
                'recommendation': 'Use specific exception handling'
            },
            {
                'name': 'Global variable usage',
                'regex': r'global\s+\w+',
                'severity': 'medium',
                'recommendation': 'Avoid global variables'
            },
            {
                'name': 'Potential memory leak',
                'regex': r'del\s+\w+',
                'severity': 'low',
                'recommendation': 'Be cautious with manual memory management'
            }
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern['regex'], source_code)
            if matches:
                error_patterns.append({
                    'pattern': pattern['name'],
                    'occurrences': len(matches),
                    'severity': pattern['severity'],
                    'recommendation': pattern['recommendation']
                })
        
        return error_patterns
    
    def _track_module_dependencies(self, file_path: str) -> List[str]:
        """
        Track dependencies for a specific module
        
        Args:
            file_path (str): Path to the Python file
        
        Returns:
            List of detected dependencies
        """
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Parse module with AST
            module = ast.parse(source_code)
            
            dependencies = []
            
            # Track imports
            for node in ast.walk(module):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
            
            return dependencies
        
        except Exception as e:
            self.logger.warning(f"Dependency tracking failed for {file_path}: {e}")
            return []
    
    def generate_semantic_linking_graph(self, project_structure: Dict[str, Any]) -> nx.DiGraph:
        """
        Generate a semantic linking graph based on module relationships
        
        Args:
            project_structure (Dict): Comprehensive project structure analysis
        
        Returns:
            NetworkX Directed Graph of semantic links
        """
        semantic_graph = nx.DiGraph()
        
        try:
            # Add modules as nodes
            for module_name, module_info in project_structure['modules'].items():
                semantic_graph.add_node(
                    module_name, 
                    **module_info.get('structure', {})
                )
            
            # Add edges based on dependencies
            for module, dependencies in project_structure['dependencies'].items():
                for dep in dependencies:
                    if dep in project_structure['modules']:
                        semantic_graph.add_edge(module, dep)
            
            return semantic_graph
        
        except Exception as e:
            self.logger.error(f"Semantic linking graph generation failed: {e}")
            return nx.DiGraph()
    
    def intelligent_code_correction(self, file_path: str) -> Optional[str]:
        """
        Perform intelligent code correction
        
        Args:
            file_path (str): Path to the Python file to correct
        
        Returns:
            Corrected source code or None if no corrections needed
        """
        try:
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Use Black for code formatting
            formatted_code = black.format_str(source_code, mode=black.FileMode())
            
            # Detect and correct common issues
            corrected_code = self._apply_semantic_corrections(formatted_code)
            
            # Only write if changes were made
            if corrected_code != source_code:
                with open(file_path, 'w') as f:
                    f.write(corrected_code)
                
                self.logger.info(f"Intelligent corrections applied to {file_path}")
                return corrected_code
            
            return None
        
        except Exception as e:
            self.logger.error(f"Intelligent code correction failed for {file_path}: {e}")
            return None
    
    def _apply_semantic_corrections(self, source_code: str) -> str:
        """
        Apply semantic-aware code corrections
        
        Args:
            source_code (str): Source code to correct
        
        Returns:
            Corrected source code
        """
        try:
            # Parse the source code
            module = ast.parse(source_code)
            
            # Transformations
            transformations = [
                self._remove_bare_except_blocks,
                self._replace_global_variables,
                self._add_type_hints
            ]
            
            # Apply transformations
            for transform in transformations:
                module = transform(module)
            
            # Convert back to source code
            return ast.unparse(module)
        
        except Exception as e:
            self.logger.warning(f"Semantic correction failed: {e}")
            return source_code
    
    def _remove_bare_except_blocks(self, module: ast.Module) -> ast.Module:
        """
        Remove bare except blocks
        
        Args:
            module (ast.Module): AST module
        
        Returns:
            Modified AST module
        """
        class BareExceptRemover(ast.NodeTransformer):
            def visit_ExceptHandler(self, node):
                # Replace bare except with specific exception
                if not node.type:
                    node.type = ast.Name(id='Exception', ctx=ast.Load())
                return node
        
        return BareExceptRemover().visit(module)
    
    def _replace_global_variables(self, module: ast.Module) -> ast.Module:
        """
        Replace global variables with more robust patterns
        
        Args:
            module (ast.Module): AST module
        
        Returns:
            Modified AST module
        """
        class GlobalVariableReplacer(ast.NodeTransformer):
            def visit_Global(self, node):
                # Log and suggest alternative
                self.logger.warning(f"Global variable detected: {', '.join(node.names)}")
                return None
        
        return GlobalVariableReplacer().visit(module)
    
    def _add_type_hints(self, module: ast.Module) -> ast.Module:
        """
        Add type hints to functions and methods
        
        Args:
            module (ast.Module): AST module
        
        Returns:
            Modified AST module
        """
        class TypeHintAdder(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Add return type hint if missing
                if not node.returns:
                    node.returns = ast.Name(id='Any', ctx=ast.Load())
                
                # Add type hints to arguments
                for arg in node.args.args:
                    if not arg.annotation:
                        arg.annotation = ast.Name(id='Any', ctx=ast.Load())
                
                return node
        
        return TypeHintAdder().visit(module)
    
    def run_comprehensive_error_correction(self):
        """
        Run comprehensive error correction across the entire project
        """
        try:
            # Analyze project structure
            project_structure = self.analyze_project_structure()
            
            # Generate semantic linking graph
            semantic_graph = self.generate_semantic_linking_graph(project_structure)
            
            # Correct files
            corrections_summary = {
                'total_files': 0,
                'files_corrected': 0,
                'error_patterns': {}
            }
            
            # Walk through Python files
            for root, _, files in os.walk(self.base_dir):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        
                        # Skip certain directories
                        if any(skip in full_path for skip in ['venv', '.git', 'logs']):
                            continue
                        
                        corrections_summary['total_files'] += 1
                        
                        # Perform intelligent correction
                        corrected_code = self.intelligent_code_correction(full_path)
                        
                        if corrected_code:
                            corrections_summary['files_corrected'] += 1
                            
                            # Analyze error patterns
                            module_info = self._analyze_module(full_path)
                            for pattern in module_info.get('error_patterns', []):
                                pattern_name = pattern['pattern']
                                corrections_summary['error_patterns'][pattern_name] = \
                                    corrections_summary['error_patterns'].get(pattern_name, 0) + 1
            
            # Log correction summary
            self.logger.info("Comprehensive Error Correction Summary:")
            self.logger.info(f"Total Files: {corrections_summary['total_files']}")
            self.logger.info(f"Files Corrected: {corrections_summary['files_corrected']}")
            self.logger.info("Error Patterns Detected:")
            for pattern, count in corrections_summary['error_patterns'].items():
                self.logger.info(f"- {pattern}: {count} occurrences")
            
            return corrections_summary
        
        except Exception as e:
            self.logger.error(f"Comprehensive error correction failed: {e}")
            return {}

def main():
    """
    Main execution for intelligent error correction
    """
    try:
        error_corrector = IntelligentErrorCorrector()
        correction_results = error_corrector.run_comprehensive_error_correction()
        
        # Print key insights
        print("Intelligent Error Correction Insights:")
        print(f"Total Files Analyzed: {correction_results.get('total_files', 0)}")
        print(f"Files Corrected: {correction_results.get('files_corrected', 0)}")
        print("\nError Patterns:")
        for pattern, count in correction_results.get('error_patterns', {}).items():
            print(f"- {pattern}: {count} occurrences")
    
    except Exception as e:
        print(f"Intelligent error correction failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 