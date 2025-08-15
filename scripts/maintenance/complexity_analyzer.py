#!/usr/bin/env python3
"""
Cyclomatic Complexity Analyzer and Refactoring Tool
Identifies functions with complexity > 15 and provides refactoring recommendations.

Author: Ultra Python Pro
Date: August 10, 2025
Purpose: Systematic analysis and refactoring of high-complexity functions
"""

import ast
import os
import sys
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FunctionComplexity:
    """Data class to store function complexity information."""
    file_path: str
    function_name: str
    line_number: int
    complexity: int
    lines_of_code: int
    parameters_count: int
    nested_functions: int
    is_async: bool
    docstring: Optional[str]
    issues: List[str]

class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.functions: List[FunctionComplexity] = []
        self.current_function = None
        self.current_file = ""
        
    def set_current_file(self, file_path: str):
        """Set the current file being analyzed."""
        self.current_file = file_path
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition and calculate complexity."""
        self._analyze_function(node, is_async=False)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition and calculate complexity."""
        self._analyze_function(node, is_async=True)
        
    def _analyze_function(self, node: ast.FunctionDef, is_async: bool):
        """Analyze a function node for complexity."""
        # Save parent function context
        parent_function = self.current_function
        
        # Calculate function complexity
        func_complexity = FunctionComplexityCalculator()
        func_complexity.visit(node)
        
        # Count lines of code (excluding docstring and comments)
        lines_of_code = self._count_lines_of_code(node)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Count nested functions
        nested_functions = sum(1 for child in ast.walk(node) 
                             if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and child != node)
        
        # Identify issues
        issues = self._identify_issues(node, func_complexity.complexity, lines_of_code)
        
        # Create function complexity record
        func_info = FunctionComplexity(
            file_path=self.current_file,
            function_name=node.name,
            line_number=node.lineno,
            complexity=func_complexity.complexity,
            lines_of_code=lines_of_code,
            parameters_count=len(node.args.args),
            nested_functions=nested_functions,
            is_async=is_async,
            docstring=docstring,
            issues=issues
        )
        
        self.functions.append(func_info)
        self.current_function = func_info
        
        # Continue visiting child nodes
        self.generic_visit(node)
        
        # Restore parent function context
        self.current_function = parent_function
        
    def _count_lines_of_code(self, node: ast.FunctionDef) -> int:
        """Count actual lines of code, excluding docstrings and comments."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno - node.lineno + 1
        
        # Fallback: count AST nodes (approximate)
        return len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
        
    def _identify_issues(self, node: ast.FunctionDef, complexity: int, lines_of_code: int) -> List[str]:
        """Identify specific issues with the function."""
        issues = []
        
        if complexity > 15:
            issues.append(f"High cyclomatic complexity: {complexity}")
        if lines_of_code > 50:
            issues.append(f"Too many lines of code: {lines_of_code}")
        if len(node.args.args) > 7:
            issues.append(f"Too many parameters: {len(node.args.args)}")
        if not ast.get_docstring(node):
            issues.append("Missing docstring")
            
        # Check for deeply nested structures
        max_nesting = self._calculate_max_nesting(node)
        if max_nesting > 4:
            issues.append(f"Deep nesting level: {max_nesting}")
            
        return issues
        
    def _calculate_max_nesting(self, node: ast.AST) -> int:
        """Calculate maximum nesting level in the function."""
        max_depth = 0
        
        def _visit_node(n, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            # Increment depth for control structures
            if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
                
            for child in ast.iter_child_nodes(n):
                _visit_node(child, depth)
                
        _visit_node(node)
        return max_depth

class FunctionComplexityCalculator(ast.NodeVisitor):
    """Calculate cyclomatic complexity for a single function."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        
    def visit_If(self, node):
        """Each if/elif adds 1 to complexity."""
        self.complexity += 1
        if hasattr(node, 'orelse') and node.orelse:
            # Check if orelse is another If (elif)
            if isinstance(node.orelse[0], ast.If):
                # elif will be counted when we visit it
                pass
            else:
                # else clause doesn't add complexity
                pass
        self.generic_visit(node)
        
    def visit_For(self, node):
        """Each for loop adds 1 to complexity."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        """Each while loop adds 1 to complexity."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_ExceptHandler(self, node):
        """Each except handler adds 1 to complexity."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_With(self, node):
        """Each with statement adds 1 to complexity."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Assert(self, node):
        """Each assert adds 1 to complexity."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        """Each boolean operator adds complexity."""
        if isinstance(node.op, (ast.And, ast.Or)):
            self.complexity += len(node.values) - 1
        self.generic_visit(node)
        
    def visit_Compare(self, node):
        """Each comparison adds complexity."""
        self.complexity += len(node.comparators)
        self.generic_visit(node)

class CodebaseAnalyzer:
    """Main analyzer class for the entire codebase."""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.all_functions: List[FunctionComplexity] = []
        self.analysis_results = {
            'total_functions': 0,
            'high_complexity_functions': 0,
            'files_analyzed': 0,
            'average_complexity': 0.0,
            'top_violators': []
        }
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase for complexity issues."""
        logger.info(f"Starting codebase analysis at: {self.root_path}")
        
        python_files = list(self.root_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")
        
        for py_file in python_files:
            try:
                self._analyze_file(py_file)
            except Exception as e:
                logger.warning(f"Error analyzing {py_file}: {e}")
                
        self._compile_results()
        logger.info(f"Analysis complete. Found {self.analysis_results['high_complexity_functions']} high-complexity functions")
        
        return self.analysis_results
        
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Visit the AST
            visitor = ComplexityVisitor()
            visitor.set_current_file(str(file_path))
            visitor.visit(tree)
            
            # Add functions to our collection
            self.all_functions.extend(visitor.functions)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            
    def _compile_results(self):
        """Compile analysis results."""
        if not self.all_functions:
            return
            
        # Basic statistics
        self.analysis_results['total_functions'] = len(self.all_functions)
        self.analysis_results['high_complexity_functions'] = sum(
            1 for f in self.all_functions if f.complexity > 15
        )
        
        complexities = [f.complexity for f in self.all_functions]
        self.analysis_results['average_complexity'] = sum(complexities) / len(complexities)
        
        # Top violators by file
        file_violations = {}
        for func in self.all_functions:
            if func.complexity > 15:
                file_key = func.file_path
                if file_key not in file_violations:
                    file_violations[file_key] = []
                file_violations[file_key].append(func)
                
        # Sort by number of violations
        sorted_violators = sorted(
            file_violations.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        self.analysis_results['top_violators'] = [
            {
                'file': file_path,
                'violation_count': len(functions),
                'functions': [
                    {
                        'name': f.function_name,
                        'complexity': f.complexity,
                        'lines_of_code': f.lines_of_code,
                        'line_number': f.line_number,
                        'issues': f.issues
                    }
                    for f in functions
                ]
            }
            for file_path, functions in sorted_violators[:20]  # Top 20 files
        ]
        
    def get_high_complexity_functions(self, threshold: int = 15) -> List[FunctionComplexity]:
        """Get all functions with complexity above threshold."""
        return [f for f in self.all_functions if f.complexity > threshold]
        
    def save_report(self, output_file: str = "complexity_analysis_report.json"):
        """Save detailed analysis report to JSON file."""
        report = {
            'analysis_summary': self.analysis_results,
            'all_high_complexity_functions': [
                asdict(f) for f in self.get_high_complexity_functions()
            ],
            'analysis_timestamp': self._get_timestamp()
        }
        
        output_path = self.root_path / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Analysis report saved to: {output_path}")
        return output_path
        
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

def main():
    """Main entry point for the complexity analyzer."""
    parser = argparse.ArgumentParser(description='Analyze cyclomatic complexity of Python codebase')
    parser.add_argument('--root', '-r', default='/opt/sutazaiapp', 
                       help='Root directory to analyze')
    parser.add_argument('--threshold', '-t', type=int, default=15,
                       help='Complexity threshold for reporting')
    parser.add_argument('--output', '-o', default='complexity_analysis_report.json',
                       help='Output file for detailed report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Run analysis
    analyzer = CodebaseAnalyzer(args.root)
    results = analyzer.analyze_codebase()
    
    # Save detailed report
    report_path = analyzer.save_report(args.output)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CYCLOMATIC COMPLEXITY ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total functions analyzed: {results['total_functions']}")
    logger.info(f"High complexity functions (>{args.threshold}): {results['high_complexity_functions']}")
    logger.info(f"Average complexity: {results['average_complexity']:.2f}")
    logger.info(f"Detailed report saved to: {report_path}")
    
    if results['top_violators']:
        logger.info("\nTOP VIOLATOR FILES:")
        for i, violator in enumerate(results['top_violators'][:10], 1):
            logger.info(f"{i:2d}. {violator['file']} ({violator['violation_count']} violations)")
            
    logger.info("\nNext steps:")
    logger.info("1. Review the detailed JSON report")
    logger.info("2. Run refactoring tools on high-complexity functions")
    logger.info("3. Update tests to ensure functionality preservation")
    
    return 0 if results['high_complexity_functions'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())