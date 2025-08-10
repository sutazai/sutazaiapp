#!/usr/bin/env python3
"""
Comprehensive Cyclomatic Complexity Analyzer
Identifies functions with high complexity for refactoring.
"""

import ast
import os
import sys
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Set
import json
import argparse
from dataclasses import dataclass


@dataclass
class ComplexFunction:
    """Data class for storing function complexity information."""
    file_path: str
    function_name: str
    line_number: int
    complexity: int
    line_count: int
    parameters: int
    nested_functions: int
    class_name: str = None


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.functions = []
        self.current_function = None
        self.current_class = None
        
    def visit_FunctionDef(self, node):
        """Visit function definitions and calculate complexity."""
        old_complexity = self.complexity
        old_function = self.current_function
        
        self.complexity = 1  # Reset for new function
        self.current_function = node.name
        
        # Count parameters
        param_count = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg:
            param_count += 1
        if node.args.kwarg:
            param_count += 1
            
        # Visit function body
        for child in ast.iter_child_nodes(node):
            self.visit(child)
            
        # Calculate line count
        if hasattr(node, 'end_lineno') and node.end_lineno:
            line_count = node.end_lineno - node.lineno + 1
        else:
            line_count = 1
            
        # Count nested functions
        nested_functions = sum(1 for child in ast.walk(node) 
                             if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                             and child != node)
        
        # Store function info
        func_info = ComplexFunction(
            file_path="",  # Will be set by caller
            function_name=node.name,
            line_number=node.lineno,
            complexity=self.complexity,
            line_count=line_count,
            parameters=param_count,
            nested_functions=nested_functions,
            class_name=self.current_class
        )
        self.functions.append(func_info)
        
        # Restore previous state
        self.complexity = old_complexity
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        self.visit_FunctionDef(node)
        
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_If(self, node):
        """Count if statements."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_While(self, node):
        """Count while loops."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        """Count for loops."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_AsyncFor(self, node):
        """Count async for loops."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_With(self, node):
        """Count with statements."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_AsyncWith(self, node):
        """Count async with statements."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_Try(self, node):
        """Count try blocks."""
        self.complexity += 1
        # Count except handlers
        self.complexity += len(node.handlers)
        self.generic_visit(node)
        
    def visit_BoolOp(self, node):
        """Count boolean operations (and, or)."""
        if isinstance(node.op, (ast.And, ast.Or)):
            self.complexity += len(node.values) - 1
        self.generic_visit(node)
        
    def visit_Compare(self, node):
        """Count comparison operations."""
        if len(node.ops) > 1:
            self.complexity += len(node.ops) - 1
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        """Count list comprehensions."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_SetComp(self, node):
        """Count set comprehensions."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_DictComp(self, node):
        """Count dict comprehensions."""
        self.complexity += 1
        self.generic_visit(node)
        
    def visit_GeneratorExp(self, node):
        """Count generator expressions."""
        self.complexity += 1
        self.generic_visit(node)


class ComplexityAnalyzer:
    """Main complexity analyzer class."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.complex_functions = []
        self.stats = defaultdict(int)
        
    def analyze_file(self, file_path: Path) -> List[ComplexFunction]:
        """Analyze a single Python file for complexity."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            tree = ast.parse(content)
            visitor = CyclomaticComplexityVisitor()
            visitor.visit(tree)
            
            # Set file path for all functions
            for func in visitor.functions:
                func.file_path = str(file_path)
                
            return visitor.functions
            
        except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
            
    def analyze_directory(self) -> None:
        """Analyze all Python files in the directory."""
        python_files = list(self.root_path.rglob("*.py"))
        
        print(f"Analyzing {len(python_files)} Python files...")
        
        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(python_files)} files")
                
            functions = self.analyze_file(file_path)
            self.complex_functions.extend(functions)
            
        self._calculate_stats()
        
    def _calculate_stats(self) -> None:
        """Calculate statistics about complexity."""
        total_functions = len(self.complex_functions)
        
        complexity_ranges = {
            'low': (1, 10),
            'medium': (11, 15),
            'high': (16, 30),
            'very_high': (31, 50),
            'extreme': (51, float('inf'))
        }
        
        for func in self.complex_functions:
            for range_name, (min_val, max_val) in complexity_ranges.items():
                if min_val <= func.complexity <= max_val:
                    self.stats[f'complexity_{range_name}'] += 1
                    break
                    
            if func.line_count > 200:
                self.stats['long_functions'] += 1
            if func.parameters > 10:
                self.stats['many_parameters'] += 1
            if func.nested_functions > 3:
                self.stats['deeply_nested'] += 1
                
        self.stats['total_functions'] = total_functions
        
    def get_high_complexity_functions(self, min_complexity: int = 15) -> List[ComplexFunction]:
        """Get functions with complexity above threshold."""
        return [func for func in self.complex_functions 
                if func.complexity >= min_complexity]
        
    def get_worst_offenders(self, limit: int = 50) -> List[ComplexFunction]:
        """Get the worst complexity offenders."""
        return sorted(self.complex_functions, 
                     key=lambda f: f.complexity, 
                     reverse=True)[:limit]
        
    def generate_report(self, output_path: str = None) -> Dict:
        """Generate comprehensive complexity report."""
        high_complexity = self.get_high_complexity_functions()
        worst_offenders = self.get_worst_offenders()
        
        report = {
            'summary': dict(self.stats),
            'high_complexity_functions': [
                {
                    'file': func.file_path,
                    'function': func.function_name,
                    'class': func.class_name,
                    'line': func.line_number,
                    'complexity': func.complexity,
                    'line_count': func.line_count,
                    'parameters': func.parameters,
                    'nested_functions': func.nested_functions
                }
                for func in high_complexity
            ],
            'worst_offenders': [
                {
                    'file': func.file_path,
                    'function': func.function_name,
                    'class': func.class_name,
                    'line': func.line_number,
                    'complexity': func.complexity,
                    'line_count': func.line_count,
                    'parameters': func.parameters,
                    'nested_functions': func.nested_functions
                }
                for func in worst_offenders
            ]
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "="*60)
        print("CYCLOMATIC COMPLEXITY ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total functions analyzed: {self.stats['total_functions']}")
        print(f"Low complexity (1-10): {self.stats['complexity_low']}")
        print(f"Medium complexity (11-15): {self.stats['complexity_medium']}")
        print(f"High complexity (16-30): {self.stats['complexity_high']}")
        print(f"Very high complexity (31-50): {self.stats['complexity_very_high']}")
        print(f"Extreme complexity (51+): {self.stats['complexity_extreme']}")
        print(f"Functions > 200 lines: {self.stats['long_functions']}")
        print(f"Functions > 10 parameters: {self.stats['many_parameters']}")
        print(f"Functions with deep nesting: {self.stats['deeply_nested']}")
        
        high_complexity_total = (self.stats['complexity_high'] + 
                               self.stats['complexity_very_high'] + 
                               self.stats['complexity_extreme'])
        
        print(f"\nFUNCTIONS REQUIRING REFACTORING: {high_complexity_total}")
        
        if high_complexity_total > 0:
            print("\nWORST OFFENDERS (Top 10):")
            worst = self.get_worst_offenders(10)
            for i, func in enumerate(worst, 1):
                rel_path = Path(func.file_path).relative_to(self.root_path)
                print(f"{i:2d}. {func.function_name} ({rel_path}:{func.line_number}) "
                      f"- Complexity: {func.complexity}, Lines: {func.line_count}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze cyclomatic complexity')
    parser.add_argument('--path', '-p', default='.', 
                       help='Path to analyze (default: current directory)')
    parser.add_argument('--output', '-o', help='Output JSON report file')
    parser.add_argument('--min-complexity', '-m', type=int, default=15,
                       help='Minimum complexity to report (default: 15)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    analyzer = ComplexityAnalyzer(args.path)
    analyzer.analyze_directory()
    
    if not args.quiet:
        analyzer.print_summary()
        
    report = analyzer.generate_report(args.output)
    
    if args.output:
        print(f"\nDetailed report saved to: {args.output}")
        
    return report


if __name__ == "__main__":
    main()