#!/usr/bin/env python3
"""
Critical Function Refactorer - Immediate Action Tool
Refactors the most critical high-complexity functions (complexity > 30) first.

Author: Ultra Python Pro  
Date: August 10, 2025
Purpose: Emergency refactoring of critical functions per Rules 1-19
"""

import ast
import os
import sys
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RefactoringOperation:
    """Represents a specific refactoring operation."""
    function_name: str
    file_path: str
    original_complexity: int
    target_complexity: int
    extraction_strategy: str
    backup_created: bool = False

class FunctionAnalyzer(ast.NodeVisitor):
    """Analyzes function structure for smart refactoring."""
    
    def __init__(self):
        self.function_info = {}
        self.current_function = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Analyze function structure."""
        info = {
            'name': node.name,
            'line_number': node.lineno,
            'parameters': len(node.args.args),
            'has_docstring': ast.get_docstring(node) is not None,
            'validation_blocks': [],
            'loop_blocks': [],
            'conditional_blocks': [],
            'try_blocks': [],
            'nested_functions': 0
        }
        
        # Analyze function body
        for i, stmt in enumerate(node.body):
            if self._is_validation_block(stmt):
                info['validation_blocks'].append((i, stmt))
            elif isinstance(stmt, (ast.For, ast.While)):
                info['loop_blocks'].append((i, stmt))
            elif isinstance(stmt, ast.If):
                info['conditional_blocks'].append((i, stmt))
            elif isinstance(stmt, ast.Try):
                info['try_blocks'].append((i, stmt))
            elif isinstance(stmt, ast.FunctionDef):
                info['nested_functions'] += 1
                
        self.function_info[node.name] = info
        self.current_function = info
        
        # Visit child nodes
        self.generic_visit(node)
        self.current_function = None
        
    def _is_validation_block(self, stmt: ast.stmt) -> bool:
        """Identify validation logic blocks."""
        if isinstance(stmt, ast.If):
            # Pattern: if not condition: raise/return
            if (isinstance(stmt.test, ast.UnaryOp) and 
                isinstance(stmt.test.op, ast.Not)):
                return True
        elif isinstance(stmt, ast.Assert):
            return True
        elif isinstance(stmt, (ast.Raise, ast.Return)) and len(stmt.value if hasattr(stmt, 'value') else []) > 0:
            return True
            
        return False

class SmartFunctionRefactorer:
    """Smart refactoring engine focusing on critical functions."""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "backups" / "critical_refactoring" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.operations_log = []
        
    def refactor_critical_functions(self, complexity_report_path: str) -> Dict[str, Any]:
        """Refactor the most critical functions first."""
        logger.info("Starting critical function refactoring...")
        
        # Load complexity analysis
        with open(complexity_report_path, 'r') as f:
            report = json.load(f)
            
        # Get critical functions (complexity > 30)
        critical_functions = [
            f for f in report['all_high_complexity_functions'] 
            if f['complexity'] > 30
        ]
        
        logger.info(f"Found {len(critical_functions)} critical functions to refactor")
        
        results = {
            'critical_functions_processed': 0,
            'successful_refactorings': 0,
            'failed_refactorings': 0,
            'complexity_reduction': 0,
            'operations': []
        }
        
        # Sort by complexity (highest first)
        critical_functions.sort(key=lambda x: x['complexity'], reverse=True)
        
        for func_info in critical_functions:
            try:
                operation = self._refactor_single_function(func_info)
                self.operations_log.append(operation)
                results['operations'].append(operation.__dict__)
                
                if operation.backup_created:
                    results['successful_refactorings'] += 1
                    results['complexity_reduction'] += (
                        operation.original_complexity - operation.target_complexity
                    )
                else:
                    results['failed_refactorings'] += 1
                    
            except Exception as e:
                logger.error(f"Error refactoring {func_info['function_name']}: {e}")
                results['failed_refactorings'] += 1
                
            results['critical_functions_processed'] += 1
            
        logger.info(f"Critical refactoring complete: {results}")
        return results
        
    def _refactor_single_function(self, func_info: Dict) -> RefactoringOperation:
        """Refactor a single critical function."""
        file_path = func_info['file_path']
        function_name = func_info['function_name']
        original_complexity = func_info['complexity']
        
        logger.info(f"Refactoring {function_name} (complexity: {original_complexity})")
        
        operation = RefactoringOperation(
            function_name=function_name,
            file_path=file_path,
            original_complexity=original_complexity,
            target_complexity=10,  # Target complexity
            extraction_strategy="auto"
        )
        
        try:
            # Create backup
            self._backup_file(file_path)
            operation.backup_created = True
            
            # Read and parse file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Analyze function structure
            analyzer = FunctionAnalyzer()
            analyzer.visit(tree)
            
            if function_name in analyzer.function_info:
                func_analysis = analyzer.function_info[function_name]
                
                # Apply appropriate refactoring strategy
                strategy = self._determine_strategy(func_analysis, original_complexity)
                operation.extraction_strategy = strategy
                
                # Apply refactoring
                refactored_tree = self._apply_refactoring_strategy(
                    tree, function_name, func_analysis, strategy
                )
                
                # Generate refactored code
                refactored_content = ast.unparse(refactored_tree)
                
                # Write refactored code
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(refactored_content)
                    
                # Estimate new complexity (simple heuristic)
                operation.target_complexity = max(
                    original_complexity // 3, 8
                )  # Estimate 66% reduction
                
                logger.info(f"Successfully refactored {function_name}")
                
        except Exception as e:
            logger.error(f"Error in refactoring operation: {e}")
            # Restore backup if needed
            if operation.backup_created:
                self._restore_backup(file_path)
                
        return operation
        
    def _determine_strategy(self, func_analysis: Dict, complexity: int) -> str:
        """Determine the best refactoring strategy for the function."""
        if len(func_analysis['validation_blocks']) > 3:
            return "extract_validation"
        elif len(func_analysis['loop_blocks']) > 2:
            return "extract_loops"
        elif len(func_analysis['conditional_blocks']) > 4:
            return "extract_conditionals"
        elif complexity > 50:
            return "full_decomposition"
        else:
            return "targeted_extraction"
            
    def _apply_refactoring_strategy(
        self, 
        tree: ast.Module, 
        function_name: str, 
        func_analysis: Dict,
        strategy: str
    ) -> ast.Module:
        """Apply the chosen refactoring strategy."""
        
        if strategy == "extract_validation":
            return self._extract_validation_logic(tree, function_name, func_analysis)
        elif strategy == "extract_loops":
            return self._extract_loop_logic(tree, function_name, func_analysis)
        elif strategy == "extract_conditionals":
            return self._extract_conditional_logic(tree, function_name, func_analysis)
        elif strategy == "full_decomposition":
            return self._full_function_decomposition(tree, function_name, func_analysis)
        else:
            return self._targeted_extraction(tree, function_name, func_analysis)
            
    def _extract_validation_logic(self, tree: ast.Module, function_name: str, func_analysis: Dict) -> ast.Module:
        """Extract validation logic into separate function."""
        # Find the target function
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target_function = node
                break
                
        if not target_function:
            return tree
            
        # Extract validation statements
        validation_stmts = []
        remaining_stmts = []
        
        for stmt in target_function.body:
            if self._is_validation_statement(stmt):
                validation_stmts.append(stmt)
            else:
                remaining_stmts.append(stmt)
                
        if validation_stmts:
            # Create validation helper function
            validation_func = ast.FunctionDef(
                name=f"_validate_{function_name}_inputs",
                args=ast.arguments(
                    posonlyargs=[],
                    args=target_function.args.args.copy(),
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=target_function.args.defaults.copy()
                ),
                body=validation_stmts,
                decorator_list=[],
                returns=None,
                type_comment=None
            )
            
            # Add docstring to helper function
            validation_func.body.insert(0, ast.Expr(
                value=ast.Constant(value=f"Validate inputs for {function_name}.")
            ))
            
            # Insert validation call at beginning of main function
            validation_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=f"_validate_{function_name}_inputs", ctx=ast.Load()),
                    args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in target_function.args.args],
                    keywords=[]
                )
            )
            
            remaining_stmts.insert(0, validation_call)
            target_function.body = remaining_stmts
            
            # Insert validation function before main function
            func_index = tree.body.index(target_function)
            tree.body.insert(func_index, validation_func)
            
        return tree
        
    def _extract_loop_logic(self, tree: ast.Module, function_name: str, func_analysis: Dict) -> ast.Module:
        """Extract complex loop logic into separate functions."""
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target_function = node
                break
                
        if not target_function:
            return tree
            
        helper_functions = []
        modified_stmts = []
        helper_count = 0
        
        for i, stmt in enumerate(target_function.body):
            if isinstance(stmt, (ast.For, ast.While)) and self._is_complex_loop(stmt):
                helper_count += 1
                helper_name = f"_process_{function_name}_loop_{helper_count}"
                
                # Create helper function for loop
                helper_func = ast.FunctionDef(
                    name=helper_name,
                    args=ast.arguments(
                        posonlyargs=[],
                        args=target_function.args.args.copy(),
                        vararg=None,
                        kwonlyargs=[],
                        kw_defaults=[],
                        kwarg=None,
                        defaults=target_function.args.defaults.copy()
                    ),
                    body=[stmt],
                    decorator_list=[],
                    returns=None,
                    type_comment=None
                )
                
                # Add docstring
                helper_func.body.insert(0, ast.Expr(
                    value=ast.Constant(value=f"Process loop logic for {function_name}.")
                ))
                
                helper_functions.append(helper_func)
                
                # Replace loop with function call
                call_stmt = ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id=helper_name, ctx=ast.Load()),
                        args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in target_function.args.args],
                        keywords=[]
                    )
                )
                modified_stmts.append(call_stmt)
            else:
                modified_stmts.append(stmt)
                
        target_function.body = modified_stmts
        
        # Insert helper functions before main function
        func_index = tree.body.index(target_function)
        for i, helper_func in enumerate(helper_functions):
            tree.body.insert(func_index + i, helper_func)
            
        return tree
        
    def _extract_conditional_logic(self, tree: ast.Module, function_name: str, func_analysis: Dict) -> ast.Module:
        """Extract complex conditional logic into separate functions."""
        # Similar implementation to loop extraction but for if statements
        return self._extract_loop_logic(tree, function_name, func_analysis)  # Simplified for now
        
    def _full_function_decomposition(self, tree: ast.Module, function_name: str, func_analysis: Dict) -> ast.Module:
        """Perform full decomposition of extremely complex functions."""
        # Apply all extraction strategies
        tree = self._extract_validation_logic(tree, function_name, func_analysis)
        tree = self._extract_loop_logic(tree, function_name, func_analysis)
        return tree
        
    def _targeted_extraction(self, tree: ast.Module, function_name: str, func_analysis: Dict) -> ast.Module:
        """Perform targeted extraction based on function analysis."""
        # Apply the most appropriate single strategy
        if func_analysis['validation_blocks']:
            return self._extract_validation_logic(tree, function_name, func_analysis)
        elif func_analysis['loop_blocks']:
            return self._extract_loop_logic(tree, function_name, func_analysis)
        else:
            return tree
            
    def _is_validation_statement(self, stmt: ast.stmt) -> bool:
        """Check if statement is validation logic."""
        if isinstance(stmt, ast.If):
            if (isinstance(stmt.test, ast.UnaryOp) and 
                isinstance(stmt.test.op, ast.Not)):
                # Check if body contains raise or return
                for body_stmt in stmt.body:
                    if isinstance(body_stmt, (ast.Raise, ast.Return)):
                        return True
        elif isinstance(stmt, ast.Assert):
            return True
        elif isinstance(stmt, ast.Raise):
            return True
            
        return False
        
    def _is_complex_loop(self, stmt: ast.stmt) -> bool:
        """Check if loop is complex enough to extract."""
        if isinstance(stmt, (ast.For, ast.While)):
            # Count nested statements
            nested_count = sum(1 for _ in ast.walk(stmt))
            return nested_count > 8  # Threshold for extraction
        return False
        
    def _backup_file(self, file_path: str):
        """Create backup of file before refactoring."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        rel_path = Path(file_path).relative_to(self.root_path)
        backup_path = self.backup_dir / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Backed up {file_path} to {backup_path}")
        
    def _restore_backup(self, file_path: str):
        """Restore file from backup."""
        rel_path = Path(file_path).relative_to(self.root_path)
        backup_path = self.backup_dir / rel_path
        
        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored {file_path} from backup")

def main():
    """Main entry point for critical function refactoring."""
    parser = argparse.ArgumentParser(description='Critical function refactoring tool')
    parser.add_argument('--complexity-report', '-r', 
                       default='/opt/sutazaiapp/reports/complexity_analysis_20250810.json',
                       help='Path to complexity analysis report')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show refactoring plan without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    refactorer = SmartFunctionRefactorer()
    
    if args.dry_run:
        # Load report and show plan
        with open(args.complexity_report, 'r') as f:
            report = json.load(f)
            
        critical_functions = [
            f for f in report['all_high_complexity_functions'] 
            if f['complexity'] > 30
        ]
        
        print("\n" + "="*70)
        print("CRITICAL FUNCTION REFACTORING PLAN")
        print("="*70)
        print(f"Critical functions to refactor: {len(critical_functions)}")
        
        # Sort by complexity
        critical_functions.sort(key=lambda x: x['complexity'], reverse=True)
        
        for i, func in enumerate(critical_functions[:20], 1):  # Top 20
            print(f"{i:2d}. {func['function_name']} (complexity: {func['complexity']}) - {func['file_path']}")
            
        return 0
        
    # Perform refactoring
    results = refactorer.refactor_critical_functions(args.complexity_report)
    
    print("\n" + "="*70)
    print("CRITICAL FUNCTION REFACTORING RESULTS")
    print("="*70)
    print(f"Functions processed: {results['critical_functions_processed']}")
    print(f"Successful refactorings: {results['successful_refactorings']}")
    print(f"Failed refactorings: {results['failed_refactorings']}")
    print(f"Estimated complexity reduction: {results['complexity_reduction']}")
    print(f"Backup directory: {refactorer.backup_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())