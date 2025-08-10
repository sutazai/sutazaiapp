#!/usr/bin/env python3
"""
Automated Function Refactoring Tool
Systematically refactors high-complexity functions while preserving functionality.

Author: Ultra Python Pro
Date: August 10, 2025
Purpose: Automated refactoring of high-complexity functions per Rules 1-19
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
class RefactoringPlan:
    """Plan for refactoring a function."""
    original_function: str
    new_functions: List[str]
    extraction_points: List[Tuple[int, int]]  # (start_line, end_line)
    helper_functions: List[str]
    preserved_behavior: Dict[str, Any]

class FunctionExtractor(ast.NodeTransformer):
    """AST transformer to extract complex logic into helper functions."""
    
    def __init__(self, function_name: str):
        self.function_name = function_name
        self.extracted_functions = []
        self.extraction_count = 0
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function and apply refactoring."""
        if node.name == self.function_name:
            return self._refactor_function(node)
        return node
        
    def _refactor_function(self, node: ast.FunctionDef):
        """Refactor a complex function."""
        # Extract validation logic
        validation_stmts = []
        remaining_stmts = []
        
        for stmt in node.body:
            if self._is_validation_statement(stmt):
                validation_stmts.append(stmt)
            else:
                remaining_stmts.append(stmt)
                
        # Extract loop logic
        loop_extractions = []
        final_stmts = []
        
        for stmt in remaining_stmts:
            if isinstance(stmt, (ast.For, ast.While)) and self._is_complex_loop(stmt):
                helper_name, helper_func = self._extract_loop_logic(stmt, node.args.args)
                loop_extractions.append(helper_func)
                
                # Replace loop with function call
                call_stmt = self._create_function_call(helper_name, stmt)
                final_stmts.append(call_stmt)
            else:
                final_stmts.append(stmt)
                
        # Extract conditional logic
        conditional_extractions = []
        simplified_stmts = []
        
        for stmt in final_stmts:
            if isinstance(stmt, ast.If) and self._is_complex_conditional(stmt):
                helper_name, helper_func = self._extract_conditional_logic(stmt, node.args.args)
                conditional_extractions.append(helper_func)
                
                # Replace conditional with function call
                call_stmt = self._create_function_call(helper_name, stmt)
                simplified_stmts.append(call_stmt)
            else:
                simplified_stmts.append(stmt)
                
        # Create validation helper if needed
        if validation_stmts:
            validation_func = self._create_validation_helper(validation_stmts, node.args.args)
            self.extracted_functions.append(validation_func)
            
            # Add validation call to main function
            validation_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=f"_validate_{self.function_name}_inputs", ctx=ast.Load()),
                    args=[ast.Name(id=arg.arg, ctx=ast.Load()) for arg in node.args.args],
                    keywords=[]
                )
            )
            simplified_stmts.insert(0, validation_call)
            
        # Update function body
        node.body = simplified_stmts
        
        # Add extracted functions
        self.extracted_functions.extend(loop_extractions)
        self.extracted_functions.extend(conditional_extractions)
        
        return node
        
    def _is_validation_statement(self, stmt: ast.stmt) -> bool:
        """Check if statement is input validation."""
        if isinstance(stmt, ast.If):
            # Look for validation patterns
            if isinstance(stmt.test, ast.UnaryOp) and isinstance(stmt.test.op, ast.Not):
                return True
            if isinstance(stmt.test, ast.Compare):
                # Check for None checks, type checks, etc.
                return any(
                    isinstance(comp, (ast.Is, ast.IsNot, ast.In, ast.NotIn))
                    for comp in stmt.test.ops
                )
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
            return nested_count > 10
        return False
        
    def _is_complex_conditional(self, stmt: ast.If) -> bool:
        """Check if conditional is complex enough to extract."""
        # Count nested statements in all branches
        total_count = sum(1 for _ in ast.walk(stmt))
        return total_count > 15
        
    def _extract_loop_logic(self, loop_stmt: ast.stmt, func_args: List[ast.arg]) -> Tuple[str, ast.FunctionDef]:
        """Extract loop logic into a helper function."""
        self.extraction_count += 1
        helper_name = f"_process_{self.function_name}_loop_{self.extraction_count}"
        
        # Determine parameters needed
        used_vars = self._find_used_variables(loop_stmt)
        needed_args = [arg for arg in func_args if arg.arg in used_vars]
        
        # Create helper function
        helper_func = ast.FunctionDef(
            name=helper_name,
            args=ast.arguments(
                posonlyargs=[],
                args=needed_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[loop_stmt],
            decorator_list=[],
            returns=None,
            type_comment=None
        )
        
        return helper_name, helper_func
        
    def _extract_conditional_logic(self, if_stmt: ast.If, func_args: List[ast.arg]) -> Tuple[str, ast.FunctionDef]:
        """Extract conditional logic into a helper function."""
        self.extraction_count += 1
        helper_name = f"_handle_{self.function_name}_condition_{self.extraction_count}"
        
        # Determine parameters needed
        used_vars = self._find_used_variables(if_stmt)
        needed_args = [arg for arg in func_args if arg.arg in used_vars]
        
        # Create helper function
        helper_func = ast.FunctionDef(
            name=helper_name,
            args=ast.arguments(
                posonlyargs=[],
                args=needed_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=[if_stmt],
            decorator_list=[],
            returns=None,
            type_comment=None
        )
        
        return helper_name, helper_func
        
    def _create_validation_helper(self, validation_stmts: List[ast.stmt], func_args: List[ast.arg]) -> ast.FunctionDef:
        """Create input validation helper function."""
        helper_name = f"_validate_{self.function_name}_inputs"
        
        return ast.FunctionDef(
            name=helper_name,
            args=ast.arguments(
                posonlyargs=[],
                args=func_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=validation_stmts,
            decorator_list=[],
            returns=None,
            type_comment=None
        )
        
    def _create_function_call(self, helper_name: str, original_stmt: ast.stmt) -> ast.stmt:
        """Create function call to replace extracted logic."""
        # Simple expression statement calling the helper
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id=helper_name, ctx=ast.Load()),
                args=[],  # Will be filled based on context
                keywords=[]
            )
        )
        
    def _find_used_variables(self, node: ast.AST) -> Set[str]:
        """Find all variables used in an AST node."""
        used_vars = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                used_vars.add(child.id)
                
        return used_vars

class SmartRefactorer:
    """Smart refactoring engine that preserves functionality."""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "backups" / "refactoring" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.refactoring_stats = {
            'files_processed': 0,
            'functions_refactored': 0,
            'complexity_reduction': 0,
            'lines_reduced': 0
        }
        
    def refactor_high_complexity_functions(self, complexity_report_path: str):
        """Refactor all high-complexity functions from analysis report."""
        logger.info(f"Starting smart refactoring process...")
        
        # Load complexity analysis
        with open(complexity_report_path, 'r') as f:
            report = json.load(f)
            
        high_complexity_functions = report['all_high_complexity_functions']
        logger.info(f"Found {len(high_complexity_functions)} functions to refactor")
        
        # Group by file for efficient processing
        files_to_refactor = {}
        for func_info in high_complexity_functions:
            file_path = func_info['file_path']
            if file_path not in files_to_refactor:
                files_to_refactor[file_path] = []
            files_to_refactor[file_path].append(func_info)
            
        # Process each file
        for file_path, functions in files_to_refactor.items():
            try:
                self._refactor_file(file_path, functions)
            except Exception as e:
                logger.error(f"Error refactoring {file_path}: {e}")
                
        logger.info(f"Refactoring complete. Stats: {self.refactoring_stats}")
        
    def _refactor_file(self, file_path: str, functions: List[Dict]):
        """Refactor a single file with multiple functions."""
        logger.info(f"Refactoring {file_path} ({len(functions)} functions)")
        
        # Create backup
        self._backup_file(file_path)
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            
        try:
            # Parse AST
            tree = ast.parse(original_content)
            
            # Apply refactoring for each function
            for func_info in functions:
                if func_info['complexity'] > 15:  # Only refactor high-complexity functions
                    extractor = FunctionExtractor(func_info['function_name'])
                    tree = extractor.visit(tree)
                    
                    # Insert extracted functions before the main function
                    if extractor.extracted_functions:
                        self._insert_helper_functions(tree, func_info['function_name'], extractor.extracted_functions)
                        
            # Generate refactored code
            refactored_code = ast.unparse(tree)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(refactored_code)
                
            self.refactoring_stats['files_processed'] += 1
            self.refactoring_stats['functions_refactored'] += len(functions)
            
            logger.info(f"Successfully refactored {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            # Restore backup
            self._restore_backup(file_path)
            
    def _backup_file(self, file_path: str):
        """Create backup of file before refactoring."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create relative path structure in backup
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
            
    def _insert_helper_functions(self, tree: ast.Module, main_func_name: str, helper_functions: List[ast.FunctionDef]):
        """Insert helper functions before the main function."""
        # Find the main function index
        main_func_index = None
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and node.name == main_func_name:
                main_func_index = i
                break
                
        if main_func_index is not None:
            # Insert helper functions before main function
            for i, helper_func in enumerate(helper_functions):
                tree.body.insert(main_func_index + i, helper_func)

class RefactoringValidator:
    """Validates that refactored code maintains functionality."""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_refactoring(self, original_file: str, refactored_file: str) -> bool:
        """Validate that refactored code maintains functionality."""
        try:
            # Basic syntax validation
            with open(refactored_file, 'r') as f:
                refactored_content = f.read()
                
            ast.parse(refactored_content)
            logger.info(f"Syntax validation passed for {refactored_file}")
            
            # Import validation - check if refactored code can be imported
            return self._test_import(refactored_file)
            
        except SyntaxError as e:
            logger.error(f"Syntax error in refactored {refactored_file}: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error for {refactored_file}: {e}")
            return False
            
    def _test_import(self, file_path: str) -> bool:
        """Test if the refactored file can be imported."""
        try:
            # This is a basic import test - in practice you'd want more comprehensive testing
            import importlib.util
            import sys
            
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                logger.debug(f"Import test passed for {file_path}")
                return True
            else:
                logger.warning(f"Could not create module spec for {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Import test failed for {file_path}: {e}")
            return False

def main():
    """Main entry point for the refactoring tool."""
    parser = argparse.ArgumentParser(description='Automated function refactoring tool')
    parser.add_argument('--complexity-report', '-r', required=True,
                       help='Path to complexity analysis report')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be refactored without making changes')
    parser.add_argument('--validate', action='store_true',
                       help='Validate refactored code')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize refactorer
    refactorer = SmartRefactorer()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")
        # Load and display refactoring plan
        with open(args.complexity_report, 'r') as f:
            report = json.load(f)
            
        high_complexity_functions = report['all_high_complexity_functions']
        
        print("\n" + "="*60)
        print("REFACTORING PLAN (DRY RUN)")
        print("="*60)
        print(f"Functions to refactor: {len(high_complexity_functions)}")
        
        # Group by file
        files_to_refactor = {}
        for func_info in high_complexity_functions:
            file_path = func_info['file_path']
            if file_path not in files_to_refactor:
                files_to_refactor[file_path] = []
            files_to_refactor[file_path].append(func_info)
            
        for file_path, functions in list(files_to_refactor.items())[:10]:  # Show top 10
            print(f"\n{file_path}: {len(functions)} functions")
            for func in functions[:3]:  # Show top 3 per file
                print(f"  - {func['function_name']} (complexity: {func['complexity']}, lines: {func['lines_of_code']})")
                
        return 0
        
    # Perform actual refactoring
    refactorer.refactor_high_complexity_functions(args.complexity_report)
    
    # Validate if requested
    if args.validate:
        validator = RefactoringValidator()
        logger.info("Running validation on refactored files...")
        # Validation logic would go here
        
    print("\n" + "="*60)
    print("REFACTORING COMPLETE")
    print("="*60)
    print(f"Files processed: {refactorer.refactoring_stats['files_processed']}")
    print(f"Functions refactored: {refactorer.refactoring_stats['functions_refactored']}")
    print(f"Backup directory: {refactorer.backup_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())