#!/usr/bin/env python3
"""
Automated Function Refactoring Tool
Breaks down high-complexity functions into smaller, maintainable units.
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import textwrap
import argparse
from collections import defaultdict


@dataclass
class RefactoredFunction:
    """Information about a refactored function."""
    original_name: str
    new_functions: List[str]
    main_function: str
    complexity_reduction: int


class FunctionExtractor(ast.NodeVisitor):
    """Extract logical blocks that can become separate functions."""
    
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.lines = source_code.splitlines()
        self.extractable_blocks = []
        self.variables_used = set()
        self.variables_defined = set()
        
    def extract_blocks(self, func_node: ast.FunctionDef) -> List[Dict]:
        """Extract refactorable blocks from a function."""
        blocks = []
        
        # Look for logical blocks
        for i, stmt in enumerate(func_node.body):
            if self._is_extractable_block(stmt):
                block_info = self._analyze_block(stmt, func_node, i)
                if block_info:
                    blocks.append(block_info)
                    
        return blocks
        
    def _is_extractable_block(self, node: ast.AST) -> bool:
        """Check if a node represents an extractable block."""
        # Large if blocks
        if isinstance(node, ast.If) and len(node.body) > 5:
            return True
            
        # Try-except blocks with substantial logic
        if isinstance(node, ast.Try) and len(node.body) > 3:
            return True
            
        # For/while loops with substantial bodies
        if isinstance(node, (ast.For, ast.While)) and len(node.body) > 4:
            return True
            
        # Complex with statements
        if isinstance(node, ast.With) and len(node.body) > 3:
            return True
            
        return False
        
    def _analyze_block(self, node: ast.AST, parent_func: ast.FunctionDef, index: int) -> Optional[Dict]:
        """Analyze a block to determine if it can be extracted."""
        # Get the variables used and defined in this block
        block_vars = VariableAnalyzer()
        block_vars.visit(node)
        
        # Get variables from the rest of the function
        rest_vars = VariableAnalyzer()
        for i, stmt in enumerate(parent_func.body):
            if i != index:
                rest_vars.visit(stmt)
                
        # Determine parameters needed
        used_vars = block_vars.used_vars - block_vars.defined_vars
        parameters = used_vars & (rest_vars.defined_vars | set(arg.arg for arg in parent_func.args.args))
        
        # Determine return values needed
        defined_vars = block_vars.defined_vars
        return_vars = defined_vars & rest_vars.used_vars
        
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', node.lineno) - 1
        
        return {
            'node': node,
            'start_line': start_line,
            'end_line': end_line,
            'parameters': list(parameters),
            'return_vars': list(return_vars),
            'block_type': type(node).__name__.lower(),
            'lines': self.lines[start_line:end_line + 1]
        }


class VariableAnalyzer(ast.NodeVisitor):
    """Analyze variable usage in AST nodes."""
    
    def __init__(self):
        self.used_vars = set()
        self.defined_vars = set()
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_vars.add(node.id)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        # Don't traverse into nested functions
        pass
        
    def visit_ClassDef(self, node):
        # Don't traverse into nested classes
        pass


class AutomaticRefactorer:
    """Main refactoring class."""
    
    def __init__(self):
        self.refactored_files = []
        self.backup_dir = Path("refactoring_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def refactor_file(self, file_path: str, function_name: str) -> Optional[RefactoredFunction]:
        """Refactor a specific function in a file."""
        path = Path(file_path)
        
        if not path.exists():
            print(f"File not found: {file_path}")
            return None
            
        # Create backup
        backup_path = self.backup_dir / f"{path.name}.backup"
        backup_path.write_text(path.read_text())
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            tree = ast.parse(source_code)
            
            # Find the target function
            target_function = None
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef) and 
                    node.name == function_name):
                    target_function = node
                    break
                    
            if not target_function:
                print(f"Function '{function_name}' not found in {file_path}")
                return None
                
            # Extract refactorable blocks
            extractor = FunctionExtractor(source_code)
            blocks = extractor.extract_blocks(target_function)
            
            if not blocks:
                print(f"No refactorable blocks found in {function_name}")
                return None
                
            # Generate refactored code
            refactored = self._generate_refactored_code(
                source_code, target_function, blocks
            )
            
            # Write refactored code
            with open(path, 'w', encoding='utf-8') as f:
                f.write(refactored.code)
                
            print(f"Refactored {function_name} in {file_path}")
            print(f"  Created {len(refactored.new_functions)} helper functions")
            print(f"  Reduced complexity by ~{refactored.complexity_reduction}")
            
            return refactored
            
        except Exception as e:
            print(f"Error refactoring {file_path}: {e}")
            # Restore backup
            if backup_path.exists():
                path.write_text(backup_path.read_text())
            return None
            
    def _generate_refactored_code(self, source_code: str, func_node: ast.FunctionDef, 
                                blocks: List[Dict]) -> RefactoredFunction:
        """Generate refactored code with extracted helper functions."""
        lines = source_code.splitlines()
        new_functions = []
        
        # Sort blocks by line number (reverse order for replacement)
        blocks.sort(key=lambda b: b['start_line'], reverse=True)
        
        # Generate helper functions
        for i, block in enumerate(blocks):
            helper_name = f"_{func_node.name}_{block['block_type']}_{i + 1}"
            helper_func = self._create_helper_function(
                helper_name, block
            )
            new_functions.append(helper_func)
            
            # Replace the block with a function call
            call_line = self._create_function_call(
                helper_name, block['parameters'], block['return_vars']
            )
            
            # Replace lines
            start, end = block['start_line'], block['end_line']
            del lines[start:end + 1]
            lines.insert(start, call_line)
            
        # Insert helper functions before the main function
        func_start = func_node.lineno - 1
        for helper in reversed(new_functions):
            lines.insert(func_start, "")
            for helper_line in reversed(helper.splitlines()):
                lines.insert(func_start, helper_line)
            lines.insert(func_start, "")
            
        refactored_code = "\n".join(lines)
        
        return RefactoredFunction(
            original_name=func_node.name,
            new_functions=[f.split('def ')[1].split('(')[0] for f in new_functions],
            main_function=func_node.name,
            code=refactored_code,
            complexity_reduction=len(blocks) * 5  # Rough estimate
        )
        
    def _create_helper_function(self, name: str, block: Dict) -> str:
        """Create a helper function from a block."""
        params = ", ".join(block['parameters'])
        return_vars = block['return_vars']
        
        # Function signature
        func_lines = [f"def {name}({params}):"]
        func_lines.append(f'    """Extracted helper function from complex function."""')
        
        # Function body
        for line in block['lines']:
            # Maintain indentation
            if line.strip():
                func_lines.append("    " + line.lstrip())
            else:
                func_lines.append("")
                
        # Return statement
        if return_vars:
            if len(return_vars) == 1:
                func_lines.append(f"    return {return_vars[0]}")
            else:
                func_lines.append(f"    return {', '.join(return_vars)}")
                
        return "\n".join(func_lines)
        
    def _create_function_call(self, name: str, parameters: List[str], 
                            return_vars: List[str]) -> str:
        """Create a function call to replace the extracted block."""
        params = ", ".join(parameters)
        
        if return_vars:
            if len(return_vars) == 1:
                return f"    {return_vars[0]} = {name}({params})"
            else:
                vars_str = ", ".join(return_vars)
                return f"    {vars_str} = {name}({params})"
        else:
            return f"    {name}({params})"
            
    def refactor_multiple_functions(self, complexity_report: Dict, 
                                  min_complexity: int = 30) -> List[RefactoredFunction]:
        """Refactor multiple functions based on complexity report."""
        results = []
        
        # Get high-complexity functions
        high_complexity = [
            func for func in complexity_report.get('high_complexity_functions', [])
            if func['complexity'] >= min_complexity
        ]
        
        print(f"Refactoring {len(high_complexity)} high-complexity functions...")
        
        for func_info in high_complexity:
            result = self.refactor_file(
                func_info['file'], 
                func_info['function']
            )
            if result:
                results.append(result)
                
        return results


def create_complexity_monitoring_tool():
    """Create a pre-commit hook to monitor complexity."""
    hook_content = '''#!/usr/bin/env python3
"""
Pre-commit hook to check cyclomatic complexity.
Prevents commits with functions exceeding complexity threshold.
"""

import sys
import json
from scripts.complexity_analyzer import ComplexityAnalyzer

def main():
    analyzer = ComplexityAnalyzer('.')
    analyzer.analyze_directory()
    
    high_complexity = analyzer.get_high_complexity_functions(15)
    
    if high_complexity:
        print("❌ COMMIT BLOCKED: High complexity functions found!")
        print("Functions with complexity > 15:")
        
        for func in high_complexity[:10]:  # Show top 10
            print(f"  - {func.function_name} ({func.file_path}:{func.line_number}) "
                  f"Complexity: {func.complexity}")
        
        if len(high_complexity) > 10:
            print(f"  ... and {len(high_complexity) - 10} more")
            
        print("\\nPlease refactor these functions before committing.")
        return 1
    
    print("✅ All functions meet complexity requirements")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    hook_path = Path(".git/hooks/pre-commit")
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)
    print("Created complexity monitoring pre-commit hook")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Automated function refactoring')
    parser.add_argument('--file', '-f', help='File to refactor')
    parser.add_argument('--function', help='Function to refactor')
    parser.add_argument('--report', '-r', help='Complexity report JSON file')
    parser.add_argument('--min-complexity', type=int, default=30,
                       help='Minimum complexity for batch refactoring')
    parser.add_argument('--create-monitor', action='store_true',
                       help='Create complexity monitoring pre-commit hook')
    
    args = parser.parse_args()
    
    refactorer = AutomaticRefactorer()
    
    if args.create_monitor:
        create_complexity_monitoring_tool()
        return
        
    if args.file and args.function:
        # Refactor single function
        result = refactorer.refactor_file(args.file, args.function)
        if result:
            print(f"Successfully refactored {args.function}")
    elif args.report:
        # Batch refactor from report
        with open(args.report) as f:
            report = json.load(f)
        results = refactorer.refactor_multiple_functions(report, args.min_complexity)
        print(f"Successfully refactored {len(results)} functions")
    else:
        print("Please specify --file and --function, or --report for batch refactoring")


if __name__ == "__main__":
    main()