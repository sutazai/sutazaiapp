#!/usr/bin/env python3.11
"""
Python 3.11 Asyncio Compatibility Fixer for SutazAI Project

This script updates asyncio code to be compatible with Python 3.11,
addressing common issues like deprecated APIs and changed behaviors.
"""

import ast
import logging
import os
import re
import sys
from typing import List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("asyncio_compatibility_fixer")

# Common asyncio patterns that need to be updated for Python 3.11
ASYNCIO_PATTERNS = {
    # Deprecated asyncio.get_event_loop() usage
    r"asyncio\.get_event_loop\(\)": "get_event_loop_pattern",
    
    # loop.create_future() can be replaced with asyncio.create_task()
    r"(\w+)\.create_future\(\)": "create_future_pattern",
    
    # Explicit coroutine running patterns
    r"loop\.run_until_complete\(([^)]+)\)": "run_until_complete_pattern",
    
    # Task creation patterns
    r"asyncio\.ensure_future\(([^)]+)\)": "ensure_future_pattern",
}

def get_event_loop_pattern(match, line: str) -> str:
    """
    Update deprecated asyncio.get_event_loop() usage.
    
    Args:
        match: Regex match object
        line: The line of code
        
    Returns:
        Updated line of code
    """
    # In Python 3.11, prefer asyncio.get_running_loop() or asyncio.new_event_loop()
    if "def " in line or "async def " in line:
        # Don't change in function definitions
        return line
    
    # Check context to determine the right replacement
    if "run_until_complete" in line or "run_forever" in line:
        # This is likely in a main function, keep as new_event_loop
        return line.replace(match.group(0), "asyncio.new_event_loop()")
    else:
        # This is likely in an async function, use get_running_loop
        return line.replace(match.group(0), "asyncio.get_running_loop()")


def create_future_pattern(match, line: str) -> str:
    """
    Update loop.create_future() usage.
    
    Args:
        match: Regex match object
        line: The line of code
        
    Returns:
        Updated line of code
    """
    # Only replace if the variable looks like a loop
    var_name = match.group(1)
    if var_name == "loop" or var_name.endswith("_loop"):
        return line.replace(match.group(0), "asyncio.Future()")
    
    return line


def run_until_complete_pattern(match, line: str) -> str:
    """
    Update loop.run_until_complete() usage.
    
    Args:
        match: Regex match object
        line: The line of code
        
    Returns:
        Updated line of code
    """
    # In Python 3.11, prefer asyncio.run() for running coroutines
    # But we need to be careful about context
    coro = match.group(1).strip()
    
    # Only update simple cases
    if "(" in coro and ")" in coro and not ("await" in coro):
        return line.replace(match.group(0), f"asyncio.run({coro})")
    
    return line


def ensure_future_pattern(match, line: str) -> str:
    """
    Update asyncio.ensure_future() to asyncio.create_task().
    
    Args:
        match: Regex match object
        line: The line of code
        
    Returns:
        Updated line of code
    """
    # In Python 3.11, prefer asyncio.create_task() over ensure_future
    coro = match.group(1).strip()
    return line.replace(match.group(0), f"asyncio.create_task({coro})")


def process_file_ast(file_path: str) -> bool:
    """
    Use AST to find asyncio patterns and check for compatibility issues.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        True if file was processed, False if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        tree = ast.parse(code)
        visitor = AsyncioVisitor()
        visitor.visit(tree)
        
        if visitor.has_issues:
            logger.info(f"AST analysis found potential asyncio issues in {file_path}")
            return True
        
        return False
    except SyntaxError:
        logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
        return False
    except Exception as e:
        logger.error(f"Error processing {file_path} with AST: {str(e)}")
        return False


class AsyncioVisitor(ast.NodeVisitor):
    """Visit Python AST nodes to find asyncio patterns."""
    
    def __init__(self):
        self.has_issues = False
    
    def visit_Call(self, node):
        """Visit function call nodes."""
        # Check for asyncio.get_event_loop()
        if (isinstance(node.func, ast.Attribute) and 
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'asyncio' and
            node.func.attr == 'get_event_loop'):
            self.has_issues = True
        
        # Continue visiting children
        self.generic_visit(node)


def process_asyncio_issues(file_path: str) -> int:
    """
    Process and fix asyncio issues in a file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Number of fixes applied
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.splitlines()
        fixed_lines = []
        fixes_applied = 0
        
        # Process each line for asyncio issues
        for line in lines:
            original_line = line
            
            # Handle each pattern
            for pattern, handler_name in ASYNCIO_PATTERNS.items():
                for match in re.finditer(pattern, line):
                    # Get the handler function dynamically
                    handler = globals().get(handler_name)
                    if handler and callable(handler):
                        line = handler(match, line)
            
            # Add the line (possibly modified)
            fixed_lines.append(line)
            
            # Count fixes
            if line != original_line:
                fixes_applied += 1
        
        # Only write back if changes were made
        if fixes_applied > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines))
            
            logger.info(f"Applied {fixes_applied} asyncio fixes to {file_path}")
        
        return fixes_applied
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0


def main() -> None:
    """Main function to run the asyncio compatibility fixer."""
    project_path = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    # Get files with asyncio imports
    asyncio_files = []
    
    logger.info(f"Scanning {project_path} for files with asyncio imports")
    
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                
                # Skip virtual environment files
                if any(p in file_path for p in ["venv/", ".venv/", "__pycache__/"]):
                    continue
                
                # Check if the file imports asyncio
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(1000)  # Just check the beginning
                        if "import asyncio" in content or "from asyncio import" in content:
                            rel_path = os.path.relpath(file_path, project_path)
                            asyncio_files.append(rel_path)
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {str(e)}")
    
    # Process each file with asyncio imports
    total_files = len(asyncio_files)
    fixed_files = 0
    total_fixes = 0
    
    logger.info(f"Found {total_files} files with asyncio imports")
    
    for file_path in asyncio_files:
        full_path = os.path.join(project_path, file_path)
        
        logger.info(f"Processing {file_path}")
        
        # Process asyncio issues
        fixes = process_asyncio_issues(full_path)
        
        if fixes > 0:
            total_fixes += fixes
            fixed_files += 1
    
    logger.info(
        f"Applied {total_fixes} asyncio fixes to {fixed_files}/{total_files} files"
    )


if __name__ == "__main__":
    main() 