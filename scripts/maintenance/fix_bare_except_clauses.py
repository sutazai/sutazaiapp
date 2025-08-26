#!/usr/bin/env python3
"""
Fix all bare except clauses in the codebase.
Replaces bare except: statements with proper exception handling.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import ast
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BareExceptFixer:
    """Automatically fix bare except clauses in Python files."""
    
    def __init__(self):
        self.fixed_count = 0
        self.error_count = 0
        self.files_processed = 0
        self.files_fixed = 0
        self.fix_patterns = {
            # Pattern: (regex to match, replacement template)
            r'(\s*)except:\s*\n(\s*)pass': (
                r'\1except Exception as e:\n'
                r'\2logger.debug(f"Suppressed exception: {e}")\n'
                r'\2pass'
            ),
            r'(\s*)except:\s*\n(\s*)continue': (
                r'\1except Exception as e:\n'
                r'\2logger.debug(f"Continuing after exception: {e}")\n'
                r'\2continue'
            ),
            r'(\s*)except:\s*\n(\s*)return\s+None': (
                r'\1except Exception as e:\n'
                r'\2logger.warning(f"Returning None due to exception: {e}")\n'
                r'\2return None'
            ),
            r'(\s*)except:\s*\n(\s*)return\s+(\S+)': (
                r'\1except Exception as e:\n'
                r'\2logger.warning(f"Returning default value due to exception: {e}")\n'
                r'\2return \3'
            ),
        }
        
    def analyze_bare_except(self, content: str, filepath: str) -> List[Dict]:
        """Analyze bare except clauses and determine appropriate fix."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            class BareExceptVisitor(ast.NodeVisitor):
                def visit_ExceptHandler(self, node):
                    if node.type is None:  # Bare except
                        line_num = node.lineno
                        
                        # Analyze what the except block does
                        body_analysis = self._analyze_except_body(node.body)
                        
                        issues.append({
                            'line': line_num,
                            'type': body_analysis['type'],
                            'context': body_analysis.get('context', 'general'),
                            'suggested_exceptions': body_analysis.get('exceptions', ['Exception'])
                        })
                    
                    self.generic_visit(node)
                
                def _analyze_except_body(self, body):
                    """Analyze what the except block does to suggest appropriate handling."""
                    if not body:
                        return {'type': 'empty', 'exceptions': ['Exception']}
                    
                    # Check for pass statement
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        return {'type': 'suppress', 'exceptions': ['Exception']}
                    
                    # Check for continue statement
                    if len(body) == 1 and isinstance(body[0], ast.Continue):
                        return {'type': 'continue_loop', 'exceptions': ['Exception']}
                    
                    # Check for return statement
                    if any(isinstance(stmt, ast.Return) for stmt in body):
                        return {'type': 'return_value', 'exceptions': ['Exception']}
                    
                    # Check for logging/print statements
                    has_logging = any(
                        isinstance(stmt, ast.Expr) and 
                        isinstance(stmt.value, ast.Call) and
                        hasattr(stmt.value.func, 'attr') and
                        stmt.value.func.attr in ['error', 'warning', 'info', 'debug', 'exception']
                        for stmt in body
                    )
                    
                    if has_logging:
                        return {'type': 'logged', 'exceptions': ['Exception']}
                    
                    # Default case
                    return {'type': 'complex', 'exceptions': ['Exception']}
            
            visitor = BareExceptVisitor()
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            logger.warning(f"Error analyzing {filepath}: {e}")
        
        return issues
    
    def fix_file(self, filepath: str) -> bool:
        """Fix bare except clauses in a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip if no bare except found
            if 'except:' not in original_content:
                return False
            
            content = original_content
            file_fixed = False
            
            # First, analyze the file to understand context
            issues = self.analyze_bare_except(content, filepath)
            
            # Apply fixes line by line
            lines = content.split('\n')
            new_lines = []
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                # Check if this is a bare except line
                if re.match(r'^\s*except:\s*(?:#.*)?$', line):
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    # Look at the next line to determine the pattern
                    next_line = lines[i + 1] if i + 1 < len(lines) else ''
                    next_line_stripped = next_line.strip()
                    
                    # Determine the appropriate exception type and handling
                    if filepath.endswith('test_') or 'test/' in filepath or '/tests/' in filepath:
                        # Test files - use AssertionError and Exception
                        exception_types = '(AssertionError, Exception)'
                    elif 'api' in filepath or 'endpoint' in filepath:
                        # API files - catch common web exceptions
                        exception_types = '(ValueError, TypeError, KeyError, AttributeError)'
                    elif 'database' in filepath or 'db' in filepath:
                        # Database files
                        exception_types = '(ConnectionError, TimeoutError, Exception)'
                    elif 'network' in filepath or 'http' in filepath or 'request' in filepath:
                        # Network-related files
                        exception_types = '(ConnectionError, TimeoutError, OSError)'
                    elif 'file' in filepath or 'io' in filepath:
                        # File I/O operations
                        exception_types = '(IOError, OSError, FileNotFoundError)'
                    else:
                        # Default case
                        exception_types = 'Exception'
                    
                    # Check what the except block does
                    if next_line_stripped == 'pass':
                        # Suppress pattern
                        new_lines.append(f'{indent_str}except {exception_types} as e:')
                        new_lines.append(f'{indent_str}    # Suppressed exception (was bare except)')
                        new_lines.append(f'{indent_str}    logger.debug(f"Suppressed exception: {{e}}")')
                        new_lines.append(next_line)
                        i += 2
                        file_fixed = True
                        self.fixed_count += 1
                        
                    elif next_line_stripped == 'continue':
                        # Continue pattern
                        new_lines.append(f'{indent_str}except {exception_types} as e:')
                        new_lines.append(f'{indent_str}    logger.debug(f"Continuing after exception: {{e}}")')
                        new_lines.append(next_line)
                        i += 2
                        file_fixed = True
                        self.fixed_count += 1
                        
                    elif next_line_stripped.startswith('return'):
                        # Return pattern
                        new_lines.append(f'{indent_str}except {exception_types} as e:')
                        new_lines.append(f'{indent_str}    logger.warning(f"Exception caught, returning: {{e}}")')
                        new_lines.append(next_line)
                        i += 2
                        file_fixed = True
                        self.fixed_count += 1
                        
                    elif next_line_stripped.startswith('raise'):
                        # Re-raise pattern
                        new_lines.append(f'{indent_str}except {exception_types} as e:')
                        new_lines.append(f'{indent_str}    logger.error(f"Re-raising exception: {{e}}")')
                        new_lines.append(next_line)
                        i += 2
                        file_fixed = True
                        self.fixed_count += 1
                        
                    else:
                        # General case - add proper exception handling
                        new_lines.append(f'{indent_str}except {exception_types} as e:')
                        new_lines.append(f'{indent_str}    logger.error(f"Unexpected exception: {{e}}", exc_info=True)')
                        i += 1
                        file_fixed = True
                        self.fixed_count += 1
                else:
                    new_lines.append(line)
                    i += 1
            
            if file_fixed:
                # Add logger import if not present
                content = '\n'.join(new_lines)
                if 'import logging' not in content and 'from logging import' not in content:
                    # Add logging import at the top
                    import_lines = []
                    content_lines = content.split('\n')
                    added_import = False
                    
                    for line in content_lines:
                        if not added_import and (line.startswith('import ') or line.startswith('from ')):
                            import_lines.append('import logging')
                            import_lines.append('')
                            import_lines.append('# Configure logger for exception handling')
                            import_lines.append('logger = logging.getLogger(__name__)')
                            import_lines.append('')
                            added_import = True
                        import_lines.append(line)
                    
                    if not added_import:
                        # No imports found, add at the beginning after docstring
                        final_lines = []
                        in_docstring = False
                        docstring_done = False
                        
                        for line in content_lines:
                            final_lines.append(line)
                            if not docstring_done:
                                if '"""' in line or "'''" in line:
                                    if not in_docstring:
                                        in_docstring = True
                                    else:
                                        docstring_done = True
                                        final_lines.append('')
                                        final_lines.append('import logging')
                                        final_lines.append('')
                                        final_lines.append('# Configure logger for exception handling')
                                        final_lines.append('logger = logging.getLogger(__name__)')
                                        final_lines.append('')
                        
                        content = '\n'.join(final_lines)
                    else:
                        content = '\n'.join(import_lines)
                
                # Write the fixed content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.files_fixed += 1
                logger.info(f"Fixed {filepath}")
                return True
                
        except Exception as e:
            logger.error(f"Error fixing {filepath}: {e}")
            self.error_count += 1
            return False
        
        return False
    
    def fix_all_files(self, root_dir: str = '/opt/sutazaiapp'):
        """Fix bare except clauses in all Python files."""
        logger.info(f"Starting bare except clause fixes in {root_dir}")
        
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(root_dir):
            # Skip virtual environments and cache directories
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.venv', 'venv', 'env', 
                '.git', 'node_modules', '.pytest_cache'
            ]]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(python_files)} Python files to process")
        
        # Process each file
        for filepath in python_files:
            self.files_processed += 1
            if self.files_processed % 100 == 0:
                logger.info(f"Progress: {self.files_processed}/{len(python_files)} files processed")
            
            self.fix_file(filepath)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("BARE EXCEPT CLAUSE FIX SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.files_processed}")
        logger.info(f"Files fixed: {self.files_fixed}")
        logger.info(f"Bare except clauses fixed: {self.fixed_count}")
        logger.info(f"Errors encountered: {self.error_count}")
        logger.info("=" * 60)
        
        return self.fixed_count > 0

def main():
    """Main function to fix all bare except clauses."""
    fixer = BareExceptFixer()
    
    # Fix all Python files
    success = fixer.fix_all_files()
    
    if success:
        logger.info("✅ Successfully fixed bare except clauses")
        
        # Run a verification check
        logger.info("\nVerifying fixes...")
        remaining = 0
        for root, dirs, files in os.walk('/opt/sutazaiapp'):
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.venv', 'venv', 'env', 
                '.git', 'node_modules', '.pytest_cache'
            ]]
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if re.search(r'^\s*except:\s*$', content, re.MULTILINE):
                            remaining += 1
                            logger.warning(f"Still has bare except: {filepath}")
                    except Exception as e:
                        logger.debug(f"Could not verify {filepath}: {e}")
        
        if remaining > 0:
            logger.warning(f"⚠️  {remaining} files still have bare except clauses")
        else:
            logger.info("✅ All bare except clauses have been fixed!")
    else:
        logger.warning("No bare except clauses were fixed")
    
    return 0 if success else 1

if __name__ == "__main__":
