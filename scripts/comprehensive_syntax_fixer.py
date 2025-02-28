#!/usr/bin/env python3
"""
Comprehensive Syntax Fixer for SutazAI Project

This script provides advanced syntax error detection and correction
across multiple Python files, addressing common syntax issues.
"""

import os
import re
import ast
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    class ComprehensiveSyntaxFixer:
    """
    A comprehensive syntax fixing utility for Python projects.
    """
    
        def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.errors_fixed = 0
        self.files_processed = 0
        
            def fix_syntax_errors(self, file_path: str) -> Optional[str]:
            """
            Attempt to fix syntax errors in a given Python file.
            """
                try:
                with open(file_path, 'r') as f:
                content = f.read()
                
                # Fix unindented except blocks
                content = re.sub(
                r'(except\s+\w+\s*:)\n(\s*pass)?$',
                r'\1\n    pass',
                content,
                flags=re.MULTILINE
            )
            
            # Ensure proper indentation
            lines = content.split('\n')
            fixed_lines = []
            indent_level = 0
            
                for line in lines:
                stripped = line.strip()
                
                # Adjust indentation for control structures
                    if stripped.startswith((
                    'def ', 'class ', 'if ', 'for ',
                    'while ', 'try:', 'except:', 'else:', 'elif '
                    )):
                    indent_level += 1
                    
                    # Reduce indent for closing blocks
                        if stripped in [')', ']', '}', 'pass', 'break', 'continue', 'return']:
                        indent_level = max(0, indent_level - 1)
                        
                        # Apply indentation
                        fixed_line = ' ' * (4 * indent_level) + stripped
                        fixed_lines.append(fixed_line)
                        
                        fixed_content = '\n'.join(fixed_lines)
                        
                        # Validate syntax
                            try:
                            ast.parse(fixed_content)
                            except SyntaxError as e:
                            logger.warning(f"Could not fully resolve syntax: {e}")
                            
                            # Write fixed content
                            with open(file_path, 'w') as f:
                            f.write(fixed_content)
                            
                            self.errors_fixed += 1
                            return fixed_content
                            
                            except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            return None
                            
                                def process_directory(self) -> Dict[str, List[str]]:
                                """
                                Process all Python files in the project directory.
                                """
                                results = {}
                                
                                    for root, _, files in os.walk(self.root_dir):
                                        for file in files:
                                            if file.endswith('.py'):
                                            file_path = os.path.join(root, file)
                                            self.files_processed += 1
                                            
                                            fixed_content = self.fix_syntax_errors(file_path)
                                                if fixed_content:
                                                results[file_path] = fixed_content.split('\n')
                                                
                                                return results
                                                
                                                    def generate_report(self, results: Dict[str, List[str]]) -> None:
                                                    """
                                                    Generate a comprehensive syntax fixing report.
                                                    """
                                                    report_path = os.path.join(
                                                    self.root_dir,
                                                    'logs',
                                                    'syntax_fix_report.md'
                                                )
                                                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                                                
                                                with open(report_path, 'w') as f:
                                                f.write("# Syntax Fixing Report\n\n")
                                                f.write(f"**Total Files Processed:** {self.files_processed}\n")
                                                f.write(f"**Errors Fixed:** {self.errors_fixed}\n\n")
                                                
                                                    for file_path, content in results.items():
                                                    f.write(f"## {file_path}\n")
                                                    f.write("```python\n")
                                                    f.write('\n'.join(content[:20]))  # First 20 lines
                                                    f.write("\n```\n\n")
                                                    
                                                        def main():
                                                        project_root = '/opt/sutazaiapp'
                                                        fixer = ComprehensiveSyntaxFixer(project_root)
                                                        results = fixer.process_directory()
                                                        fixer.generate_report(results)
                                                        
                                                            if __name__ == '__main__':
                                                            main()
                                                            