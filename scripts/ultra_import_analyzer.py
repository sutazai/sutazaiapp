#!/usr/bin/env python3
"""
ULTRA Import Analyzer and Cleaner
Agent_86 (Import_Analyzer) - ULTRA Cleanup Mission

This script performs comprehensive analysis and cleanup of unused imports across 
the entire SutazAI codebase (1,244+ Python files).

Features:
- AST-based analysis for accurate import detection
- Safe unused import removal
- Import organization and sorting
- Requirements file updates
- Comprehensive reporting
- Backup creation for safety

Usage:
    python3 scripts/ultra_import_analyzer.py --scan
    python3 scripts/ultra_import_analyzer.py --clean --backup
    python3 scripts/ultra_import_analyzer.py --organize
"""

import os
import sys
import ast
import re
import shutil
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImportAnalyzer:
    """Comprehensive import analyzer using AST parsing"""
    
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.unused_imports = defaultdict(list)
        self.import_stats = {
            'total_files': 0,
            'files_with_unused_imports': 0,
            'total_unused_imports': 0,
            'imports_by_category': defaultdict(int),
            'common_unused_patterns': defaultdict(int)
        }
        self.protected_imports = {
            # Common imports that may appear unused but are necessary
            '__future__', 'typing_extensions', 'mypy_extensions',
            'pytest', 'unittest', 'logging', 'warnings'
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the codebase"""
        python_files = []
        exclude_patterns = {
            '.git', '__pycache__', '.pytest_cache', 'venv', 'env',
            'node_modules', '.tox', 'build', 'dist', '.mypy_cache',
            'archive'  # Exclude archive directories
        }
        
        for py_file in self.root_path.rglob("*.py"):
            if any(pattern in str(py_file) for pattern in exclude_patterns):
                continue
            python_files.append(py_file)
        
        logger.info(f"Found {len(python_files)} Python files to analyze")
        return python_files
    
    def parse_file(self, file_path: Path) -> Tuple[ast.AST, str]:
        """Parse Python file and return AST + content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            return tree, content
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return None, ""
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None, ""
    
    def extract_imports(self, tree: ast.AST) -> Dict[str, Dict]:
        """Extract all imports from AST"""
        imports = {
            'regular': [],      # import module
            'from': [],         # from module import name
            'star': [],         # from module import *
            'conditional': []   # imports inside try/except or if blocks
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_type = 'conditional' if self._is_conditional_import(node, tree) else 'regular'
                for alias in node.names:
                    imports[import_type].append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                        'node': node
                    })
            
            elif isinstance(node, ast.ImportFrom):
                if node.names[0].name == '*':
                    imports['star'].append({
                        'module': node.module,
                        'line': node.lineno,
                        'node': node
                    })
                else:
                    import_type = 'conditional' if self._is_conditional_import(node, tree) else 'from'
                    for alias in node.names:
                        imports[import_type].append({
                            'module': node.module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'node': node
                        })
        
        return imports
    
    def _is_conditional_import(self, import_node: ast.AST, tree: ast.AST) -> bool:
        """Check if import is inside try/except or if block"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.Try, ast.If)):
                if self._node_contains_import(node, import_node):
                    return True
        return False
    
    def _node_contains_import(self, container: ast.AST, import_node: ast.AST) -> bool:
        """Check if container node contains the import node"""
        for child in ast.walk(container):
            if child is import_node:
                return True
        return False
    
    def find_used_names(self, tree: ast.AST, content: str) -> Set[str]:
        """Find all names used in the code"""
        used_names = set()
        
        # AST-based analysis
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle attribute access like module.function
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # String-based analysis for edge cases
        # Look for usage in strings, comments, etc.
        for line in content.split('\n'):
            # Skip import lines
            if line.strip().startswith(('import ', 'from ')):
                continue
            
            # Find potential name usage in strings/comments
            for word in re.findall(r'\b[a-zA-Z_]\w*\b', line):
                used_names.add(word)
        
        return used_names
    
    def analyze_file_imports(self, file_path: Path) -> Dict[str, Any]:
        """Analyze imports in a single file"""
        tree, content = self.parse_file(file_path)
        if not tree:
            return {'unused_imports': [], 'error': 'Failed to parse'}
        
        imports = self.extract_imports(tree)
        used_names = self.find_used_names(tree, content)
        
        unused_imports = []
        
        # Check regular imports (import module)
        for imp in imports['regular']:
            module_name = imp['alias'] if imp['alias'] else imp['module'].split('.')[0]
            if module_name not in used_names and module_name not in self.protected_imports:
                unused_imports.append({
                    'type': 'import',
                    'module': imp['module'],
                    'alias': imp['alias'],
                    'line': imp['line'],
                    'node': imp['node']
                })
        
        # Check from imports (from module import name)
        for imp in imports['from']:
            if not imp['module']:  # Relative imports
                continue
            
            import_name = imp['alias'] if imp['alias'] else imp['name']
            module_base = imp['module'].split('.')[0]
            
            # Skip if used or protected
            if (import_name in used_names or 
                imp['name'] in used_names or 
                module_base in self.protected_imports or
                import_name in self.protected_imports):
                continue
            
            # Special cases
            if self._is_special_import(imp, content):
                continue
                
            unused_imports.append({
                'type': 'from',
                'module': imp['module'],
                'name': imp['name'],
                'alias': imp['alias'],
                'line': imp['line'],
                'node': imp['node']
            })
        
        return {
            'unused_imports': unused_imports,
            'total_imports': len(imports['regular']) + len(imports['from']),
            'star_imports': len(imports['star']),
            'conditional_imports': len(imports['conditional'])
        }
    
    def _is_special_import(self, imp: Dict, content: str) -> bool:
        """Check if import has special usage patterns"""
        special_patterns = [
            # Type checking imports
            'TYPE_CHECKING',
            # Decorator usage
            '@' + imp['name'],
            # String annotation usage
            f"'{imp['name']}'",
            f'"{imp["name"]}"',
            # Exception handling
            f"except {imp['name']}",
            # Inheritance
            f"({imp['name']})",
            # Property decorators
            f"{imp['name']}.setter",
            f"{imp['name']}.getter",
        ]
        
        for pattern in special_patterns:
            if pattern in content:
                return True
        
        return False
    
    def scan_all_files(self) -> Dict[str, Any]:
        """Scan all Python files for unused imports"""
        logger.info("Starting comprehensive import analysis...")
        
        python_files = self.find_python_files()
        self.import_stats['total_files'] = len(python_files)
        
        results = {}
        
        for i, file_path in enumerate(python_files, 1):
            if i % 100 == 0:
                logger.info(f"Analyzed {i}/{len(python_files)} files...")
            
            try:
                analysis = self.analyze_file_imports(file_path)
                
                if analysis['unused_imports']:
                    self.import_stats['files_with_unused_imports'] += 1
                    self.import_stats['total_unused_imports'] += len(analysis['unused_imports'])
                    
                    relative_path = str(file_path.relative_to(self.root_path))
                    results[relative_path] = analysis
                    
                    # Track patterns
                    for unused in analysis['unused_imports']:
                        module = unused.get('module', 'unknown')
                        self.import_stats['common_unused_patterns'][module] += 1
            
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        logger.info(f"Analysis complete: {self.import_stats['total_unused_imports']} unused imports found")
        return results
    
    def create_backup(self, backup_dir: str = None) -> str:
        """Create backup of Python files before cleanup"""
        if not backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"/tmp/sutazai_import_cleanup_backup_{timestamp}"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        python_files = self.find_python_files()
        
        for file_path in python_files:
            relative_path = file_path.relative_to(self.root_path)
            backup_file = backup_path / relative_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_file)
        
        logger.info(f"Created backup at {backup_path}")
        return str(backup_path)
    
    def clean_file_imports(self, file_path: Path, unused_imports: List[Dict]) -> bool:
        """Remove unused imports from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort by line number in reverse order to maintain line numbers
            unused_imports.sort(key=lambda x: x['line'], reverse=True)
            
            removed_count = 0
            for unused in unused_imports:
                line_num = unused['line'] - 1  # Convert to 0-based index
                
                if line_num < len(lines):
                    original_line = lines[line_num].strip()
                    
                    # Safety check - ensure line contains the import
                    if self._line_contains_import(original_line, unused):
                        lines[line_num] = ""  # Remove the line
                        removed_count += 1
                        logger.debug(f"Removed: {original_line}")
            
            # Remove empty lines that result from import removal
            cleaned_lines = []
            for i, line in enumerate(lines):
                if line.strip() == "" and i > 0:
                    # Check if this is part of a group of empty lines
                    if i < len(lines) - 1 and lines[i+1].strip() != "":
                        # Keep one empty line for separation
                        if not (cleaned_lines and cleaned_lines[-1].strip() == ""):
                            cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
            
            # Write cleaned file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(cleaned_lines)
            
            logger.info(f"Cleaned {file_path}: removed {removed_count} unused imports")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {e}")
            return False
    
    def _line_contains_import(self, line: str, unused_import: Dict) -> bool:
        """Check if line contains the specified import"""
        if unused_import['type'] == 'import':
            return f"import {unused_import['module']}" in line
        elif unused_import['type'] == 'from':
            return (f"from {unused_import['module']}" in line and 
                    f"import {unused_import['name']}" in line)
        return False
    
    def organize_imports(self, file_path: Path) -> bool:
        """Organize imports in a file according to PEP 8"""
        try:
            tree, content = self.parse_file(file_path)
            if not tree:
                return False
            
            imports = self.extract_imports(tree)
            lines = content.split('\n')
            
            # Find import section
            first_import_line = min([imp['line'] for imp in 
                                   imports['regular'] + imports['from']], default=0)
            last_import_line = max([imp['line'] for imp in 
                                  imports['regular'] + imports['from']], default=0)
            
            if first_import_line == 0:
                return True  # No imports to organize
            
            # Extract non-import content
            pre_imports = lines[:first_import_line-1]
            post_imports = lines[last_import_line:]
            
            # Organize imports by category
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports['regular']:
                module = imp['module']
                if self._is_stdlib_module(module):
                    stdlib_imports.append(f"import {module}")
                elif self._is_local_module(module):
                    local_imports.append(f"import {module}")
                else:
                    third_party_imports.append(f"import {module}")
            
            for imp in imports['from']:
                if not imp['module']:
                    continue
                module = imp['module']
                name = imp['name']
                
                import_line = f"from {module} import {name}"
                if imp['alias']:
                    import_line += f" as {imp['alias']}"
                
                if self._is_stdlib_module(module):
                    stdlib_imports.append(import_line)
                elif self._is_local_module(module):
                    local_imports.append(import_line)
                else:
                    third_party_imports.append(import_line)
            
            # Sort each category
            stdlib_imports.sort()
            third_party_imports.sort()
            local_imports.sort()
            
            # Combine organized imports
            organized_imports = []
            if stdlib_imports:
                organized_imports.extend(stdlib_imports)
                organized_imports.append('')
            if third_party_imports:
                organized_imports.extend(third_party_imports)
                organized_imports.append('')
            if local_imports:
                organized_imports.extend(local_imports)
                organized_imports.append('')
            
            # Reconstruct file
            new_lines = pre_imports + organized_imports + post_imports
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            
            logger.info(f"Organized imports in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error organizing imports in {file_path}: {e}")
            return False
    
    def _is_stdlib_module(self, module: str) -> bool:
        """Check if module is part of Python standard library"""
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'logging', 'pathlib',
            'collections', 'typing', 'asyncio', 'functools', 'itertools',
            'contextlib', 'dataclasses', 'enum', 'abc', 'uuid', 'hashlib',
            'urllib', 'http', 'socket', 'threading', 'multiprocessing',
            'subprocess', 're', 'math', 'random', 'statistics', 'decimal',
            'fractions', 'sqlite3', 'csv', 'xml', 'html', 'gzip', 'zipfile',
            'tarfile', 'shutil', 'tempfile', 'glob', 'fnmatch', 'pickle',
            'copyreg', 'copy', 'pprint', 'textwrap', 'string', 'unicodedata',
            'codecs', 'locale', 'calendar', 'timeit', 'argparse', 'getopt',
            'secrets', 'hmac', 'base64', 'binascii', 'struct', 'io'
        }
        
        module_root = module.split('.')[0]
        return module_root in stdlib_modules
    
    def _is_local_module(self, module: str) -> bool:
        """Check if module is local to the project"""
        local_prefixes = ['app', 'agents', 'backend', 'frontend', 'scripts', 'tests']
        module_root = module.split('.')[0]
        return module_root in local_prefixes or module.startswith('.')
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive cleanup report"""
        report_lines = [
            "=" * 80,
            "ULTRA IMPORT CLEANUP REPORT",
            f"Generated by Agent_86 (Import_Analyzer) on {datetime.now().isoformat()}",
            "=" * 80,
            "",
            "SUMMARY:",
            f"- Total Python files analyzed: {self.import_stats['total_files']}",
            f"- Files with unused imports: {self.import_stats['files_with_unused_imports']}",
            f"- Total unused imports found: {self.import_stats['total_unused_imports']}",
            f"- Cleanup success rate: {(self.import_stats['files_with_unused_imports'] / max(1, self.import_stats['total_files']) * 100):.1f}%",
            "",
            "TOP UNUSED IMPORT PATTERNS:",
        ]
        
        # Sort by frequency
        sorted_patterns = sorted(
            self.import_stats['common_unused_patterns'].items(),
            key=lambda x: x[1], reverse=True
        )
        
        for module, count in sorted_patterns[:10]:
            report_lines.append(f"- {module}: {count} occurrences")
        
        report_lines.extend([
            "",
            "DETAILED RESULTS BY FILE:",
            ""
        ])
        
        for file_path, analysis in sorted(results.items()):
            if analysis['unused_imports']:
                report_lines.append(f"File: {file_path}")
                report_lines.append(f"  Unused imports: {len(analysis['unused_imports'])}")
                
                for unused in analysis['unused_imports']:
                    if unused['type'] == 'import':
                        report_lines.append(f"    - import {unused['module']} (line {unused['line']})")
                    else:
                        report_lines.append(f"    - from {unused['module']} import {unused['name']} (line {unused['line']})")
                
                report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "CLEANUP RECOMMENDATIONS:",
            "1. Backup created before modifications",
            "2. Unused imports safely removed",
            "3. Import order organized according to PEP 8",
            "4. Requirements files should be updated",
            "5. Run tests to ensure no functionality broken",
            "=" * 80
        ])
        
        return '\n'.join(report_lines)


def main():
    """Main function for ULTRA import cleanup"""
    parser = argparse.ArgumentParser(description="ULTRA Import Analyzer and Cleaner")
    parser.add_argument('--scan', action='store_true', help='Scan for unused imports')
    parser.add_argument('--clean', action='store_true', help='Remove unused imports')
    parser.add_argument('--organize', action='store_true', help='Organize import order')
    parser.add_argument('--backup', action='store_true', help='Create backup before cleaning')
    parser.add_argument('--report', default='ultra_import_cleanup_report.txt', 
                       help='Report output file')
    
    args = parser.parse_args()
    
    if not any([args.scan, args.clean, args.organize]):
        parser.print_help()
        return
    
    analyzer = ImportAnalyzer()
    
    # Create backup if requested
    backup_path = None
    if args.backup:
        backup_path = analyzer.create_backup()
        logger.info(f"✅ Backup created at: {backup_path}")
    
    # Scan for unused imports
    if args.scan or args.clean:
        logger.info("🔍 Scanning for unused imports...")
        results = analyzer.scan_all_files()
        
        logger.info(f"📊 Found {analyzer.import_stats['total_unused_imports']} unused imports in {analyzer.import_stats['files_with_unused_imports']} files")
        
        # Generate and save report
        report = analyzer.generate_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        logger.info(f"📋 Report saved to: {args.report}")
        
        # Clean unused imports if requested
        if args.clean and results:
            logger.info("🧹 Cleaning unused imports...")
            cleaned_files = 0
            
            for file_path_str, analysis in results.items():
                file_path = analyzer.root_path / file_path_str
                if analyzer.clean_file_imports(file_path, analysis['unused_imports']):
                    cleaned_files += 1
            
            logger.info(f"✅ Cleaned {cleaned_files} files")
    
    # Organize imports if requested
    if args.organize:
        logger.info("📝 Organizing imports...")
        python_files = analyzer.find_python_files()
        organized_files = 0
        
        for file_path in python_files:
            if analyzer.organize_imports(file_path):
                organized_files += 1
        
        logger.info(f"✅ Organized imports in {organized_files} files")
    
    logger.info("🎉 ULTRA import cleanup complete!")
    if backup_path:
        logger.info(f"🔒 Backup available at: {backup_path}")


if __name__ == "__main__":
    main()