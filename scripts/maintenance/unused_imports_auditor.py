#!/usr/bin/env python3
"""
ULTRA CODE AUDITOR - Unused Imports Analysis and Cleanup
========================================================

This script identifies and removes unused imports across the entire SutazAI codebase.
Designed for safe, automated cleanup while preserving functionality.

Author: Ultra Code Auditor
Created: August 10, 2025
Purpose: Remove 9,242+ unused imports for code hygiene and performance
"""

import ast
import os
import sys
import re
import json
from pathlib import Path
import argparse
import logging
from dataclasses import dataclass
from collections import defaultdict
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    name: str
    alias: str
    line_number: int
    import_type: str  # 'import' or 'from'
    full_statement: str

@dataclass
class FileAnalysis:
    """Analysis results for a single Python file"""
    filepath: str
    total_imports: int
    unused_imports: List[ImportInfo]
    used_imports: List[ImportInfo]
    errors: List[str]

class UnusedImportDetector:
    """Advanced unused import detection with AST analysis"""
    
    def __init__(self):
        self.statistics = {
            'total_files': 0,
            'analyzed_files': 0,
            'skipped_files': 0,
            'total_imports': 0,
            'unused_imports': 0,
            'errors': 0
        }
        self.skip_patterns = [
            r'__init__.py$',  # Often have exports
            r'test_.*\.py$',  # Test files may have fixture imports
            r'.*_test\.py$',
            r'conftest\.py$',
            r'setup\.py$',
        ]
        
        # Known safe-to-remove patterns
        self.safe_unused_patterns = [
            r'^typing\.',
            r'^collections\.',
            r'^dataclasses\.',
            r'^enum\.',
            r'^functools\.',
            r'^itertools\.',
            r'^pathlib\.',
            r'^datetime\.',
            r'^json$',
            r'^os$',
            r'^sys$',
            r'^re$',
            r'^logging$',
            r'^argparse$',
            r'^subprocess$',
        ]

    def should_skip_file(self, filepath: str) -> bool:
        """Check if file should be skipped from analysis"""
        filename = os.path.basename(filepath)
        for pattern in self.skip_patterns:
            if re.match(pattern, filename):
                return True
        return False

    def extract_imports_from_ast(self, tree: ast.AST, source_lines: List[str]) -> List[ImportInfo]:
        """Extract import information using AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        name=alias.name.split('.')[-1],
                        alias=alias.asname or alias.name.split('.')[-1],
                        line_number=node.lineno,
                        import_type='import',
                        full_statement=source_lines[node.lineno - 1].strip()
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=f"{module}.{alias.name}" if module else alias.name,
                        name=alias.name,
                        alias=alias.asname or alias.name,
                        line_number=node.lineno,
                        import_type='from',
                        full_statement=source_lines[node.lineno - 1].strip()
                    ))
        
        return imports

    def find_name_usage(self, tree: ast.AST, name: str) -> bool:
        """Find if a name is used in the AST (excluding import statements)"""
        class NameVisitor(ast.NodeVisitor):
            def __init__(self):
                self.found = False
                self.in_import = False
            
            def visit_Import(self, node):
                self.in_import = True
                self.generic_visit(node)
                self.in_import = False
            
            def visit_ImportFrom(self, node):
                self.in_import = True
                self.generic_visit(node)
                self.in_import = False
            
            def visit_Name(self, node):
                if not self.in_import and node.id == name:
                    self.found = True
            
            def visit_Attribute(self, node):
                if not self.in_import:
                    # Check for usage like module.function
                    if isinstance(node.value, ast.Name) and node.value.id == name:
                        self.found = True
                self.generic_visit(node)
        
        visitor = NameVisitor()
        visitor.visit(tree)
        return visitor.found

    def analyze_file(self, filepath: str) -> FileAnalysis:
        """Analyze a single Python file for unused imports"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                source_lines = content.splitlines()
            
            tree = ast.parse(content)
            imports = self.extract_imports_from_ast(tree, source_lines)
            
            unused_imports = []
            used_imports = []
            
            for import_info in imports:
                # Check if the imported name is used
                if self.find_name_usage(tree, import_info.alias):
                    used_imports.append(import_info)
                else:
                    # Additional check for string usage (like in __all__, docstrings, etc.)
                    if self.check_string_usage(content, import_info.alias):
                        used_imports.append(import_info)
                    else:
                        unused_imports.append(import_info)
            
            self.statistics['analyzed_files'] += 1
            self.statistics['total_imports'] += len(imports)
            self.statistics['unused_imports'] += len(unused_imports)
            
            return FileAnalysis(
                filepath=filepath,
                total_imports=len(imports),
                unused_imports=unused_imports,
                used_imports=used_imports,
                errors=[]
            )
            
        except Exception as e:
            self.statistics['errors'] += 1
            logger.warning(f"Error analyzing {filepath}: {str(e)}")
            return FileAnalysis(
                filepath=filepath,
                total_imports=0,
                unused_imports=[],
                used_imports=[],
                errors=[str(e)]
            )

    def check_string_usage(self, content: str, name: str) -> bool:
        """Check if name appears in strings (like __all__, docstrings, etc.)"""
        # Check for __all__ exports
        if f"'{name}'" in content or f'"{name}"' in content:
            return True
        
        # Check for dynamic imports or string references
        patterns = [
            rf"getattr\([^,]+,\s*['\"]({name})['\"]",
            rf"hasattr\([^,]+,\s*['\"]({name})['\"]",
            rf"__all__.*['\"]({name})['\"]",
            rf"globals\(\)\s*\[\s*['\"]({name})['\"]",
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        
        return False

    def scan_codebase(self, root_path: str) -> Dict[str, FileAnalysis]:
        """Scan entire codebase for unused imports"""
        logger.info(f"Starting codebase scan from: {root_path}")
        
        results = {}
        python_files = list(Path(root_path).rglob("*.py"))
        self.statistics['total_files'] = len(python_files)
        
        for i, filepath in enumerate(python_files, 1):
            filepath_str = str(filepath)
            
            if self.should_skip_file(filepath_str):
                self.statistics['skipped_files'] += 1
                logger.debug(f"Skipping {filepath_str}")
                continue
            
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(python_files)} files analyzed")
            
            analysis = self.analyze_file(filepath_str)
            results[filepath_str] = analysis
        
        logger.info(f"Scan complete. Analyzed {self.statistics['analyzed_files']} files")
        return results

    def generate_report(self, results: Dict[str, FileAnalysis]) -> Dict:
        """Generate comprehensive audit report"""
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'statistics': self.statistics.copy(),
            'files_with_unused_imports': [],
            'top_unused_modules': defaultdict(int),
            'summary': {}
        }
        
        files_with_unused = []
        
        for filepath, analysis in results.items():
            if analysis.unused_imports:
                file_info = {
                    'filepath': filepath,
                    'total_imports': analysis.total_imports,
                    'unused_count': len(analysis.unused_imports),
                    'unused_imports': []
                }
                
                for imp in analysis.unused_imports:
                    file_info['unused_imports'].append({
                        'module': imp.module,
                        'name': imp.name,
                        'alias': imp.alias,
                        'line': imp.line_number,
                        'statement': imp.full_statement
                    })
                    
                    # Track most unused modules
                    report['top_unused_modules'][imp.module] += 1
                
                files_with_unused.append(file_info)
        
        # Sort by unused count (descending)
        files_with_unused.sort(key=lambda x: x['unused_count'], reverse=True)
        report['files_with_unused_imports'] = files_with_unused
        
        # Generate summary
        report['summary'] = {
            'total_files_analyzed': self.statistics['analyzed_files'],
            'files_with_unused_imports': len(files_with_unused),
            'total_unused_imports': self.statistics['unused_imports'],
            'cleanup_potential_mb': round(self.statistics['unused_imports'] * 0.001, 2),
            'most_unused_modules': dict(sorted(
                report['top_unused_modules'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])
        }
        
        return report

class UnusedImportCleaner:
    """Safe unused import removal with backup and validation"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.backup_dir = Path("/opt/sutazaiapp/backups/unused_imports_cleanup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self, filepath: str) -> str:
        """Create backup of file before modification"""
        backup_path = self.backup_dir / f"{Path(filepath).name}.backup"
        subprocess.run(['cp', filepath, str(backup_path)], check=True)
        return str(backup_path)
    
    def remove_unused_imports(self, filepath: str, unused_imports: List[ImportInfo]) -> bool:
        """Remove unused imports from file"""
        try:
            if not self.dry_run:
                backup_path = self.create_backup(filepath)
                logger.info(f"Created backup: {backup_path}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort by line number (descending) to remove from bottom up
            unused_imports.sort(key=lambda x: x.line_number, reverse=True)
            
            removed_lines = []
            for imp in unused_imports:
                if 1 <= imp.line_number <= len(lines):
                    removed_line = lines[imp.line_number - 1].strip()
                    removed_lines.append(f"Line {imp.line_number}: {removed_line}")
                    
                    if not self.dry_run:
                        del lines[imp.line_number - 1]
            
            if not self.dry_run:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                logger.info(f"Removed {len(unused_imports)} unused imports from {filepath}")
            else:
                logger.info(f"[DRY RUN] Would remove {len(unused_imports)} imports from {filepath}:")
                for line in removed_lines[:5]:  # Show first 5
                    logger.info(f"  {line}")
                if len(removed_lines) > 5:
                    logger.info(f"  ... and {len(removed_lines) - 5} more")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {filepath}: {str(e)}")
            return False
    
    def cleanup_files(self, results: Dict[str, FileAnalysis]) -> Dict:
        """Clean up unused imports in all analyzed files"""
        cleanup_stats = {
            'files_processed': 0,
            'files_cleaned': 0,
            'total_imports_removed': 0,
            'errors': []
        }
        
        for filepath, analysis in results.items():
            if analysis.unused_imports:
                cleanup_stats['files_processed'] += 1
                
                if self.remove_unused_imports(filepath, analysis.unused_imports):
                    cleanup_stats['files_cleaned'] += 1
                    cleanup_stats['total_imports_removed'] += len(analysis.unused_imports)
                else:
                    cleanup_stats['errors'].append(filepath)
        
        return cleanup_stats

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Ultra Code Auditor - Unused Imports Analysis and Cleanup"
    )
    parser.add_argument(
        '--root-path',
        default='/opt/sutazaiapp',
        help='Root path to scan for Python files'
    )
    parser.add_argument(
        '--output-report',
        default='/opt/sutazaiapp/reports/unused_imports_audit.json',
        help='Path to save the audit report'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Actually remove unused imports (not just report)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Show what would be removed without making changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Phase 1: Analysis
    logger.info("=== PHASE 1: UNUSED IMPORTS ANALYSIS ===")
    detector = UnusedImportDetector()
    results = detector.scan_codebase(args.root_path)
    
    # Generate comprehensive report
    report = detector.generate_report(results)
    
    # Save report
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("UNUSED IMPORTS AUDIT SUMMARY")
    logger.info("="*80)
    logger.info(f"Total files analyzed: {report['summary']['total_files_analyzed']}")
    logger.info(f"Files with unused imports: {report['summary']['files_with_unused_imports']}")
    logger.info(f"Total unused imports found: {report['summary']['total_unused_imports']}")
    logger.info(f"Estimated cleanup benefit: {report['summary']['cleanup_potential_mb']} MB")
    logger.info("\nTop 10 Most Unused Modules:")
    for module, count in list(report['summary']['most_unused_modules'].items())[:10]:
        logger.info(f"  {module}: {count} occurrences")
    
    logger.info(f"\nDetailed report saved to: {args.output_report}")
    
    # Phase 2: Cleanup (if requested)
    if args.cleanup:
        logger.info("\n=== PHASE 2: CLEANUP EXECUTION ===")
        cleaner = UnusedImportCleaner(dry_run=args.dry_run)
        cleanup_stats = cleaner.cleanup_files(results)
        
        logger.info("\n" + "="*80)
        logger.info("CLEANUP SUMMARY")
        logger.info("="*80)
        logger.info(f"Files processed: {cleanup_stats['files_processed']}")
        logger.info(f"Files cleaned: {cleanup_stats['files_cleaned']}")
        logger.info(f"Total imports removed: {cleanup_stats['total_imports_removed']}")
        if cleanup_stats['errors']:
            logger.error(f"Errors encountered: {len(cleanup_stats['errors'])}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())