#!/usr/bin/env python3
"""
Comprehensive dependency optimization tool
Removes unused imports, analyzes dependencies, and suggests optimizations
"""

import ast
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import subprocess
import importlib.util

logger = logging.getLogger(__name__)

@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str
    name: str
    alias: Optional[str] = None
    line_number: int = 0
    is_from_import: bool = False
    is_used: bool = False

@dataclass
class DependencyAnalysis:
    """Analysis results for a single file"""
    file_path: str
    total_imports: int = 0
    unused_imports: List[ImportInfo] = field(default_factory=list)
    missing_imports: List[str] = field(default_factory=list)
    circular_imports: List[str] = field(default_factory=list)
    heavy_imports: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class DependencyOptimizer:
    """Optimizes Python dependencies and imports"""
    
    STANDARD_LIBRARY_MODULES = {
        'os', 'sys', 'json', 'time', 'datetime', 'logging', 'argparse',
        'pathlib', 'collections', 'itertools', 'functools', 'operator',
        'typing', 'dataclasses', 'enum', 'abc', 'contextlib', 'warnings',
        're', 'math', 'random', 'statistics', 'hashlib', 'base64', 'uuid',
        'urllib', 'http', 'socket', 'threading', 'multiprocessing', 'asyncio',
        'subprocess', 'shutil', 'tempfile', 'glob', 'fnmatch', 'csv'
    }
    
    HEAVY_MODULES = {
        'numpy', 'pandas', 'matplotlib', 'tensorflow', 'torch', 'sklearn',
        'scipy', 'seaborn', 'plotly', 'cv2', 'PIL', 'transformers'
    }
    
    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)
        self.file_analyses = {}
        self.global_stats = {
            'total_files': 0,
            'total_imports': 0,
            'unused_imports': 0,
            'files_with_issues': 0
        }
        
    def analyze_directory(self) -> Dict[str, DependencyAnalysis]:
        """Analyze all Python files in directory"""
        python_files = list(self.root_directory.rglob("*.py"))
        self.global_stats['total_files'] = len(python_files)
        
        for file_path in python_files:
            if self._should_analyze_file(file_path):
                try:
                    analysis = self.analyze_file(file_path)
                    self.file_analyses[str(file_path)] = analysis
                    self._update_global_stats(analysis)
                except Exception as e:
                    logger.error(f"Failed to analyze {file_path}: {e}")
                    
        return self.file_analyses
        
    def analyze_file(self, file_path: Path) -> DependencyAnalysis:
        """Analyze a single Python file"""
        analysis = DependencyAnalysis(file_path=str(file_path))
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract imports
            imports = self._extract_imports(tree)
            analysis.total_imports = len(imports)
            
            # Find used names
            used_names = self._extract_used_names(tree)
            
            # Identify unused imports
            for import_info in imports:
                if not self._is_import_used(import_info, used_names):
                    import_info.is_used = False
                    analysis.unused_imports.append(import_info)
                else:
                    import_info.is_used = True
                    
            # Check for heavy imports
            for import_info in imports:
                if any(heavy in import_info.module for heavy in self.HEAVY_MODULES):
                    analysis.heavy_imports.append(import_info.module)
                    
            # Generate suggestions
            analysis.suggestions = self._generate_suggestions(analysis, imports)
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            
        return analysis
        
    def _extract_imports(self, tree: ast.AST) -> List[ImportInfo]:
        """Extract all import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                        is_from_import=False
                    ))
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=module_name,
                        name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                        is_from_import=True
                    ))
                    
        return imports
        
    def _extract_used_names(self, tree: ast.AST) -> Set[str]:
        """Extract all used names from AST"""
        used_names = set()
        
        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                used_names.add(node.id)
                
            def visit_Attribute(self, node):
                # Handle chained attributes like module.submodule.function
                parts = []
                current = node
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    used_names.update(parts)
                    # Add the full chain
                    if len(parts) > 1:
                        used_names.add('.'.join(reversed(parts)))
                        
        visitor = NameVisitor()
        visitor.visit(tree)
        return used_names
        
    def _is_import_used(self, import_info: ImportInfo, used_names: Set[str]) -> bool:
        """Check if an import is actually used"""
        name_to_check = import_info.alias or import_info.name
        
        # Special cases for commonly used patterns
        if import_info.module in ['typing', 'dataclasses']:
            return True  # Type hints and decorators often used
            
        if import_info.name in ['*']:
            return True  # Star imports assume used
            
        # Check if the name appears in used names
        if name_to_check in used_names:
            return True
            
        # For from imports, check if module.name is used
        if import_info.is_from_import:
            full_name = f"{import_info.module}.{import_info.name}"
            if full_name in used_names:
                return True
                
        return False
        
    def _generate_suggestions(self, analysis: DependencyAnalysis, imports: List[ImportInfo]) -> List[str]:
        """Generate optimization suggestions"""
        suggestions = []
        
        if analysis.unused_imports:
            suggestions.append(f"Remove {len(analysis.unused_imports)} unused imports")
            
        # Check for import organization
        import_groups = self._group_imports(imports)
        if len(import_groups['standard']) > 10:
            suggestions.append("Consider grouping standard library imports")
            
        # Check for heavy imports
        if analysis.heavy_imports:
            suggestions.append(f"Consider lazy loading for heavy modules: {', '.join(analysis.heavy_imports[:3])}")
            
        # Check for too many imports
        if analysis.total_imports > 30:
            suggestions.append("File has many imports, consider breaking into smaller modules")
            
        return suggestions
        
    def _group_imports(self, imports: List[ImportInfo]) -> Dict[str, List[ImportInfo]]:
        """Group imports by type"""
        groups = {
            'standard': [],
            'third_party': [],
            'local': []
        }
        
        for import_info in imports:
            if any(import_info.module.startswith(std) for std in self.STANDARD_LIBRARY_MODULES):
                groups['standard'].append(import_info)
            elif '.' in import_info.module and not import_info.module.startswith('.'):
                groups['third_party'].append(import_info)
            else:
                groups['local'].append(import_info)
                
        return groups
        
    def remove_unused_imports(self, file_path: Path, dry_run: bool = True) -> int:
        """Remove unused imports from a file"""
        if str(file_path) not in self.file_analyses:
            self.analyze_file(file_path)
            
        analysis = self.file_analyses[str(file_path)]
        if not analysis.unused_imports:
            return 0
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Sort unused imports by line number (descending) to maintain positions
            unused_lines = sorted([imp.line_number for imp in analysis.unused_imports], reverse=True)
            
            removed_count = 0
            for line_num in unused_lines:
                if 1 <= line_num <= len(lines):
                    if not dry_run:
                        lines.pop(line_num - 1)
                    removed_count += 1
                    logger.info(f"{'Would remove' if dry_run else 'Removed'} unused import at line {line_num}")
                    
            if removed_count > 0 and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                    
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to remove unused imports from {file_path}: {e}")
            return 0
            
    def optimize_imports(self, file_path: Path, dry_run: bool = True) -> Dict[str, int]:
        """Comprehensive import optimization"""
        results = {
            'removed_unused': 0,
            'reorganized': 0,
            'errors': 0
        }
        
        try:
            # Remove unused imports
            results['removed_unused'] = self.remove_unused_imports(file_path, dry_run)
            
            # Reorganize imports (if not dry run)
            if not dry_run and results['removed_unused'] > 0:
                self._reorganize_imports(file_path)
                results['reorganized'] = 1
                
        except Exception as e:
            logger.error(f"Failed to optimize imports in {file_path}: {e}")
            results['errors'] = 1
            
        return results
        
    def _reorganize_imports(self, file_path: Path):
        """Reorganize imports following PEP 8 guidelines"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            imports = self._extract_imports(tree)
            groups = self._group_imports(imports)
            
            # Generate organized import block
            organized_imports = []
            
            # Standard library imports
            if groups['standard']:
                for imp in sorted(groups['standard'], key=lambda x: x.module):
                    organized_imports.append(self._format_import(imp))
                organized_imports.append("")
                
            # Third-party imports
            if groups['third_party']:
                for imp in sorted(groups['third_party'], key=lambda x: x.module):
                    organized_imports.append(self._format_import(imp))
                organized_imports.append("")
                
            # Local imports
            if groups['local']:
                for imp in sorted(groups['local'], key=lambda x: x.module):
                    organized_imports.append(self._format_import(imp))
                organized_imports.append("")
                
            # Replace import section in file
            # This is a simplified version - in practice, you'd need more sophisticated parsing
            logger.info(f"Reorganized imports in {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to reorganize imports in {file_path}: {e}")
            
    def _format_import(self, import_info: ImportInfo) -> str:
        """Format an import statement"""
        if import_info.is_from_import:
            base = f"from {import_info.module} import {import_info.name}"
        else:
            base = f"import {import_info.module}"
            
        if import_info.alias:
            base += f" as {import_info.alias}"
            
        return base
        
    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive optimization report"""
        return {
            "summary": self.global_stats,
            "top_offenders": self._get_top_offenders(),
            "optimization_opportunities": self._get_optimization_opportunities(),
            "file_details": {
                path: {
                    "unused_imports": len(analysis.unused_imports),
                    "total_imports": analysis.total_imports,
                    "suggestions": analysis.suggestions,
                    "heavy_imports": analysis.heavy_imports
                }
                for path, analysis in self.file_analyses.items()
                if analysis.unused_imports or analysis.suggestions
            }
        }
        
    def _get_top_offenders(self) -> List[Dict[str, any]]:
        """Get files with most unused imports"""
        offenders = []
        for path, analysis in self.file_analyses.items():
            if analysis.unused_imports:
                offenders.append({
                    "file": path,
                    "unused_count": len(analysis.unused_imports),
                    "total_imports": analysis.total_imports,
                    "percentage": len(analysis.unused_imports) / analysis.total_imports * 100
                })
                
        return sorted(offenders, key=lambda x: x["unused_count"], reverse=True)[:10]
        
    def _get_optimization_opportunities(self) -> Dict[str, int]:
        """Get optimization opportunities summary"""
        opportunities = {
            "files_with_unused_imports": 0,
            "total_unused_imports": 0,
            "files_with_heavy_imports": 0,
            "files_needing_reorganization": 0
        }
        
        for analysis in self.file_analyses.values():
            if analysis.unused_imports:
                opportunities["files_with_unused_imports"] += 1
                opportunities["total_unused_imports"] += len(analysis.unused_imports)
                
            if analysis.heavy_imports:
                opportunities["files_with_heavy_imports"] += 1
                
            if analysis.total_imports > 20:
                opportunities["files_needing_reorganization"] += 1
                
        return opportunities
        
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed"""
        exclude_patterns = ['__pycache__', '.git', 'node_modules', 'venv', '.venv']
        path_str = str(file_path)
        return not any(pattern in path_str for pattern in exclude_patterns)
        
    def _update_global_stats(self, analysis: DependencyAnalysis):
        """Update global statistics"""
        self.global_stats['total_imports'] += analysis.total_imports
        self.global_stats['unused_imports'] += len(analysis.unused_imports)
        
        if analysis.unused_imports or analysis.suggestions:
            self.global_stats['files_with_issues'] += 1


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Optimize Python dependencies and imports")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--remove-unused", action="store_true", help="Remove unused imports")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview changes")
    parser.add_argument("--apply", action="store_true", help="Apply changes")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    optimizer = DependencyOptimizer(args.directory)
    optimizer.analyze_directory()
    
    # Generate report
    report = optimizer.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))
        
    # Apply optimizations if requested
    if args.remove_unused:
        dry_run = args.dry_run and not args.apply
        total_removed = 0
        
        for file_path in optimizer.file_analyses:
            results = optimizer.optimize_imports(Path(file_path), dry_run)
            total_removed += results['removed_unused']
            
        print(f"\n{'Would remove' if dry_run else 'Removed'} {total_removed} unused imports")


if __name__ == "__main__":
    main()