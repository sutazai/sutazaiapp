#!/usr/bin/env python3
"""
Purpose: Verify that file deletions are safe and won't break functionality (Rule 10)
Usage: python check-safe-deletion.py
Requirements: Python 3.8+, git
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import sys
import subprocess
from pathlib import Path

class SafeDeletionChecker:
    """Checks if file deletions are safe."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.unsafe_deletions = []
        self.warnings = []
        
    def get_deleted_files(self) -> List[Path]:
        """Get files marked for deletion in git."""
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only', '--diff-filter=D'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            return [Path(f) for f in result.stdout.strip().split('\n') if f]
        except (IOError, OSError, FileNotFoundError) as e:
            logger.warning(f"Exception caught, returning: {e}")
            return []
    
    def find_references(self, filepath: Path) -> Dict[str, List[str]]:
        """Find all references to a file in the codebase."""
        references = {
            'imports': [],
            'includes': [],
            'requires': [],
            'other': []
        }
        
        # Get filename without extension for import searches
        module_name = filepath.stem
        
        # Search patterns for different types of references
        patterns = {
            'imports': [
                f"import.*{module_name}",
                f"from.*{module_name}.*import",
                f"require.*{module_name}",
                f"include.*{module_name}"
            ],
            'path_references': [
                str(filepath),
                str(filepath.relative_to(self.project_root)),
                filepath.name
            ]
        }
        
        # Search for import references
        for pattern in patterns['imports']:
            try:
                result = subprocess.run(
                    ['git', 'grep', '-l', '-E', pattern],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    files = result.stdout.strip().split('\n')
                    references['imports'].extend(files)
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        # Search for path references
        for pattern in patterns['path_references']:
            try:
                result = subprocess.run(
                    ['git', 'grep', '-l', '-F', pattern],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    files = result.stdout.strip().split('\n')
                    references['other'].extend(files)
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        # Remove duplicates
        for key in references:
            references[key] = list(set(references[key]))
        
        return references
    
    def check_test_coverage(self, filepath: Path) -> bool:
        """Check if the file being deleted has associated tests."""
        # Look for test files
        test_patterns = [
            f"test_{filepath.stem}",
            f"{filepath.stem}_test",
            f"test_*{filepath.stem}*"
        ]
        
        for pattern in test_patterns:
            try:
                result = subprocess.run(
                    ['find', str(self.project_root), '-name', f"{pattern}.py", '-type', 'f'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        return False
    
    def check_documentation_references(self, filepath: Path) -> List[str]:
        """Check if file is referenced in documentation."""
        doc_references = []
        doc_patterns = ['*.md', '*.rst', '*.txt']
        
        for pattern in doc_patterns:
            try:
                result = subprocess.run(
                    ['git', 'grep', '-l', filepath.name, '--', f'*{pattern}'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    doc_references.extend(result.stdout.strip().split('\n'))
            except (IOError, OSError, FileNotFoundError) as e:
                # Suppressed exception (was bare except)
                logger.debug(f"Suppressed exception: {e}")
                pass
        
        return list(set(doc_references))
    
    def analyze_deletion_safety(self, filepath: Path) -> Dict[str, any]:
        """Analyze if a file deletion is safe."""
        analysis = {
            'filepath': filepath,
            'safe': True,
            'reasons': [],
            'references': {},
            'has_tests': False,
            'doc_references': []
        }
        
        # Find all references
        references = self.find_references(filepath)
        analysis['references'] = references
        
        # Check if file has references
        total_refs = sum(len(refs) for refs in references.values())
        if total_refs > 0:
            analysis['safe'] = False
            analysis['reasons'].append(f"Found {total_refs} references to this file")
        
        # Check test coverage
        has_tests = self.check_test_coverage(filepath)
        analysis['has_tests'] = has_tests
        if has_tests:
            analysis['safe'] = False
            analysis['reasons'].append("File has associated test files")
        
        # Check documentation references
        doc_refs = self.check_documentation_references(filepath)
        analysis['doc_references'] = doc_refs
        if doc_refs:
            analysis['safe'] = False
            analysis['reasons'].append(f"Referenced in {len(doc_refs)} documentation files")
        
        # Special checks for certain file types
        if filepath.suffix == '.py' and filepath.name == '__init__.py':
            analysis['safe'] = False
            analysis['reasons'].append("Package initialization file - deletion may break imports")
        
        if filepath.name in ['setup.py', 'setup.cfg', 'pyproject.toml', 'package.json']:
            analysis['safe'] = False
            analysis['reasons'].append("Critical configuration file")
        
        if 'config' in str(filepath) or 'settings' in str(filepath):
            analysis['safe'] = False
            analysis['reasons'].append("Configuration file - verify all settings are migrated")
        
        return analysis

def main():
    """Main function to check deletion safety."""
    project_root = Path("/opt/sutazaiapp")
    checker = SafeDeletionChecker(project_root)
    
    print("🔍 Checking deletion safety (Rule 10)...")
    
    # Get deleted files
    deleted_files = checker.get_deleted_files()
    
    if not deleted_files:
        print("No files marked for deletion")
        return 0
    
    print(f"Found {len(deleted_files)} files marked for deletion")
    
    # Analyze each deletion
    unsafe_deletions = []
    warnings = []
    
    for filepath in deleted_files:
        analysis = checker.analyze_deletion_safety(filepath)
        
        if not analysis['safe']:
            unsafe_deletions.append(analysis)
        elif analysis['references'] or analysis['doc_references']:
            warnings.append(analysis)
    
    # Report results
    if unsafe_deletions:
        print("\n❌ Rule 10 Violation: Unsafe file deletions detected")
        print("\n📋 Files that should NOT be deleted without verification:")
        
        for analysis in unsafe_deletions:
            print(f"\n  File: {analysis['filepath']}")
            print("  Reasons:")
            for reason in analysis['reasons']:
                print(f"    - {reason}")
            
            if analysis['references']['imports']:
                print(f"  Import references ({len(analysis['references']['imports'])}):")
                for ref in analysis['references']['imports'][:5]:
                    print(f"    - {ref}")
                if len(analysis['references']['imports']) > 5:
                    print(f"    ... and {len(analysis['references']['imports']) - 5} more")
            
            if analysis['doc_references']:
                print(f"  Documentation references:")
                for ref in analysis['doc_references'][:3]:
                    print(f"    - {ref}")
        
        print("\n📋 How to proceed safely:")
        print("  1. Update all files that reference the deleted files")
        print("  2. Run tests to ensure nothing breaks")
        print("  3. Update documentation to remove references")
        print("  4. Consider deprecation instead of immediate deletion")
        print("  5. If deletion is necessary, document in PR why it's safe")
        
        return 1
    
    if warnings:
        print("\n⚠️  Deletion warnings:")
        for analysis in warnings:
            print(f"  - {analysis['filepath']}: May have minor references")
    
    print("✅ Rule 10: All deletions appear safe")
    return 0

if __name__ == "__main__":
    sys.exit(main())