#!/usr/bin/env python3
"""
ULTRA CODE AUDITOR - Safe Unused Imports Cleanup
=================================================

This script safely removes unused imports while preserving critical functionality.
Uses Conservative approach with extensive validation and rollback capabilities.

Author: Ultra Code Auditor
Created: August 10, 2025
Purpose: Clean up 2,264 unused imports identified in audit
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SafeImportCleaner:
    """Safe import cleanup with validation and rollback"""
    
    def __init__(self):
        self.backup_dir = Path("/opt/sutazaiapp/backups/import_cleanup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Conservative cleanup - only remove obviously safe imports
        self.safe_to_remove = {
            'typing.Optional', 'typing.List', 'typing.Dict', 'typing.Tuple',
            'typing.Set', 'typing.Union', 'typing.Any', 'typing.Callable',
            'typing.NamedTuple', 'typing.TypeVar', 'typing.Generic',
            'datetime.timedelta', 'datetime.timezone', 'datetime.date',
            'os', 'sys', 'json', 're', 'pathlib.Path', 'pathlib.PurePath',
            'collections.defaultdict', 'collections.OrderedDict',
            'functools.partial', 'functools.wraps', 'itertools.chain',
            'logging', 'argparse', 'subprocess', 'time', 'random',
            'string', 'hashlib', 'base64', 'uuid', 'copy', 'math'
        }
        
        # Critical files to skip completely
        self.critical_files = {
            'main.py', 'app.py', '__init__.py', 'conftest.py', 
            'settings.py', 'config.py', 'setup.py'
        }
        
        # Files that require special handling
        self.test_file_patterns = {'test_', '_test.py', '/tests/'}
        
    def is_safe_cleanup_candidate(self, filepath: str, unused_imports: List) -> bool:
        """Determine if file is safe for automated cleanup"""
        filename = os.path.basename(filepath)
        
        # Skip critical files
        if filename in self.critical_files:
            return False
            
        # Only clean test files with very conservative approach
        if any(pattern in filepath for pattern in self.test_file_patterns):
            # Only remove typing imports from test files
            return all(imp.get('module', '').startswith('typing.') for imp in unused_imports)
        
        # For non-test files, check if all unused imports are in safe list
        return all(imp.get('module', '') in self.safe_to_remove for imp in unused_imports)
    
    def create_backup(self, filepath: str) -> str:
        """Create timestamped backup of file"""
        backup_name = f"{Path(filepath).name}.{os.getpid()}.backup"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(filepath, backup_path)
        return str(backup_path)
    
    def remove_imports_from_file(self, filepath: str, unused_imports: List) -> bool:
        """Safely remove unused imports from file"""
        try:
            # Create backup
            backup_path = self.create_backup(filepath)
            logger.info(f"Created backup: {backup_path}")
            
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Sort by line number (descending) to remove from bottom up
            unused_imports.sort(key=lambda x: x.get('line', 0), reverse=True)
            
            removed_count = 0
            for imp in unused_imports:
                line_num = imp.get('line', 0)
                if 1 <= line_num <= len(lines):
                    statement = imp.get('statement', '').strip()
                    actual_line = lines[line_num - 1].strip()
                    
                    # Double-check the line matches what we expect
                    if statement in actual_line or actual_line in statement:
                        logger.debug(f"Removing line {line_num}: {actual_line}")
                        del lines[line_num - 1]
                        removed_count += 1
                    else:
                        logger.warning(f"Line mismatch in {filepath}:{line_num}")
                        logger.warning(f"Expected: {statement}")
                        logger.warning(f"Actual: {actual_line}")
            
            # Write cleaned file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            logger.info(f"Removed {removed_count} unused imports from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning {filepath}: {e}")
            return False
    
    def validate_file_syntax(self, filepath: str) -> bool:
        """Validate Python file syntax after cleanup"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            compile(content, filepath, 'exec')
            return True
            
        except SyntaxError as e:
            logger.error(f"Syntax error in {filepath}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating {filepath}: {e}")
            return False
    
    def rollback_file(self, filepath: str) -> bool:
        """Rollback file from backup if validation fails"""
        try:
            backup_name = f"{Path(filepath).name}.{os.getpid()}.backup"
            backup_path = self.backup_dir / backup_name
            
            if backup_path.exists():
                shutil.copy2(backup_path, filepath)
                logger.info(f"Rolled back {filepath} from backup")
                return True
            else:
                logger.error(f"Backup not found for {filepath}")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back {filepath}: {e}")
            return False
    
    def process_cleanup_batch(self, audit_report: Dict, batch_size: int = 20) -> Dict:
        """Process cleanup in small batches for safety"""
        results = {
            'processed': 0,
            'cleaned': 0,
            'skipped': 0,
            'failed': 0,
            'rollbacks': 0,
            'errors': []
        }
        
        files_with_unused = audit_report.get('files_with_unused_imports', [])
        
        # Process in batches
        for i in range(0, len(files_with_unused), batch_size):
            batch = files_with_unused[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
            
            for file_info in batch:
                filepath = file_info['filepath']
                unused_imports = file_info['unused_imports']
                results['processed'] += 1
                
                # Safety check
                if not self.is_safe_cleanup_candidate(filepath, unused_imports):
                    logger.info(f"Skipping {filepath} - not safe for automated cleanup")
                    results['skipped'] += 1
                    continue
                
                # Perform cleanup
                if self.remove_imports_from_file(filepath, unused_imports):
                    # Validate syntax
                    if self.validate_file_syntax(filepath):
                        results['cleaned'] += 1
                        logger.info(f"✅ Successfully cleaned {filepath}")
                    else:
                        # Rollback on syntax error
                        if self.rollback_file(filepath):
                            results['rollbacks'] += 1
                            logger.warning(f"⚠️ Rolled back {filepath} due to syntax error")
                        else:
                            results['failed'] += 1
                            results['errors'].append(f"Failed to rollback {filepath}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to clean {filepath}")
        
        return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Safe unused imports cleanup")
    parser.add_argument('--audit-report', 
                       default='/opt/sutazaiapp/reports/unused_imports_audit.json',
                       help='Path to audit report JSON file')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Number of files to process in each batch')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without making changes')
    
    args = parser.parse_args()
    
    # Load audit report
    try:
        with open(args.audit_report, 'r') as f:
            audit_report = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load audit report: {e}")
        return 1
    
    cleaner = SafeImportCleaner()
    
    if args.dry_run:
        # Dry run analysis
        safe_files = 0
        safe_imports = 0
        
        for file_info in audit_report.get('files_with_unused_imports', []):
            if cleaner.is_safe_cleanup_candidate(file_info['filepath'], file_info['unused_imports']):
                safe_files += 1
                safe_imports += len(file_info['unused_imports'])
        
        print(f"\n=== DRY RUN ANALYSIS ===")
        print(f"Files safe for automated cleanup: {safe_files}")
        print(f"Unused imports that can be safely removed: {safe_imports}")
        print(f"Total files in audit: {len(audit_report.get('files_with_unused_imports', []))}")
        print(f"Total unused imports in audit: {audit_report.get('statistics', {}).get('unused_imports', 0)}")
        
    else:
        # Execute cleanup
        logger.info("Starting safe import cleanup...")
        results = cleaner.process_cleanup_batch(audit_report, args.batch_size)
        
        print(f"\n=== CLEANUP RESULTS ===")
        print(f"Files processed: {results['processed']}")
        print(f"Files cleaned successfully: {results['cleaned']}")
        print(f"Files skipped (not safe): {results['skipped']}")
        print(f"Files failed: {results['failed']}")
        print(f"Files rolled back: {results['rollbacks']}")
        
        if results['errors']:
            print(f"\nErrors encountered:")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())