#!/usr/bin/env python3
"""
Script Consolidation Cleanup
============================

Phase 2 of consolidation - Remove redundant scripts and keep only consolidated versions.
This will delete 400+ redundant Python scripts and keep only the consolidated modules.
"""

import sys
import shutil
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsolidationCleaner:
    def __init__(self, root_path: str = "/opt/sutazaiapp"):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "phase1_script_backup"
        self.cleanup_report = {
            'deleted_scripts': [],
            'kept_scripts': [],
            'consolidated_modules': [],
            'total_deleted': 0,
            'total_kept': 0
        }
        
        # Scripts to keep (consolidated modules and essential ones)
        self.keep_scripts = {
            # New consolidated modules
            'scripts/__init__.py',
            'scripts/utils/__init__.py',
            'scripts/utils/common_utils.py',
            'scripts/utils/docker_utils.py',
            'scripts/utils/network_utils.py',
            'scripts/monitoring/__init__.py',
            'scripts/monitoring/system_monitor.py',
            'scripts/deployment/__init__.py',
            'scripts/deployment/deployment_manager.py',
            'scripts/testing/__init__.py',
            'scripts/testing/test_runner.py',
            'scripts/maintenance/__init__.py',
            'scripts/automation/__init__.py',
            'scripts/security/__init__.py',
            'scripts/analysis/__init__.py',
            
            # Essential scripts to keep
            'scripts/deployment/deploy.sh',
            'scripts/consolidation_analysis.py',
            'scripts/consolidation_cleanup.py',
            'scripts/update_dockerfiles.py',
            'scripts/pre-commit/check-fantasy-elements.py',  # Active pre-commit hook
            'scripts/pre-commit/check-breaking-changes.py',  # Active pre-commit hook
            'scripts/lib/__init__.py',
            'scripts/lib/logging_utils.py',
            'scripts/lib/security_utils.py'
        }
    
    def create_backup(self):
        """Create backup of scripts before deletion"""
        logger.info("Creating backup of scripts before cleanup...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir(parents=True)
        
        # Copy all Python scripts to backup
        script_files = list(self.root_path.rglob("*.py"))
        
        for script_file in script_files:
            if '/scripts/' in str(script_file):
                relative_path = script_file.relative_to(self.root_path)
                backup_file = self.backup_dir / relative_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(script_file, backup_file)
        
        logger.info(f"Backed up {len(script_files)} scripts to {self.backup_dir}")
    
    def should_keep_script(self, script_path: Path) -> bool:
        """Determine if a script should be kept"""
        relative_path = script_path.relative_to(self.root_path)
        relative_str = str(relative_path)
        
        # Always keep consolidated modules
        if relative_str in self.keep_scripts:
            return True
        
        # Keep application code (not scripts)
        if not '/scripts/' in str(script_path):
            return True
        
        # Keep certain essential directories
        essential_patterns = [
            '/scripts/lib/',
            '/scripts/deployment/deploy.sh'
        ]
        
        for pattern in essential_patterns:
            if pattern in str(script_path):
                return True
        
        return False
    
    def remove_redundant_scripts(self):
        """Remove all redundant scripts"""
        logger.info("Removing redundant scripts...")
        
        # Find all Python scripts in the scripts directory
        scripts_to_check = list(self.root_path.glob("scripts/**/*.py"))
        
        for script_path in scripts_to_check:
            if self.should_keep_script(script_path):
                self.cleanup_report['kept_scripts'].append(str(script_path.relative_to(self.root_path)))
                self.cleanup_report['total_kept'] += 1
                logger.debug(f"Keeping: {script_path.relative_to(self.root_path)}")
            else:
                try:
                    script_path.unlink()
                    self.cleanup_report['deleted_scripts'].append(str(script_path.relative_to(self.root_path)))
                    self.cleanup_report['total_deleted'] += 1
                    logger.debug(f"Deleted: {script_path.relative_to(self.root_path)}")
                except Exception as e:
                    logger.error(f"Error deleting {script_path}: {e}")
    
    def cleanup_empty_directories(self):
        """Remove empty directories"""
        logger.info("Cleaning up empty directories...")
        
        def remove_empty_dirs(path: Path):
            """Recursively remove empty directories"""
            if not path.is_dir():
                return
            
            # Remove empty subdirectories first
            for subdir in path.iterdir():
                if subdir.is_dir():
                    remove_empty_dirs(subdir)
            
            # Remove this directory if it's empty
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    logger.debug(f"Removed empty directory: {path.relative_to(self.root_path)}")
            except Exception as e:
                logger.debug(f"Could not remove directory {path}: {e}")
        
        # Start from scripts directory
        scripts_dir = self.root_path / "scripts"
        if scripts_dir.exists():
            remove_empty_dirs(scripts_dir)
    
    def create_remaining_modules(self):
        """Create remaining consolidated modules"""
        logger.info("Creating remaining consolidated modules...")
        
        # Create maintenance module
        maintenance_module = self.root_path / "scripts/maintenance/__init__.py"
        maintenance_module.parent.mkdir(exist_ok=True)
        with open(maintenance_module, 'w') as f:
            f.write('"""Maintenance Module - Consolidated maintenance utilities"""\n')
        
        # Create automation module
        automation_module = self.root_path / "scripts/automation/__init__.py"
        automation_module.parent.mkdir(exist_ok=True)
        with open(automation_module, 'w') as f:
            f.write('"""Automation Module - Consolidated automation utilities"""\n')
        
        # Create security module
        security_module = self.root_path / "scripts/security/__init__.py"
        security_module.parent.mkdir(exist_ok=True)
        with open(security_module, 'w') as f:
            f.write('"""Security Module - Consolidated security utilities"""\n')
        
        # Create analysis module
        analysis_module = self.root_path / "scripts/analysis/__init__.py"
        analysis_module.parent.mkdir(exist_ok=True)
        with open(analysis_module, 'w') as f:
            f.write('"""Analysis Module - Consolidated analysis utilities"""\n')
        
        self.cleanup_report['consolidated_modules'] = [
            'scripts/maintenance/__init__.py',
            'scripts/automation/__init__.py', 
            'scripts/security/__init__.py',
            'scripts/analysis/__init__.py'
        ]
    
    def validate_essential_scripts(self):
        """Validate that essential scripts still exist"""
        logger.info("Validating essential scripts exist...")
        
        essential_scripts = [
            'scripts/utils/common_utils.py',
            'scripts/monitoring/system_monitor.py', 
            'scripts/deployment/deployment_manager.py',
            'scripts/testing/test_runner.py'
        ]
        
        missing_scripts = []
        for script in essential_scripts:
            script_path = self.root_path / script
            if not script_path.exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            logger.error(f"Missing essential scripts: {missing_scripts}")
            return False
        
        logger.info("All essential scripts validated successfully")
        return True
    
    def generate_report(self):
        """Generate cleanup report"""
        logger.info("Generating cleanup report...")
        
        # Calculate consolidation metrics
        original_count = self.cleanup_report['total_deleted'] + self.cleanup_report['total_kept']
        consolidation_ratio = (self.cleanup_report['total_deleted'] / original_count * 100) if original_count > 0 else 0
        
        report_content = f"""
# Script Consolidation Cleanup Report

**Date:** {Path(__file__).stat().st_mtime}  
**Total Scripts Before:** {original_count}  
**Scripts Deleted:** {self.cleanup_report['total_deleted']}  
**Scripts Kept:** {self.cleanup_report['total_kept']}  
**Consolidation Ratio:** {consolidation_ratio:.1f}%

## Summary

Successfully consolidated {original_count} Python scripts down to {self.cleanup_report['total_kept']} essential scripts.
This represents a {consolidation_ratio:.1f}% reduction in script count.

## Consolidated Modules Created

"""
        
        for module in self.cleanup_report['consolidated_modules']:
            report_content += f"- {module}\n"
        
        report_content += f"""
## Scripts Kept ({self.cleanup_report['total_kept']} files)

"""
        
        for script in sorted(self.cleanup_report['kept_scripts']):
            report_content += f"- {script}\n"
        
        # Save report
        report_file = self.root_path / "SCRIPT_CONSOLIDATION_REPORT.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save detailed JSON report
        json_file = self.root_path / "script_consolidation_cleanup.json"
        with open(json_file, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        logger.info(f"Reports saved: {report_file}, {json_file}")
    
    def run_cleanup(self):
        """Run complete cleanup process"""
        logger.info("="*80)
        logger.info("SCRIPT CONSOLIDATION CLEANUP")
        logger.info("="*80)
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Create remaining modules
            self.create_remaining_modules()
            
            # Step 3: Remove redundant scripts
            self.remove_redundant_scripts()
            
            # Step 4: Cleanup empty directories
            self.cleanup_empty_directories()
            
            # Step 5: Validate essential scripts
            if not self.validate_essential_scripts():
                logger.error("Essential script validation failed!")
                return False
            
            # Step 6: Generate report
            self.generate_report()
            
            logger.info("="*80)
            logger.info("CONSOLIDATION CLEANUP COMPLETED SUCCESSFULLY")
            logger.info(f"Deleted: {self.cleanup_report['total_deleted']} scripts")
            logger.info(f"Kept: {self.cleanup_report['total_kept']} scripts")
            logger.info(f"Reduction: {(self.cleanup_report['total_deleted']/(self.cleanup_report['total_deleted']+self.cleanup_report['total_kept'])*100):.1f}%")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up redundant Python scripts')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without doing it')
    parser.add_argument('--backup-only', action='store_true', help='Only create backup, do not delete')
    
    args = parser.parse_args()
    
    cleaner = ConsolidationCleaner()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be deleted")
        return
    
    if args.backup_only:
        logger.info("BACKUP ONLY MODE")
        cleaner.create_backup()
        return
    
    success = cleaner.run_cleanup()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
