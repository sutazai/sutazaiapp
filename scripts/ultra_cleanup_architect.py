#!/usr/bin/env python3
"""
ULTRA-CLEANUP ARCHITECTURE SYSTEM
Lead System Architect for Perfect SutazAI Cleanup
Following all 19 CLAUDE.md rules with zero compromise
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/cleanup_architect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltraCleanupArchitect:
    """Master architect for perfect cleanup execution"""
    
    def __init__(self):
        self.base_dir = Path('/opt/sutazaiapp')
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir = self.base_dir / f'backups/ultracleanup_backup_{self.timestamp}'
        
        # Essential files to preserve (from CLAUDE.md analysis)
        self.essential_dockerfiles = {
            'docker-compose.yml',
            'docker/faiss/Dockerfile',  # Keep FAISS as specified
            'backend/Dockerfile',
            'frontend/Dockerfile',
            'agents/ai_agent_orchestrator/Dockerfile'
        }
        
        # Essential Python patterns
        self.essential_python_patterns = {
            'backend/app/',
            'frontend/',
            'agents/',
            'scripts/deploy',
            'scripts/health',
            'scripts/backup'
        }
        
        # Directories to completely remove
        self.remove_dirs = {
            'archive',
            'chaos', 
            'jenkins',
            'terraform',
            'node_modules',
            'ultracleanup_backup_20250811_133308',  # Old backup
            'ultracleanup_backup_20250811_145137'   # Old backup
        }
        
        self.stats = {
            'dockerfiles_found': 0,
            'dockerfiles_deleted': 0,
            'python_files_found': 0,
            'python_files_deleted': 0,
            'directories_removed': 0,
            'space_freed_mb': 0
        }

    def calculate_file_hash(self, filepath: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return "ERROR"

    def create_backup_structure(self):
        """Create perfect backup structure"""
        logger.info("=== PHASE 1: Creating Perfect Backup Structure ===")
        
        # Create organized backup directories
        backup_dirs = [
            self.backup_dir / 'dockerfiles',
            self.backup_dir / 'python_files',
            self.backup_dir / 'configs',
            self.backup_dir / 'scripts',
            self.backup_dir / 'deleted_directories'
        ]
        
        for dir_path in backup_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created backup directory: {dir_path}")
        
        # Create manifest file
        self.manifest_file = self.backup_dir / 'manifest.txt'
        with open(self.manifest_file, 'w') as f:
            f.write(f"ULTRA-CLEANUP BACKUP MANIFEST\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"System: SutazAI v81\n")
            f.write(f"{'='*60}\n\n")
        
        logger.info(f"Backup structure created at: {self.backup_dir}")
        return True

    def analyze_dockerfiles(self) -> Dict[str, List[Path]]:
        """Analyze and categorize all Dockerfiles"""
        logger.info("=== Analyzing Dockerfiles ===")
        
        dockerfiles = {
            'essential': [],
            'delete': [],
            'variants': []
        }
        
        # Find all Dockerfiles
        for dockerfile in self.base_dir.rglob('Dockerfile*'):
            relative_path = dockerfile.relative_to(self.base_dir)
            
            # Check if essential
            is_essential = False
            for essential in self.essential_dockerfiles:
                if str(relative_path) == essential or str(relative_path).startswith(essential):
                    is_essential = True
                    break
            
            if is_essential:
                dockerfiles['essential'].append(dockerfile)
            elif 'Dockerfile.' in dockerfile.name:
                dockerfiles['variants'].append(dockerfile)
            else:
                dockerfiles['delete'].append(dockerfile)
        
        self.stats['dockerfiles_found'] = sum(len(v) for v in dockerfiles.values())
        
        logger.info(f"Dockerfiles found: {self.stats['dockerfiles_found']}")
        logger.info(f"  Essential: {len(dockerfiles['essential'])}")
        logger.info(f"  To delete: {len(dockerfiles['delete'])}")
        logger.info(f"  Variants: {len(dockerfiles['variants'])}")
        
        return dockerfiles

    def analyze_python_files(self) -> Dict[str, List[Path]]:
        """Analyze Python files for cleanup"""
        logger.info("=== Analyzing Python Files ===")
        
        python_files = {
            'essential': [],
            'has_todos': [],
            'tests': [],
            'duplicates': [],
            'delete': []
        }
        
        seen_contents = {}
        
        for py_file in self.base_dir.rglob('*.py'):
            if 'site-packages' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            relative_path = py_file.relative_to(self.base_dir)
            
            # Check if essential
            is_essential = any(
                str(relative_path).startswith(pattern) 
                for pattern in self.essential_python_patterns
            )
            
            if is_essential:
                python_files['essential'].append(py_file)
                continue
            
            # Check for TODOs
            try:
                content = py_file.read_text()
                file_hash = hashlib.md5(content.encode()).hexdigest()
                
                if 'TODO' in content or 'FIXME' in content:
                    python_files['has_todos'].append(py_file)
                
                # Check for duplicates
                if file_hash in seen_contents:
                    python_files['duplicates'].append((py_file, seen_contents[file_hash]))
                else:
                    seen_contents[file_hash] = py_file
                
                # Check if test file
                if 'test_' in py_file.name or '_test.py' in py_file.name:
                    python_files['tests'].append(py_file)
                
                # Mark for deletion if not essential
                if not is_essential and ('TODO' in content or 'test' in str(py_file)):
                    python_files['delete'].append(py_file)
                    
            except Exception as e:
                logger.error(f"Error analyzing {py_file}: {e}")
        
        self.stats['python_files_found'] = sum(len(v) if not isinstance(v[0], tuple) else len(v) 
                                               for v in python_files.values())
        
        logger.info(f"Python files analyzed: {self.stats['python_files_found']}")
        logger.info(f"  Essential: {len(python_files['essential'])}")
        logger.info(f"  With TODOs: {len(python_files['has_todos'])}")
        logger.info(f"  Tests: {len(python_files['tests'])}")
        logger.info(f"  Duplicates: {len(python_files['duplicates'])}")
        
        return python_files

    def backup_files(self, files: List[Path], category: str):
        """Backup files before deletion"""
        logger.info(f"Backing up {len(files)} {category} files...")
        
        backup_subdir = self.backup_dir / category
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        manifest_entries = []
        
        for file_path in files:
            try:
                relative_path = file_path.relative_to(self.base_dir)
                backup_path = backup_subdir / relative_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Calculate hash before backup
                file_hash = self.calculate_file_hash(file_path)
                
                # Copy file
                shutil.copy2(file_path, backup_path)
                
                # Add to manifest
                manifest_entries.append(f"{file_hash}  {relative_path}")
                
            except Exception as e:
                logger.error(f"Error backing up {file_path}: {e}")
        
        # Append to manifest
        with open(self.manifest_file, 'a') as f:
            f.write(f"\n{category.upper()} FILES:\n")
            f.write('\n'.join(manifest_entries))
            f.write('\n')
        
        logger.info(f"Backed up {len(manifest_entries)} {category} files")
        return len(manifest_entries)

    def execute_cleanup_phase1(self, dockerfiles: Dict):
        """Phase 1: Dockerfile cleanup"""
        logger.info("=== PHASE 1: Dockerfile Cleanup ===")
        
        # Backup dockerfiles to delete
        files_to_delete = dockerfiles['delete'] + dockerfiles['variants']
        backed_up = self.backup_files(files_to_delete, 'dockerfiles')
        
        if backed_up != len(files_to_delete):
            logger.error("Backup count mismatch! Aborting cleanup.")
            return False
        
        # Delete dockerfiles
        for dockerfile in files_to_delete:
            try:
                dockerfile.unlink()
                self.stats['dockerfiles_deleted'] += 1
                logger.debug(f"Deleted: {dockerfile}")
            except Exception as e:
                logger.error(f"Error deleting {dockerfile}: {e}")
        
        # Clean up entire docker directory except faiss
        docker_dir = self.base_dir / 'docker'
        if docker_dir.exists():
            for item in docker_dir.iterdir():
                if item.name != 'faiss':
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        logger.info(f"Removed docker/{item.name}")
                    except Exception as e:
                        logger.error(f"Error removing {item}: {e}")
        
        logger.info(f"Phase 1 complete: Deleted {self.stats['dockerfiles_deleted']} Dockerfiles")
        return True

    def execute_cleanup_phase2(self, python_files: Dict):
        """Phase 2: Python cleanup"""
        logger.info("=== PHASE 2: Python File Cleanup ===")
        
        # Backup python files to delete
        files_to_delete = python_files['delete']
        backed_up = self.backup_files(files_to_delete, 'python_files')
        
        if backed_up != len(files_to_delete):
            logger.error("Backup count mismatch! Aborting cleanup.")
            return False
        
        # Delete python files
        for py_file in files_to_delete:
            try:
                py_file.unlink()
                self.stats['python_files_deleted'] += 1
                logger.debug(f"Deleted: {py_file}")
            except Exception as e:
                logger.error(f"Error deleting {py_file}: {e}")
        
        logger.info(f"Phase 2 complete: Deleted {self.stats['python_files_deleted']} Python files")
        return True

    def execute_cleanup_phase3(self):
        """Phase 3: Directory cleanup"""
        logger.info("=== PHASE 3: Directory Cleanup ===")
        
        for dir_name in self.remove_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                try:
                    # Calculate size before deletion
                    size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024*1024)
                    
                    # Backup directory structure
                    backup_path = self.backup_dir / 'deleted_directories' / dir_name
                    logger.info(f"Backing up directory: {dir_name}")
                    shutil.copytree(dir_path, backup_path, symlinks=True)
                    
                    # Remove directory
                    shutil.rmtree(dir_path)
                    self.stats['directories_removed'] += 1
                    self.stats['space_freed_mb'] += size_mb
                    logger.info(f"Removed directory: {dir_name} (freed {size_mb:.2f} MB)")
                    
                except Exception as e:
                    logger.error(f"Error removing {dir_path}: {e}")
        
        logger.info(f"Phase 3 complete: Removed {self.stats['directories_removed']} directories")
        return True

    def verify_system_health(self) -> bool:
        """Verify system health after cleanup"""
        logger.info("=== Verifying System Health ===")
        
        health_checks = [
            ("Backend API", "curl -s http://localhost:10010/health"),
            ("Frontend UI", "curl -s http://localhost:10011/"),
            ("Ollama", "curl -s http://localhost:10104/api/tags"),
            ("PostgreSQL", "docker exec sutazai-postgres pg_isready"),
            ("Redis", "docker exec sutazai-redis redis-cli ping")
        ]
        
        all_healthy = True
        for service, command in health_checks:
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"✅ {service}: HEALTHY")
                else:
                    logger.error(f"❌ {service}: FAILED")
                    all_healthy = False
            except Exception as e:
                logger.error(f"❌ {service}: ERROR - {e}")
                all_healthy = False
        
        return all_healthy

    def create_rollback_script(self):
        """Create rollback script for emergency recovery"""
        rollback_script = self.backup_dir / 'rollback.sh'
        
        script_content = f'''#!/bin/bash
# ULTRA-CLEANUP ROLLBACK SCRIPT
# Generated: {datetime.now().isoformat()}
# Backup: {self.backup_dir}

set -e

echo "=== ULTRA-CLEANUP ROLLBACK ==="
echo "Restoring from: {self.backup_dir}"

# Restore dockerfiles
echo "Restoring Dockerfiles..."
rsync -av {self.backup_dir}/dockerfiles/ /opt/sutazaiapp/

# Restore Python files
echo "Restoring Python files..."
rsync -av {self.backup_dir}/python_files/ /opt/sutazaiapp/

# Restore deleted directories
echo "Restoring deleted directories..."
rsync -av {self.backup_dir}/deleted_directories/ /opt/sutazaiapp/

echo "=== ROLLBACK COMPLETE ==="
echo "Please restart services: docker-compose restart"
'''
        
        with open(rollback_script, 'w') as f:
            f.write(script_content)
        
        rollback_script.chmod(0o755)
        logger.info(f"Rollback script created: {rollback_script}")

    def generate_final_report(self):
        """Generate comprehensive cleanup report"""
        report_file = self.backup_dir / 'cleanup_report.json'
        
        report = {
            'timestamp': self.timestamp,
            'stats': self.stats,
            'backup_location': str(self.backup_dir),
            'manifest_file': str(self.manifest_file),
            'rollback_script': str(self.backup_dir / 'rollback.sh'),
            'system_health': self.verify_system_health()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=== FINAL CLEANUP REPORT ===")
        logger.info(f"Dockerfiles deleted: {self.stats['dockerfiles_deleted']}")
        logger.info(f"Python files deleted: {self.stats['python_files_deleted']}")
        logger.info(f"Directories removed: {self.stats['directories_removed']}")
        logger.info(f"Space freed: {self.stats['space_freed_mb']:.2f} MB")
        logger.info(f"Backup location: {self.backup_dir}")
        logger.info(f"Report saved: {report_file}")
        
        return report

    def execute_ultra_cleanup(self, dry_run=False):
        """Execute the complete ultra-cleanup process"""
        logger.info("=== STARTING ULTRA-CLEANUP ARCHITECTURE ===")
        logger.info(f"Dry run: {dry_run}")
        
        # Step 1: Create backup structure
        if not self.create_backup_structure():
            logger.error("Failed to create backup structure")
            return False
        
        # Step 2: Analyze files
        dockerfiles = self.analyze_dockerfiles()
        python_files = self.analyze_python_files()
        
        if dry_run:
            logger.info("=== DRY RUN COMPLETE ===")
            logger.info("No files were deleted. Run without --dry-run to execute cleanup.")
            return True
        
        # Step 3: Execute cleanup phases
        if not self.execute_cleanup_phase1(dockerfiles):
            logger.error("Phase 1 failed! Aborting cleanup.")
            return False
        
        # Verify health after Phase 1
        if not self.verify_system_health():
            logger.error("System unhealthy after Phase 1! Aborting remaining cleanup.")
            self.create_rollback_script()
            return False
        
        if not self.execute_cleanup_phase2(python_files):
            logger.error("Phase 2 failed! Creating rollback script.")
            self.create_rollback_script()
            return False
        
        # Verify health after Phase 2
        if not self.verify_system_health():
            logger.error("System unhealthy after Phase 2! Aborting remaining cleanup.")
            self.create_rollback_script()
            return False
        
        if not self.execute_cleanup_phase3():
            logger.error("Phase 3 failed! Creating rollback script.")
            self.create_rollback_script()
            return False
        
        # Final verification
        if not self.verify_system_health():
            logger.error("System unhealthy after cleanup! Rollback script available.")
            self.create_rollback_script()
            return False
        
        # Create rollback script anyway for safety
        self.create_rollback_script()
        
        # Generate final report
        self.generate_final_report()
        
        logger.info("=== ULTRA-CLEANUP COMPLETE ===")
        return True


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Cleanup Architecture System')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only, no deletions')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    args = parser.parse_args()
    
    architect = UltraCleanupArchitect()
    
    if not args.dry_run and not args.force:
        print("\n⚠️  ULTRA-CLEANUP WARNING ⚠️")
        print("This will permanently delete files and directories.")
        print("A backup will be created, but please confirm.")
        response = input("\nType 'CLEAN' to proceed: ")
        if response != 'CLEAN':
            print("Cleanup cancelled.")
            sys.exit(0)
    
    success = architect.execute_ultra_cleanup(dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()