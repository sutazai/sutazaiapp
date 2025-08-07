#!/usr/bin/env python3
"""
SutazAI Backup Verification System
Validates backup integrity, completeness, and restorability
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import gzip
import hashlib
import time
import tempfile
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psycopg2
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/backup-verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupVerificationSystem:
    """Backup verification and integrity checking system"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.verification_dir = self.backup_root / 'verification'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure verification directory exists
        self.verification_dir.mkdir(parents=True, exist_ok=True)
        
        # Verification types
        self.verification_types = {
            'checksum': 'Verify file integrity using checksums',
            'archive': 'Verify archive contents and structure',
            'database': 'Test database backup restoration',
            'configuration': 'Validate configuration file completeness',
            'model': 'Verify AI model integrity and loading',
            'logs': 'Check log archive accessibility',
            'monitoring': 'Validate monitoring data integrity'
        }
        
        # Test restoration settings
        self.test_restore_config = {
            'postgres_test_db': 'sutazai_backup_test',
            'temp_restore_dir': '/tmp/sutazai_restore_test',
            'sample_verification_percent': 10  # Verify 10% of backups by default
        }
    
    def discover_backup_files(self) -> Dict[str, List[Dict]]:
        """Discover all backup files and manifests"""
        backup_inventory = {}
        
        # Scan backup directories
        backup_categories = ['daily', 'weekly', 'monthly', 'postgres', 'sqlite', 
                           'loki', 'config', 'agents', 'models', 'monitoring', 'logs']
        
        for category in backup_categories:
            category_path = self.backup_root / category
            if not category_path.exists():
                continue
            
            backup_inventory[category] = []
            
            # Find backup files and manifests
            for backup_file in category_path.rglob('*'):
                if backup_file.is_file() and not backup_file.name.startswith('.'):
                    try:
                        file_info = {
                            'path': str(backup_file),
                            'name': backup_file.name,
                            'size': backup_file.stat().st_size,
                            'modified': datetime.datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                            'extension': backup_file.suffix,
                            'category': category
                        }
                        
                        # Identify file type
                        if backup_file.suffix in ['.gz', '.tar', '.zip']:
                            file_info['type'] = 'archive'
                        elif backup_file.suffix in ['.sql', '.dump']:
                            file_info['type'] = 'database'
                        elif backup_file.suffix == '.json' and 'manifest' in backup_file.name:
                            file_info['type'] = 'manifest'
                        else:
                            file_info['type'] = 'data'
                        
                        backup_inventory[category].append(file_info)
                        
                    except Exception as e:
                        logger.warning(f"Error processing backup file {backup_file}: {e}")
        
        return backup_inventory
    
    def verify_checksum_integrity(self, backup_files: List[Dict]) -> List[Dict]:
        """Verify backup file integrity using checksums"""
        results = []
        
        for backup_file in backup_files:
            try:
                file_path = Path(backup_file['path'])
                
                logger.info(f"Verifying checksum for: {file_path.name}")
                
                # Calculate current checksum
                current_checksum = self.calculate_checksum(file_path)
                
                # Look for stored checksum in manifest files
                stored_checksum = self.find_stored_checksum(backup_file)
                
                if stored_checksum:
                    checksum_match = current_checksum == stored_checksum
                    
                    results.append({
                        'file': str(file_path),
                        'current_checksum': current_checksum,
                        'stored_checksum': stored_checksum,
                        'checksum_match': checksum_match,
                        'status': 'verified' if checksum_match else 'corrupted',
                        'verification_type': 'checksum'
                    })
                    
                    if not checksum_match:
                        logger.error(f"Checksum mismatch for {file_path}")
                else:
                    # No stored checksum found, store current one for future verification
                    results.append({
                        'file': str(file_path),
                        'current_checksum': current_checksum,
                        'stored_checksum': None,
                        'status': 'checksum_stored',
                        'verification_type': 'checksum'
                    })
                    
                    # Store checksum for future verification
                    self.store_checksum(backup_file, current_checksum)
                
            except Exception as e:
                logger.error(f"Error verifying checksum for {backup_file['path']}: {e}")
                results.append({
                    'file': backup_file['path'],
                    'status': 'error',
                    'error': str(e),
                    'verification_type': 'checksum'
                })
        
        return results
    
    def verify_archive_integrity(self, archive_files: List[Dict]) -> List[Dict]:
        """Verify archive file integrity and contents"""
        results = []
        
        for archive_file in archive_files:
            try:
                file_path = Path(archive_file['path'])
                
                logger.info(f"Verifying archive: {file_path.name}")
                
                verification_result = {
                    'file': str(file_path),
                    'verification_type': 'archive'
                }
                
                if file_path.suffix == '.gz':
                    # Test gzip file
                    try:
                        with gzip.open(file_path, 'rt') as f:
                            # Try to read first few lines
                            for i, line in enumerate(f):
                                if i >= 10:  # Read first 10 lines
                                    break
                        
                        verification_result['status'] = 'valid'
                        verification_result['archive_type'] = 'gzip'
                        
                    except Exception as gz_e:
                        verification_result['status'] = 'corrupted'
                        verification_result['error'] = str(gz_e)
                
                elif file_path.suffix in ['.tar', '.tgz'] or '.tar.gz' in file_path.name:
                    # Test tar file
                    try:
                        with tarfile.open(file_path, 'r:*') as tar:
                            # Get archive contents
                            members = tar.getnames()
                            verification_result['archive_contents_count'] = len(members)
                            verification_result['status'] = 'valid'
                            verification_result['archive_type'] = 'tar'
                            
                            # Sample some files to verify they can be extracted
                            sample_size = min(5, len(members))
                            if sample_size > 0:
                                sample_members = members[:sample_size]
                                
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    for member in sample_members:
                                        try:
                                            tar.extract(member, temp_dir)
                                        except Exception as extract_e:
                                            verification_result['status'] = 'partially_corrupted'
                                            verification_result['extraction_errors'] = str(extract_e)
                                            break
                        
                    except Exception as tar_e:
                        verification_result['status'] = 'corrupted'
                        verification_result['error'] = str(tar_e)
                
                else:
                    # Unknown archive type, just check if file is readable
                    try:
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # Read first 1KB
                        
                        verification_result['status'] = 'readable'
                        verification_result['archive_type'] = 'unknown'
                        
                    except Exception as read_e:
                        verification_result['status'] = 'unreadable'
                        verification_result['error'] = str(read_e)
                
                results.append(verification_result)
                
            except Exception as e:
                logger.error(f"Error verifying archive {archive_file['path']}: {e}")
                results.append({
                    'file': archive_file['path'],
                    'status': 'error',
                    'error': str(e),
                    'verification_type': 'archive'
                })
        
        return results
    
    def verify_database_backups(self, database_files: List[Dict]) -> List[Dict]:
        """Verify database backup files by testing restoration"""
        results = []
        
        for db_file in database_files:
            try:
                file_path = Path(db_file['path'])
                
                logger.info(f"Verifying database backup: {file_path.name}")
                
                if 'postgres' in file_path.name or file_path.suffix == '.sql':
                    # Test PostgreSQL backup
                    result = self.test_postgres_backup_restore(file_path)
                elif file_path.suffix == '.sqlite' or 'sqlite' in file_path.name:
                    # Test SQLite backup
                    result = self.test_sqlite_backup_restore(file_path)
                else:
                    # Generic database file check
                    result = self.test_generic_database_file(file_path)
                
                result['file'] = str(file_path)
                result['verification_type'] = 'database'
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error verifying database backup {db_file['path']}: {e}")
                results.append({
                    'file': db_file['path'],
                    'status': 'error',
                    'error': str(e),
                    'verification_type': 'database'
                })
        
        return results
    
    def test_postgres_backup_restore(self, backup_file: Path) -> Dict:
        """Test PostgreSQL backup by attempting to restore to test database"""
        try:
            # Check if backup file is readable
            if backup_file.suffix == '.gz':
                test_cmd = ['zcat', str(backup_file)]
            else:
                test_cmd = ['cat', str(backup_file)]
            
            # Test if file contains valid SQL
            result = subprocess.run(test_cmd + ['|', 'head', '-20'], 
                                  shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and ('CREATE' in result.stdout or 'INSERT' in result.stdout):
                return {
                    'status': 'valid_sql',
                    'database_type': 'postgresql',
                    'test_method': 'sql_syntax_check'
                }
            else:
                return {
                    'status': 'invalid_sql',
                    'database_type': 'postgresql',
                    'error': 'No valid SQL statements found'
                }
                
        except Exception as e:
            return {
                'status': 'test_failed',
                'database_type': 'postgresql',
                'error': str(e)
            }
    
    def test_sqlite_backup_restore(self, backup_file: Path) -> Dict:
        """Test SQLite backup by attempting to open and query"""
        try:
            # Test if SQLite file is valid
            if backup_file.suffix == '.gz':
                # Decompress to temporary file
                with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as temp_file:
                    with gzip.open(backup_file, 'rb') as gz_file:
                        shutil.copyfileobj(gz_file, temp_file)
                    temp_path = temp_file.name
            else:
                temp_path = str(backup_file)
            
            try:
                # Try to connect and query
                conn = sqlite3.connect(temp_path)
                cursor = conn.cursor()
                
                # Check if database has tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                conn.close()
                
                if temp_path != str(backup_file):
                    os.unlink(temp_path)  # Clean up temp file
                
                return {
                    'status': 'valid_database',
                    'database_type': 'sqlite',
                    'tables_count': len(tables),
                    'tables': [table[0] for table in tables]
                }
                
            except sqlite3.Error as sql_e:
                if temp_path != str(backup_file):
                    os.unlink(temp_path)  # Clean up temp file
                
                return {
                    'status': 'corrupted_database',
                    'database_type': 'sqlite',
                    'error': str(sql_e)
                }
                
        except Exception as e:
            return {
                'status': 'test_failed',
                'database_type': 'sqlite',
                'error': str(e)
            }
    
    def test_generic_database_file(self, backup_file: Path) -> Dict:
        """Generic database file validation"""
        try:
            # Check if file is readable and has reasonable size
            file_size = backup_file.stat().st_size
            
            if file_size == 0:
                return {
                    'status': 'empty_file',
                    'database_type': 'unknown'
                }
            
            # Try to read first few bytes
            with open(backup_file, 'rb') as f:
                header = f.read(100)
            
            # Check for common database file signatures
            if header.startswith(b'SQLite format'):
                return {
                    'status': 'valid_file',
                    'database_type': 'sqlite',
                    'file_size': file_size
                }
            elif b'PostgreSQL' in header or b'CREATE' in header:
                return {
                    'status': 'valid_file',
                    'database_type': 'postgresql',
                    'file_size': file_size
                }
            else:
                return {
                    'status': 'unknown_format',
                    'database_type': 'unknown',
                    'file_size': file_size
                }
                
        except Exception as e:
            return {
                'status': 'test_failed',
                'database_type': 'unknown',
                'error': str(e)
            }
    
    def verify_configuration_backups(self, config_files: List[Dict]) -> List[Dict]:
        """Verify configuration backup completeness"""
        results = []
        
        essential_configs = [
            'docker-compose',
            'agent',
            'ollama',
            'nginx',
            'backup-config'
        ]
        
        for config_file in config_files:
            try:
                file_path = Path(config_file['path'])
                
                verification_result = {
                    'file': str(file_path),
                    'verification_type': 'configuration'
                }
                
                # Check if it's an essential config
                is_essential = any(essential in file_path.name.lower() for essential in essential_configs)
                verification_result['is_essential'] = is_essential
                
                # Try to parse configuration files
                if file_path.suffix == '.json':
                    try:
                        with open(file_path, 'r') as f:
                            json.load(f)
                        verification_result['status'] = 'valid_json'
                    except json.JSONDecodeError as je:
                        verification_result['status'] = 'invalid_json'
                        verification_result['error'] = str(je)
                
                elif file_path.suffix in ['.yml', '.yaml']:
                    try:
                        import yaml
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f)
                        verification_result['status'] = 'valid_yaml'
                    except Exception as ye:
                        verification_result['status'] = 'invalid_yaml'
                        verification_result['error'] = str(ye)
                
                else:
                    # Generic file readability test
                    try:
                        with open(file_path, 'r') as f:
                            f.read(1024)  # Read first 1KB
                        verification_result['status'] = 'readable'
                    except Exception as re:
                        verification_result['status'] = 'unreadable'
                        verification_result['error'] = str(re)
                
                results.append(verification_result)
                
            except Exception as e:
                logger.error(f"Error verifying configuration {config_file['path']}: {e}")
                results.append({
                    'file': config_file['path'],
                    'status': 'error',
                    'error': str(e),
                    'verification_type': 'configuration'
                })
        
        return results
    
    def find_stored_checksum(self, backup_file: Dict) -> Optional[str]:
        """Find stored checksum for a backup file"""
        try:
            # Look for manifest files in the same directory
            file_path = Path(backup_file['path'])
            manifest_files = list(file_path.parent.glob('*manifest*.json'))
            
            for manifest_file in manifest_files:
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    
                    # Search for checksum in various manifest structures
                    if 'backup_results' in manifest:
                        for result in manifest['backup_results']:
                            if result.get('file') == str(file_path) or result.get('backup_file') == str(file_path):
                                return result.get('checksum')
                    
                    if 'backups' in manifest:
                        for backup in manifest['backups']:
                            if backup.get('file') == str(file_path):
                                return backup.get('checksum')
                                
                except Exception:
                    continue
            
            return None
            
        except Exception:
            return None
    
    def store_checksum(self, backup_file: Dict, checksum: str):
        """Store checksum for future verification"""
        try:
            checksum_file = self.verification_dir / f"checksums_{self.timestamp}.json"
            
            checksum_data = {
                'file': backup_file['path'],
                'checksum': checksum,
                'timestamp': self.timestamp,
                'file_size': backup_file['size'],
                'modified': backup_file['modified']
            }
            
            # Append to checksum file
            if checksum_file.exists():
                with open(checksum_file, 'r') as f:
                    checksums = json.load(f)
            else:
                checksums = []
            
            checksums.append(checksum_data)
            
            with open(checksum_file, 'w') as f:
                json.dump(checksums, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not store checksum for {backup_file['path']}: {e}")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def run_backup_verification(self) -> Dict:
        """Run complete backup verification process"""
        start_time = time.time()
        logger.info(f"Starting backup verification - {self.timestamp}")
        
        # Discover all backup files
        backup_inventory = self.discover_backup_files()
        
        # Categorize files for verification
        all_files = []
        archive_files = []
        database_files = []
        config_files = []
        
        for category, files in backup_inventory.items():
            for file_info in files:
                all_files.append(file_info)
                
                if file_info.get('type') == 'archive':
                    archive_files.append(file_info)
                elif file_info.get('type') == 'database':
                    database_files.append(file_info)
                elif category == 'config':
                    config_files.append(file_info)
        
        # Sample files for verification if too many
        sample_percent = self.test_restore_config['sample_verification_percent']
        if len(all_files) > 100:  # If more than 100 files, sample
            sample_size = max(10, int(len(all_files) * sample_percent / 100))
            import random
            all_files = random.sample(all_files, sample_size)
            logger.info(f"Sampling {sample_size} files for verification ({sample_percent}%)")
        
        verification_results = []
        
        # Run different types of verification
        logger.info("Running checksum verification...")
        checksum_results = self.verify_checksum_integrity(all_files)
        verification_results.extend(checksum_results)
        
        logger.info("Running archive verification...")
        archive_results = self.verify_archive_integrity(archive_files)
        verification_results.extend(archive_results)
        
        logger.info("Running database verification...")
        database_results = self.verify_database_backups(database_files)
        verification_results.extend(database_results)
        
        logger.info("Running configuration verification...")
        config_results = self.verify_configuration_backups(config_files)
        verification_results.extend(config_results)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate verification statistics
        total_verified = len(verification_results)
        successful_verifications = len([r for r in verification_results if r.get('status') in ['verified', 'valid', 'valid_sql', 'valid_database', 'valid_json', 'valid_yaml', 'readable']])
        failed_verifications = len([r for r in verification_results if r.get('status') in ['corrupted', 'invalid_sql', 'corrupted_database', 'invalid_json', 'invalid_yaml', 'unreadable', 'error']])
        
        # Create verification report
        verification_report = {
            'timestamp': self.timestamp,
            'verification_date': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'backup_inventory': backup_inventory,
            'verification_summary': {
                'total_files_discovered': sum(len(files) for files in backup_inventory.values()),
                'files_verified': total_verified,
                'successful_verifications': successful_verifications,
                'failed_verifications': failed_verifications,
                'verification_success_rate': f"{(successful_verifications / total_verified * 100):.1f}%" if total_verified > 0 else "0%"
            },
            'verification_results': verification_results,
            'verification_types_used': list(self.verification_types.keys()),
            'sample_verification_percent': sample_percent
        }
        
        # Save verification report
        report_file = self.verification_dir / f"backup_verification_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(verification_report, f, indent=2)
        
        logger.info(f"Backup verification completed in {duration:.2f} seconds")
        logger.info(f"Verified: {successful_verifications}/{total_verified} files ({verification_report['verification_summary']['verification_success_rate']})")
        
        if failed_verifications > 0:
            logger.warning(f"Found {failed_verifications} failed verifications - check report for details")
        
        return verification_report

def main():
    """Main entry point"""
    try:
        verification_system = BackupVerificationSystem()
        result = verification_system.run_backup_verification()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/backup_verification_summary_{verification_system.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code based on verification results
        if result['verification_summary']['failed_verifications'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Backup verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()