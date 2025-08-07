#!/usr/bin/env python3
"""
SutazAI Restore Testing System
Automated testing of backup restoration to ensure data recoverability
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import gzip
import subprocess
import tempfile
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psycopg2
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/restore-testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RestoreTestingSystem:
    """Automated backup restore testing system"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.restore_test_dir = self.backup_root / 'restore_tests'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure restore test directory exists
        self.restore_test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test environment settings
        self.test_config = {
            'test_database_prefix': 'sutazai_restore_test',
            'test_temp_dir': '/tmp/sutazai_restore_tests',
            'max_test_duration_minutes': 30,
            'sample_restore_percent': 20,  # Test 20% of backups
            'docker_test_network': 'sutazai_restore_test'
        }
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Could not initialize Docker client: {e}")
            self.docker_client = None
        
        # Restore test types
        self.restore_tests = {
            'database': self.test_database_restore,
            'configuration': self.test_configuration_restore,
            'archive': self.test_archive_restore,
            'agent_state': self.test_agent_state_restore,
            'models': self.test_model_restore,
            'logs': self.test_log_restore
        }
    
    def discover_backups_for_testing(self) -> Dict[str, List[Dict]]:
        """Discover backup files suitable for restore testing"""
        test_candidates = {}
        
        # Categories of backups to test
        backup_categories = {
            'database': ['postgres', 'sqlite'],
            'configuration': ['config'],
            'archive': ['agents', 'monitoring'],
            'models': ['models'],
            'logs': ['logs']
        }
        
        for test_type, categories in backup_categories.items():
            test_candidates[test_type] = []
            
            for category in categories:
                category_path = self.backup_root / category
                if not category_path.exists():
                    continue
                
                # Find recent backup files
                for backup_file in category_path.rglob('*'):
                    if backup_file.is_file() and not backup_file.name.startswith('.'):
                        try:
                            # Only test recent backups (last 7 days)
                            file_age_days = (time.time() - backup_file.stat().st_mtime) / (24 * 3600)
                            
                            if file_age_days <= 7:
                                file_info = {
                                    'path': str(backup_file),
                                    'name': backup_file.name,
                                    'category': category,
                                    'size': backup_file.stat().st_size,
                                    'modified': datetime.datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                                    'age_days': file_age_days,
                                    'test_type': test_type
                                }
                                
                                test_candidates[test_type].append(file_info)
                                
                        except Exception as e:
                            logger.warning(f"Error processing backup file {backup_file}: {e}")
        
        # Sample files for testing if too many
        sample_percent = self.test_config['sample_restore_percent']
        
        for test_type, files in test_candidates.items():
            if len(files) > 10:  # If more than 10 files, sample
                import random
                sample_size = max(2, int(len(files) * sample_percent / 100))
                test_candidates[test_type] = random.sample(files, sample_size)
                logger.info(f"Sampling {sample_size} {test_type} backups for testing")
        
        return test_candidates
    
    def setup_test_environment(self) -> Dict:
        """Set up isolated test environment"""
        try:
            test_env = {
                'temp_dir': Path(self.test_config['test_temp_dir']) / self.timestamp,
                'postgres_test_db': f"{self.test_config['test_database_prefix']}_{self.timestamp}",
                'docker_network': None
            }
            
            # Create temporary directory
            test_env['temp_dir'].mkdir(parents=True, exist_ok=True)
            
            # Create Docker test network if Docker is available
            if self.docker_client:
                try:
                    network_name = f"{self.test_config['docker_test_network']}_{self.timestamp}"
                    network = self.docker_client.networks.create(
                        network_name,
                        driver="bridge",
                        labels={"purpose": "sutazai-restore-testing"}
                    )
                    test_env['docker_network'] = network
                    logger.info(f"Created Docker test network: {network_name}")
                except Exception as e:
                    logger.warning(f"Could not create Docker test network: {e}")
            
            return {
                'status': 'success',
                'test_env': test_env
            }
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def cleanup_test_environment(self, test_env: Dict):
        """Clean up test environment"""
        try:
            # Remove temporary directory
            if 'temp_dir' in test_env and test_env['temp_dir'].exists():
                shutil.rmtree(test_env['temp_dir'], ignore_errors=True)
                logger.info("Cleaned up temporary test directory")
            
            # Remove Docker test network
            if 'docker_network' in test_env and test_env['docker_network']:
                try:
                    test_env['docker_network'].remove()
                    logger.info("Removed Docker test network")
                except Exception as e:
                    logger.warning(f"Could not remove Docker test network: {e}")
            
            # Clean up test databases
            if 'postgres_test_db' in test_env:
                self.cleanup_test_database(test_env['postgres_test_db'])
                
        except Exception as e:
            logger.error(f"Error cleaning up test environment: {e}")
    
    def test_database_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test database backup restoration"""
        results = []
        
        for backup_file in backup_files:
            try:
                file_path = Path(backup_file['path'])
                
                logger.info(f"Testing database restore: {file_path.name}")
                
                if 'postgres' in backup_file['category'] or file_path.suffix == '.sql':
                    result = self.test_postgres_restore(file_path, test_env)
                elif 'sqlite' in backup_file['category'] or file_path.name.endswith('.sqlite'):
                    result = self.test_sqlite_restore(file_path, test_env)
                else:
                    result = {
                        'status': 'skipped',
                        'reason': 'Unknown database type'
                    }
                
                result.update({
                    'backup_file': str(file_path),
                    'test_type': 'database',
                    'file_size': backup_file['size']
                })
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error testing database restore for {backup_file['path']}: {e}")
                results.append({
                    'backup_file': backup_file['path'],
                    'test_type': 'database',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def test_postgres_restore(self, backup_file: Path, test_env: Dict) -> Dict:
        """Test PostgreSQL backup restoration"""
        try:
            test_db_name = f"{test_env['postgres_test_db']}_pg"
            
            # Create test database
            create_db_result = self.create_test_postgres_database(test_db_name)
            if create_db_result['status'] != 'success':
                return create_db_result
            
            # Restore backup to test database
            restore_start = time.time()
            
            if backup_file.suffix == '.gz':
                restore_cmd = f"zcat {backup_file} | psql -h localhost -U postgres -d {test_db_name}"
            else:
                restore_cmd = f"psql -h localhost -U postgres -d {test_db_name} -f {backup_file}"
            
            env = os.environ.copy()
            env['PGPASSWORD'] = os.environ.get('POSTGRES_PASSWORD', 'postgres')
            
            result = subprocess.run(
                restore_cmd,
                shell=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            restore_duration = time.time() - restore_start
            
            if result.returncode == 0:
                # Verify restoration by checking table count
                verification_result = self.verify_postgres_restoration(test_db_name)
                
                # Clean up test database
                self.cleanup_test_database(test_db_name)
                
                return {
                    'status': 'success',
                    'database_type': 'postgresql',
                    'restore_duration_seconds': restore_duration,
                    'verification': verification_result
                }
            else:
                # Clean up test database
                self.cleanup_test_database(test_db_name)
                
                return {
                    'status': 'failed',
                    'database_type': 'postgresql',
                    'error': result.stderr,
                    'restore_duration_seconds': restore_duration
                }
                
        except subprocess.TimeoutExpired:
            self.cleanup_test_database(test_db_name)
            return {
                'status': 'failed',
                'database_type': 'postgresql',
                'error': 'Restore operation timed out'
            }
        except Exception as e:
            return {
                'status': 'error',
                'database_type': 'postgresql',
                'error': str(e)
            }
    
    def test_sqlite_restore(self, backup_file: Path, test_env: Dict) -> Dict:
        """Test SQLite backup restoration"""
        try:
            test_db_path = test_env['temp_dir'] / f"test_sqlite_{int(time.time())}.db"
            
            restore_start = time.time()
            
            # Handle compressed backups
            if backup_file.suffix == '.gz':
                with gzip.open(backup_file, 'rb') as gz_file:
                    with open(test_db_path, 'wb') as test_file:
                        shutil.copyfileobj(gz_file, test_file)
            else:
                shutil.copy2(backup_file, test_db_path)
            
            restore_duration = time.time() - restore_start
            
            # Verify SQLite database
            verification_result = self.verify_sqlite_restoration(test_db_path)
            
            # Clean up test file
            if test_db_path.exists():
                test_db_path.unlink()
            
            if verification_result['status'] == 'success':
                return {
                    'status': 'success',
                    'database_type': 'sqlite',
                    'restore_duration_seconds': restore_duration,
                    'verification': verification_result
                }
            else:
                return {
                    'status': 'failed',
                    'database_type': 'sqlite',
                    'restore_duration_seconds': restore_duration,
                    'verification': verification_result
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'database_type': 'sqlite',
                'error': str(e)
            }
    
    def test_configuration_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test configuration backup restoration"""
        results = []
        
        for backup_file in backup_files:
            try:
                file_path = Path(backup_file['path'])
                
                logger.info(f"Testing configuration restore: {file_path.name}")
                
                restore_start = time.time()
                
                # Extract configuration to test directory
                test_extract_dir = test_env['temp_dir'] / 'config_test' / file_path.stem
                test_extract_dir.mkdir(parents=True, exist_ok=True)
                
                if file_path.suffix == '.gz' and '.tar' in file_path.name:
                    # Extract tar.gz archive
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(test_extract_dir)
                elif file_path.suffix in ['.json', '.yaml', '.yml']:
                    # Copy single config file
                    shutil.copy2(file_path, test_extract_dir)
                else:
                    # Try to copy as regular file
                    shutil.copy2(file_path, test_extract_dir)
                
                restore_duration = time.time() - restore_start
                
                # Verify configuration files
                verification_result = self.verify_configuration_restoration(test_extract_dir)
                
                results.append({
                    'backup_file': str(file_path),
                    'test_type': 'configuration',
                    'status': 'success' if verification_result['status'] == 'success' else 'failed',
                    'restore_duration_seconds': restore_duration,
                    'verification': verification_result,
                    'file_size': backup_file['size']
                })
                
            except Exception as e:
                logger.error(f"Error testing configuration restore for {backup_file['path']}: {e}")
                results.append({
                    'backup_file': backup_file['path'],
                    'test_type': 'configuration',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def test_archive_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test archive backup restoration"""
        results = []
        
        for backup_file in backup_files:
            try:
                file_path = Path(backup_file['path'])
                
                logger.info(f"Testing archive restore: {file_path.name}")
                
                restore_start = time.time()
                
                # Extract archive to test directory
                test_extract_dir = test_env['temp_dir'] / 'archive_test' / file_path.stem
                test_extract_dir.mkdir(parents=True, exist_ok=True)
                
                extraction_success = False
                
                if file_path.suffix == '.gz':
                    if '.tar' in file_path.name:
                        # tar.gz archive
                        with tarfile.open(file_path, 'r:gz') as tar:
                            tar.extractall(test_extract_dir)
                            extraction_success = True
                    else:
                        # gzip compressed file
                        with gzip.open(file_path, 'rb') as gz_file:
                            extracted_file = test_extract_dir / file_path.stem
                            with open(extracted_file, 'wb') as out_file:
                                shutil.copyfileobj(gz_file, out_file)
                            extraction_success = True
                elif '.tar' in file_path.name:
                    # tar archive
                    with tarfile.open(file_path, 'r') as tar:
                        tar.extractall(test_extract_dir)
                        extraction_success = True
                else:
                    # Regular file copy
                    shutil.copy2(file_path, test_extract_dir)
                    extraction_success = True
                
                restore_duration = time.time() - restore_start
                
                if extraction_success:
                    # Verify extracted contents
                    verification_result = self.verify_archive_restoration(test_extract_dir)
                    
                    results.append({
                        'backup_file': str(file_path),
                        'test_type': 'archive',
                        'status': 'success',
                        'restore_duration_seconds': restore_duration,
                        'verification': verification_result,
                        'file_size': backup_file['size']
                    })
                else:
                    results.append({
                        'backup_file': str(file_path),
                        'test_type': 'archive',
                        'status': 'failed',
                        'error': 'Could not extract archive',
                        'restore_duration_seconds': restore_duration
                    })
                
            except Exception as e:
                logger.error(f"Error testing archive restore for {backup_file['path']}: {e}")
                results.append({
                    'backup_file': backup_file['path'],
                    'test_type': 'archive',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def test_agent_state_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test agent state backup restoration"""
        results = []
        
        # Agent state testing is similar to archive testing
        return self.test_archive_restore(backup_files, test_env)
    
    def test_model_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test model backup restoration"""
        results = []
        
        for backup_file in backup_files:
            try:
                file_path = Path(backup_file['path'])
                
                logger.info(f"Testing model restore: {file_path.name}")
                
                restore_start = time.time()
                
                # Extract model to test directory
                test_extract_dir = test_env['temp_dir'] / 'model_test' / file_path.stem
                test_extract_dir.mkdir(parents=True, exist_ok=True)
                
                # Handle different model backup formats
                if '.tar.gz' in file_path.name:
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(test_extract_dir)
                elif file_path.suffix == '.gguf':
                    # Ollama model file
                    shutil.copy2(file_path, test_extract_dir)
                else:
                    # Generic file copy
                    shutil.copy2(file_path, test_extract_dir)
                
                restore_duration = time.time() - restore_start
                
                # Basic verification - check if files exist and are not empty
                verification_result = self.verify_model_restoration(test_extract_dir)
                
                results.append({
                    'backup_file': str(file_path),
                    'test_type': 'models',
                    'status': 'success' if verification_result['status'] == 'success' else 'failed',
                    'restore_duration_seconds': restore_duration,
                    'verification': verification_result,
                    'file_size': backup_file['size']
                })
                
            except Exception as e:
                logger.error(f"Error testing model restore for {backup_file['path']}: {e}")
                results.append({
                    'backup_file': backup_file['path'],
                    'test_type': 'models',
                    'status': 'error',
                    'error': str(e)
                })
        
        return results
    
    def test_log_restore(self, backup_files: List[Dict], test_env: Dict) -> List[Dict]:
        """Test log backup restoration"""
        results = []
        
        # Log restoration testing is similar to archive testing
        return self.test_archive_restore(backup_files, test_env)
    
    def create_test_postgres_database(self, db_name: str) -> Dict:
        """Create a test PostgreSQL database"""
        try:
            # Connect to default database to create test database
            conn = psycopg2.connect(
                host='localhost',
                database='postgres',
                user='postgres',
                password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
            )
            conn.autocommit = True
            
            cursor = conn.cursor()
            
            # Drop database if it exists
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            
            # Create test database
            cursor.execute(f"CREATE DATABASE {db_name}")
            
            cursor.close()
            conn.close()
            
            return {
                'status': 'success',
                'database': db_name
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def cleanup_test_database(self, db_name: str):
        """Clean up test PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host='localhost',
                database='postgres',
                user='postgres',
                password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
            )
            conn.autocommit = True
            
            cursor = conn.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            cursor.close()
            conn.close()
            
            logger.info(f"Cleaned up test database: {db_name}")
            
        except Exception as e:
            logger.warning(f"Could not clean up test database {db_name}: {e}")
    
    def verify_postgres_restoration(self, db_name: str) -> Dict:
        """Verify PostgreSQL database restoration"""
        try:
            conn = psycopg2.connect(
                host='localhost',
                database=db_name,
                user='postgres',
                password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
            )
            
            cursor = conn.cursor()
            
            # Check table count
            cursor.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            table_count = cursor.fetchone()[0]
            
            # Check if we can query tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                LIMIT 5
            """)
            sample_tables = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return {
                'status': 'success',
                'table_count': table_count,
                'sample_tables': sample_tables
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def verify_sqlite_restoration(self, db_path: Path) -> Dict:
        """Verify SQLite database restoration"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 5")
            sample_tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                'status': 'success',
                'table_count': table_count,
                'sample_tables': sample_tables
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def verify_configuration_restoration(self, config_dir: Path) -> Dict:
        """Verify configuration file restoration"""
        try:
            files_found = list(config_dir.rglob('*'))
            file_count = len([f for f in files_found if f.is_file()])
            
            # Check for essential config files
            essential_files = []
            for file_path in files_found:
                if file_path.is_file():
                    if any(keyword in file_path.name.lower() for keyword in 
                          ['docker-compose', 'config', 'env', '.json', '.yaml']):
                        essential_files.append(file_path.name)
            
            return {
                'status': 'success',
                'files_extracted': file_count,
                'essential_files': essential_files[:10]  # Limit to first 10
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def verify_archive_restoration(self, extract_dir: Path) -> Dict:
        """Verify archive extraction"""
        try:
            files_found = list(extract_dir.rglob('*'))
            file_count = len([f for f in files_found if f.is_file()])
            
            # Calculate total extracted size
            total_size = sum(f.stat().st_size for f in files_found if f.is_file())
            
            return {
                'status': 'success',
                'files_extracted': file_count,
                'total_size': total_size
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def verify_model_restoration(self, model_dir: Path) -> Dict:
        """Verify model file restoration"""
        try:
            model_files = list(model_dir.rglob('*'))
            file_count = len([f for f in model_files if f.is_file()])
            
            # Check for model-specific files
            model_types = []
            for file_path in model_files:
                if file_path.is_file():
                    if file_path.suffix in ['.gguf', '.bin', '.safetensors', '.pt', '.pth']:
                        model_types.append(file_path.suffix)
            
            return {
                'status': 'success',
                'files_extracted': file_count,
                'model_file_types': list(set(model_types))
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_restore_testing(self) -> Dict:
        """Run complete restore testing process"""
        start_time = time.time()
        logger.info(f"Starting restore testing - {self.timestamp}")
        
        # Set up test environment
        env_setup = self.setup_test_environment()
        if env_setup['status'] != 'success':
            return {
                'timestamp': self.timestamp,
                'status': 'failed',
                'error': 'Could not set up test environment'
            }
        
        test_env = env_setup['test_env']
        
        try:
            # Discover backups for testing
            test_candidates = self.discover_backups_for_testing()
            
            total_candidates = sum(len(files) for files in test_candidates.values())
            logger.info(f"Found {total_candidates} backup files for restore testing")
            
            if total_candidates == 0:
                return {
                    'timestamp': self.timestamp,
                    'status': 'no_backups',
                    'message': 'No backup files found for testing'
                }
            
            # Run restore tests for each type
            all_test_results = []
            
            for test_type, backup_files in test_candidates.items():
                if not backup_files:
                    continue
                
                logger.info(f"Running {test_type} restore tests on {len(backup_files)} files")
                
                try:
                    test_function = self.restore_tests.get(test_type)
                    if test_function:
                        test_results = test_function(backup_files, test_env)
                        all_test_results.extend(test_results)
                    else:
                        logger.warning(f"No test function available for {test_type}")
                        
                except Exception as e:
                    logger.error(f"Error running {test_type} restore tests: {e}")
                    all_test_results.append({
                        'test_type': test_type,
                        'status': 'error',
                        'error': str(e)
                    })
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate test statistics
            total_tests = len(all_test_results)
            successful_tests = len([r for r in all_test_results if r.get('status') == 'success'])
            failed_tests = len([r for r in all_test_results if r.get('status') == 'failed'])
            error_tests = len([r for r in all_test_results if r.get('status') == 'error'])
            
            # Create test report
            test_report = {
                'timestamp': self.timestamp,
                'test_date': datetime.datetime.now().isoformat(),
                'duration_seconds': duration,
                'test_summary': {
                    'total_backups_discovered': total_candidates,
                    'total_tests_run': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'error_tests': error_tests,
                    'success_rate': f"{(successful_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
                },
                'test_candidates': test_candidates,
                'test_results': all_test_results,
                'test_environment': {
                    'temp_dir': str(test_env['temp_dir']),
                    'docker_available': self.docker_client is not None,
                    'sample_percent': self.test_config['sample_restore_percent']
                }
            }
            
            # Save test report
            report_file = self.restore_test_dir / f"restore_test_report_{self.timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(test_report, f, indent=2)
            
            logger.info(f"Restore testing completed in {duration:.2f} seconds")
            logger.info(f"Test results: {successful_tests} successful, {failed_tests} failed, {error_tests} errors")
            
            if failed_tests > 0 or error_tests > 0:
                logger.warning(f"Some restore tests failed - check report for details")
            
            return test_report
            
        finally:
            # Always clean up test environment
            self.cleanup_test_environment(test_env)

def main():
    """Main entry point"""
    try:
        restore_testing = RestoreTestingSystem()
        result = restore_testing.run_restore_testing()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/restore_testing_summary_{restore_testing.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code based on test results
        if result.get('test_summary', {}).get('failed_tests', 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Restore testing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()