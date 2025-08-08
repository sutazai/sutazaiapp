#!/usr/bin/env python3
"""
SutazAI Configuration Backup System
Backs up all configuration files, Docker compose files, and environment settings
"""

import os
import sys
import json
import logging
import datetime
import shutil
import tarfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/config-backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigBackupSystem:
    """Configuration backup system for SutazAI"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.config_backup_dir = self.backup_root / 'config'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure backup directory exists
        self.config_backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Define configuration paths to backup
        self.config_paths = {
            'main_config': [
                '/opt/sutazaiapp/config/',
                '/opt/sutazaiapp/agents/configs/',
                '/opt/sutazaiapp/secrets/',
                '/opt/sutazaiapp/secrets_secure/'
            ],
            'docker_configs': [
                '/opt/sutazaiapp/docker-compose*.yml',
                '/opt/sutazaiapp/docker/',
                '/opt/sutazaiapp/configs/',
                '/opt/sutazaiapp/nginx/',
                '/opt/sutazaiapp/deployment/'
            ],
            'service_configs': [
                '/opt/sutazaiapp/monitoring/',
                '/opt/sutazaiapp/self-healing/',
                '/opt/sutazaiapp/security/',
                '/opt/sutazaiapp/disaster-recovery/'
            ],
            'environment_files': [
                '/opt/sutazaiapp/.env*',
                '/opt/sutazaiapp/pyproject.toml',
                '/opt/sutazaiapp/requirements*.txt',
                '/opt/sutazaiapp/pytest.ini'
            ]
        }
    
    def backup_config_directory(self, source_path: str, category: str) -> Dict:
        """Backup a configuration directory or file"""
        source = Path(source_path)
        
        if not source.exists():
            logger.warning(f"Configuration path not found: {source_path}")
            return {
                'source': source_path,
                'status': 'skipped',
                'reason': 'path_not_found'
            }
        
        try:
            # Create category subdirectory
            category_dir = self.config_backup_dir / category / self.timestamp
            category_dir.mkdir(parents=True, exist_ok=True)
            
            if source.is_file():
                # Handle single file
                dest_file = category_dir / source.name
                shutil.copy2(source, dest_file)
                
                # Calculate checksum
                checksum = self.calculate_checksum(dest_file)
                
                return {
                    'source': source_path,
                    'destination': str(dest_file),
                    'type': 'file',
                    'size': dest_file.stat().st_size,
                    'checksum': checksum,
                    'status': 'success'
                }
            
            elif source.is_dir():
                # Handle directory
                dest_dir = category_dir / source.name
                shutil.copytree(source, dest_dir, dirs_exist_ok=True)
                
                # Calculate directory checksum (sum of all file checksums)
                total_size = 0
                checksums = []
                
                for file_path in dest_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        checksums.append(self.calculate_checksum(file_path))
                
                # Create combined checksum
                combined_checksum = hashlib.sha256(''.join(sorted(checksums)).encode()).hexdigest()
                
                return {
                    'source': source_path,
                    'destination': str(dest_dir),
                    'type': 'directory',
                    'size': total_size,
                    'files_count': len(checksums),
                    'checksum': combined_checksum,
                    'status': 'success'
                }
            
        except Exception as e:
            logger.error(f"Error backing up {source_path}: {e}")
            return {
                'source': source_path,
                'status': 'failed',
                'error': str(e)
            }
    
    def backup_docker_compose_files(self) -> List[Dict]:
        """Backup all Docker Compose files"""
        results = []
        
        # Find all docker-compose files
        docker_compose_files = list(Path('/opt/sutazaiapp').glob('docker-compose*.yml'))
        docker_compose_files.extend(list(Path('/opt/sutazaiapp').glob('docker-compose*.yaml')))
        
        for compose_file in docker_compose_files:
            result = self.backup_config_directory(str(compose_file), 'docker_configs')
            results.append(result)
        
        return results
    
    def backup_environment_variables(self) -> Dict:
        """Backup environment variables and system configuration"""
        try:
            env_backup_file = self.config_backup_dir / 'environment' / self.timestamp / 'environment_snapshot.json'
            env_backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Capture relevant environment variables (excluding sensitive data)
            safe_env_vars = {}
            sensitive_patterns = ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'PRIVATE']
            
            for key, value in os.environ.items():
                if not any(pattern in key.upper() for pattern in sensitive_patterns):
                    safe_env_vars[key] = value
                else:
                    safe_env_vars[key] = f"<REDACTED_{len(value)}_CHARS>"
            
            # Add system information
            system_info = {
                'timestamp': self.timestamp,
                'hostname': os.uname().nodename,
                'system': os.uname().sysname,
                'release': os.uname().release,
                'python_version': sys.version,
                'working_directory': os.getcwd(),
                'user': os.environ.get('USER', 'unknown'),
                'environment_variables': safe_env_vars
            }
            
            with open(env_backup_file, 'w') as f:
                json.dump(system_info, f, indent=2)
            
            checksum = self.calculate_checksum(env_backup_file)
            
            return {
                'type': 'environment',
                'destination': str(env_backup_file),
                'size': env_backup_file.stat().st_size,
                'checksum': checksum,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error backing up environment: {e}")
            return {
                'type': 'environment',
                'status': 'failed',
                'error': str(e)
            }
    
    def create_config_archive(self, backup_results: List[Dict]) -> Dict:
        """Create compressed archive of all configuration backups"""
        try:
            archive_file = self.config_backup_dir / f"config_archive_{self.timestamp}.tar.gz"
            
            with tarfile.open(archive_file, 'w:gz') as tar:
                # Add all backed up configurations
                for category in self.config_paths.keys():
                    category_path = self.config_backup_dir / category / self.timestamp
                    if category_path.exists():
                        tar.add(category_path, arcname=f"{category}_{self.timestamp}")
                
                # Add environment backup
                env_path = self.config_backup_dir / 'environment' / self.timestamp
                if env_path.exists():
                    tar.add(env_path, arcname=f"environment_{self.timestamp}")
            
            checksum = self.calculate_checksum(archive_file)
            
            return {
                'type': 'archive',
                'file': str(archive_file),
                'size': archive_file.stat().st_size,
                'checksum': checksum,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error creating configuration archive: {e}")
            return {
                'type': 'archive',
                'status': 'failed',
                'error': str(e)
            }
    
    # calculate_checksum centralized in scripts.lib.file_utils
    
    def run_config_backup(self) -> Dict:
        """Run complete configuration backup"""
        logger.info(f"Starting configuration backup - {self.timestamp}")
        
        all_results = []
        
        # Backup each category of configurations
        for category, paths in self.config_paths.items():
            logger.info(f"Backing up {category} configurations")
            
            for path_pattern in paths:
                if '*' in path_pattern:
                    # Handle glob patterns
                    matching_paths = list(Path('/').glob(path_pattern.lstrip('/')))
                    for matching_path in matching_paths:
                        result = self.backup_config_directory(str(matching_path), category)
                        all_results.append(result)
                else:
                    # Handle direct paths
                    result = self.backup_config_directory(path_pattern, category)
                    all_results.append(result)
        
        # Backup environment variables
        env_result = self.backup_environment_variables()
        all_results.append(env_result)
        
        # Create compressed archive
        archive_result = self.create_config_archive(all_results)
        all_results.append(archive_result)
        
        # Create manifest
        manifest = {
            'timestamp': self.timestamp,
            'backup_date': datetime.datetime.now().isoformat(),
            'total_backups': len(all_results),
            'successful_backups': len([r for r in all_results if r.get('status') == 'success']),
            'failed_backups': len([r for r in all_results if r.get('status') == 'failed']),
            'backup_results': all_results
        }
        
        manifest_file = self.config_backup_dir / f"config_manifest_{self.timestamp}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Configuration backup completed")
        logger.info(f"Successful: {manifest['successful_backups']}, Failed: {manifest['failed_backups']}")
        
        return manifest

def main():
    """Main entry point"""
    try:
        config_backup = ConfigBackupSystem()
        result = config_backup.run_config_backup()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/config_backup_summary_{config_backup.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result['failed_backups'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Configuration backup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
