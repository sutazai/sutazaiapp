#!/usr/bin/env python3
"""
SutazAI Offsite Backup Replication System
Implements 3-2-1 backup strategy with offsite replication
"""

import os
import sys
import json
import logging
import datetime
import shutil
import subprocess
import hashlib
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/offsite-backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OffsiteBackupReplicationSystem:
    """Offsite backup replication system implementing 3-2-1 strategy"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/data/backups"):
        self.backup_root = Path(backup_root)
        self.offsite_dir = self.backup_root / 'offsite'
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure offsite directory exists
        self.offsite_dir.mkdir(parents=True, exist_ok=True)
        
        # Load offsite configuration
        self.config = self.load_offsite_config()
        
        # Supported offsite methods
        self.replication_methods = {
            'rsync': self.replicate_via_rsync,
            's3': self.replicate_via_s3,
            'sftp': self.replicate_via_sftp,
            'nfs': self.replicate_via_nfs,
            'local_drive': self.replicate_to_local_drive
        }
        
        # Initialize cloud clients
        self.s3_client = None
        if self.config.get('s3', {}).get('enabled', False):
            self.initialize_s3_client()
    
    def load_offsite_config(self) -> Dict:
        """Load offsite backup configuration"""
        config_file = Path('/opt/sutazaiapp/config/backup-config.json')
        
        default_config = {
            'offsite': {
                'enabled': True,
                'methods': ['local_drive'],
                'retention_days': 90,
                'encryption': True,
                'compression': True,
                'bandwidth_limit': None,
                'schedule': 'daily'
            },
            'local_drive': {
                'enabled': True,
                'destination': '/mnt/backup-drive/sutazai-offsite',
                'mount_check': True
            },
            'rsync': {
                'enabled': False,
                'remote_host': None,
                'remote_path': '/backups/sutazai',
                'ssh_key': '/root/.ssh/backup_key',
                'user': 'backup',
                'options': ['-avz', '--delete', '--compress']
            },
            's3': {
                'enabled': False,
                'bucket': 'sutazai-backups',
                'region': 'us-east-1',
                'storage_class': 'STANDARD_IA',
                'server_side_encryption': True
            },
            'sftp': {
                'enabled': False,
                'host': None,
                'port': 22,
                'username': 'backup',
                'private_key': '/root/.ssh/sftp_key',
                'remote_path': '/backups/sutazai'
            },
            'nfs': {
                'enabled': False,
                'mount_point': '/mnt/nfs-backup',
                'server': None,
                'path': '/exports/backups'
            }
        }
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge configurations
                return {**default_config, **loaded_config}
            else:
                return default_config
        except Exception as e:
            logger.error(f"Error loading offsite config: {e}")
            return default_config
    
    def initialize_s3_client(self):
        """Initialize AWS S3 client"""
        try:
            s3_config = self.config.get('s3', {})
            
            # Check for AWS credentials
            if not (os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY')):
                logger.warning("AWS credentials not found in environment")
                return
            
            self.s3_client = boto3.client(
                's3',
                region_name=s3_config.get('region', 'us-east-1')
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=s3_config.get('bucket', 'sutazai-backups'))
            logger.info("S3 client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def discover_backups_for_offsite(self) -> List[Dict]:
        """Discover backup files that need offsite replication"""
        backup_files = []
        
        # Scan backup directories for recent backups
        backup_categories = ['daily', 'weekly', 'monthly', 'postgres', 'sqlite', 
                           'config', 'agents', 'models', 'monitoring']
        
        for category in backup_categories:
            category_path = self.backup_root / category
            if not category_path.exists():
                continue
            
            for backup_file in category_path.rglob('*'):
                if backup_file.is_file() and not backup_file.name.startswith('.'):
                    try:
                        # Check if file is recent enough for offsite backup
                        file_age_hours = (time.time() - backup_file.stat().st_mtime) / 3600
                        
                        # Include files from last 48 hours for offsite backup
                        if file_age_hours <= 48:
                            file_info = {
                                'path': str(backup_file),
                                'name': backup_file.name,
                                'category': category,
                                'size': backup_file.stat().st_size,
                                'modified': datetime.datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                                'age_hours': file_age_hours
                            }
                            
                            backup_files.append(file_info)
                            
                    except Exception as e:
                        logger.warning(f"Error processing file {backup_file}: {e}")
        
        return backup_files
    
    def encrypt_backup_file(self, file_path: Path, encryption_key: Optional[str] = None) -> Path:
        """Encrypt backup file using GPG"""
        try:
            if not self.config['offsite'].get('encryption', False):
                return file_path
            
            encrypted_file = file_path.with_suffix(file_path.suffix + '.gpg')
            
            # Use symmetric encryption with passphrase
            passphrase = encryption_key or os.environ.get('BACKUP_ENCRYPTION_PASSPHRASE', 'sutazai-backup-key')
            
            cmd = [
                'gpg', '--cipher-algo', 'AES256', '--compress-algo', '1',
                '--s2k-mode', '3', '--s2k-digest-algo', 'SHA512',
                '--s2k-count', '65011712', '--force-mdc', '--quiet',
                '--pinentry-mode', 'loopback', '--passphrase', passphrase,
                '--symmetric', '--output', str(encrypted_file), str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Encrypted backup file: {file_path.name}")
                return encrypted_file
            else:
                logger.error(f"Encryption failed for {file_path}: {result.stderr}")
                return file_path
                
        except Exception as e:
            logger.error(f"Error encrypting file {file_path}: {e}")
            return file_path
    
    def replicate_to_local_drive(self, backup_files: List[Dict]) -> Dict:
        """Replicate backups to local external drive"""
        try:
            local_config = self.config.get('local_drive', {})
            destination = Path(local_config.get('destination', '/mnt/backup-drive/sutazai-offsite'))
            
            # Check if mount point is available
            if local_config.get('mount_check', True):
                if not destination.exists() or not os.path.ismount(str(destination.parent)):
                    return {
                        'method': 'local_drive',
                        'status': 'failed',
                        'error': 'Backup drive not mounted or accessible'
                    }
            
            # Ensure destination directory exists
            destination.mkdir(parents=True, exist_ok=True)
            
            replicated_files = []
            total_size = 0
            
            for backup_file in backup_files:
                try:
                    source_path = Path(backup_file['path'])
                    dest_path = destination / backup_file['category'] / source_path.name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Encrypt if enabled
                    if self.config['offsite'].get('encryption', False):
                        encrypted_source = self.encrypt_backup_file(source_path)
                        dest_path = dest_path.with_suffix(dest_path.suffix + '.gpg')
                        source_to_copy = encrypted_source
                    else:
                        source_to_copy = source_path
                    
                    # Copy file
                    shutil.copy2(source_to_copy, dest_path)
                    
                    # Verify copy
                    if dest_path.exists() and dest_path.stat().st_size == source_to_copy.stat().st_size:
                        replicated_files.append({
                            'source': str(source_path),
                            'destination': str(dest_path),
                            'size': dest_path.stat().st_size,
                            'encrypted': self.config['offsite'].get('encryption', False)
                        })
                        
                        total_size += dest_path.stat().st_size
                    
                    # Clean up temporary encrypted file
                    if self.config['offsite'].get('encryption', False) and encrypted_source != source_path:
                        encrypted_source.unlink()
                        
                except Exception as e:
                    logger.error(f"Error replicating {backup_file['path']} to local drive: {e}")
                    continue
            
            return {
                'method': 'local_drive',
                'status': 'success',
                'files_replicated': len(replicated_files),
                'total_size': total_size,
                'destination': str(destination),
                'replicated_files': replicated_files
            }
            
        except Exception as e:
            logger.error(f"Local drive replication failed: {e}")
            return {
                'method': 'local_drive',
                'status': 'failed',
                'error': str(e)
            }
    
    def replicate_via_rsync(self, backup_files: List[Dict]) -> Dict:
        """Replicate backups via rsync to remote server"""
        try:
            rsync_config = self.config.get('rsync', {})
            
            if not rsync_config.get('remote_host'):
                return {
                    'method': 'rsync',
                    'status': 'failed',
                    'error': 'Remote host not configured'
                }
            
            remote_host = rsync_config['remote_host']
            remote_path = rsync_config.get('remote_path', '/backups/sutazai')
            ssh_key = rsync_config.get('ssh_key', '/root/.ssh/backup_key')
            user = rsync_config.get('user', 'backup')
            options = rsync_config.get('options', ['-avz', '--delete'])
            
            # Build rsync command
            cmd = ['rsync'] + options
            cmd.extend(['-e', f'ssh -i {ssh_key} -o StrictHostKeyChecking=no'])
            
            # Create temporary directory with backup files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy files to temp directory with category structure
                for backup_file in backup_files:
                    source_path = Path(backup_file['path'])
                    category_dir = temp_path / backup_file['category']
                    category_dir.mkdir(parents=True, exist_ok=True)
                    
                    dest_file = category_dir / source_path.name
                    
                    # Encrypt if enabled
                    if self.config['offsite'].get('encryption', False):
                        encrypted_source = self.encrypt_backup_file(source_path)
                        dest_file = dest_file.with_suffix(dest_file.suffix + '.gpg')
                        shutil.copy2(encrypted_source, dest_file)
                        if encrypted_source != source_path:
                            encrypted_source.unlink()
                    else:
                        shutil.copy2(source_path, dest_file)
                
                # Sync temp directory to remote
                cmd.extend([f"{temp_path}/", f"{user}@{remote_host}:{remote_path}/"])
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    return {
                        'method': 'rsync',
                        'status': 'success',
                        'files_replicated': len(backup_files),
                        'remote_host': remote_host,
                        'remote_path': remote_path,
                        'rsync_output': result.stdout
                    }
                else:
                    return {
                        'method': 'rsync',
                        'status': 'failed',
                        'error': result.stderr
                    }
                    
        except Exception as e:
            logger.error(f"Rsync replication failed: {e}")
            return {
                'method': 'rsync',
                'status': 'failed',
                'error': str(e)
            }
    
    def replicate_via_s3(self, backup_files: List[Dict]) -> Dict:
        """Replicate backups to AWS S3"""
        try:
            if not self.s3_client:
                return {
                    'method': 's3',
                    'status': 'failed',
                    'error': 'S3 client not initialized'
                }
            
            s3_config = self.config.get('s3', {})
            bucket = s3_config.get('bucket', 'sutazai-backups')
            storage_class = s3_config.get('storage_class', 'STANDARD_IA')
            
            replicated_files = []
            total_size = 0
            
            for backup_file in backup_files:
                try:
                    source_path = Path(backup_file['path'])
                    s3_key = f"sutazai-backups/{backup_file['category']}/{source_path.name}"
                    
                    # Encrypt if enabled
                    if self.config['offsite'].get('encryption', False):
                        encrypted_source = self.encrypt_backup_file(source_path)
                        s3_key += '.gpg'
                        file_to_upload = encrypted_source
                    else:
                        file_to_upload = source_path
                    
                    # Upload to S3
                    extra_args = {
                        'StorageClass': storage_class
                    }
                    
                    if s3_config.get('server_side_encryption', True):
                        extra_args['ServerSideEncryption'] = 'AES256'
                    
                    self.s3_client.upload_file(
                        str(file_to_upload),
                        bucket,
                        s3_key,
                        ExtraArgs=extra_args
                    )
                    
                    # Verify upload
                    try:
                        response = self.s3_client.head_object(Bucket=bucket, Key=s3_key)
                        if response['ContentLength'] == file_to_upload.stat().st_size:
                            replicated_files.append({
                                'source': str(source_path),
                                's3_key': s3_key,
                                'size': response['ContentLength'],
                                'storage_class': storage_class,
                                'encrypted': self.config['offsite'].get('encryption', False)
                            })
                            
                            total_size += response['ContentLength']
                    except ClientError as e:
                        logger.error(f"S3 verification failed for {s3_key}: {e}")
                    
                    # Clean up temporary encrypted file
                    if self.config['offsite'].get('encryption', False) and encrypted_source != source_path:
                        encrypted_source.unlink()
                        
                except Exception as e:
                    logger.error(f"Error uploading {backup_file['path']} to S3: {e}")
                    continue
            
            return {
                'method': 's3',
                'status': 'success',
                'files_replicated': len(replicated_files),
                'total_size': total_size,
                'bucket': bucket,
                'storage_class': storage_class,
                'replicated_files': replicated_files
            }
            
        except Exception as e:
            logger.error(f"S3 replication failed: {e}")
            return {
                'method': 's3',
                'status': 'failed',
                'error': str(e)
            }
    
    def replicate_via_sftp(self, backup_files: List[Dict]) -> Dict:
        """Replicate backups via SFTP"""
        try:
            import paramiko
            
            sftp_config = self.config.get('sftp', {})
            
            if not sftp_config.get('host'):
                return {
                    'method': 'sftp',
                    'status': 'failed',
                    'error': 'SFTP host not configured'
                }
            
            # Establish SFTP connection
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            private_key = paramiko.RSAKey.from_private_key_file(sftp_config['private_key'])
            
            ssh.connect(
                hostname=sftp_config['host'],
                port=sftp_config.get('port', 22),
                username=sftp_config.get('username', 'backup'),
                pkey=private_key
            )
            
            sftp = ssh.open_sftp()
            
            remote_path = sftp_config.get('remote_path', '/backups/sutazai')
            
            replicated_files = []
            total_size = 0
            
            for backup_file in backup_files:
                try:
                    source_path = Path(backup_file['path'])
                    remote_file_path = f"{remote_path}/{backup_file['category']}/{source_path.name}"
                    
                    # Ensure remote directory exists
                    remote_dir = f"{remote_path}/{backup_file['category']}"
                    try:
                        sftp.mkdir(remote_dir)
                    except IOError:
                        pass  # Directory might already exist
                    
                    # Encrypt if enabled
                    if self.config['offsite'].get('encryption', False):
                        encrypted_source = self.encrypt_backup_file(source_path)
                        remote_file_path += '.gpg'
                        file_to_upload = encrypted_source
                    else:
                        file_to_upload = source_path
                    
                    # Upload file
                    sftp.put(str(file_to_upload), remote_file_path)
                    
                    # Verify upload
                    remote_stat = sftp.stat(remote_file_path)
                    if remote_stat.st_size == file_to_upload.stat().st_size:
                        replicated_files.append({
                            'source': str(source_path),
                            'remote_path': remote_file_path,
                            'size': remote_stat.st_size,
                            'encrypted': self.config['offsite'].get('encryption', False)
                        })
                        
                        total_size += remote_stat.st_size
                    
                    # Clean up temporary encrypted file
                    if self.config['offsite'].get('encryption', False) and encrypted_source != source_path:
                        encrypted_source.unlink()
                        
                except Exception as e:
                    logger.error(f"Error uploading {backup_file['path']} via SFTP: {e}")
                    continue
            
            sftp.close()
            ssh.close()
            
            return {
                'method': 'sftp',
                'status': 'success',
                'files_replicated': len(replicated_files),
                'total_size': total_size,
                'remote_host': sftp_config['host'],
                'remote_path': remote_path,
                'replicated_files': replicated_files
            }
            
        except Exception as e:
            logger.error(f"SFTP replication failed: {e}")
            return {
                'method': 'sftp',
                'status': 'failed',
                'error': str(e)
            }
    
    def replicate_via_nfs(self, backup_files: List[Dict]) -> Dict:
        """Replicate backups to NFS mount"""
        try:
            nfs_config = self.config.get('nfs', {})
            mount_point = Path(nfs_config.get('mount_point', '/mnt/nfs-backup'))
            
            # Check if NFS is mounted
            if not mount_point.exists() or not os.path.ismount(str(mount_point)):
                # Try to mount NFS
                nfs_server = nfs_config.get('server')
                nfs_path = nfs_config.get('path', '/exports/backups')
                
                if nfs_server:
                    mount_cmd = ['mount', '-t', 'nfs', f"{nfs_server}:{nfs_path}", str(mount_point)]
                    result = subprocess.run(mount_cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        return {
                            'method': 'nfs',
                            'status': 'failed',
                            'error': f'NFS mount failed: {result.stderr}'
                        }
                else:
                    return {
                        'method': 'nfs',
                        'status': 'failed',
                        'error': 'NFS server not configured'
                    }
            
            destination = mount_point / 'sutazai-backups'
            destination.mkdir(parents=True, exist_ok=True)
            
            replicated_files = []
            total_size = 0
            
            for backup_file in backup_files:
                try:
                    source_path = Path(backup_file['path'])
                    dest_path = destination / backup_file['category'] / source_path.name
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Encrypt if enabled
                    if self.config['offsite'].get('encryption', False):
                        encrypted_source = self.encrypt_backup_file(source_path)
                        dest_path = dest_path.with_suffix(dest_path.suffix + '.gpg')
                        source_to_copy = encrypted_source
                    else:
                        source_to_copy = source_path
                    
                    # Copy file
                    shutil.copy2(source_to_copy, dest_path)
                    
                    # Verify copy
                    if dest_path.exists() and dest_path.stat().st_size == source_to_copy.stat().st_size:
                        replicated_files.append({
                            'source': str(source_path),
                            'destination': str(dest_path),
                            'size': dest_path.stat().st_size,
                            'encrypted': self.config['offsite'].get('encryption', False)
                        })
                        
                        total_size += dest_path.stat().st_size
                    
                    # Clean up temporary encrypted file
                    if self.config['offsite'].get('encryption', False) and encrypted_source != source_path:
                        encrypted_source.unlink()
                        
                except Exception as e:
                    logger.error(f"Error replicating {backup_file['path']} to NFS: {e}")
                    continue
            
            return {
                'method': 'nfs',
                'status': 'success',
                'files_replicated': len(replicated_files),
                'total_size': total_size,
                'mount_point': str(mount_point),
                'destination': str(destination),
                'replicated_files': replicated_files
            }
            
        except Exception as e:
            logger.error(f"NFS replication failed: {e}")
            return {
                'method': 'nfs',
                'status': 'failed',
                'error': str(e)
            }
    
    def cleanup_old_offsite_backups(self) -> Dict:
        """Clean up old offsite backups based on retention policy"""
        cleanup_results = {
            'cleaned_files': 0,
            'freed_space': 0,
            'methods': {}
        }
        
        retention_days = self.config['offsite'].get('retention_days', 90)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        
        # Clean up each enabled method
        for method in self.config['offsite'].get('methods', []):
            try:
                method_cleanup = self.cleanup_method_specific(method, cutoff_date)
                cleanup_results['methods'][method] = method_cleanup
                
                if method_cleanup.get('status') == 'success':
                    cleanup_results['cleaned_files'] += method_cleanup.get('files_cleaned', 0)
                    cleanup_results['freed_space'] += method_cleanup.get('space_freed', 0)
                    
            except Exception as e:
                logger.error(f"Error cleaning up {method} offsite backups: {e}")
                cleanup_results['methods'][method] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return cleanup_results
    
    def cleanup_method_specific(self, method: str, cutoff_date: datetime.datetime) -> Dict:
        """Clean up old backups for specific method"""
        if method == 'local_drive':
            return self.cleanup_local_drive_backups(cutoff_date)
        elif method == 's3':
            return self.cleanup_s3_backups(cutoff_date)
        else:
            return {
                'status': 'skipped',
                'reason': f'Cleanup not implemented for {method}'
            }
    
    def cleanup_local_drive_backups(self, cutoff_date: datetime.datetime) -> Dict:
        """Clean up old local drive backups"""
        try:
            local_config = self.config.get('local_drive', {})
            destination = Path(local_config.get('destination', '/mnt/backup-drive/sutazai-offsite'))
            
            if not destination.exists():
                return {
                    'status': 'skipped',
                    'reason': 'Destination not found'
                }
            
            files_cleaned = 0
            space_freed = 0
            
            for backup_file in destination.rglob('*'):
                if backup_file.is_file():
                    file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
                    
                    if file_time < cutoff_date:
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        files_cleaned += 1
                        space_freed += file_size
                        logger.info(f"Cleaned up old offsite backup: {backup_file}")
            
            return {
                'status': 'success',
                'files_cleaned': files_cleaned,
                'space_freed': space_freed
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def cleanup_s3_backups(self, cutoff_date: datetime.datetime) -> Dict:
        """Clean up old S3 backups"""
        try:
            if not self.s3_client:
                return {
                    'status': 'skipped',
                    'reason': 'S3 client not available'
                }
            
            s3_config = self.config.get('s3', {})
            bucket = s3_config.get('bucket', 'sutazai-backups')
            
            files_cleaned = 0
            space_freed = 0
            
            # List objects in bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix='sutazai-backups/')
            
            objects_to_delete = []
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        last_modified = obj['LastModified'].replace(tzinfo=None)
                        
                        if last_modified < cutoff_date:
                            objects_to_delete.append({'Key': obj['Key']})
                            space_freed += obj['Size']
                            files_cleaned += 1
            
            # Delete old objects in batches
            if objects_to_delete:
                for i in range(0, len(objects_to_delete), 1000):  # S3 limit is 1000 objects per request
                    batch = objects_to_delete[i:i+1000]
                    
                    self.s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': batch}
                    )
            
            return {
                'status': 'success',
                'files_cleaned': files_cleaned,
                'space_freed': space_freed
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_offsite_replication(self) -> Dict:
        """Run complete offsite backup replication"""
        start_time = time.time()
        logger.info(f"Starting offsite backup replication - {self.timestamp}")
        
        # Check if offsite backup is enabled
        if not self.config['offsite'].get('enabled', False):
            return {
                'timestamp': self.timestamp,
                'status': 'disabled',
                'message': 'Offsite backup is disabled in configuration'
            }
        
        # Discover backups for offsite replication
        backup_files = self.discover_backups_for_offsite()
        
        if not backup_files:
            logger.info("No recent backup files found for offsite replication")
            return {
                'timestamp': self.timestamp,
                'status': 'no_backups',
                'message': 'No backup files found for offsite replication'
            }
        
        logger.info(f"Found {len(backup_files)} backup files for offsite replication")
        
        # Run replication for each configured method
        replication_results = []
        enabled_methods = self.config['offsite'].get('methods', ['local_drive'])
        
        for method in enabled_methods:
            if method in self.replication_methods:
                method_config = self.config.get(method, {})
                
                if method_config.get('enabled', True):
                    logger.info(f"Starting {method} replication...")
                    
                    try:
                        result = self.replication_methods[method](backup_files)
                        replication_results.append(result)
                        
                        if result.get('status') == 'success':
                            logger.info(f"{method} replication completed successfully")
                        else:
                            logger.error(f"{method} replication failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        logger.error(f"Error during {method} replication: {e}")
                        replication_results.append({
                            'method': method,
                            'status': 'failed',
                            'error': str(e)
                        })
        
        # Clean up old offsite backups
        cleanup_results = self.cleanup_old_offsite_backups()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate summary statistics
        successful_replications = len([r for r in replication_results if r.get('status') == 'success'])
        failed_replications = len([r for r in replication_results if r.get('status') == 'failed'])
        total_files_replicated = sum(r.get('files_replicated', 0) for r in replication_results if r.get('status') == 'success')
        
        # Create final report
        offsite_report = {
            'timestamp': self.timestamp,
            'replication_date': datetime.datetime.now().isoformat(),
            'duration_seconds': duration,
            'backup_files_discovered': len(backup_files),
            'replication_summary': {
                'methods_attempted': len(replication_results),
                'successful_replications': successful_replications,
                'failed_replications': failed_replications,
                'total_files_replicated': total_files_replicated
            },
            'replication_results': replication_results,
            'cleanup_results': cleanup_results,
            'configuration': {
                'enabled_methods': enabled_methods,
                'encryption_enabled': self.config['offsite'].get('encryption', False),
                'retention_days': self.config['offsite'].get('retention_days', 90)
            }
        }
        
        # Save report
        report_file = self.offsite_dir / f"offsite_replication_report_{self.timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(offsite_report, f, indent=2)
        
        logger.info(f"Offsite replication completed in {duration:.2f} seconds")
        logger.info(f"Successfully replicated to {successful_replications}/{len(replication_results)} methods")
        
        return offsite_report

def main():
    """Main entry point"""
    try:
        offsite_system = OffsiteBackupReplicationSystem()
        result = offsite_system.run_offsite_replication()
        
        # Write summary to log
        summary_file = f"/opt/sutazaiapp/logs/offsite_replication_summary_{offsite_system.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result.get('replication_summary', {}).get('failed_replications', 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Offsite backup replication failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()