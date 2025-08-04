#!/usr/bin/env python3
"""
SutazAI Backup Coordinator
Comprehensive backup system for all critical data and system state preservation.
"""

import os
import sys
import json
import time
import hashlib
import logging
import schedule
import sqlite3
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tarfile
import shutil
import psutil
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/backup-coordinator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BackupJob:
    """Backup job configuration"""
    name: str
    source_path: str
    backup_type: str  # 'file', 'database', 'container', 'state'
    frequency: str  # 'hourly', 'daily', 'weekly', 'monthly'
    retention_days: int
    encryption: bool = True
    compression: bool = True
    incremental: bool = True
    priority: int = 1  # 1=critical, 2=important, 3=standard
    pre_backup_commands: List[str] = None
    post_backup_commands: List[str] = None
    verification_commands: List[str] = None

@dataclass
class BackupMetadata:
    """Backup metadata for tracking and recovery"""
    backup_id: str
    job_name: str
    timestamp: datetime
    size_bytes: int
    checksum_sha256: str
    incremental: bool
    base_backup_id: Optional[str]
    source_path: str
    backup_path: str
    encryption_key_id: str
    verification_status: str
    recovery_instructions: str

class BackupCoordinator:
    """Centralized backup coordination and state management"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/backup-config.json"):
        self.config_path = config_path
        self.base_backup_dir = "/opt/sutazaiapp/backups"
        self.offsite_backup_dir = "/mnt/offsite-backups"
        self.metadata_db_path = "/opt/sutazaiapp/data/backup-metadata.db"
        self.encryption_key_path = "/opt/sutazaiapp/secrets/backup-encryption.key"
        
        # Create directories
        os.makedirs(self.base_backup_dir, exist_ok=True)
        os.makedirs(self.offsite_backup_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_db_path), exist_ok=True)
        
        # Initialize components
        self._init_encryption()
        self._init_database()
        self._load_backup_jobs()
        
        # State tracking
        self.active_backups = {}
        self.backup_queue = []
        self.shutdown_event = threading.Event()
        
        # Resource monitoring
        self.max_concurrent_backups = 3
        self.max_disk_usage_percent = 80
        
        logger.info("Backup Coordinator initialized")

    def _init_encryption(self):
        """Initialize encryption system"""
        if os.path.exists(self.encryption_key_path):
            with open(self.encryption_key_path, 'rb') as f:
                self.encryption_key = f.read()
        else:
            # Generate new encryption key
            password = b"sutazai-backup-encryption-key-2024"
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            self.encryption_key = base64.urlsafe_b64encode(kdf.derive(password))
            
            # Save encryption key securely
            os.makedirs(os.path.dirname(self.encryption_key_path), exist_ok=True)
            with open(self.encryption_key_path, 'wb') as f:
                f.write(self.encryption_key)
            os.chmod(self.encryption_key_path, 0o600)
        
        self.cipher_suite = Fernet(self.encryption_key)

    def _init_database(self):
        """Initialize backup metadata database"""
        self.conn = sqlite3.connect(self.metadata_db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS backup_metadata (
                backup_id TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                incremental BOOLEAN NOT NULL,
                base_backup_id TEXT,
                source_path TEXT NOT NULL,
                backup_path TEXT NOT NULL,
                encryption_key_id TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                recovery_instructions TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS backup_jobs (
                name TEXT PRIMARY KEY,
                config TEXT NOT NULL,
                last_backup DATETIME,
                next_backup DATETIME,
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS backup_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_id TEXT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()

    def _load_backup_jobs(self):
        """Load backup job configurations"""
        default_jobs = [
            BackupJob(
                name="critical-databases",
                source_path="/opt/sutazaiapp/data",
                backup_type="database",
                frequency="hourly",
                retention_days=30,
                priority=1,
                verification_commands=["sqlite3 {backup_path} 'PRAGMA integrity_check;'"]
            ),
            BackupJob(
                name="system-configuration",
                source_path="/opt/sutazaiapp/config",
                backup_type="file",
                frequency="daily",
                retention_days=90,
                priority=1
            ),
            BackupJob(
                name="agent-code",
                source_path="/opt/sutazaiapp/agents",
                backup_type="file",
                frequency="daily", 
                retention_days=60,
                priority=2
            ),
            BackupJob(
                name="logs-archive",
                source_path="/opt/sutazaiapp/logs",
                backup_type="file",
                frequency="daily",
                retention_days=180,
                priority=3,
                compression=True
            ),
            BackupJob(
                name="container-state",
                source_path="/var/lib/docker/volumes",
                backup_type="container",
                frequency="daily",
                retention_days=14,
                priority=2,
                pre_backup_commands=["docker system prune -f"]
            ),
            BackupJob(
                name="application-state",
                source_path="/opt/sutazaiapp",
                backup_type="state",
                frequency="weekly",
                retention_days=365,
                priority=1,
                incremental=False  # Full backup weekly
            )
        ]
        
        self.backup_jobs = {}
        
        # Load from database first
        cursor = self.conn.execute("SELECT name, config FROM backup_jobs WHERE status = 'active'")
        for row in cursor.fetchall():
            job_config = json.loads(row[1])
            self.backup_jobs[row[0]] = BackupJob(**job_config)
        
        # Add default jobs if not present
        for job in default_jobs:
            if job.name not in self.backup_jobs:
                self.backup_jobs[job.name] = job
                self._save_job_config(job)

    def _save_job_config(self, job: BackupJob):
        """Save backup job configuration to database"""
        self.conn.execute(
            "INSERT OR REPLACE INTO backup_jobs (name, config) VALUES (?, ?)",
            (job.name, json.dumps(asdict(job)))
        )
        self.conn.commit()

    def create_backup(self, job_name: str) -> Optional[str]:
        """Create a backup for the specified job"""
        if job_name not in self.backup_jobs:
            logger.error(f"Backup job '{job_name}' not found")
            return None
        
        job = self.backup_jobs[job_name]
        backup_id = f"{job_name}_{int(time.time())}"
        
        try:
            logger.info(f"Starting backup: {backup_id}")
            
            # Check system resources
            if not self._check_system_resources():
                logger.warning(f"System resources insufficient for backup: {backup_id}")
                return None
            
            # Execute pre-backup commands
            if job.pre_backup_commands:
                for cmd in job.pre_backup_commands:
                    subprocess.run(cmd, shell=True, check=True)
            
            # Create backup based on type
            if job.backup_type == "database":
                backup_path = self._backup_database(job, backup_id)
            elif job.backup_type == "file":
                backup_path = self._backup_files(job, backup_id)
            elif job.backup_type == "container":
                backup_path = self._backup_containers(job, backup_id)
            elif job.backup_type == "state":
                backup_path = self._backup_system_state(job, backup_id)
            else:
                logger.error(f"Unknown backup type: {job.backup_type}")
                return None
            
            if not backup_path:
                logger.error(f"Backup failed: {backup_id}")
                return None
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                job_name=job_name,
                timestamp=datetime.now(),
                size_bytes=os.path.getsize(backup_path),
                checksum_sha256=checksum,
                incremental=job.incremental,
                base_backup_id=self._get_last_backup_id(job_name) if job.incremental else None,
                source_path=job.source_path,
                backup_path=backup_path,
                encryption_key_id="default",
                verification_status="pending",
                recovery_instructions=self._generate_recovery_instructions(job, backup_path)
            )
            
            # Save metadata
            self._save_backup_metadata(metadata)
            
            # Verify backup
            if self._verify_backup(job, backup_path):
                metadata.verification_status = "verified"
            else:
                metadata.verification_status = "failed"
            
            self._update_backup_metadata(metadata)
            
            # Execute post-backup commands
            if job.post_backup_commands:
                for cmd in job.post_backup_commands:
                    subprocess.run(cmd, shell=True, check=True)
            
            # Copy to offsite storage
            self._copy_to_offsite(backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups(job_name, job.retention_days)
            
            logger.info(f"Backup completed successfully: {backup_id}")
            return backup_id
            
        except Exception as e:
            logger.error(f"Backup failed: {backup_id} - {str(e)}")
            self._log_backup_event(backup_id, "ERROR", f"Backup failed: {str(e)}")
            return None

    def _backup_database(self, job: BackupJob, backup_id: str) -> Optional[str]:
        """Backup database files with consistency"""
        backup_dir = os.path.join(self.base_backup_dir, job.name, datetime.now().strftime("%Y%m%d"))
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"{backup_id}.tar.gz.enc")
        
        # Create consistent backup
        with tarfile.open(backup_file.replace('.enc', ''), 'w:gz') as tar:
            for root, dirs, files in os.walk(job.source_path):
                for file in files:
                    if file.endswith('.db') or file.endswith('.sqlite'):
                        file_path = os.path.join(root, file)
                        # Lock database for consistent backup
                        try:
                            conn = sqlite3.connect(file_path)
                            conn.execute("BEGIN IMMEDIATE;")
                            tar.add(file_path, arcname=os.path.relpath(file_path, job.source_path))
                            conn.rollback()
                            conn.close()
                        except Exception as e:
                            logger.warning(f"Could not lock database {file_path}: {e}")
                            tar.add(file_path, arcname=os.path.relpath(file_path, job.source_path))
        
        # Encrypt backup
        if job.encryption:
            self._encrypt_file(backup_file.replace('.enc', ''), backup_file)
            os.remove(backup_file.replace('.enc', ''))
            return backup_file
        
        return backup_file.replace('.enc', '')

    def _backup_files(self, job: BackupJob, backup_id: str) -> Optional[str]:
        """Backup file system with incremental support"""
        backup_dir = os.path.join(self.base_backup_dir, job.name, datetime.now().strftime("%Y%m%d"))
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"{backup_id}.tar.gz.enc")
        
        # Create backup archive
        compression = 'w:gz' if job.compression else 'w'
        with tarfile.open(backup_file.replace('.enc', ''), compression) as tar:
            if job.incremental:
                # Incremental backup - only changed files
                last_backup_time = self._get_last_backup_time(job.name)
                for root, dirs, files in os.walk(job.source_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) > last_backup_time:
                            tar.add(file_path, arcname=os.path.relpath(file_path, job.source_path))
            else:
                # Full backup
                tar.add(job.source_path, arcname=os.path.basename(job.source_path))
        
        # Encrypt if required
        if job.encryption:
            self._encrypt_file(backup_file.replace('.enc', ''), backup_file)
            os.remove(backup_file.replace('.enc', ''))
            return backup_file
        
        return backup_file.replace('.enc', '')

    def _backup_containers(self, job: BackupJob, backup_id: str) -> Optional[str]:
        """Backup Docker container volumes and images"""
        backup_dir = os.path.join(self.base_backup_dir, job.name, datetime.now().strftime("%Y%m%d"))
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"{backup_id}.tar.gz.enc")
        
        # Export container images and volumes
        temp_dir = f"/tmp/container_backup_{backup_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Backup running containers
            result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], 
                                  capture_output=True, text=True)
            containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Save container images
            for container in containers:
                if container:
                    subprocess.run([
                        "docker", "commit", container, f"backup-{container}-{backup_id}"
                    ], check=True)
                    
                    subprocess.run([
                        "docker", "save", "-o", 
                        os.path.join(temp_dir, f"{container}.tar"),
                        f"backup-{container}-{backup_id}"
                    ], check=True)
            
            # Backup volumes
            subprocess.run([
                "cp", "-r", "/var/lib/docker/volumes/", 
                os.path.join(temp_dir, "volumes")
            ], check=True)
            
            # Create archive
            with tarfile.open(backup_file.replace('.enc', ''), 'w:gz') as tar:
                tar.add(temp_dir, arcname=f"container_backup_{backup_id}")
            
            # Encrypt if required
            if job.encryption:
                self._encrypt_file(backup_file.replace('.enc', ''), backup_file)
                os.remove(backup_file.replace('.enc', ''))
                result_file = backup_file
            else:
                result_file = backup_file.replace('.enc', '')
            
            return result_file
            
        finally:
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Remove temporary images
            for container in containers:
                if container:
                    subprocess.run([
                        "docker", "rmi", f"backup-{container}-{backup_id}"
                    ], stderr=subprocess.DEVNULL)

    def _backup_system_state(self, job: BackupJob, backup_id: str) -> Optional[str]:
        """Backup complete system state"""
        backup_dir = os.path.join(self.base_backup_dir, job.name, datetime.now().strftime("%Y%m%d"))
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"{backup_id}.tar.gz.enc")
        
        # Collect system state
        state_dir = f"/tmp/system_state_{backup_id}"
        os.makedirs(state_dir, exist_ok=True)
        
        try:
            # System information
            system_info = {
                "timestamp": datetime.now().isoformat(),
                "hostname": os.uname().nodename,
                "system": os.uname().sysname,
                "release": os.uname().release,
                "version": os.uname().version,
                "machine": os.uname().machine,
                "disk_usage": dict(psutil.disk_usage('/')._asdict()),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "cpu_count": psutil.cpu_count(),
                "network_interfaces": list(psutil.net_if_addrs().keys()),
                "processes": len(psutil.pids())
            }
            
            with open(os.path.join(state_dir, "system_info.json"), 'w') as f:
                json.dump(system_info, f, indent=2)
            
            # Docker state
            subprocess.run([
                "docker", "system", "df", "--format", "json"
            ], stdout=open(os.path.join(state_dir, "docker_state.json"), 'w'))
            
            # Network configuration
            subprocess.run([
                "ip", "addr", "show"
            ], stdout=open(os.path.join(state_dir, "network_config.txt"), 'w'))
            
            # Process list
            subprocess.run([
                "ps", "aux"
            ], stdout=open(os.path.join(state_dir, "processes.txt"), 'w'))
            
            # Environment variables
            with open(os.path.join(state_dir, "environment.json"), 'w') as f:
                json.dump(dict(os.environ), f, indent=2)
            
            # Create archive
            with tarfile.open(backup_file.replace('.enc', ''), 'w:gz') as tar:
                tar.add(job.source_path, arcname="application")
                tar.add(state_dir, arcname="system_state")
            
            # Encrypt if required
            if job.encryption:
                self._encrypt_file(backup_file.replace('.enc', ''), backup_file)
                os.remove(backup_file.replace('.enc', ''))
                return backup_file
            
            return backup_file.replace('.enc', '')
            
        finally:
            shutil.rmtree(state_dir, ignore_errors=True)

    def _encrypt_file(self, source_file: str, target_file: str):
        """Encrypt a file using Fernet encryption"""
        with open(source_file, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher_suite.encrypt(data)
        
        with open(target_file, 'wb') as f:
            f.write(encrypted_data)

    def _decrypt_file(self, encrypted_file: str, output_file: str):
        """Decrypt a file using Fernet encryption"""
        with open(encrypted_file, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        
        with open(output_file, 'wb') as f:
            f.write(decrypted_data)

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _verify_backup(self, job: BackupJob, backup_path: str) -> bool:
        """Verify backup integrity"""
        try:
            # Basic file existence and readability
            if not os.path.exists(backup_path) or not os.access(backup_path, os.R_OK):
                return False
            
            # Verify archive integrity
            if backup_path.endswith('.tar.gz') or backup_path.endswith('.tar.gz.enc'):
                test_file = backup_path
                if backup_path.endswith('.enc'):
                    # Decrypt to temporary file for testing
                    test_file = backup_path + '.test'
                    self._decrypt_file(backup_path, test_file)
                
                try:
                    with tarfile.open(test_file, 'r:gz') as tar:
                        # Test archive integrity
                        for member in tar.getmembers():
                            if member.isfile():
                                tar.extractfile(member).read(1024)  # Test read
                finally:
                    if test_file != backup_path:
                        os.remove(test_file)
            
            # Run custom verification commands
            if job.verification_commands:
                for cmd in job.verification_commands:
                    cmd = cmd.format(backup_path=backup_path)
                    result = subprocess.run(cmd, shell=True, capture_output=True)
                    if result.returncode != 0:
                        logger.warning(f"Verification command failed: {cmd}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            return False

    def _check_system_resources(self) -> bool:
        """Check if system has sufficient resources for backup"""
        # Check disk space
        disk_usage = psutil.disk_usage(self.base_backup_dir)
        if (disk_usage.used / disk_usage.total) * 100 > self.max_disk_usage_percent:
            logger.warning("Disk usage too high for backup")
            return False
        
        # Check concurrent backups
        if len(self.active_backups) >= self.max_concurrent_backups:
            logger.warning("Too many concurrent backups")
            return False
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning("Memory usage too high for backup")
            return False
        
        return True

    def _copy_to_offsite(self, backup_path: str):
        """Copy backup to offsite storage"""
        try:
            if os.path.exists(self.offsite_backup_dir):
                relative_path = os.path.relpath(backup_path, self.base_backup_dir)
                offsite_path = os.path.join(self.offsite_backup_dir, relative_path)
                os.makedirs(os.path.dirname(offsite_path), exist_ok=True)
                shutil.copy2(backup_path, offsite_path)
                logger.info(f"Copied to offsite storage: {offsite_path}")
        except Exception as e:
            logger.error(f"Failed to copy to offsite storage: {e}")

    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database"""
        self.conn.execute('''
            INSERT INTO backup_metadata (
                backup_id, job_name, timestamp, size_bytes, checksum_sha256,
                incremental, base_backup_id, source_path, backup_path,
                encryption_key_id, verification_status, recovery_instructions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata.backup_id, metadata.job_name, metadata.timestamp.isoformat(),
            metadata.size_bytes, metadata.checksum_sha256, metadata.incremental,
            metadata.base_backup_id, metadata.source_path, metadata.backup_path,
            metadata.encryption_key_id, metadata.verification_status,
            metadata.recovery_instructions
        ))
        self.conn.commit()

    def _update_backup_metadata(self, metadata: BackupMetadata):
        """Update backup metadata in database"""
        self.conn.execute('''
            UPDATE backup_metadata 
            SET verification_status = ?
            WHERE backup_id = ?
        ''', (metadata.verification_status, metadata.backup_id))
        self.conn.commit()

    def _get_last_backup_id(self, job_name: str) -> Optional[str]:
        """Get the most recent backup ID for a job"""
        cursor = self.conn.execute('''
            SELECT backup_id FROM backup_metadata 
            WHERE job_name = ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (job_name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def _get_last_backup_time(self, job_name: str) -> float:
        """Get timestamp of last backup for incremental comparison"""
        cursor = self.conn.execute('''
            SELECT timestamp FROM backup_metadata 
            WHERE job_name = ? 
            ORDER BY timestamp DESC LIMIT 1
        ''', (job_name,))
        row = cursor.fetchone()
        if row:
            return datetime.fromisoformat(row[0]).timestamp()
        return 0

    def _cleanup_old_backups(self, job_name: str, retention_days: int):
        """Remove old backups beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Get old backups
        cursor = self.conn.execute('''
            SELECT backup_id, backup_path FROM backup_metadata 
            WHERE job_name = ? AND timestamp < ?
        ''', (job_name, cutoff_date.isoformat()))
        
        for row in cursor.fetchall():
            backup_id, backup_path = row
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                
                # Remove from offsite if exists
                if os.path.exists(self.offsite_backup_dir):
                    relative_path = os.path.relpath(backup_path, self.base_backup_dir)
                    offsite_path = os.path.join(self.offsite_backup_dir, relative_path)
                    if os.path.exists(offsite_path):
                        os.remove(offsite_path)
                
                logger.info(f"Cleaned up old backup: {backup_id}")
                
            except Exception as e:
                logger.error(f"Failed to cleanup backup {backup_id}: {e}")
        
        # Remove metadata for old backups
        self.conn.execute('''
            DELETE FROM backup_metadata 
            WHERE job_name = ? AND timestamp < ?
        ''', (job_name, cutoff_date.isoformat()))
        self.conn.commit()

    def _generate_recovery_instructions(self, job: BackupJob, backup_path: str) -> str:
        """Generate recovery instructions for backup"""
        instructions = [
            f"# Recovery Instructions for {job.name}",
            f"Backup Path: {backup_path}",
            f"Original Source: {job.source_path}",
            "",
            "## Recovery Steps:",
        ]
        
        if job.backup_type == "database":
            instructions.extend([
                "1. Stop services accessing the database",
                "2. Decrypt backup if encrypted:",
                f"   python3 /opt/sutazaiapp/disaster-recovery/backup-coordinator.py decrypt {backup_path}",
                "3. Extract backup:",
                f"   tar -xzf {backup_path.replace('.enc', '')} -C /tmp/",
                "4. Copy database files to original location",
                "5. Restart services",
                "6. Verify database integrity"
            ])
        elif job.backup_type == "file":
            instructions.extend([
                "1. Stop services if necessary",
                "2. Decrypt and extract backup:",
                f"   python3 /opt/sutazaiapp/disaster-recovery/backup-coordinator.py restore {backup_path} {job.source_path}",
                "3. Set proper permissions",
                "4. Restart services"
            ])
        elif job.backup_type == "container":
            instructions.extend([
                "1. Stop all containers",
                "2. Decrypt and extract backup",
                "3. Load container images:",
                "   docker load -i <image.tar>",
                "4. Restore volumes",
                "5. Restart containers"
            ])
        elif job.backup_type == "state":
            instructions.extend([
                "1. This is a full system state backup",
                "2. Requires complete system restore",
                "3. Extract to temporary location first",
                "4. Review system_info.json for compatibility",
                "5. Restore application files",
                "6. Restore system configuration if needed"
            ])
        
        return "\n".join(instructions)

    def _log_backup_event(self, backup_id: str, level: str, message: str):
        """Log backup event to database"""
        self.conn.execute(
            "INSERT INTO backup_logs (backup_id, level, message) VALUES (?, ?, ?)",
            (backup_id, level, message)
        )
        self.conn.commit()

    def schedule_backups(self):
        """Schedule all backup jobs"""
        for job_name, job in self.backup_jobs.items():
            if job.frequency == "hourly":
                schedule.every().hour.do(self.create_backup, job_name)
            elif job.frequency == "daily":
                schedule.every().day.at("02:00").do(self.create_backup, job_name)
            elif job.frequency == "weekly":
                schedule.every().week.do(self.create_backup, job_name)
            elif job.frequency == "monthly":
                schedule.every().month.do(self.create_backup, job_name)
        
        logger.info(f"Scheduled {len(self.backup_jobs)} backup jobs")

    def run_scheduler(self):
        """Run the backup scheduler"""
        logger.info("Starting backup scheduler")
        
        while not self.shutdown_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def restore_backup(self, backup_id: str, restore_path: str = None) -> bool:
        """Restore a backup by ID"""
        cursor = self.conn.execute('''
            SELECT * FROM backup_metadata WHERE backup_id = ?
        ''', (backup_id,))
        
        row = cursor.fetchone()
        if not row:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        # Extract metadata
        metadata = BackupMetadata(
            backup_id=row[0],
            job_name=row[1],
            timestamp=datetime.fromisoformat(row[2]),
            size_bytes=row[3],
            checksum_sha256=row[4],
            incremental=bool(row[5]),
            base_backup_id=row[6],
            source_path=row[7],
            backup_path=row[8],
            encryption_key_id=row[9],
            verification_status=row[10],
            recovery_instructions=row[11]
        )
        
        try:
            logger.info(f"Starting restore: {backup_id}")
            
            restore_target = restore_path or metadata.source_path
            
            # Verify backup before restore
            if not self._verify_backup(self.backup_jobs[metadata.job_name], metadata.backup_path):
                logger.error(f"Backup verification failed: {backup_id}")
                return False
            
            # Create restore directory
            os.makedirs(restore_target, exist_ok=True)
            
            # Decrypt if necessary
            if metadata.backup_path.endswith('.enc'):
                temp_file = metadata.backup_path + '.restore'
                self._decrypt_file(metadata.backup_path, temp_file)
                backup_file = temp_file
            else:
                backup_file = metadata.backup_path
            
            try:
                # Extract backup
                with tarfile.open(backup_file, 'r:gz') as tar:
                    tar.extractall(path=restore_target)
                
                logger.info(f"Restore completed: {backup_id} to {restore_target}")
                return True
                
            finally:
                if backup_file != metadata.backup_path:
                    os.remove(backup_file)
            
        except Exception as e:
            logger.error(f"Restore failed: {backup_id} - {str(e)}")
            return False

    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup system status"""
        cursor = self.conn.execute('''
            SELECT job_name, COUNT(*) as backup_count, 
                   MAX(timestamp) as last_backup,
                   SUM(size_bytes) as total_size
            FROM backup_metadata 
            GROUP BY job_name
        ''')
        
        job_status = {}
        for row in cursor.fetchall():
            job_status[row[0]] = {
                "backup_count": row[1],
                "last_backup": row[2],
                "total_size_bytes": row[3] or 0
            }
        
        # System status
        disk_usage = psutil.disk_usage(self.base_backup_dir)
        
        return {
            "jobs": job_status,
            "active_backups": len(self.active_backups),
            "disk_usage": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": (disk_usage.used / disk_usage.total) * 100
            },
            "offsite_available": os.path.exists(self.offsite_backup_dir),
            "scheduler_running": not self.shutdown_event.is_set()
        }

    def emergency_backup_all(self) -> List[str]:
        """Create emergency backups of all critical jobs"""
        logger.warning("Emergency backup initiated")
        
        critical_jobs = [name for name, job in self.backup_jobs.items() if job.priority == 1]
        backup_ids = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_backups) as executor:
            futures = {executor.submit(self.create_backup, job_name): job_name 
                      for job_name in critical_jobs}
            
            for future in as_completed(futures):
                job_name = futures[future]
                try:
                    backup_id = future.result()
                    if backup_id:
                        backup_ids.append(backup_id)
                        logger.info(f"Emergency backup completed: {job_name} -> {backup_id}")
                except Exception as e:
                    logger.error(f"Emergency backup failed for {job_name}: {e}")
        
        return backup_ids

    def shutdown(self):
        """Graceful shutdown of backup system"""
        logger.info("Shutting down backup coordinator")
        self.shutdown_event.set()
        
        # Wait for active backups to complete
        timeout = 300  # 5 minutes
        start_time = time.time()
        
        while self.active_backups and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self.active_backups)} active backups to complete")
            time.sleep(10)
        
        if self.active_backups:
            logger.warning(f"Shutdown timeout - {len(self.active_backups)} backups still active")
        
        self.conn.close()
        logger.info("Backup coordinator shutdown complete")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SutazAI Backup Coordinator")
    parser.add_argument("command", choices=["start", "backup", "restore", "status", "emergency"])
    parser.add_argument("--job", help="Job name for backup command")
    parser.add_argument("--backup-id", help="Backup ID for restore command")
    parser.add_argument("--restore-path", help="Restore path for restore command")
    
    args = parser.parse_args()
    
    coordinator = BackupCoordinator()
    
    try:
        if args.command == "start":
            coordinator.schedule_backups()
            coordinator.run_scheduler()
        elif args.command == "backup":
            if not args.job:
                print("Error: --job required for backup command")
                sys.exit(1)
            backup_id = coordinator.create_backup(args.job)
            if backup_id:
                print(f"Backup created: {backup_id}")
            else:
                print("Backup failed")
                sys.exit(1)
        elif args.command == "restore":
            if not args.backup_id:
                print("Error: --backup-id required for restore command")
                sys.exit(1)
            success = coordinator.restore_backup(args.backup_id, args.restore_path)
            if success:
                print("Restore completed successfully")
            else:
                print("Restore failed")
                sys.exit(1)
        elif args.command == "status":
            status = coordinator.get_backup_status()
            print(json.dumps(status, indent=2))
        elif args.command == "emergency":
            backup_ids = coordinator.emergency_backup_all()
            print(f"Emergency backups created: {backup_ids}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        coordinator.shutdown()

if __name__ == "__main__":
    main()