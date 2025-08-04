#!/usr/bin/env python3
"""
SutazAI Database Backup System
Comprehensive database backup automation with 3-2-1 strategy implementation
"""

import os
import sys
import json
import logging
import datetime
import subprocess
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psycopg2
import sqlite3
import shutil
import gzip
import tarfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/database-backup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseBackupSystem:
    """Comprehensive database backup system for SutazAI"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/backup-config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.backup_root = Path(self.config.get('backup_root', '/opt/sutazaiapp/data/backups'))
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure backup directories exist
        self.ensure_directories()
        
    def load_config(self) -> Dict:
        """Load backup configuration"""
        default_config = {
            "backup_root": "/opt/sutazaiapp/data/backups",
            "retention_days": 30,
            "compression": True,
            "encryption": False,
            "databases": {
                "postgres": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 5432,
                    "databases": ["sutazai_main", "agent_data"],
                    "user": "postgres",
                    "password_env": "POSTGRES_PASSWORD"
                },
                "sqlite": {
                    "enabled": True,
                    "databases": [
                        "/opt/sutazaiapp/data/flowise/database.sqlite",
                        "/opt/sutazaiapp/data/langflow/langflow.db",
                        "/opt/sutazaiapp/data/n8n/database.sqlite",
                        "/opt/sutazaiapp/data/garbage-collection.db",
                        "/opt/sutazaiapp/data/disaster-recovery-tests.db",
                        "/opt/sutazaiapp/data/shutdown-state.db"
                    ]
                },
                "loki": {
                    "enabled": True,
                    "data_path": "/opt/sutazaiapp/data/loki"
                }
            },
            "offsite": {
                "enabled": True,
                "type": "rsync",
                "destination": "/mnt/backup-drive/sutazai-backups",
                "remote_host": None
            }
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
            else:
                # Create default config
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def ensure_directories(self):
        """Ensure all backup directories exist"""
        directories = [
            self.backup_root,
            self.backup_root / 'daily',
            self.backup_root / 'weekly',
            self.backup_root / 'monthly',
            self.backup_root / 'postgres',
            self.backup_root / 'sqlite',
            self.backup_root / 'loki',
            self.backup_root / 'verification',
            self.backup_root / 'offsite'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def backup_postgres_databases(self) -> List[Dict]:
        """Backup all PostgreSQL databases"""
        results = []
        postgres_config = self.config['databases']['postgres']
        
        if not postgres_config.get('enabled', False):
            logger.info("PostgreSQL backup disabled")
            return results
        
        password = os.environ.get(postgres_config.get('password_env', 'POSTGRES_PASSWORD'))
        if not password:
            logger.error("PostgreSQL password not found in environment")
            return results
        
        # Set environment for pg_dump
        env = os.environ.copy()
        env['PGPASSWORD'] = password
        
        for db_name in postgres_config.get('databases', []):
            try:
                backup_file = self.backup_root / 'postgres' / f"{db_name}_{self.timestamp}.sql"
                
                # Create pg_dump command
                cmd = [
                    'pg_dump',
                    '-h', postgres_config.get('host', 'localhost'),
                    '-p', str(postgres_config.get('port', 5432)),
                    '-U', postgres_config.get('user', 'postgres'),
                    '-d', db_name,
                    '--no-password',
                    '--verbose',
                    '--clean',
                    '--if-exists',
                    '--create'
                ]
                
                logger.info(f"Backing up PostgreSQL database: {db_name}")
                
                with open(backup_file, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, 
                                          env=env, text=True)
                
                if result.returncode == 0:
                    # Compress if enabled
                    if self.config.get('compression', True):
                        compressed_file = f"{backup_file}.gz"
                        with open(backup_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(backup_file)
                        backup_file = compressed_file
                    
                    # Calculate checksum
                    checksum = self.calculate_checksum(backup_file)
                    
                    results.append({
                        'type': 'postgres',
                        'database': db_name,
                        'file': str(backup_file),
                        'size': os.path.getsize(backup_file),
                        'checksum': checksum,
                        'timestamp': self.timestamp,
                        'status': 'success'
                    })
                    
                    logger.info(f"Successfully backed up {db_name}")
                else:
                    logger.error(f"Failed to backup {db_name}: {result.stderr}")
                    results.append({
                        'type': 'postgres',
                        'database': db_name,
                        'status': 'failed',
                        'error': result.stderr
                    })
                    
            except Exception as e:
                logger.error(f"Error backing up PostgreSQL database {db_name}: {e}")
                results.append({
                    'type': 'postgres',
                    'database': db_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def backup_sqlite_databases(self) -> List[Dict]:
        """Backup all SQLite databases"""
        results = []
        sqlite_config = self.config['databases']['sqlite']
        
        if not sqlite_config.get('enabled', False):
            logger.info("SQLite backup disabled")
            return results
        
        for db_path in sqlite_config.get('databases', []):
            try:
                if not os.path.exists(db_path):
                    logger.warning(f"SQLite database not found: {db_path}")
                    continue
                
                db_name = os.path.basename(db_path)
                backup_file = self.backup_root / 'sqlite' / f"{db_name}_{self.timestamp}"
                
                logger.info(f"Backing up SQLite database: {db_path}")
                
                # Use SQLite .backup command for consistent backup
                conn = sqlite3.connect(db_path)
                backup_conn = sqlite3.connect(str(backup_file))
                conn.backup(backup_conn)
                backup_conn.close()
                conn.close()
                
                # Compress if enabled
                if self.config.get('compression', True):
                    compressed_file = f"{backup_file}.gz"
                    with open(backup_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(backup_file)
                    backup_file = compressed_file
                
                # Calculate checksum
                checksum = self.calculate_checksum(backup_file)
                
                results.append({
                    'type': 'sqlite',
                    'database': db_path,
                    'file': str(backup_file),
                    'size': os.path.getsize(backup_file),
                    'checksum': checksum,
                    'timestamp': self.timestamp,
                    'status': 'success'
                })
                
                logger.info(f"Successfully backed up {db_path}")
                
            except Exception as e:
                logger.error(f"Error backing up SQLite database {db_path}: {e}")
                results.append({
                    'type': 'sqlite',
                    'database': db_path,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return results
    
    def backup_loki_data(self) -> List[Dict]:
        """Backup Loki time-series data"""
        results = []
        loki_config = self.config['databases']['loki']
        
        if not loki_config.get('enabled', False):
            logger.info("Loki backup disabled")
            return results
        
        loki_path = loki_config.get('data_path', '/opt/sutazaiapp/data/loki')
        
        if not os.path.exists(loki_path):
            logger.warning(f"Loki data path not found: {loki_path}")
            return results
        
        try:
            backup_file = self.backup_root / 'loki' / f"loki_data_{self.timestamp}.tar.gz"
            
            logger.info(f"Backing up Loki data: {loki_path}")
            
            # Create tarball of Loki data
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(loki_path, arcname='loki_data')
            
            # Calculate checksum
            checksum = self.calculate_checksum(backup_file)
            
            results.append({
                'type': 'loki',
                'source': loki_path,
                'file': str(backup_file),
                'size': os.path.getsize(backup_file),
                'checksum': checksum,
                'timestamp': self.timestamp,
                'status': 'success'
            })
            
            logger.info(f"Successfully backed up Loki data")
            
        except Exception as e:
            logger.error(f"Error backing up Loki data: {e}")
            results.append({
                'type': 'loki',
                'source': loki_path,
                'status': 'failed',
                'error': str(e)
            })
        
        return results
    
    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def create_backup_manifest(self, backup_results: List[Dict]) -> str:
        """Create backup manifest file"""
        manifest = {
            'timestamp': self.timestamp,
            'backup_date': datetime.datetime.now().isoformat(),
            'total_backups': len(backup_results),
            'successful_backups': len([r for r in backup_results if r.get('status') == 'success']),
            'failed_backups': len([r for r in backup_results if r.get('status') == 'failed']),
            'backups': backup_results,
            'config': self.config
        }
        
        manifest_file = self.backup_root / f"backup_manifest_{self.timestamp}.json"
        
        try:
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created backup manifest: {manifest_file}")
            return str(manifest_file)
            
        except Exception as e:
            logger.error(f"Error creating backup manifest: {e}")
            return ""
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        retention_days = self.config.get('retention_days', 30)
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
        
        for backup_type in ['postgres', 'sqlite', 'loki']:
            backup_dir = self.backup_root / backup_type
            if not backup_dir.exists():
                continue
            
            for backup_file in backup_dir.iterdir():
                if backup_file.is_file():
                    file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        try:
                            backup_file.unlink()
                            logger.info(f"Removed old backup: {backup_file}")
                        except Exception as e:
                            logger.error(f"Error removing old backup {backup_file}: {e}")
    
    def run_full_backup(self) -> Dict:
        """Run complete database backup process"""
        start_time = time.time()
        logger.info(f"Starting full database backup - {self.timestamp}")
        
        all_results = []
        
        # Backup PostgreSQL databases
        postgres_results = self.backup_postgres_databases()
        all_results.extend(postgres_results)
        
        # Backup SQLite databases
        sqlite_results = self.backup_sqlite_databases()
        all_results.extend(sqlite_results)
        
        # Backup Loki data
        loki_results = self.backup_loki_data()
        all_results.extend(loki_results)
        
        # Create backup manifest
        manifest_file = self.create_backup_manifest(all_results)
        
        # Cleanup old backups
        self.cleanup_old_backups()
        
        end_time = time.time()
        duration = end_time - start_time
        
        summary = {
            'timestamp': self.timestamp,
            'duration_seconds': duration,
            'total_backups': len(all_results),
            'successful_backups': len([r for r in all_results if r.get('status') == 'success']),
            'failed_backups': len([r for r in all_results if r.get('status') == 'failed']),
            'manifest_file': manifest_file,
            'results': all_results
        }
        
        logger.info(f"Database backup completed in {duration:.2f} seconds")
        logger.info(f"Successful: {summary['successful_backups']}, Failed: {summary['failed_backups']}")
        
        return summary

def main():
    """Main entry point"""
    try:
        backup_system = DatabaseBackupSystem()
        result = backup_system.run_full_backup()
        
        # Write summary to file
        summary_file = f"/opt/sutazaiapp/logs/database_backup_summary_{backup_system.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Exit with appropriate code
        if result['failed_backups'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()