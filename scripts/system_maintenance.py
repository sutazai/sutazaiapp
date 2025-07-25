#!/usr/bin/env python3.11
"""
SutazAI System Maintenance Script

This script provides comprehensive system maintenance capabilities including:
- System health checks
- Performance optimization
- Security validation
- Dependency management
- Log rotation
- Backup management
"""

import json
import logging
import os
import psutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

# Configure logging
# Use project relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "maintenance.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SutazAI.Maintenance")

class SystemMaintainer:
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'log_dir': LOG_DIR,
            'backup_dir': os.path.join(PROJECT_ROOT, 'backups'),
            'max_log_age_days': 7,
            'max_backup_age_days': 30,
            'backup_retention': 5
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['log_dir'], 'maintenance.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SystemMaintainer')

    def check_system_health(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            health_status = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.config['log_dir'], 'health_status.json'), 'w') as f:
                json.dump(health_status, f)
                
            return True
        except Exception as e:
            self.logger.error(f"Error checking system health: {str(e)}")
            return False

    def rotate_logs(self):
        try:
            log_dir = self.config['log_dir']
            max_age = timedelta(days=self.config['max_log_age_days'])
            current_time = datetime.now()
            
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)
            
            # Get all log files
            log_files = []
            for filename in os.listdir(log_dir):
                # Check for both standard log files and rotated log files with dates
                if (filename.endswith('.log') or 
                    filename.endswith('.log.old') or 
                    '.log.' in filename):  # Catch patterns like app.log.20000101
                    
                    filepath = os.path.join(log_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    log_files.append((filepath, file_time))
            
            # Sort by modification time (oldest first)
            log_files.sort(key=lambda x: x[1])
            
            # Remove old logs
            for filepath, file_time in log_files:
                if current_time - file_time > max_age:
                    try:
                        os.remove(filepath)
                        self.logger.info(f"Removed old log file: {filepath}")
                    except Exception as e:
                        self.logger.error(f"Error removing log file {filepath}: {str(e)}")
                        continue
            
            # Rotate current logs
            for filename in os.listdir(log_dir):
                if filename.endswith('.log'):
                    filepath = os.path.join(log_dir, filename)
                    if os.path.getsize(filepath) > 10 * 1024 * 1024:  # 10MB
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_filepath = os.path.join(log_dir, f"{filename}.{timestamp}")
                        os.rename(filepath, new_filepath)
                        self.logger.info(f"Rotated log file: {filepath} -> {new_filepath}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error rotating logs: {str(e)}")
            return False

    def manage_backups(self):
        try:
            backup_dir = self.config['backup_dir']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
            
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup application files
            subprocess.run([
                "rsync",
                "-av",
                "--exclude=venv",
                "--exclude=__pycache__",
                PROJECT_ROOT + "/",
                backup_path
            ], check=True)
            
            # Backup database
            subprocess.run([
                "pg_dump",
                "-U", "sutazai",
                "sutazai",
                "-f", os.path.join(backup_path, "db_backup.sql")
            ], check=True)
            
            # Clean old backups
            self._clean_old_backups()
            
            return True
        except Exception as e:
            self.logger.error(f"Error managing backups: {str(e)}")
            return False

    def _clean_old_backups(self):
        backup_dir = self.config['backup_dir']
        max_age = timedelta(days=self.config['max_backup_age_days'])
        current_time = datetime.now()
        
        # Get list of backups sorted by modification time
        backups = []
        for dirname in os.listdir(backup_dir):
            path = os.path.join(backup_dir, dirname)
            if os.path.isdir(path) and dirname.startswith('backup_'):
                backups.append((path, os.path.getmtime(path)))
        
        backups.sort(key=lambda x: x[1], reverse=True)
        
        # Keep only the specified number of most recent backups
        for backup_path, mtime in backups[self.config['backup_retention']:]:
            backup_time = datetime.fromtimestamp(mtime)
            if current_time - backup_time > max_age:
                shutil.rmtree(backup_path)
                self.logger.info(f"Removed old backup: {backup_path}")

    def optimize_performance(self):
        try:
            # Clear system cache
            subprocess.run(["sync"], check=True)
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("3")
            
            # Optimize database
            subprocess.run([
                "psql",
                "-U", "sutazai",
                "-d", "sutazai",
                "-c", "VACUUM ANALYZE;"
            ], check=True)
            
            return True
        except Exception as e:
            self.logger.error(f"Error optimizing performance: {str(e)}")
            return False

    def validate_security(self):
        try:
            # Check file permissions
            critical_paths = [
                os.path.join(PROJECT_ROOT, 'config'),
                os.path.join(PROJECT_ROOT, 'logs'),
                os.path.join(PROJECT_ROOT, 'backups')
            ]
            
            for path in critical_paths:
                if os.path.exists(path):
                    mode = os.stat(path).st_mode & 0o777
                    if mode != 0o750:
                        os.chmod(path, 0o750)
                        self.logger.warning(f"Fixed permissions for {path}")
            
            # Check running processes
            for proc in psutil.process_iter(['name', 'username']):
                if proc.info['name'].startswith('sutazai') and proc.info['username'] != 'sutazaiapp_dev':
                    self.logger.warning(f"Suspicious process found: {proc.info}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error validating security: {str(e)}")
            return False

    def run_maintenance(self) -> bool:
        """Run complete system maintenance."""
        logger.info("Starting system maintenance...")
        
        try:
            # Check system health
            health_status = self.check_system_health()
            if not health_status:
                logger.warning("System health issues detected")
            
            # Run maintenance tasks
            tasks = [
                (self.optimize_performance, "Performance optimization"),
                (self.validate_security, "Security validation"),
                (self.rotate_logs, "Log rotation"),
                (self.manage_backups, "Backup management"),
            ]
            
            for task, description in tasks:
                logger.info(f"Running {description}...")
                if not task():
                    logger.error(f"{description} failed")
                    return False
            
            logger.info("System maintenance completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during system maintenance: {e}")
            return False

def main():
    """Main entry point."""
    maintainer = SystemMaintainer()
    success = maintainer.run_maintenance()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
