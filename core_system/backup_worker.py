#!/usr/bin/env python3
"""
SutazAI Backup Worker Module
"""

import logging
import subprocess
from typing import List, Optional

class BackupWorker:
    """Worker for executing and managing backup tasks"""
    
    def __init__(self, 
                 backup_scripts: Optional[List[str]] = None,
                 verify_scripts: Optional[List[str]] = None,
                 interval: int = 86400):
        """
        Initialize the backup worker
        
        Args:
            backup_scripts: List of backup script paths
            verify_scripts: List of verification script paths
            interval: Execution interval in seconds
        """
        self.logger = logging.getLogger("BackupWorker")
        self.backup_scripts = backup_scripts if backup_scripts is not None else ["./scripts/backup_manager.sh"]
        self.verify_scripts = verify_scripts if verify_scripts is not None else ["./scripts/backup_verify.sh"]
        self.interval = interval
    
    def start(self) -> None:
        """Start the backup worker process"""
        self.logger.info("Starting backup worker with interval %d seconds", self.interval)
        self._execute_backup()
    
    def _execute_backup(self) -> bool:
        """
        Execute the backup process
        
        Returns:
            bool: True if backup was successful, False otherwise
        """
        try:
            for script in self.backup_scripts:
                self.logger.info("Running backup script: %s", script)
                result = subprocess.run(script, shell=True, check=True, 
                                      capture_output=True, text=True)
                self.logger.debug("Backup output: %s", result.stdout)
            
            self._verify_backup()
            return True
        except Exception as e:
            self.logger.error("Backup failed: %s", str(e))
            return False
    
    def _verify_backup(self) -> bool:
        """
        Verify backup integrity
        
        Returns:
            bool: True if verification was successful, False otherwise
        """
        try:
            for script in self.verify_scripts:
                self.logger.info("Running verification script: %s", script)
                result = subprocess.run(script, shell=True, check=True,
                                      capture_output=True, text=True)
                self.logger.debug("Verification output: %s", result.stdout)
            return True
        except Exception as e:
            self.logger.error("Backup verification failed: %s", str(e))
            return False
