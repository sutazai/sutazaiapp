"""
Automated Backup System for SutazAI
Comprehensive backup and recovery with versioning
"""

import asyncio
import json
import logging
import shutil
import tarfile
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_root: str = "/opt/sutazaiapp/backups"
    retention_days: int = 30
    compression_enabled: bool = True
    incremental_enabled: bool = True
    max_backup_size_gb: float = 10.0
    backup_schedule: Dict[str, int] = None  # {"daily": 7, "weekly": 4, "monthly": 12}
    
    def __post_init__(self):
        if self.backup_schedule is None:
            self.backup_schedule = {"daily": 7, "weekly": 4, "monthly": 12}

class AutomatedBackupManager:
    """Automated backup and recovery system"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        self.backup_root = Path(self.config.backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        
        # Backup directories
        self.daily_dir = self.backup_root / "daily"
        self.weekly_dir = self.backup_root / "weekly"
        self.monthly_dir = self.backup_root / "monthly"
        self.incremental_dir = self.backup_root / "incremental"
        
        for directory in [self.daily_dir, self.weekly_dir, self.monthly_dir, self.incremental_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.backup_lock = threading.Lock()
        self.backup_history = []
        
        # Statistics
        self.stats = {
            "total_backups": 0,
            "successful_backups": 0,
            "failed_backups": 0,
            "total_backup_size": 0,
            "last_backup_time": 0,
            "average_backup_time": 0.0
        }
    
    async def initialize(self):
        """Initialize backup manager"""
        logger.info("🔄 Initializing Automated Backup Manager")
        
        # Load backup history
        await self._load_backup_history()
        
        # Start backup scheduler
        asyncio.create_task(self._backup_scheduler())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info("✅ Backup manager initialized")
    
    async def _load_backup_history(self):
        """Load backup history from disk"""
        try:
            history_file = self.backup_root / "backup_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.backup_history = data.get("backups", [])
                    self.stats.update(data.get("stats", {}))
                
                logger.info(f"Loaded backup history with {len(self.backup_history)} entries")
        except Exception as e:
            logger.warning(f"Failed to load backup history: {e}")
    
    async def create_backup(self, backup_type: str = "manual", include_paths: List[str] = None) -> Dict[str, Any]:
        """Create a new backup"""
        start_time = time.time()
        backup_id = f"{backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Creating backup: {backup_id}")
        
        with self.backup_lock:
            try:
                # Default paths to backup
                if include_paths is None:
                    include_paths = [
                        "/opt/sutazaiapp/data",
                        "/opt/sutazaiapp/config",
                        "/opt/sutazaiapp/models",
                        "/opt/sutazaiapp/logs",
                        "/opt/sutazaiapp/.env"
                    ]
                
                # Determine backup directory
                if backup_type == "daily":
                    backup_dir = self.daily_dir
                elif backup_type == "weekly":
                    backup_dir = self.weekly_dir
                elif backup_type == "monthly":
                    backup_dir = self.monthly_dir
                else:
                    backup_dir = self.backup_root / "manual"
                    backup_dir.mkdir(exist_ok=True)
                
                # Create backup archive
                backup_file = backup_dir / f"{backup_id}.tar.gz"
                
                # Create manifest
                manifest = {
                    "backup_id": backup_id,
                    "backup_type": backup_type,
                    "created_at": time.time(),
                    "include_paths": include_paths,
                    "compression_enabled": self.config.compression_enabled,
                    "incremental": False
                }
                
                # Create tar archive
                with tarfile.open(backup_file, 'w:gz' if self.config.compression_enabled else 'w') as tar:
                    for path_str in include_paths:
                        path = Path(path_str)
                        if path.exists():
                            if path.is_file():
                                tar.add(path, arcname=path.name)
                            else:
                                tar.add(path, arcname=path.name)
                
                # Get backup size
                backup_size = backup_file.stat().st_size
                manifest["backup_size"] = backup_size
                
                # Save manifest
                manifest_file = backup_dir / f"{backup_id}.json"
                with open(manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Update statistics
                backup_time = time.time() - start_time
                self.stats["total_backups"] += 1
                self.stats["successful_backups"] += 1
                self.stats["total_backup_size"] += backup_size
                self.stats["last_backup_time"] = time.time()
                self.stats["average_backup_time"] = (
                    (self.stats["average_backup_time"] * (self.stats["total_backups"] - 1) + backup_time) /
                    self.stats["total_backups"]
                )
                
                # Add to history
                backup_record = {
                    **manifest,
                    "backup_time": backup_time,
                    "status": "completed"
                }
                self.backup_history.append(backup_record)
                
                # Save history
                await self._save_backup_history()
                
                logger.info(f"Backup completed: {backup_id} ({backup_size / 1024 / 1024:.1f}MB)")
                
                return {
                    "backup_id": backup_id,
                    "status": "completed",
                    "backup_file": str(backup_file),
                    "backup_size": backup_size,
                    "backup_time": backup_time
                }
                
            except Exception as e:
                logger.error(f"Backup failed: {e}")
                
                self.stats["failed_backups"] += 1
                
                # Add failed backup to history
                backup_record = {
                    "backup_id": backup_id,
                    "backup_type": backup_type,
                    "created_at": time.time(),
                    "status": "failed",
                    "error": str(e)
                }
                self.backup_history.append(backup_record)
                
                return {
                    "backup_id": backup_id,
                    "status": "failed",
                    "error": str(e)
                }
    
    async def restore_backup(self, backup_id: str, restore_path: str = None) -> Dict[str, Any]:
        """Restore from backup"""
        logger.info(f"Restoring backup: {backup_id}")
        
        try:
            # Find backup file
            backup_file = None
            manifest_file = None
            
            for backup_dir in [self.daily_dir, self.weekly_dir, self.monthly_dir, self.backup_root / "manual"]:
                potential_backup = backup_dir / f"{backup_id}.tar.gz"
                potential_manifest = backup_dir / f"{backup_id}.json"
                
                if potential_backup.exists() and potential_manifest.exists():
                    backup_file = potential_backup
                    manifest_file = potential_manifest
                    break
            
            if not backup_file:
                return {"status": "failed", "error": "Backup not found"}
            
            # Load manifest
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Determine restore path
            if restore_path is None:
                restore_path = "/opt/sutazaiapp/restore"
            
            restore_dir = Path(restore_path)
            restore_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz' if self.config.compression_enabled else 'r') as tar:
                tar.extractall(restore_dir)
            
            logger.info(f"Backup restored to: {restore_dir}")
            
            return {
                "status": "completed",
                "backup_id": backup_id,
                "restore_path": str(restore_dir),
                "manifest": manifest
            }
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def list_backups(self, backup_type: str = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for backup_record in self.backup_history:
            if backup_type is None or backup_record.get("backup_type") == backup_type:
                backups.append(backup_record)
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return backups
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        try:
            # Find and delete backup files
            deleted = False
            
            for backup_dir in [self.daily_dir, self.weekly_dir, self.monthly_dir, self.backup_root / "manual"]:
                backup_file = backup_dir / f"{backup_id}.tar.gz"
                manifest_file = backup_dir / f"{backup_id}.json"
                
                if backup_file.exists():
                    backup_file.unlink()
                    deleted = True
                
                if manifest_file.exists():
                    manifest_file.unlink()
            
            # Remove from history
            self.backup_history = [
                record for record in self.backup_history
                if record.get("backup_id") != backup_id
            ]
            
            await self._save_backup_history()
            
            if deleted:
                logger.info(f"Deleted backup: {backup_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def _backup_scheduler(self):
        """Automated backup scheduler"""
        last_daily = 0
        last_weekly = 0
        last_monthly = 0
        
        while True:
            try:
                current_time = time.time()
                
                # Daily backup (once per day)
                if current_time - last_daily > 24 * 3600:
                    await self.create_backup("daily")
                    last_daily = current_time
                
                # Weekly backup (once per week)
                if current_time - last_weekly > 7 * 24 * 3600:
                    await self.create_backup("weekly")
                    last_weekly = current_time
                
                # Monthly backup (once per month)
                if current_time - last_monthly > 30 * 24 * 3600:
                    await self.create_backup("monthly")
                    last_monthly = current_time
                
                # Sleep for 1 hour before next check
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_task(self):
        """Clean up old backups based on retention policy"""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                
                current_time = time.time()
                
                # Clean up based on retention policy
                for backup_type, retention_count in self.config.backup_schedule.items():
                    if backup_type == "daily":
                        backup_dir = self.daily_dir
                    elif backup_type == "weekly":
                        backup_dir = self.weekly_dir
                    elif backup_type == "monthly":
                        backup_dir = self.monthly_dir
                    else:
                        continue
                    
                    # Get all backups of this type
                    backup_files = list(backup_dir.glob("*.tar.gz"))
                    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    # Keep only the most recent ones
                    for backup_file in backup_files[retention_count:]:
                        try:
                            backup_id = backup_file.stem.replace('.tar', '')
                            await self.delete_backup(backup_id)
                        except Exception as e:
                            logger.warning(f"Failed to delete old backup {backup_file}: {e}")
                
            except Exception as e:
                logger.error(f"Backup cleanup error: {e}")
    
    async def _save_backup_history(self):
        """Save backup history to disk"""
        try:
            history_file = self.backup_root / "backup_history.json"
            
            data = {
                "backups": self.backup_history[-1000:],  # Keep last 1000 records
                "stats": self.stats,
                "saved_at": time.time()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup system statistics"""
        success_rate = (
            self.stats["successful_backups"] / max(1, self.stats["total_backups"]) * 100
        )
        
        return {
            **self.stats,
            "success_rate_percent": success_rate,
            "total_backup_size_gb": self.stats["total_backup_size"] / (1024 ** 3),
            "average_backup_size_mb": (
                self.stats["total_backup_size"] / max(1, self.stats["successful_backups"]) / (1024 ** 2)
            ),
            "retention_policy": self.config.backup_schedule,
            "backup_history_count": len(self.backup_history)
        }

# Global backup manager instance
backup_manager = AutomatedBackupManager()
