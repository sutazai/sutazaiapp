#!/usr/bin/env python3
"""
Database Backup Retention Policy and Disaster Recovery Manager
============================================================

Implements a comprehensive backup retention strategy with:
- Multiple retention tiers (daily, weekly, monthly, yearly)
- Automated cleanup with safety checks
- Disaster recovery validation
- Backup integrity verification
- Cross-site backup management (if configured)

Retention Policy:
- Daily backups: Keep last 7 days
- Weekly backups: Keep last 4 weeks  
- Monthly backups: Keep last 12 months
- Yearly backups: Keep last 3 years
- Critical checkpoints: Never delete (marked backups)
"""

import os
import sys
import json
import logging
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import subprocess
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BackupFile:
    """Represents a database backup file with metadata"""
    path: str
    database: str
    timestamp: datetime
    size_bytes: int
    backup_type: str  # daily, weekly, monthly, yearly, critical
    checksum: Optional[str] = None
    verified: bool = False
    retention_date: Optional[datetime] = None


@dataclass
class RetentionReport:
    """Report on retention policy execution"""
    timestamp: str
    total_files_processed: int
    files_deleted: int
    files_retained: int
    space_freed_mb: int
    errors: List[str]
    retention_summary: Dict[str, int]


class BackupRetentionManager:
    """Manages backup retention policies and disaster recovery validation"""
    
    def __init__(self, backup_root: str = "/opt/sutazaiapp/backups"):
        self.backup_root = Path(backup_root)
        self.config_file = self.backup_root / "retention_config.json"
        
        # Default retention policy (days)
        self.retention_policy = {
            'daily': 7,
            'weekly': 28,     # 4 weeks
            'monthly': 365,   # 12 months
            'yearly': 1095,   # 3 years
            'critical': -1    # Never delete
        }
        
        # RTO/RPO requirements
        self.rto_hours = 2      # Recovery Time Objective: 2 hours
        self.rpo_minutes = 60   # Recovery Point Objective: 1 hour max data loss
        
        # Safety checks
        self.min_backups_per_db = 3  # Always keep at least 3 backups per database
        self.safety_margin_days = 1  # Keep extra day as safety margin
        
        self.load_config()
    
    def load_config(self):
        """Load retention configuration if exists"""
        try:
            if self.config_file.exists():
                with open(self.config_file) as f:
                    config = json.load(f)
                    self.retention_policy.update(config.get('retention_policy', {}))
                    self.rto_hours = config.get('rto_hours', self.rto_hours)
                    self.rpo_minutes = config.get('rpo_minutes', self.rpo_minutes)
                    logger.info("Loaded custom retention configuration")
        except Exception as e:
            logger.warning(f"Could not load retention config: {e}, using defaults")
    
    def save_config(self):
        """Save current configuration"""
        try:
            config = {
                'retention_policy': self.retention_policy,
                'rto_hours': self.rto_hours,
                'rpo_minutes': self.rpo_minutes,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved retention configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save retention config: {e}")
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for backup integrity"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""
    
    def verify_backup_integrity(self, backup: BackupFile) -> bool:
        """Verify backup file integrity"""
        try:
            file_path = Path(backup.path)
            
            # Check file exists and is not empty
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.error(f"Backup file missing or empty: {backup.path}")
                return False
            
            # Verify based on file type
            if backup.path.endswith('.gz'):
                # Test gzip integrity
                result = subprocess.run(['gzip', '-t', backup.path], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Corrupted gzip backup: {backup.path}")
                    return False
            
            elif backup.path.endswith('.tar.gz'):
                # Test tar.gz integrity
                result = subprocess.run(['tar', '-tzf', backup.path], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Corrupted tar.gz backup: {backup.path}")
                    return False
            
            # Verify checksum if available
            if backup.checksum:
                current_checksum = self.calculate_checksum(file_path)
                if current_checksum != backup.checksum:
                    logger.error(f"Checksum mismatch for {backup.path}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying backup {backup.path}: {e}")
            return False
    
    def discover_backup_files(self) -> List[BackupFile]:
        """Discover all backup files and categorize them"""
        backup_files = []
        
        # Database directories to scan
        db_dirs = ['postgres', 'postgresql', 'redis', 'neo4j', 'vector-databases']
        
        for db_dir in db_dirs:
            db_path = self.backup_root / db_dir
            if not db_path.exists():
                continue
            
            logger.info(f"Scanning {db_path} for backup files...")
            
            # Find all backup files (various extensions)
            patterns = ['*.gz', '*.sql', '*.tar.gz', '*.json', '*.dump']
            
            for pattern in patterns:
                for file_path in db_path.glob(pattern):
                    try:
                        # Extract timestamp from filename
                        timestamp = self.extract_timestamp_from_filename(file_path.name)
                        if not timestamp:
                            logger.warning(f"Could not extract timestamp from {file_path.name}")
                            continue
                        
                        # Determine backup type based on age
                        backup_type = self.classify_backup_by_age(timestamp)
                        
                        backup = BackupFile(
                            path=str(file_path),
                            database=db_dir,
                            timestamp=timestamp,
                            size_bytes=file_path.stat().st_size,
                            backup_type=backup_type,
                            checksum=None,
                            verified=False
                        )
                        
                        backup_files.append(backup)
                        
                    except Exception as e:
                        logger.warning(f"Error processing backup file {file_path}: {e}")
        
        logger.info(f"Discovered {len(backup_files)} backup files")
        return backup_files
    
    def extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from backup filename"""
        import re
        
        # Common timestamp patterns in backup filenames
        patterns = [
            r'(\d{8}_\d{6})',           # 20240815_143022
            r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})',  # 2024-08-15_14-30-22
            r'(\d{4}\d{2}\d{2}_\d{6})',  # 20240815_143022
            r'(\d{4}-\d{2}-\d{2})',      # 2024-08-15
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                timestamp_str = match.group(1)
                
                try:
                    # Try different parsing formats
                    formats = [
                        '%Y%m%d_%H%M%S',
                        '%Y-%m-%d_%H-%M-%S',
                        '%Y-%m-%d',
                        '%Y%m%d'
                    ]
                    
                    for fmt in formats:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
                    
                except Exception as e:
                    logger.debug(f"Failed to parse timestamp {timestamp_str}: {e}")
        
        # Fallback to file modification time
        try:
            file_path = self.backup_root / filename
            if file_path.exists():
                return datetime.fromtimestamp(file_path.stat().st_mtime)
        except Exception:
            pass
        
        return None
    
    def classify_backup_by_age(self, timestamp: datetime) -> str:
        """Classify backup type based on age"""
        now = datetime.now()
        age_days = (now - timestamp).days
        
        if age_days <= 7:
            return 'daily'
        elif age_days <= 28:
            return 'weekly'
        elif age_days <= 365:
            return 'monthly'
        else:
            return 'yearly'
    
    def calculate_retention_date(self, backup: BackupFile) -> datetime:
        """Calculate when this backup should be deleted"""
        if backup.backup_type == 'critical':
            # Never delete critical backups
            return datetime.max
        
        retention_days = self.retention_policy.get(backup.backup_type, 30)
        if retention_days == -1:
            return datetime.max
        
        # Add safety margin
        retention_days += self.safety_margin_days
        
        return backup.timestamp + timedelta(days=retention_days)
    
    def select_backups_for_deletion(self, backups: List[BackupFile]) -> List[BackupFile]:
        """Select which backups can be safely deleted"""
        to_delete = []
        now = datetime.now()
        
        # Group backups by database
        db_groups = {}
        for backup in backups:
            if backup.database not in db_groups:
                db_groups[backup.database] = []
            db_groups[backup.database].append(backup)
        
        for database, db_backups in db_groups.items():
            logger.info(f"Analyzing {len(db_backups)} backups for {database}")
            
            # Sort by timestamp (newest first)
            db_backups.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Always keep minimum number of backups
            if len(db_backups) <= self.min_backups_per_db:
                logger.info(f"Keeping all {len(db_backups)} backups for {database} (below minimum)")
                continue
            
            # Keep the newest backups regardless of age
            safe_backups = db_backups[:self.min_backups_per_db]
            candidate_backups = db_backups[self.min_backups_per_db:]
            
            for backup in candidate_backups:
                backup.retention_date = self.calculate_retention_date(backup)
                
                # Mark for deletion if past retention date
                if backup.retention_date <= now:
                    # Additional safety check - ensure we have a newer backup
                    newer_backups = [b for b in safe_backups if b.timestamp > backup.timestamp]
                    if newer_backups:
                        to_delete.append(backup)
                        logger.info(f"Marked for deletion: {backup.path} (age: {(now - backup.timestamp).days} days)")
                    else:
                        logger.warning(f"Keeping {backup.path} - no newer backup found")
        
        return to_delete
    
    def execute_retention_policy(self, dry_run: bool = False) -> RetentionReport:
        """Execute the retention policy"""
        logger.info(f"Executing retention policy (dry_run={dry_run})")
        
        start_time = datetime.now()
        errors = []
        space_freed = 0
        files_deleted = 0
        
        try:
            # Discover all backup files
            all_backups = self.discover_backup_files()
            
            if not all_backups:
                logger.warning("No backup files found")
                return RetentionReport(
                    timestamp=start_time.isoformat(),
                    total_files_processed=0,
                    files_deleted=0,
                    files_retained=0,
                    space_freed_mb=0,
                    errors=["No backup files found"],
                    retention_summary={}
                )
            
            # Verify backup integrity for recent backups
            logger.info("Verifying backup integrity...")
            recent_backups = [b for b in all_backups if (datetime.now() - b.timestamp).days <= 7]
            
            for backup in recent_backups:
                if self.verify_backup_integrity(backup):
                    backup.verified = True
                    logger.info(f"✓ Verified: {Path(backup.path).name}")
                else:
                    errors.append(f"Failed integrity check: {backup.path}")
                    logger.error(f"✗ Failed: {Path(backup.path).name}")
            
            # Select backups for deletion
            to_delete = self.select_backups_for_deletion(all_backups)
            
            logger.info(f"Selected {len(to_delete)} backups for deletion")
            
            # Execute deletions
            for backup in to_delete:
                try:
                    file_path = Path(backup.path)
                    file_size = file_path.stat().st_size
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete: {backup.path} ({file_size / 1024 / 1024:.1f}MB)")
                    else:
                        # Safety check - verify file still exists and is old enough
                        if file_path.exists() and (datetime.now() - backup.timestamp).days >= 7:
                            file_path.unlink()
                            logger.info(f"Deleted: {backup.path} ({file_size / 1024 / 1024:.1f}MB)")
                            space_freed += file_size
                            files_deleted += 1
                        else:
                            logger.warning(f"Skipping deletion of {backup.path} - safety check failed")
                
                except Exception as e:
                    error_msg = f"Failed to delete {backup.path}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Calculate retention summary
            retention_summary = {}
            for backup in all_backups:
                backup_type = backup.backup_type
                if backup_type not in retention_summary:
                    retention_summary[backup_type] = 0
                retention_summary[backup_type] += 1
            
            return RetentionReport(
                timestamp=start_time.isoformat(),
                total_files_processed=len(all_backups),
                files_deleted=files_deleted,
                files_retained=len(all_backups) - files_deleted,
                space_freed_mb=space_freed // (1024 * 1024),
                errors=errors,
                retention_summary=retention_summary
            )
        
        except Exception as e:
            error_msg = f"Retention policy execution failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            
            return RetentionReport(
                timestamp=start_time.isoformat(),
                total_files_processed=0,
                files_deleted=0,
                files_retained=0,
                space_freed_mb=0,
                errors=errors,
                retention_summary={}
            )
    
    def validate_disaster_recovery_readiness(self) -> Dict[str, Any]:
        """Validate disaster recovery capabilities"""
        logger.info("Validating disaster recovery readiness...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'rto_compliance': True,
            'rpo_compliance': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check backup freshness (RPO compliance)
        all_backups = self.discover_backup_files()
        db_groups = {}
        
        for backup in all_backups:
            if backup.database not in db_groups:
                db_groups[backup.database] = []
            db_groups[backup.database].append(backup)
        
        for database, backups in db_groups.items():
            if not backups:
                report['issues'].append(f"No backups found for {database}")
                report['rpo_compliance'] = False
                continue
            
            # Find most recent backup
            latest_backup = max(backups, key=lambda x: x.timestamp)
            backup_age_minutes = (datetime.now() - latest_backup.timestamp).total_seconds() / 60
            
            if backup_age_minutes > self.rpo_minutes:
                report['issues'].append(f"{database} backup is {backup_age_minutes:.0f} minutes old (RPO: {self.rpo_minutes} minutes)")
                report['rpo_compliance'] = False
            
            # Verify backup integrity for latest backup
            if not self.verify_backup_integrity(latest_backup):
                report['issues'].append(f"Latest {database} backup failed integrity check")
                report['rto_compliance'] = False
        
        # Check backup diversity (multiple retention tiers)
        for database, backups in db_groups.items():
            backup_types = set(b.backup_type for b in backups)
            
            if len(backup_types) < 2:
                report['recommendations'].append(f"Consider implementing multiple backup retention tiers for {database}")
            
            if len(backups) < 3:
                report['recommendations'].append(f"Increase backup frequency for {database} (only {len(backups)} backups found)")
        
        # Storage capacity check
        try:
            backup_size = sum(Path(b.path).stat().st_size for b in all_backups) / (1024 * 1024 * 1024)  # GB
            disk_usage = shutil.disk_usage(self.backup_root)
            free_space_gb = disk_usage.free / (1024 * 1024 * 1024)
            
            if free_space_gb < backup_size * 2:  # Less than 2x current backup size free
                report['issues'].append(f"Low disk space for backups: {free_space_gb:.1f}GB free")
                report['recommendations'].append("Increase backup storage capacity or implement backup archival")
        
        except Exception as e:
            report['issues'].append(f"Could not check storage capacity: {str(e)}")
        
        return report
    
    def generate_retention_report(self, report: RetentionReport) -> str:
        """Generate human-readable retention report"""
        lines = [
            "Database Backup Retention Report",
            "=" * 40,
            f"Timestamp: {report.timestamp}",
            f"Total Files Processed: {report.total_files_processed}",
            f"Files Deleted: {report.files_deleted}",
            f"Files Retained: {report.files_retained}",
            f"Space Freed: {report.space_freed_mb}MB",
            "",
            "Retention Summary by Type:",
        ]
        
        for backup_type, count in report.retention_summary.items():
            lines.append(f"  {backup_type}: {count} files")
        
        if report.errors:
            lines.extend([
                "",
                "Errors Encountered:",
                *[f"  • {error}" for error in report.errors]
            ])
        
        return "\n".join(lines)


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database backup retention manager")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be deleted without actually deleting")
    parser.add_argument('--validate-dr', action='store_true', help="Validate disaster recovery readiness")
    parser.add_argument('--backup-root', default="/opt/sutazaiapp/backups", help="Backup root directory")
    
    args = parser.parse_args()
    
    manager = BackupRetentionManager(args.backup_root)
    
    try:
        if args.validate_dr:
            # Disaster recovery validation
            dr_report = manager.validate_disaster_recovery_readiness()
            
            logger.info("Disaster Recovery Readiness Report")
            logger.info("=" * 40)
            logger.info(f"RTO Compliance: {'✓' if dr_report['rto_compliance'] else '✗'}")
            logger.info(f"RPO Compliance: {'✓' if dr_report['rpo_compliance'] else '✗'}")
            
            if dr_report['issues']:
                logger.info("\nIssues Found:")
                for issue in dr_report['issues']:
                    logger.info(f"  🚨 {issue}")
            
            if dr_report['recommendations']:
                logger.info("\nRecommendations:")
                for rec in dr_report['recommendations']:
                    logger.info(f"  💡 {rec}")
        
        else:
            # Execute retention policy
            report = manager.execute_retention_policy(dry_run=args.dry_run)
            
            # Print report
            logger.info(manager.generate_retention_report(report))
            
            # Save report
            report_file = Path(args.backup_root) / f"retention_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            
            logger.info(f"\nDetailed report saved to: {report_file}")
            
            # Return error code if there were errors
            if report.errors:
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Retention manager interrupted by user")
    except Exception as e:
        logger.error(f"Retention manager failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()