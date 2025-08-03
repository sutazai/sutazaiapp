#!/usr/bin/env python3
"""
Comprehensive Garbage Collection System for Sutazai App
Purpose: Automatically manage all report files, logs, and temporary files
Usage: python garbage-collection-system.py [--daemon] [--dry-run]
Requirements: Python 3.8+, psutil, schedule
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import gzip
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import threading
import signal
import hashlib
import psutil
import schedule
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

# Configuration constants
BASE_DIR = Path("/opt/sutazaiapp")
ARCHIVE_DIR = BASE_DIR / "archive" / "garbage-collection"
DB_PATH = BASE_DIR / "data" / "garbage-collection.db"
CONFIG_FILE = BASE_DIR / "config" / "garbage-collection.json"

# Default retention policies (in days)
DEFAULT_POLICIES = {
    "logs": {
        "deployment_*.log": 7,
        "health_*.log": 3,
        "compliance-*.log": 14,
        "*.log": 30,
        "max_size_mb": 100
    },
    "reports": {
        "*_report_*.json": 7,
        "*_report_*.md": 7,
        "latest.json": -1,  # Never delete
        "*.json": 14,
        "*.md": 14
    },
    "compliance-reports": {
        "latest.json": -1,  # Never delete
        "report_*.json": 7
    },
    "temporary": {
        "*.tmp": 1,
        "*.temp": 1,
        "*.bak": 3,
        "*.swp": 1,
        "*.pyc": 7,
        "__pycache__": 7
    },
    "archives": {
        "*.tar.gz": 30,
        "*.zip": 30,
        "retention_days": 90
    }
}

class FileStatus(Enum):
    """Status of files in the garbage collection system"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    COMPRESSED = "compressed"
    ERROR = "error"

@dataclass
class FileMetadata:
    """Metadata for tracked files"""
    path: str
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    status: FileStatus
    checksum: str
    archive_path: Optional[str] = None
    compressed: bool = False
    
class GarbageCollectionDB:
    """Database for tracking garbage collection activities"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS file_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    size INTEGER,
                    created TIMESTAMP,
                    modified TIMESTAMP,
                    accessed TIMESTAMP,
                    status TEXT,
                    checksum TEXT,
                    archive_path TEXT,
                    compressed BOOLEAN,
                    last_action TIMESTAMP,
                    action_type TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collection_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_time TIMESTAMP,
                    files_processed INTEGER,
                    files_archived INTEGER,
                    files_deleted INTEGER,
                    space_freed_mb REAL,
                    errors INTEGER,
                    duration_seconds REAL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS disk_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    total_gb REAL,
                    used_gb REAL,
                    free_gb REAL,
                    percent_used REAL,
                    logs_size_mb REAL,
                    reports_size_mb REAL,
                    archives_size_mb REAL
                )
            """)
    
    def track_file(self, metadata: FileMetadata, action: str):
        """Track file in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO file_tracking 
                (path, size, created, modified, accessed, status, checksum, 
                 archive_path, compressed, last_action, action_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.path, metadata.size,
                metadata.created, metadata.modified, metadata.accessed,
                metadata.status.value, metadata.checksum,
                metadata.archive_path, metadata.compressed,
                datetime.now(), action
            ))
    
    def record_run(self, stats: Dict):
        """Record garbage collection run statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO collection_runs 
                (run_time, files_processed, files_archived, files_deleted,
                 space_freed_mb, errors, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), stats['files_processed'],
                stats['files_archived'], stats['files_deleted'],
                stats['space_freed_mb'], stats['errors'],
                stats['duration_seconds']
            ))
    
    def record_disk_usage(self, usage: Dict):
        """Record current disk usage statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO disk_usage 
                (timestamp, total_gb, used_gb, free_gb, percent_used,
                 logs_size_mb, reports_size_mb, archives_size_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(), usage['total_gb'], usage['used_gb'],
                usage['free_gb'], usage['percent_used'],
                usage['logs_size_mb'], usage['reports_size_mb'],
                usage['archives_size_mb']
            ))

class GarbageCollector:
    """Main garbage collection system"""
    
    def __init__(self, config_path: Optional[Path] = None, dry_run: bool = False):
        self.base_dir = BASE_DIR
        self.archive_dir = ARCHIVE_DIR
        self.dry_run = dry_run
        self.config = self._load_config(config_path)
        self.db = GarbageCollectionDB(DB_PATH)
        self.logger = self._setup_logging()
        self._stop_event = threading.Event()
        
        # Create necessary directories
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'files_archived': 0,
            'files_deleted': 0,
            'space_freed_mb': 0,
            'errors': 0,
            'start_time': None,
            'duration_seconds': 0
        }
    
    def _load_config(self, config_path: Optional[Path] = None) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        elif CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
        else:
            # Save default config
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_POLICIES, f, indent=2)
            return DEFAULT_POLICIES
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("GarbageCollector")
        logger.setLevel(logging.INFO)
        
        # Rotate log file if it exists and is too large
        log_file = log_dir / "garbage-collection.log"
        if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
            log_file.rename(log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
        
        # File handler with rotation
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if not self.dry_run else logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def _get_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Get metadata for a file"""
        try:
            stat = file_path.stat()
            return FileMetadata(
                path=str(file_path),
                size=stat.st_size,
                created=datetime.fromtimestamp(stat.st_ctime),
                modified=datetime.fromtimestamp(stat.st_mtime),
                accessed=datetime.fromtimestamp(stat.st_atime),
                status=FileStatus.ACTIVE,
                checksum=self._calculate_checksum(file_path),
                compressed=False
            )
        except Exception as e:
            self.logger.error(f"Error getting metadata for {file_path}: {e}")
            return None
    
    def _should_process_file(self, file_path: Path, pattern_config: Dict) -> Tuple[bool, int]:
        """Check if file should be processed based on retention policies"""
        for pattern, retention_days in pattern_config.items():
            if pattern == "max_size_mb":
                continue
                
            if retention_days == -1:  # Never delete
                if file_path.match(pattern):
                    return False, -1
                    
            if file_path.match(pattern):
                file_age = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
                if file_age > retention_days:
                    return True, retention_days
                return False, retention_days
        
        return False, -1
    
    def _compress_file(self, file_path: Path) -> Optional[Path]:
        """Compress a file using gzip"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would compress: {file_path}")
            return None
            
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        try:
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Preserve timestamps
            shutil.copystat(file_path, compressed_path)
            
            # Remove original
            file_path.unlink()
            
            self.logger.info(f"Compressed: {file_path} -> {compressed_path}")
            return compressed_path
            
        except Exception as e:
            self.logger.error(f"Error compressing {file_path}: {e}")
            return None
    
    def _archive_file(self, file_path: Path, category: str) -> Optional[Path]:
        """Archive a file to the archive directory"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would archive: {file_path}")
            return None
            
        try:
            # Create archive subdirectory
            date_dir = self.archive_dir / category / datetime.now().strftime("%Y/%m/%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive path
            archive_path = date_dir / file_path.name
            
            # Handle existing archives
            if archive_path.exists():
                timestamp = datetime.now().strftime("%H%M%S")
                archive_path = date_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            
            # Move file
            shutil.move(str(file_path), str(archive_path))
            
            self.logger.info(f"Archived: {file_path} -> {archive_path}")
            return archive_path
            
        except Exception as e:
            self.logger.error(f"Error archiving {file_path}: {e}")
            self.stats['errors'] += 1
            return None
    
    def _delete_file(self, file_path: Path):
        """Delete a file"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would delete: {file_path}")
            return
            
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_path.unlink()
            self.stats['files_deleted'] += 1
            self.stats['space_freed_mb'] += size_mb
            self.logger.info(f"Deleted: {file_path} (freed {size_mb:.2f} MB)")
            
        except Exception as e:
            self.logger.error(f"Error deleting {file_path}: {e}")
            self.stats['errors'] += 1
    
    def _process_logs(self):
        """Process log files"""
        self.logger.info("Processing log files...")
        log_dir = self.base_dir / "logs"
        
        if not log_dir.exists():
            return
            
        log_config = self.config.get("logs", {})
        max_size_mb = log_config.get("max_size_mb", 100)
        
        for log_file in log_dir.rglob("*.log"):
            # Skip our own log
            if "garbage-collection" in log_file.name:
                continue
                
            self.stats['files_processed'] += 1
            
            # Check size limit
            size_mb = log_file.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                compressed = self._compress_file(log_file)
                if compressed:
                    log_file = compressed
                    
            # Check retention
            should_process, retention = self._should_process_file(log_file, log_config)
            if should_process:
                # Archive important logs, delete others
                if any(keyword in log_file.name for keyword in ["deployment", "health", "compliance"]):
                    archived = self._archive_file(log_file, "logs")
                    if archived:
                        self.stats['files_archived'] += 1
                        metadata = self._get_file_metadata(archived)
                        if metadata:
                            metadata.status = FileStatus.ARCHIVED
                            metadata.archive_path = str(archived)
                            self.db.track_file(metadata, "archive")
                else:
                    self._delete_file(log_file)
    
    def _process_reports(self):
        """Process report files"""
        self.logger.info("Processing report files...")
        
        # Process main reports directory
        reports_dir = self.base_dir / "reports"
        if reports_dir.exists():
            report_config = self.config.get("reports", {})
            for report_file in reports_dir.rglob("*"):
                if report_file.is_file():
                    self.stats['files_processed'] += 1
                    should_process, retention = self._should_process_file(report_file, report_config)
                    if should_process:
                        # Archive all reports before deletion
                        archived = self._archive_file(report_file, "reports")
                        if archived:
                            self.stats['files_archived'] += 1
                            metadata = self._get_file_metadata(archived)
                            if metadata:
                                metadata.status = FileStatus.ARCHIVED
                                metadata.archive_path = str(archived)
                                self.db.track_file(metadata, "archive")
        
        # Process compliance reports
        compliance_dir = self.base_dir / "compliance-reports"
        if compliance_dir.exists():
            compliance_config = self.config.get("compliance-reports", {})
            for report_file in compliance_dir.glob("*.json"):
                self.stats['files_processed'] += 1
                should_process, retention = self._should_process_file(report_file, compliance_config)
                if should_process:
                    archived = self._archive_file(report_file, "compliance-reports")
                    if archived:
                        self.stats['files_archived'] += 1
    
    def _process_temporary_files(self):
        """Process temporary files"""
        self.logger.info("Processing temporary files...")
        temp_config = self.config.get("temporary", {})
        
        # Common temporary file locations
        temp_dirs = [
            self.base_dir,
            self.base_dir / "tmp",
            self.base_dir / "temp",
            self.base_dir / "cache"
        ]
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
                
            for pattern, retention_days in temp_config.items():
                for temp_file in temp_dir.rglob(pattern):
                    if temp_file.is_file():
                        self.stats['files_processed'] += 1
                        file_age = (datetime.now() - datetime.fromtimestamp(temp_file.stat().st_mtime)).days
                        if file_age > retention_days:
                            self._delete_file(temp_file)
    
    def _clean_old_archives(self):
        """Clean old archives based on retention policy"""
        self.logger.info("Cleaning old archives...")
        archive_config = self.config.get("archives", {})
        retention_days = archive_config.get("retention_days", 90)
        
        if not self.archive_dir.exists():
            return
            
        for archive_file in self.archive_dir.rglob("*"):
            if archive_file.is_file():
                file_age = (datetime.now() - datetime.fromtimestamp(archive_file.stat().st_mtime)).days
                if file_age > retention_days:
                    self._delete_file(archive_file)
    
    def _clean_empty_directories(self):
        """Remove empty directories"""
        self.logger.info("Cleaning empty directories...")
        
        for root, dirs, files in os.walk(self.base_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        if not self.dry_run:
                            dir_path.rmdir()
                            self.logger.info(f"Removed empty directory: {dir_path}")
                        else:
                            self.logger.info(f"[DRY RUN] Would remove empty directory: {dir_path}")
                except Exception as e:
                    self.logger.debug(f"Could not remove directory {dir_path}: {e}")
    
    def _calculate_disk_usage(self) -> Dict:
        """Calculate current disk usage statistics"""
        disk_usage = psutil.disk_usage(str(self.base_dir))
        
        # Calculate size of specific directories
        def get_dir_size(path: Path) -> float:
            total = 0
            if path.exists():
                for file in path.rglob("*"):
                    if file.is_file():
                        total += file.stat().st_size
            return total / (1024 * 1024)  # Convert to MB
        
        return {
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3),
            'free_gb': disk_usage.free / (1024**3),
            'percent_used': disk_usage.percent,
            'logs_size_mb': get_dir_size(self.base_dir / "logs"),
            'reports_size_mb': get_dir_size(self.base_dir / "reports") + 
                              get_dir_size(self.base_dir / "compliance-reports"),
            'archives_size_mb': get_dir_size(self.archive_dir)
        }
    
    def _generate_report(self) -> Dict:
        """Generate garbage collection report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'dry_run' if self.dry_run else 'live',
            'statistics': self.stats,
            'disk_usage': self._calculate_disk_usage(),
            'configuration': self.config
        }
        
        # Save report
        report_path = self.base_dir / "reports" / f"garbage_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        if not self.dry_run:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
    
    def run_collection(self):
        """Run a complete garbage collection cycle"""
        self.logger.info("Starting garbage collection cycle...")
        self.stats['start_time'] = time.time()
        
        try:
            # Record initial disk usage
            initial_usage = self._calculate_disk_usage()
            self.db.record_disk_usage(initial_usage)
            
            # Process different file categories
            self._process_logs()
            self._process_reports()
            self._process_temporary_files()
            self._clean_old_archives()
            self._clean_empty_directories()
            
            # Calculate duration
            self.stats['duration_seconds'] = time.time() - self.stats['start_time']
            
            # Record final disk usage
            final_usage = self._calculate_disk_usage()
            self.db.record_disk_usage(final_usage)
            
            # Record run statistics
            self.db.record_run(self.stats)
            
            # Generate report
            report = self._generate_report()
            
            self.logger.info(f"Garbage collection completed: "
                           f"{self.stats['files_processed']} files processed, "
                           f"{self.stats['files_archived']} archived, "
                           f"{self.stats['files_deleted']} deleted, "
                           f"{self.stats['space_freed_mb']:.2f} MB freed")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")
            self.stats['errors'] += 1
            raise
    
    def start_daemon(self):
        """Start garbage collector as a daemon with scheduled runs"""
        self.logger.info("Starting garbage collection daemon...")
        
        # Schedule runs
        schedule.every(1).hours.do(self.run_collection)
        schedule.every().day.at("02:00").do(self._clean_old_archives)
        
        # Run initial collection
        self.run_collection()
        
        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_daemon(self):
        """Stop the daemon"""
        self.logger.info("Stopping garbage collection daemon...")
        self._stop_event.set()

class GarbageCollectionDashboard:
    """Dashboard for monitoring garbage collection status"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
    
    def get_status(self) -> Dict:
        """Get current garbage collection status"""
        with sqlite3.connect(self.db_path) as conn:
            # Get latest run
            latest_run = conn.execute("""
                SELECT * FROM collection_runs 
                ORDER BY run_time DESC LIMIT 1
            """).fetchone()
            
            # Get latest disk usage
            latest_usage = conn.execute("""
                SELECT * FROM disk_usage 
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            
            # Get statistics for last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            daily_stats = conn.execute("""
                SELECT 
                    COUNT(*) as runs,
                    SUM(files_processed) as total_processed,
                    SUM(files_archived) as total_archived,
                    SUM(files_deleted) as total_deleted,
                    SUM(space_freed_mb) as total_freed_mb
                FROM collection_runs
                WHERE run_time > ?
            """, (yesterday,)).fetchone()
            
        return {
            'latest_run': latest_run,
            'disk_usage': latest_usage,
            'daily_statistics': daily_stats
        }
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        status = self.get_status()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Garbage Collection Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .card {{ background: #f0f0f0; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .warning {{ color: #ff9800; }}
                .error {{ color: #f44336; }}
                .success {{ color: #4CAF50; }}
            </style>
            <meta http-equiv="refresh" content="60">
        </head>
        <body>
            <h1>Garbage Collection Dashboard</h1>
            <div class="card">
                <h2>Disk Usage</h2>
                <div class="metric">
                    <div class="metric-value">{status['disk_usage'][4]:.1f}%</div>
                    <div class="metric-label">Disk Used</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['disk_usage'][3]:.1f} GB</div>
                    <div class="metric-label">Free Space</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['disk_usage'][5]:.1f} MB</div>
                    <div class="metric-label">Logs Size</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['disk_usage'][6]:.1f} MB</div>
                    <div class="metric-label">Reports Size</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Last 24 Hours</h2>
                <div class="metric">
                    <div class="metric-value">{status['daily_statistics'][0]}</div>
                    <div class="metric-label">Collection Runs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['daily_statistics'][1] or 0}</div>
                    <div class="metric-label">Files Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status['daily_statistics'][4] or 0:.1f} MB</div>
                    <div class="metric-label">Space Freed</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Latest Run</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Files Processed</th>
                        <th>Archived</th>
                        <th>Deleted</th>
                        <th>Space Freed</th>
                        <th>Duration</th>
                        <th>Errors</th>
                    </tr>
                    <tr>
                        <td>{status['latest_run'][1]}</td>
                        <td>{status['latest_run'][2]}</td>
                        <td>{status['latest_run'][3]}</td>
                        <td>{status['latest_run'][4]}</td>
                        <td>{status['latest_run'][5]:.1f} MB</td>
                        <td>{status['latest_run'][7]:.1f}s</td>
                        <td class="{'error' if status['latest_run'][6] > 0 else 'success'}">{status['latest_run'][6]}</td>
                    </tr>
                </table>
            </div>
            
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        return html

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logging.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Garbage Collection System for Sutazai App")
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no changes)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard')
    parser.add_argument('--status', action='store_true', help='Show current status')
    
    args = parser.parse_args()
    
    if args.dashboard:
        dashboard = GarbageCollectionDashboard()
        html = dashboard.generate_html_dashboard()
        dashboard_path = BASE_DIR / "dashboard" / "garbage-collection.html"
        dashboard_path.parent.mkdir(exist_ok=True)
        with open(dashboard_path, 'w') as f:
            f.write(html)
        print(f"Dashboard generated: {dashboard_path}")
        return
    
    if args.status:
        dashboard = GarbageCollectionDashboard()
        status = dashboard.get_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create garbage collector
    config_path = Path(args.config) if args.config else None
    collector = GarbageCollector(config_path=config_path, dry_run=args.dry_run)
    
    if args.daemon:
        collector.start_daemon()
    else:
        report = collector.run_collection()
        print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()