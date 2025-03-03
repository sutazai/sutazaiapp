#!/opt/sutazaiapp/venv/bin/python3"""System Maintenance ModuleHandles system maintenance tasks including log rotation and disk space monitoring."""import loggingimport osimport shutilimport sysfrom datetime import datetime, timedeltafrom pathlib import Pathfrom typing import Dict, List, Optional, Setlogging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s",handlers=[logging.StreamHandler(),logging.FileHandler("/opt/sutazaiapp/logs/system_maintenance.log"),],)logger = logging.getLogger(__name__)class SystemMaintenance:    """
Handles comprehensive system maintenance tasks for SutazAI.
Provides methods for:
- Disk space monitoring and cleanup
- Log file rotation and archival
- Cache and temporary file cleanup
- System optimization
"""
def __init__(self, base_path: str = "/opt/sutazaiapp"):        """
    Initialize the system maintenance handler.
    Args:
    base_path: Base directory path for the application
    """
    self.base_path = Path(base_path)
    self.log_dir = self.base_path / "logs"
    self.cache_dir = self.base_path / "cache"
    self.temp_dir = self.base_path / "temp"
    # Create necessary directories
    for directory in [self.log_dir, self.cache_dir, self.temp_dir]:
    directory.mkdir(parents=True, exist_ok=True)
    # Configure thresholds
    self.thresholds = {
    "disk_warning": 80.0,  # Warn at 80% usage
    "disk_critical": 90.0,  # Critical at 90% usage
    "log_max_size_mb": 100,  # Max log file size in MB
    "log_max_age_days": 30,  # Max age for log files
    "cache_max_age_days": 7,  # Max age for cache files
    }
    def check_disk_space(self) -> Dict[str, float]:            """
        Check disk space usage with detailed metrics.
        Returns:
        Dictionary containing disk space metrics
        """
        try:
            disk_info = os.statvfs(self.base_path)
            total = disk_info.f_blocks * disk_info.f_frsize
            free = disk_info.f_bfree * disk_info.f_frsize
            used = total - free
            metrics = {
            "total_gb": total / (1024**3),
            "used_gb": used / (1024**3),
            "free_gb": free / (1024**3),
            "usage_percent": (used / total) * 100,
            }
            # Log warnings based on thresholds
            if metrics["usage_percent"] >= self.thresholds["disk_critical"]:
                logger.critical(
                f"Critical disk usage: {metrics['usage_percent']:.1f}% "
                f"({metrics['free_gb']:.1f}GB free)"
                )
                elif metrics["usage_percent"] >= self.thresholds["disk_warning"]:
                    logger.warning(
                    f"High disk usage: {metrics['usage_percent']:.1f}% "
                    f"({metrics['free_gb']:.1f}GB free)"
                    )
                    return metrics
                except Exception as e:                        logger.error(f"Failed to check disk space: {e}")                        return {}                    def rotate_logs(self) -> None:                        """
                    Rotate and archive log files based on size and age.
                    """
                    try:
                        current_time = datetime.now()
                        max_size = self.thresholds["log_max_size_mb"] * 1024 * 1024  # Convert to bytes
                        max_age = current_time - timedelta(days=self.thresholds["log_max_age_days"])
                        for log_file in self.log_dir.rglob("*.log"):
                        try:
                            # Skip if file is a symlink
                            if log_file.is_symlink():
                                continue
                            stats = log_file.stat()
                            file_time = datetime.fromtimestamp(stats.st_mtime)
                            # Check if file needs rotation
                            if stats.st_size > max_size or file_time < max_age:
                                # Create archive filename with timestamp
                                archive_name = (
                                log_file.with_name(
                                f"{log_file.stem}_{file_time.strftime('%Y%m%d_%H%M%S')}.log.gz"
                                )
                                )
                                # Compress the log file
                                import gzip
                                with log_file.open("rb") as f_in:
                                with gzip.open(archive_name, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                                # Truncate original file
                                log_file.write_text("")
                                logger.info(f"Rotated log file: {log_file} -> {archive_name}")
                                except Exception as e:
                                    logger.error(f"Failed to rotate log file {log_file}: {e}")
                                    except Exception as e:
                                        logger.error(f"Failed to rotate logs: {e}")
                                        def cleanup_cache(self) -> None:                                                """
                                            Clean up old cache files and temporary data.
                                            """
                                            try:
                                                current_time = datetime.now()
                                                max_age = current_time - timedelta(days=self.thresholds["cache_max_age_days"])
                                                # Clean cache directory
                                                cleaned_files = 0
                                                cleaned_size = 0
                                                for cache_file in self.cache_dir.rglob("*"):
                                                try:
                                                    if cache_file.is_file() and not cache_file.is_symlink():
                                                        if datetime.fromtimestamp(cache_file.stat().st_mtime) < max_age:
                                                            file_size = cache_file.stat().st_size
                                                            cache_file.unlink()
                                                            cleaned_files += 1
                                                            cleaned_size += file_size
                                                            except Exception as e:
                                                                logger.error(f"Failed to clean cache file {cache_file}: {e}")
                                                                if cleaned_files > 0:
                                                                    logger.info(
                                                                    f"Cleaned {cleaned_files} cache files "
                                                                    f"(freed {cleaned_size / 1024**2:.1f}MB)"
                                                                    )
                                                                    # Clean temp directory
                                                                    shutil.rmtree(self.temp_dir)
                                                                    self.temp_dir.mkdir(parents=True, exist_ok=True)
                                                                    logger.info("Cleaned temporary directory")
                                                                    except Exception as e:
                                                                        logger.error(f"Failed to clean cache: {e}")
                                                                        def optimize_storage(self) -> None:                                                                                """
                                                                            Perform storage optimization tasks.
                                                                            """
                                                                            try:
                                                                                # Remove empty directories
                                                                                empty_dirs = 0
                                                                                for dirpath, dirnames, filenames in os.walk(self.base_path, topdown=False):
                                                                                if not dirnames and not filenames and dirpath != str(self.base_path):
                                                                                    try:
                                                                                        os.rmdir(dirpath)
                                                                                        empty_dirs += 1
                                                                                        except OSError:
                                                                                            continue
                                                                                        if empty_dirs > 0:
                                                                                            logger.info(f"Removed {empty_dirs} empty directories")
                                                                                            # Clean __pycache__ directories
                                                                                            pycache_dirs = 0
                                                                                            pycache_files = 0
                                                                                            for pycache in self.base_path.rglob("__pycache__"):
                                                                                            if pycache.is_dir():
                                                                                                try:
                                                                                                    shutil.rmtree(pycache)
                                                                                                    pycache_dirs += 1
                                                                                                    pycache_files += len(list(pycache.rglob("*.pyc")))
                                                                                                    except Exception:
                                                                                                        continue
                                                                                                    if pycache_dirs > 0:
                                                                                                        logger.info(
                                                                                                        f"Cleaned {pycache_dirs} __pycache__ directories "
                                                                                                        f"({pycache_files} files)"
                                                                                                        )
                                                                                                        except Exception as e:
                                                                                                            logger.error(f"Failed to optimize storage: {e}")
                                                                                                            def run_maintenance(self) -> None:                                                                                                                    """Execute all maintenance tasks."""
                                                                                                                logger.info("Starting system maintenance tasks...")
                                                                                                                try:
                                                                                                                    # Check disk space first
                                                                                                                    disk_space = self.check_disk_space()
                                                                                                                    if disk_space:
                                                                                                                        logger.info(
                                                                                                                        f"Initial Disk Status: {disk_space['usage_percent']:.1f}% used "
                                                                                                                        f"({disk_space['free_gb']:.1f}/{disk_space['total_gb']:.1f} GB free)"
                                                                                                                        )
                                                                                                                        # Run maintenance tasks
                                                                                                                        self.rotate_logs()
                                                                                                                        self.cleanup_cache()
                                                                                                                        self.optimize_storage()
                                                                                                                        # Check disk space after maintenance
                                                                                                                        disk_space = self.check_disk_space()
                                                                                                                        if disk_space:
                                                                                                                            logger.info(
                                                                                                                            f"Final Disk Status: {disk_space['usage_percent']:.1f}% used "
                                                                                                                            f"({disk_space['free_gb']:.1f}/{disk_space['total_gb']:.1f} GB free)"
                                                                                                                            )
                                                                                                                            logger.info("System maintenance completed successfully")
                                                                                                                            except Exception as e:
                                                                                                                                logger.error(f"System maintenance failed: {e}")
                                                                                                                                raise
                                                                                                                                def main() -> None:                                                                                                                                        """Run system maintenance with error handling."""
                                                                                                                                    try:
                                                                                                                                        maintenance = SystemMaintenance()
                                                                                                                                        maintenance.run_maintenance()
                                                                                                                                        except Exception as e:
                                                                                                                                            logger.error(f"Fatal error in system maintenance: {e}")
                                                                                                                                            sys.exit(1)
                                                                                                                                            if __name__ == "__main__":
                                                                                                                                                main()

