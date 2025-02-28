#!/opt/sutazaiapp/venv/bin/python3
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class AdvancedSystemMaintenance:    def __init__(self, project_root: str):        # Ensure project_root is set first
        self.project_root = project_root

        # Create backup and log directory paths
        self.backup_dir = os.path.join(project_root, 'misc', 'system_backups')
        self.log_dir = os.path.join(project_root, 'logs')

        # Ensure backup and log directories exist
        try:            os.makedirs(self.backup_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:            logger.error(f"Failed to create backup or log directories: {e}")
            # Fallback to temporary directories if creation fails
            import tempfile
            self.backup_dir = tempfile.mkdtemp(prefix='sutazai_backups_')
            self.log_dir = tempfile.mkdtemp(prefix='sutazai_logs_')
            logger.warning(
                f"Using temporary directories: {self.backup_dir}, {self.log_dir}")

    def create_comprehensive_backup(self) -> str:        """Create a comprehensive backup of the entire project."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"sutazai_backup_{timestamp}"
        backup_path = os.path.join(self.backup_dir, backup_name)

        try:            # Use tar for efficient compression
            subprocess.run([
                "tar",
                "-czf",
                f"{backup_path}.tar.gz",
                "-C",
                self.project_root,
                ".",
            ], check=True)

            logger.info(f"Comprehensive backup created: {backup_path}.tar.gz")
            return backup_path
        except subprocess.CalledProcessError as e:            logger.error(f"Backup creation failed: {e}")
            return ""

    def clean_old_backups(self, days_to_keep: int = 30):        """Remove backups older than specified days."""
        current_time = datetime.now()

        for backup in os.listdir(self.backup_dir):            backup_path = os.path.join(self.backup_dir, backup)

            # Skip if not a backup file
            if not backup.startswith(
                    "sutazai_backup_") or not backup.endswith(".tar.gz"):                continue

            try:                # Extract timestamp from filename
                timestamp_str = backup.split("_", 2)[2].split(".")[0]
                backup_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                # Calculate age of backup
                backup_age = (current_time - backup_time).days

                if backup_age > days_to_keep:                    os.remove(backup_path)
                    logger.info(f"Removed old backup: {backup}")
            except Exception as e:                logger.error(f"Error processing backup {backup}: {e}")

    def rotate_log_files(
        self,
        max_log_size_mb: int = 100,
        max_log_files: int = 5
    ):        """Rotate and compress log files."""
        for log_file in os.listdir(self.log_dir):            log_path = os.path.join(self.log_dir, log_file)

            # Check log file size
            if os.path.getsize(log_path) > max_log_size_mb * 1024 * 1024:                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_log = f"{log_file}.{timestamp}"

                # Compress old log
                subprocess.run(["gzip", log_path], check=True)

                # Remove excess log files
                log_files = sorted(
                    [f for f in os.listdir(self.log_dir)
                    if f.startswith(log_file)],
                    reverse=True,
                )
                for old_log in log_files[max_log_files:]:                    os.remove(os.path.join(self.log_dir, old_log))

    def check_disk_health(self) -> Dict[str, Any]:        """Check disk health and usage."""
        try:            disk_usage = shutil.disk_usage(self.project_root)
            return {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent_used": disk_usage.used / disk_usage.total * 100,
            }
        except Exception as e:            logger.error(f"Disk health check failed: {e}")
            return {}

    def run_maintenance(self):        """Run comprehensive system maintenance."""
        # Create comprehensive backup
        self.create_comprehensive_backup()

        # Clean old backups
        self.clean_old_backups()

        # Rotate log files
        self.rotate_log_files()

        # Check disk health
        disk_health = self.check_disk_health()
        if disk_health:            logger.info("Disk Health Status:")
            for key, value in disk_health.items():                logger.info(f"{key}: {value}")

            # Alert if disk usage is high
            if disk_health["percent_used"] > 90:                logger.warning("CRITICAL: Disk usage exceeds 90%!")


def main():    project_root = "/opt/sutazaiapp"
    maintenance = AdvancedSystemMaintenance(project_root)
    maintenance.run_maintenance()


if __name__ == "__main__":    main()
