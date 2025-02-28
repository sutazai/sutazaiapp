#!/usr/bin/env python3
"""System maintenance module for SutazAI."""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class SystemMaintenance:
    """Handles system maintenance tasks for SutazAI."""

    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """Initialize the system maintenance handler.

        Args:
            base_path: Base directory path for the application
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        self.backup_dir = os.path.join(base_path, "backups")

        # Ensure required directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def check_disk_space(self, min_space_gb: float = 5.0) -> bool:
        """Check if available disk space meets minimum requirements.

        Args:
            min_space_gb: Minimum required disk space in GB

        Returns:
            bool: True if sufficient space available, False otherwise
        """
        try:
            stat = os.statvfs(self.base_path)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

            if available_gb < min_space_gb:
                self.logger.warning(
                    "Low disk space: %.2f GB available, %.2f GB required",
                    available_gb,
                    min_space_gb,
                )
                return False

            self.logger.info("Sufficient disk space: %.2f GB available", available_gb)
            return True

        except Exception as e:
            self.logger.error("Failed to check disk space: %s", str(e))
            return False

    def rotate_logs(self, max_size_mb: int = 100) -> None:
        """Rotate log files that exceed maximum size.

        Args:
            max_size_mb: Maximum log file size in MB
        """
        try:
            for log_file in os.listdir(self.log_dir):
                log_path = os.path.join(self.log_dir, log_file)
                if os.path.getsize(log_path) > (max_size_mb * 1024 * 1024):
                    self.archive_log(log_path)

        except Exception as e:
            self.logger.error("Failed to rotate logs: %s", str(e))

    def archive_log(self, log_path: str) -> None:
        """Archive a log file by moving it to the backup directory.

        Args:
            log_path: Path to the log file to archive
        """
        try:
            filename = os.path.basename(log_path)
            archive_path = os.path.join(
                self.backup_dir,
                f"{filename}.{self.get_timestamp()}",
            )

            os.rename(log_path, archive_path)
            self.logger.info("Archived log file: %s -> %s", log_path, archive_path)

        except Exception as e:
            self.logger.error("Failed to archive log file: %s", str(e))

    def get_timestamp(self) -> str:
        """Get current timestamp string.

        Returns:
            str: Formatted timestamp string
        """
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Main entry point for system maintenance."""
    maintenance = SystemMaintenance()

    # Run maintenance tasks
    maintenance.check_disk_space()
    maintenance.rotate_logs()


if __name__ == "__main__":
    main()
