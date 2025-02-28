#!/usr/bin/env python3.11
"""
Comprehensive System Maintenance Module for SutazAI

This script provides advanced system maintenance capabilities,
including log rotation, disk space monitoring, and backup management.
"""

import logging
import os
import shutil
from datetime import datetime
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/opt/sutazaiapp/logs/system_maintenance.log"),
    ],
)


class SystemMaintenance:    """
    Handles comprehensive system maintenance tasks for SutazAI.

    Provides methods for:    - Disk space monitoring
    - Log file rotation
    - Backup management
    """

    def __init__(self, base_path: str = "/opt/sutazaiapp"):        """
        Initialize the system maintenance handler.

        Args:        base_path: Base directory path for the application
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        self.backup_dir = os.path.join(base_path, "backups")

        # Ensure required directories exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

        def check_disk_space(self, min_space_gb: float = 5.0) -> bool:            """
            Check if available disk space meets minimum requirements.

            Args:            min_space_gb: Minimum required disk space in GB

            Returns:            bool: True if sufficient space available, False otherwise
            """
            try:                stat = os.statvfs(self.base_path)
                available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)

                if available_gb < min_space_gb:                    self.logger.warning(
                        "Low disk space: %.2f GB available, %.2f GB required",
                        available_gb,
                        min_space_gb,
                    )
                return False

                self.logger.info(
                    "Sufficient disk space: %.2f GB available",
                    available_gb,
                )
            return True

            except Exception as e:                self.logger.error("Failed to check disk space: %s", str(e))
            return False

            def rotate_logs(
                    self,
                    max_size_mb: int = 100,
                    max_backup_count: int = 5) -> None:                """
                Rotate log files that exceed maximum size.

                Args:                max_size_mb: Maximum log file size in MB
                max_backup_count: Maximum number of backup log files to keep
                """
                try:                    log_files = [
                        os.path.join(self.log_dir, f)
                        for f in os.listdir(self.log_dir)
                        if os.path.isfile(os.path.join(self.log_dir, f))
                    ]

                    # Sort log files by modification time
                    log_files.sort(key=os.path.getmtime)

                    for log_path in log_files:                        try:                            if os.path.getsize(log_path) > (
                                    max_size_mb * 1024 * 1024):                                self.archive_log(
                                    log_path, max_backup_count)
                                except Exception as file_error:                                    self.logger.error(
                                        f"Error processing log file {log_path}: {file_error}")

                                    except Exception as e:                                        self.logger.error(
                                            f"Failed to rotate logs: {e}")

                                        def archive_log(
                                            self, log_path: str, max_backup_count: int = 5) -> None:                                            """
                                                    Archive a log file by moving it to the backup directory.

                                                    Args:                                                    log_path: Path to the log file to archive
                                                    max_backup_count: Maximum number of backup log files to keep
                                                    """
                                                try:                                                    filename = os.path.basename(
                                                        log_path)
                                                    timestamp = self.get_timestamp()
                                                    archive_path = os.path.join(
                                                        self.backup_dir,
                                                        f"{filename}.{timestamp}",
                                                    )

                                                    # Move the log file to
                                                    # backup
                                                    shutil.move(
                                                        log_path, archive_path)
                                                    self.logger.info(
                                                        f"Archived log file: {log_path} -> {archive_path}")

                                                    # Manage backup log files
                                                    self._manage_backup_logs(
                                                        filename, max_backup_count)

                                                    except Exception as e:                                                        self.logger.error(
                                                            f"Failed to archive log file {log_path}: {e}")

                                                        def _manage_backup_logs(
                                                                self,
                                                                base_filename: str,
                                                                max_backup_count: int) -> None:                                                            """
                                                                Manage backup log files, keeping only the most recent ones.

                                                                Args:                                                                base_filename: Base name of the log file
                                                                max_backup_count: Maximum number of backup log files to keep
                                                                """
                                                            try:                                                                backup_logs = [
                                                                    os.path.join(
                                                                        self.backup_dir, f)
                                                                    for f in os.listdir(self.backup_dir)
                                                                    if f.startswith(base_filename)
                                                                ]

                                                                # Sort backup
                                                                # logs by
                                                                # modification
                                                                # time
                                                                backup_logs.sort(
                                                                    key=os.path.getmtime, reverse=True)

                                                                # Remove
                                                                # excess
                                                                # backup
                                                                # logs
                                                                for log_path in backup_logs[max_backup_count:]:                                                                    try:                                                                        os.remove(
                                                                            log_path)
                                                                        self.logger.info(
                                                                            f"Removed old backup log: {log_path}")
                                                                        except Exception as remove_error:                                                                            self.logger.error(
                                                                                f"Failed to remove backup log {log_path}: {remove_error}")

                                                                            except Exception as e:                                                                                self.logger.error(
                                                                                    f"Failed to manage backup logs: {e}")

                                                                                @staticmethod
                                                                                def get_timestamp() -> str:                                                                                        """
                                                                                                Get current timestamp string.

                                                                                                Returns:                                                                                                str: Formatted timestamp string
                                                                                                """
                                                                                    return datetime.now().strftime("%Y%m%d_%H%M%S")

                                                                                    def perform_maintenance(
                                                                                            self) -> None:                                                                                        """
                                                                                                Perform comprehensive system maintenance tasks.
                                                                                                """
                                                                                        self.logger.info(
                                                                                            "Starting system maintenance...")

                                                                                        try:                                                                                            # Check
                                                                                            # disk
                                                                                            # space
                                                                                            if not self.check_disk_space():                                                                                                self.logger.warning(
                                                                                                    "Low disk space detected. Consider cleaning up.")

                                                                                                # Rotate
                                                                                                # logs
                                                                                                self.rotate_logs()

                                                                                                self.logger.info(
                                                                                                    "System maintenance completed successfully.")
                                                                                                except Exception as e:                                                                                                    self.logger.error(
                                                                                                        f"System maintenance failed: {e}")

                                                                                                    def main() -> None:                                                                                                        """Main entry point for system maintenance."""
                                                                                                        try:                                                                                                            maintenance = SystemMaintenance()
                                                                                                            maintenance.perform_maintenance()
                                                                                                            except Exception as e:                                                                                                                logging.exception(
                                                                                                                    f"System maintenance script failed: {e}")

                                                                                                                if __name__ == "__main__":                                                                                                                    main()
