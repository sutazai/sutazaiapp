#!/usr/bin/env python3.11
"""
System Maintenance Module
Handles system maintenance tasks including log rotation and disk space monitoring.
"""
import logging
import os
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SystemMaintenance:
    """
    Handles comprehensive system maintenance tasks for SutazAI.
    Provides methods for:
    - Disk space monitoring
    - Log file rotation
    """

    def __init__(self, base_path: str = "/opt/sutazaiapp"):
        """
        Initialize the system maintenance handler.
        Args:
            base_path: Base directory path for the application
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path
        self.log_dir = os.path.join(base_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

    def check_disk_space(self) -> Dict[str, float]:
        """
        Check disk space usage.
        Returns:
            Dictionary containing disk space metrics
        """
        try:
            disk_info = os.statvfs(self.base_path)
            total = disk_info.f_blocks * disk_info.f_frsize
            free = disk_info.f_bfree * disk_info.f_frsize
            used = total - free

            return {
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "usage_percent": (used / total) * 100,
            }
        except Exception as e:
            self.logger.error("Failed to check disk space: %s", e)
            return {}

    def rotate_logs(
        self,
        max_size_mb: int = 100,
        max_log_files: int = 5,
    ) -> None:
        """
        Rotate log files that exceed maximum size.
        Args:
            max_size_mb: Maximum log file size in MB
            max_log_files: Maximum number of log files to keep
        """
        try:
            for root, _, files in os.walk(self.log_dir):
                for file in files:
                    if not file.endswith(".log"):
                        continue

                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB

                    if file_size > max_size_mb:
                        # Rotate existing log files
                        for i in range(max_log_files - 1, 0, -1):
                            old_path = f"{file_path}.{i}"
                            new_path = f"{file_path}.{i + 1}"
                            if os.path.exists(old_path):
                                os.rename(old_path, new_path)

                        # Rename current log file
                        os.rename(file_path, f"{file_path}.1")

                        # Create new empty log file
                        open(file_path, "w").close()
                        self.logger.info("Rotated log file: %s", file_path)

        except Exception as e:
            self.logger.error("Failed to rotate logs: %s", e)

    def run_maintenance(self) -> None:
        """Execute all maintenance tasks."""
        self.logger.info("Starting system maintenance tasks...")

        # Check disk space
        disk_space = self.check_disk_space()
        if disk_space:
            self.logger.info(
                "Disk Status: %.1f%% used (%.1f/%.1f GB free)",
                disk_space["usage_percent"],
                disk_space["free_gb"],
                disk_space["total_gb"],
            )

            # Warn if disk usage is high
            if disk_space["usage_percent"] > 85:
                self.logger.warning("High disk usage detected!")

        # Rotate logs
        self.rotate_logs()

        self.logger.info("System maintenance completed")


def main() -> None:
    """Run system maintenance."""
    try:
        maintenance = SystemMaintenance()
        maintenance.run_maintenance()
    except Exception as e:
        logger.error("System maintenance failed: %s", e)


if __name__ == "__main__":
    main()
