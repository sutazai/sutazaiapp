#!/usr/bin/env python3
"""
Log Rotation Script
Manages log files to prevent disk space issues.
"""

import gzip
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LogRotator:
    def __init__(
        self,
        log_dir: str = "/opt/sutazai/logs",
        max_size_mb: int = 100,
        retention_days: int = 30,
    ):
        self.log_dir = Path(log_dir)
        self.max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.retention_days = retention_days

    def get_log_files(self) -> List[Path]:
        """Get all log files in the directory."""
        return [f for f in self.log_dir.glob("**/*.log") if f.is_file()]

    def compress_file(self, file_path: Path) -> None:
        """Compress a log file using gzip."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            compressed_path = file_path.with_suffix(f".{timestamp}.log.gz")

            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Truncate original file
            with open(file_path, "w") as f:
                f.truncate(0)

            logger.info(f"Compressed {file_path} to {compressed_path}")
        except Exception as e:
            logger.error(f"Failed to compress {file_path}: {str(e)}")

    def remove_old_logs(self) -> None:
        """Remove log files older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for file in self.log_dir.glob("**/*.log.gz"):
            try:
                if file.stat().st_mtime < cutoff_date.timestamp():
                    file.unlink()
                    logger.info(f"Removed old log file: {file}")
            except Exception as e:
                logger.error(f"Failed to remove {file}: {str(e)}")

    def rotate_logs(self) -> None:
        """Perform log rotation."""
        try:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Get all log files
            log_files = self.get_log_files()

            for log_file in log_files:
                try:
                    # Check file size
                    if log_file.stat().st_size > self.max_size:
                        self.compress_file(log_file)
                except Exception as e:
                    logger.error(f"Failed to process {log_file}: {str(e)}")

            # Remove old logs
            self.remove_old_logs()

        except Exception as e:
            logger.error(f"Log rotation failed: {str(e)}")


def main():
    try:
        rotator = LogRotator()
        rotator.rotate_logs()
        logger.info("Log rotation completed successfully")
    except Exception as e:
        logger.error(f"Log rotation script failed: {str(e)}")


if __name__ == "__main__":
    main()
