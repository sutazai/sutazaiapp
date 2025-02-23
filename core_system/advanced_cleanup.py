#!/usr/bin/env python3
import logging
import os
import shutil
import subprocess
from typing import List


class AdvancedCleanupManager:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler("/var/log/sutazai/advanced_cleanup.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def cleanup_timeshift_snapshots(self, keep_recent: int = 3):
        """
        Clean up Timeshift snapshots, keeping only the most recent ones.
        """
        try:
            # List snapshots sorted by date
            snapshots_cmd = (
                "timeshift --list | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -r"
            )
            snapshots = (
                subprocess.check_output(snapshots_cmd, shell=True).decode().splitlines()
            )

            # Delete older snapshots
            for snapshot in snapshots[keep_recent:]:
                try:
                    subprocess.run(
                        [
                            "sudo",
                            "timeshift",
                            "--delete",
                            "--snapshot",
                            snapshot.split()[0],
                        ],
                        check=True,
                    )
                    self.logger.info(f"Deleted Timeshift snapshot: {snapshot}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to delete snapshot {snapshot}: {e}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Timeshift snapshot cleanup failed: {e}")

    def reduce_swap_and_hibernate(self):
        """
        Optimize swap and hibernation file sizes.
        """
        try:
            # Check current swap size
            swap_info = subprocess.check_output(["free", "-h"]).decode()
            self.logger.info(f"Current Swap Info:\n{swap_info}")

            # Reduce swap size (example: set to 4GB)
            subprocess.run(["sudo", "swapoff", "-a"], check=True)
            subprocess.run(
                ["sudo", "dd", "if=/dev/zero", "of=/swapfile", "bs=1G", "count=4"],
                check=True,
            )
            subprocess.run(["sudo", "mkswap", "/swapfile"], check=True)
            subprocess.run(["sudo", "swapon", "/swapfile"], check=True)

            # Disable hibernation if not needed
            subprocess.run(
                ["sudo", "systemctl", "mask", "hibernate.target"], check=True
            )

            self.logger.info("Swap and hibernation optimized")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Swap and hibernation optimization failed: {e}")

    def clean_cache_directories(self):
        """
        Clean various cache directories with user permission.
        """
        cache_dirs = [
            os.path.expanduser("~/.cache"),
            "/var/cache",
            os.path.expanduser("~/.npm"),
            os.path.expanduser("~/.docker/cache"),
        ]

        for cache_dir in cache_dirs:
            try:
                if os.path.exists(cache_dir):
                    # Use du to get directory size before cleaning
                    du_output = (
                        subprocess.check_output(["du", "-sh", cache_dir])
                        .decode()
                        .strip()
                    )
                    self.logger.info(
                        f"Cache directory size before cleanup: {cache_dir} - {du_output}"
                    )

                    # Remove cache contents, keeping directory structure
                    for root, dirs, files in os.walk(cache_dir):
                        for f in files:
                            try:
                                os.unlink(os.path.join(root, f))
                            except Exception as e:
                                self.logger.warning(f"Could not remove {f}: {e}")

                    self.logger.info(f"Cleaned cache directory: {cache_dir}")

            except Exception as e:
                self.logger.error(f"Error cleaning {cache_dir}: {e}")

    def run_comprehensive_cleanup(self):
        """
        Execute comprehensive system cleanup.
        """
        print("🧹 Starting Advanced System Cleanup 🧹")

        self.cleanup_timeshift_snapshots()
        self.reduce_swap_and_hibernate()
        self.clean_cache_directories()

        print("✅ Advanced Cleanup Complete!")


def main():
    cleanup_manager = AdvancedCleanupManager()
    cleanup_manager.run_comprehensive_cleanup()


if __name__ == "__main__":
    main()
