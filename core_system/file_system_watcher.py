#!/usr/bin/env python3
"""
SutazAI File System Watcher

Monitors project directory for changes and updates structure documentation.
"""

import logging
import time
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from core_system.file_structure_tracker import FileStructureTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger("SutazAI.FileSystemWatcher")


class ProjectStructureHandler(FileSystemEventHandler):
    """
    Custom file system event handler for tracking project structure changes
    """

    def __init__(self, base_dir: str):
        """
        Initialize the structure handler

        Args:
            base_dir (str): Base directory to monitor
        """
        self.base_dir = base_dir
        self.structure_tracker = FileStructureTracker(base_dir)
        self.last_update = 0
        self.update_interval = 5  # seconds

    def _should_update(self) -> bool:
        """
        Determine if an update should be performed

        Returns:
            bool: Whether to update the structure
        """
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False

    def on_any_event(self, event):
        """
        Handle any file system event

        Args:
            event: Watchdog file system event
        """
        if not event.is_directory and self._should_update():
            try:
                logger.info(f"Updating project structure after event: {event.src_path}")
                self.structure_tracker.update_structure_files()

                # Update README structure
                self.structure_tracker.update_readme_structure()
            except Exception as e:
                logger.error(f"Error updating project structure: {e}")


def start_file_system_watcher(base_dir: str) -> Optional[Observer]:
    """
    Start file system watcher for the project

    Args:
        base_dir (str): Base directory to monitor

    Returns:
        Optional[Observer]: File system observer instance
    """
    try:
        event_handler = ProjectStructureHandler(base_dir)
        observer = Observer()
        observer.schedule(event_handler, base_dir, recursive=True)
        observer.start()
        logger.info(f"File system watcher started for {base_dir}")
        return observer
    except Exception as e:
        logger.error(f"Failed to start file system watcher: {e}")
        return None


def stop_file_system_watcher(observer: Optional[Observer]):
    """
    Stop the file system watcher

    Args:
        observer (Optional[Observer]): File system observer to stop
    """
    if observer:
        try:
            observer.stop()
            observer.join()
            logger.info("File system watcher stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping file system watcher: {e}")


def main():
    """
    Main execution for file system watcher
    """
    base_dir = "/opt/SutazAI"
    observer = start_file_system_watcher(base_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_file_system_watcher(observer)


if __name__ == "__main__":
    main()
