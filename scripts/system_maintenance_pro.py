#!/opt/sutazaiapp/venv/bin/python3import loggingimport osimport subprocessfrom datetime import datetime, timedeltafrom typing import Any, Dict, Listlogging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s: %(message)s",)logger = logging.getLogger(__name__)class AdvancedSystemMaintenance:    def __init__(self, project_root: str):        # Ensure project_root is set first
self.project_root = project_root
# Create log directory path
self.log_dir = os.path.join(project_root, "logs")
# Ensure log directory exists
try:
    os.makedirs(self.log_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create log directory: {e}")
        # Use a temporary directory as fallback
        self.log_dir = os.path.join("/tmp", "sutazai_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        logger.warning(f"Using temporary directory: {self.log_dir}")
        def rotate_log_files(                    self,
            max_log_size_mb: int = 100,
            max_log_files: int = 5,
            ):
            """Rotate log files that exceed the size limit."""
            try:
                for root, _, files in os.walk(self.log_dir):
                for file in files:
                if not file.endswith(".log"):
                    continue
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                if file_size > max_log_size_mb:
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
                        logger.info(f"Rotated log file: {file_path}")
                        except Exception as e:
                            logger.error(f"Log rotation failed: {e}")
                            def check_disk_health(self) -> Dict[str, Any]:                                        """Check disk space and health metrics."""
                                try:
                                    disk_info = os.statvfs(self.project_root)
                                    total_space = disk_info.f_blocks * disk_info.f_frsize
                                    free_space = disk_info.f_bfree * disk_info.f_frsize
                                    used_space = total_space - free_space
                                    usage_percent = (used_space / total_space) * 100
                                    return {
                                "total_gb": total_space / (1024**3),                                        "used_gb": used_space / (1024**3),                                        "free_gb": free_space / (1024**3),                                        "usage_percent": usage_percent,                                        }                                        except Exception as e:                                            logger.error(f"Disk health check failed: {e}")                                            return {}                                        def run_maintenance(self):                                            """Run all maintenance tasks."""
                                logger.info("Starting system maintenance tasks...")
                                try:
                                    # Rotate log files
                                    self.rotate_log_files()
                                    # Check disk health
                                    disk_status = self.check_disk_health()
                                    if disk_status:
                                        logger.info(
                                        "Disk Status: %.1f%% used (%.1f/%.1f GB free)",
                                        disk_status["usage_percent"],
                                        disk_status["free_gb"],
                                        disk_status["total_gb"],
                                        )
                                        # Warn if disk usage is high
                                        if disk_status["usage_percent"] > 85:
                                            logger.warning("High disk usage detected!")
                                            logger.info("System maintenance completed")
                                            except Exception as e:
                                                logger.error(f"Maintenance tasks failed: {e}")
                                                def main():                                                                """Run system maintenance."""
                                                    try:
                                                        maintenance = AdvancedSystemMaintenance("/opt/sutazaiapp")
                                                        maintenance.run_maintenance()
                                                        except Exception as e:
                                                            logger.error(f"System maintenance failed: {e}")
                                                            sys.exit(1)
                                                            if __name__ == "__main__":
                                                                main()

