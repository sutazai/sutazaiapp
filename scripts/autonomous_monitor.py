#!/opt/sutazaiapp/venv/bin/python3
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
from filelock import FileLock

# Configure logging
log_dir = Path("/opt/sutazaiapp/logs/autonomous_monitor")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / "monitor.log"),
    ],
)
logger = logging.getLogger("SutazAIMonitor")

# Constants
PID_FILE = "/tmp/sutazai_monitor.pid"
LOCK_FILE = "/tmp/sutazai_monitor.lock"
CONFIG_FILE = "/opt/sutazaiapp/config/monitor_config.json"

class MonitoringSystem:
    def __init__(self):
        self.config = self.load_config()
        self.lock = FileLock(LOCK_FILE)
        self.should_run = True
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, cleaning up...")
        self.should_run = False

    def load_config(self) -> Dict:
        """Load monitoring configuration with defaults."""
        default_config = {
            "interval": 300,  # 5 minutes
            "cpu_max": 50.0,  # 50% CPU threshold
            "memory_max": 70.0,  # 70% memory threshold
            "disk_max": 85.0,  # 85% disk threshold
            "process_cpu_max": 30.0,  # 30% per process
            "critical_services": [
                "autonomous_monitor.py",
                "performance_manager.py",
            ],
        }
        
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE) as f:
                    config = json.load(f)
                # Update defaults with loaded config
                default_config.update(config)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return default_config

    def check_single_instance(self) -> bool:
        """Ensure only one instance of the monitor is running."""
        try:
            with open(PID_FILE, "w") as f:
                f.write(str(os.getpid()))
            return True
        except Exception as e:
            logger.error(f"Error checking instance: {e}")
            return False

    def check_system_resources(self) -> Dict:
        """Check system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            
            status = {
                "cpu": cpu_percent,
                "memory": memory.percent,
                "memory_used": memory.used,
                "memory_total": memory.total,
                "disk": disk.percent,
            }
            
            # Log warnings for high resource usage
            if cpu_percent > self.config["cpu_max"]:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > self.config["memory_max"]:
                logger.warning(f"High memory usage: {memory.percent}%")
            
            if disk.percent > self.config["disk_max"]:
                logger.warning(f"High disk usage: {disk.percent}%")
            
            return status
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return {}

    def check_critical_processes(self) -> None:
        """Monitor critical processes and their resource usage."""
        try:
            for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent"]):
                try:
                    proc_cpu = proc.info["cpu_percent"]
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    
                    # Check if process is critical
                    if any(svc in cmdline for svc in self.config["critical_services"]):
                        if proc_cpu > self.config["process_cpu_max"]:
                            logger.warning(
                                f"High CPU usage ({proc_cpu}%) in process: "
                                f"PID {proc.info['pid']} ({cmdline})"
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error checking processes: {e}")

    def run(self) -> None:
        """Main monitoring loop."""
        if not self.check_single_instance():
            sys.exit(1)
        
        logger.info("Starting lightweight monitoring system")
        logger.info(
            f"Configured with: interval={self.config['interval']}s, "
            f"cpu_max={self.config['cpu_max']}%, "
            f"memory_max={self.config['memory_max']}%"
        )
        
        while self.should_run:
            try:
                with self.lock:
                    status = self.check_system_resources()
                    if status:
                        logger.info(
                            f"System Status - CPU: {status['cpu']}%, "
                            f"Memory: {status['memory']}% "
                            f"(Used: {status['memory_used'] / 1024**3:.1f} GB/"
                            f"Total: {status['memory_total'] / 1024**3:.1f} GB), "
                            f"Disk: {status.get('disk', 'N/A')}%"
                        )
                    
                    self.check_critical_processes()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            # Sleep for the configured interval
            time.sleep(self.config["interval"])


def main():
    """Main entry point with error handling."""
    try:
        monitor = MonitoringSystem()
        monitor.run()
    except Exception as e:
        logger.error(f"Fatal error in monitoring system: {e}")
        sys.exit(1)
    finally:
        # Cleanup on exit
        try:
            if os.path.exists(PID_FILE):
                os.unlink(PID_FILE)
            if os.path.exists(LOCK_FILE):
                os.unlink(LOCK_FILE)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()

