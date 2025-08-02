#!/usr/bin/env python3
"""
SutazAI System Monitor

This script monitors system resources and health for the SutazAI automation/advanced automation system,
tracking CPU, memory, disk usage, and service status.
"""

import os
import json
import time
import logging
import psutil
import argparse
import signal
import datetime
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
LOG_DIR = os.environ.get("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/system_monitor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SystemMonitor")

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
PID_DIR = PROJECT_ROOT / "pids"
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_FILE = DATA_DIR / "system_history.json"
SERVICES = ["backend-api", "web-ui", "vector-db", "cpu-monitor"]
SAMPLING_INTERVAL = 60  # seconds
HISTORY_MAX_SAMPLES = 24 * 60  # 24 hours of minute samples


class SystemMonitor:
    """
    Monitor system resources and health for the SutazAI system.
    """

    def __init__(self, history_file: str = str(HISTORY_FILE)):
        """
        Initialize the system monitor.

        Args:
            history_file: File to store historical data
        """
        self.history_file = history_file
        self.history = self._load_history()
        self.start_time = time.time()
        self.running = True

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._handle_exit)
        signal.signal(signal.SIGINT, self._handle_exit)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(history_file), exist_ok=True)

        logger.info(f"System monitor initialized, history file: {history_file}")

    def _handle_exit(self, signum, frame):
        """Handle exit signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def _load_history(self) -> Dict[str, Any]:
        """
        Load historical data from file.

        Returns:
            Dictionary containing historical data
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    history = json.load(f)
                    logger.info(
                        f"Loaded history with {len(history.get('timestamps', []))} samples"
                    )
                    return history
            except Exception as e:
                logger.error(f"Error loading history: {str(e)}")

        # Initialize new history if file doesn't exist or loading failed
        return {
            "timestamps": [],
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "services": [],
        }

    def _save_history(self):
        """Save historical data to file"""
        try:
            # Trim history to max samples
            if len(self.history["timestamps"]) > HISTORY_MAX_SAMPLES:
                for key in self.history:
                    if isinstance(self.history[key], list):
                        self.history[key] = self.history[key][-HISTORY_MAX_SAMPLES:]

            # Save to file
            with open(self.history_file, "w") as f:
                json.dump(self.history, f)

            logger.debug("Saved history to file")
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")

    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get CPU usage statistics.

        Returns:
            Dictionary with CPU statistics
        """
        try:
            # Get overall CPU percent
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get per-core CPU percent
            per_cpu = psutil.cpu_percent(interval=1, percpu=True)

            # Get CPU count
            cpu_count = psutil.cpu_count()

            # Get CPU frequency
            try:
                freq = psutil.cpu_freq()
                if freq:
                    freq_current = freq.current
                    freq_max = freq.max
                else:
                    freq_current = None
                    freq_max = None
            except Exception:
                freq_current = None
                freq_max = None

            # Get load average (unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (AttributeError, OSError):
                load_avg = (0, 0, 0)

            return {
                "percent": cpu_percent,
                "per_cpu": per_cpu,
                "count": cpu_count,
                "frequency": {"current": freq_current, "max": freq_max},
                "load_avg": load_avg,
            }
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
            return {"percent": 0, "error": str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        try:
            # Get virtual memory info
            mem = psutil.virtual_memory()

            # Get swap memory info
            swap = psutil.swap_memory()

            return {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "free": mem.free,
                "percent": mem.percent,
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent,
                },
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {"percent": 0, "error": str(e)}

    def get_disk_usage(self) -> Dict[str, Any]:
        """
        Get disk usage statistics.

        Returns:
            Dictionary with disk statistics
        """
        try:
            # Get disk usage for root partition
            disk = psutil.disk_usage("/")

            # Get disk I/O stats
            try:
                io_stats = psutil.disk_io_counters()
                if io_stats:
                    read_bytes = io_stats.read_bytes
                    write_bytes = io_stats.write_bytes
                else:
                    read_bytes = 0
                    write_bytes = 0
            except Exception:
                read_bytes = 0
                write_bytes = 0

            # Get disk usage for data directory
            try:
                data_disk = psutil.disk_usage(str(DATA_DIR))
                data_disk_info = {
                    "total": data_disk.total,
                    "used": data_disk.used,
                    "free": data_disk.free,
                    "percent": data_disk.percent,
                }
            except Exception:
                data_disk_info = None

            return {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
                "io": {"read_bytes": read_bytes, "write_bytes": write_bytes},
                "data_dir": data_disk_info,
            }
        except Exception as e:
            logger.error(f"Error getting disk usage: {str(e)}")
            return {"percent": 0, "error": str(e)}

    def get_network_usage(self) -> Dict[str, Any]:
        """
        Get network usage statistics.

        Returns:
            Dictionary with network statistics
        """
        try:
            # Get network I/O stats
            net = psutil.net_io_counters()

            # Get network connections count
            conn_count = len(psutil.net_connections())

            return {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
                "connections": conn_count,
            }
        except Exception as e:
            logger.error(f"Error getting network usage: {str(e)}")
            return {"error": str(e)}

    def get_service_status(self) -> List[Dict[str, Any]]:
        """
        Get status of system services.

        Returns:
            List of dictionaries with service status
        """
        services = []

        for service_name in SERVICES:
            pid_file = PID_DIR / f"{service_name}.pid"

            service_info = {
                "name": service_name,
                "status": "stopped",
                "pid": None,
                "uptime": 0,
                "memory_usage": 0,
                "cpu_percent": 0,
            }

            if pid_file.exists():
                try:
                    # Read PID
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())

                    # Check if process exists
                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)

                        # Get process info
                        service_info["status"] = "running"
                        service_info["pid"] = pid
                        service_info["uptime"] = time.time() - proc.create_time()
                        service_info["memory_usage"] = proc.memory_info().rss / (
                            1024 * 1024
                        )  # MB
                        service_info["cpu_percent"] = proc.cpu_percent(interval=0.1)
                    else:
                        # Process doesn't exist, but PID file does
                        service_info["status"] = "dead"

                except Exception as e:
                    logger.error(f"Error checking service {service_name}: {str(e)}")
                    service_info["status"] = "error"
                    service_info["error"] = str(e)

            services.append(service_info)

        return services

    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """
        Get GPU information if available.

        Returns:
            List of dictionaries with GPU information
        """
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()

            gpu_info = []
            for gpu in gpus:
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,  # Convert to percentage
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "temperature": gpu.temperature,
                        "uuid": gpu.uuid,
                    }
                )

            return gpu_info
        except ImportError:
            logger.debug("GPUtil not installed, skipping GPU info")
            return []
        except Exception as e:
            logger.error(f"Error getting GPU info: {str(e)}")
            return []

    def collect_system_status(self) -> Dict[str, Any]:
        """
        Collect current system status.

        Returns:
            Dictionary with all system status information
        """
        # Collect all metrics
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        disk = self.get_disk_usage()
        network = self.get_network_usage()
        services = self.get_service_status()
        gpu = self.get_gpu_info()

        # Determine overall system status
        if any(service["status"] != "running" for service in services):
            status = "degraded"
        elif (
            cpu.get("percent", 0) > 90
            or memory.get("percent", 0) > 90
            or disk.get("percent", 0) > 90
        ):
            status = "warning"
        else:
            status = "healthy"

        # Calculate uptime
        uptime = time.time() - self.start_time

        # Compile status report
        status_report = {
            "timestamp": time.time(),
            "formatted_time": datetime.datetime.now().isoformat(),
            "status": status,
            "uptime": uptime,
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "network": network,
            "services": services,
            "gpu": gpu,
        }

        return status_report

    def update_history(self, status: Dict[str, Any]):
        """
        Update historical data with current status.

        Args:
            status: Current system status
        """
        # Add timestamp
        self.history["timestamps"].append(status["timestamp"])

        # Add CPU data
        self.history["cpu"].append(status["cpu"]["percent"])

        # Add memory data
        self.history["memory"].append(status["memory"]["percent"])

        # Add disk data
        self.history["disk"].append(status["disk"]["percent"])

        # Add network data
        self.history["network"].append(
            {
                "sent": status["network"].get("bytes_sent", 0),
                "recv": status["network"].get("bytes_recv", 0),
            }
        )

        # Add service status
        service_status = {
            service["name"]: service["status"] for service in status["services"]
        }
        self.history["services"].append(service_status)

    def check_alerts(self, status: Dict[str, Any]):
        """
        Check for alert conditions and log warnings.

        Args:
            status: Current system status
        """
        # Check CPU usage
        if status["cpu"]["percent"] > 90:
            logger.warning(f"High CPU usage: {status['cpu']['percent']}%")

        # Check memory usage
        if status["memory"]["percent"] > 90:
            logger.warning(f"High memory usage: {status['memory']['percent']}%")

        # Check disk usage
        if status["disk"]["percent"] > 90:
            logger.warning(f"High disk usage: {status['disk']['percent']}%")

        # Check service status
        for service in status["services"]:
            if service["status"] != "running":
                logger.warning(f"Service {service['name']} is {service['status']}")

    def run(self):
        """Run the system monitor continuously"""
        logger.info("Starting system monitor")

        while self.running:
            try:
                # Collect system status
                status = self.collect_system_status()

                # Update history
                self.update_history(status)

                # Save history periodically
                self._save_history()

                # Check for alerts
                self.check_alerts(status)

                # Log status summary
                logger.info(
                    f"Status: {status['status']}, CPU: {status['cpu']['percent']}%, "
                    f"Memory: {status['memory']['percent']}%, "
                    f"Disk: {status['disk']['percent']}%"
                )

                # Sleep until next sample
                for _ in range(int(SAMPLING_INTERVAL)):
                    if not self.running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(10)  # Wait a bit longer on error

        logger.info("System monitor stopped")


def main():
    """Main entry point"""
    # Define global sampling interval
    global SAMPLING_INTERVAL

    parser = argparse.ArgumentParser(description="SutazAI System Monitor")
    parser.add_argument(
        "--history-file",
        type=str,
        default=str(HISTORY_FILE),
        help="File to store historical data",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=SAMPLING_INTERVAL,
        help="Sampling interval in seconds",
    )
    args = parser.parse_args()

    # Update global sampling interval if provided via arguments
    SAMPLING_INTERVAL = args.interval

    # Create and run system monitor
    monitor = SystemMonitor(history_file=args.history_file)
    monitor.run()


if __name__ == "__main__":
    main()
