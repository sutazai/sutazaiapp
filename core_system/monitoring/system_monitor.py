import logging
import time
from typing import Any, Dict

import psutil


class SystemMonitor:
    """Advanced system monitoring and resource management class"""

    def __init__(self, log_path: str = "logs/system_monitor.log"):
        """
        Initialize the system monitor with logging configuration

        :param log_path: Path to log file for system monitoring
        """
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive system metrics

        :return: Dictionary of system performance metrics
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory()
            disk_usage = psutil.disk_usage("/")

            metrics = {
                "cpu": {
                    "usage_percent": cpu_usage,
                    "cores": psutil.cpu_count(logical=False),
                    "logical_cores": psutil.cpu_count(logical=True),
                },
                "memory": {
                    "total_gb": round(memory_usage.total / (1024**3), 2),
                    "available_gb": round(memory_usage.available / (1024**3), 2),
                    "used_percent": memory_usage.percent,
                },
                "disk": {
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "free_gb": round(disk_usage.free / (1024**3), 2),
                    "used_percent": disk_usage.percent,
                },
                "network": self._get_network_stats(),
            }

            self.logger.info(f"System Metrics: {metrics}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error retrieving system metrics: {e}")
            return {}

    def _get_network_stats(self) -> Dict[str, Any]:
        """
        Retrieve network interface statistics

        :return: Dictionary of network interface metrics
        """
        try:
            net_stats = psutil.net_io_counters()
            return {
                "bytes_sent": net_stats.bytes_sent,
                "bytes_recv": net_stats.bytes_recv,
                "packets_sent": net_stats.packets_sent,
                "packets_recv": net_stats.packets_recv,
            }
        except Exception as e:
            self.logger.warning(f"Could not retrieve network stats: {e}")
            return {}

    def optimize_memory(self) -> None:
        """
        Attempt to optimize system memory usage
        """
        try:
            # Placeholder for memory optimization techniques
            self.logger.info("Initiating memory optimization")
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")

    def cleanup_disk(self) -> None:
        """
        Perform disk cleanup and maintenance
        """
        try:
            # Placeholder for disk cleanup techniques
            self.logger.info("Initiating disk cleanup")
        except Exception as e:
            self.logger.error(f"Disk cleanup failed: {e}")

    def monitor_continuously(self, interval: int = 60) -> None:
        """
        Continuously monitor system resources

        :param interval: Monitoring interval in seconds
        """
        try:
            while True:
                metrics = self.get_system_metrics()

                # Basic alerting mechanism
                if metrics.get("cpu", {}).get("usage_percent", 0) > 80:
                    self.logger.warning("High CPU usage detected!")

                if metrics.get("memory", {}).get("used_percent", 0) > 90:
                    self.logger.critical("Critical memory usage!")

                time.sleep(interval)

        except KeyboardInterrupt:
            self.logger.info("System monitoring stopped.")
        except Exception as e:
            self.logger.error(f"Continuous monitoring error: {e}")


def main():
    """
    Standalone execution for system monitoring
    """
    monitor = SystemMonitor()
    monitor.monitor_continuously()


if __name__ == "__main__":
    main()
