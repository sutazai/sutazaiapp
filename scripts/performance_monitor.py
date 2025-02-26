#!/usr/bin/env python3
"""
Performance Monitoring Script for SutazAI

Monitors system and application performance, logs metrics, and provides insights.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    filename="/opt/sutazaiapp/logs/performance_monitor.log",
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(
        self,
        log_dir="/opt/sutazaiapp/performance_logs",
        interval=60,
        duration=3600,
    ):
        """
        Initialize performance monitor.

        Args:
            log_dir (str): Directory to store performance logs
            interval (int): Monitoring interval in seconds
            duration (int): Total monitoring duration in seconds
        """
        self.log_dir = log_dir
        self.interval = interval
        self.duration = duration

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Performance metrics storage
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
        }

        # Monitoring flags
        self.is_monitoring = False
        self.monitor_thread = None

    def _log_system_metrics(self):
        """
        Collect and log system performance metrics.
        """
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["cpu_usage"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "percent": cpu_percent,
                }
            )

            # Memory Usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                }
            )

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            self.metrics["disk_io"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes,
                }
            )

            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics["network_io"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                }
            )

            logger.info(f"Performance metrics collected: CPU {cpu_percent}%")

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def start_monitoring(self):
        """
        Start performance monitoring in a separate thread.
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop
            )
            self.monitor_thread.start()
            logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """
        Stop performance monitoring and save results.
        """
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        # Save performance metrics
        self._save_metrics()
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """
        Continuous monitoring loop.
        """
        start_time = time.time()

        while self.is_monitoring and time.time() - start_time < self.duration:
            self._log_system_metrics()
            time.sleep(self.interval)

    def _save_metrics(self):
        """
        Save collected performance metrics to a JSON file.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = os.path.join(
                self.log_dir, f"performance_metrics_{timestamp}.json"
            )

            with open(metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=4)

            logger.info(f"Performance metrics saved to {metrics_file}")

        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")


def main():
    """
    Main function to demonstrate performance monitoring.
    """
    monitor = PerformanceMonitor(
        log_dir="/opt/sutazaiapp/performance_logs",
        interval=30,  # Check every 30 seconds
        duration=1800,  # Monitor for 30 minutes
    )

    try:
        monitor.start_monitoring()

        # Keep main thread alive
        while monitor.is_monitoring:
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")

    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()
