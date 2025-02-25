#!/usr/bin/env python3
"""
🌐 SutazAI Continuous Performance Monitoring Framework 🌐

Advanced real-time system and application performance monitoring toolkit:
- Continuous resource tracking
- Performance bottleneck detection
- Automated alerting
- Historical performance analysis
- Adaptive optimization recommendations
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import GPUtil
import psutil
import schedule

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] 🔍 %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "/opt/sutazai/logs/continuous_performance_monitor.log"
        ),
    ],
)
logger = logging.getLogger("ContinuousPerformanceMonitor")


class SutazAIPerformanceMonitor:
    def __init__(self, project_root: str = "/opt/SutazAI"):
        self.project_root = project_root
        self.performance_history = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
            "bottlenecks": [],
        }
        self.alert_thresholds = {
            "cpu_percent": 85,
            "memory_percent": 85,
            "disk_io_read_time": 100,
            "disk_io_write_time": 100,
        }

    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system performance metrics."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "per_core": psutil.cpu_percent(interval=1, percpu=True),
            },
            "memory": {
                "total_gb": round(
                    psutil.virtual_memory().total / (1024**3), 2
                ),
                "available_gb": round(
                    psutil.virtual_memory().available / (1024**3), 2
                ),
                "percent": psutil.virtual_memory().percent,
            },
            "disk_io": dict(psutil.disk_io_counters()._asdict()),
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "gpu": self._get_gpu_metrics(),
            "top_processes": self._get_top_processes(),
        }
        return metrics

    def _get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Collect GPU performance metrics."""
        try:
            gpus = GPUtil.getGPUs()
            return [
                {
                    "name": gpu.name,
                    "memory_total_mb": gpu.memoryTotal,
                    "memory_used_mb": gpu.memoryUsed,
                    "memory_free_mb": gpu.memoryFree,
                    "gpu_load": gpu.load * 100,
                    "temperature": gpu.temperature,
                }
                for gpu in gpus
            ]
        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}")
            return []

    def _get_top_processes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top resource-consuming processes."""
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_percent": proc.info["cpu_percent"],
                            "memory_percent": proc.info["memory_percent"],
                        }
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return sorted(
                processes, key=lambda x: x["cpu_percent"], reverse=True
            )[:limit]
        except Exception as e:
            logger.error(f"Top processes retrieval failed: {e}")
            return []

    def detect_performance_bottlenecks(
        self, metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks based on predefined thresholds."""
        bottlenecks = []

        if metrics["cpu"]["percent"] > self.alert_thresholds["cpu_percent"]:
            bottlenecks.append(
                {
                    "type": "CPU_OVERLOAD",
                    "current_usage": metrics["cpu"]["percent"],
                    "recommendation": "High CPU usage detected. Investigate resource-intensive processes.",
                }
            )

        if (
            metrics["memory"]["percent"]
            > self.alert_thresholds["memory_percent"]
        ):
            bottlenecks.append(
                {
                    "type": "MEMORY_PRESSURE",
                    "current_usage": metrics["memory"]["percent"],
                    "recommendation": "High memory usage. Consider optimizing memory-intensive applications.",
                }
            )

        disk_read_time = metrics["disk_io"].get("read_time", 0)
        disk_write_time = metrics["disk_io"].get("write_time", 0)

        if disk_read_time > self.alert_thresholds["disk_io_read_time"]:
            bottlenecks.append(
                {
                    "type": "DISK_IO_READ_BOTTLENECK",
                    "read_time": disk_read_time,
                    "recommendation": "High disk read time. Consider I/O optimization or SSD upgrade.",
                }
            )

        if disk_write_time > self.alert_thresholds["disk_io_write_time"]:
            bottlenecks.append(
                {
                    "type": "DISK_IO_WRITE_BOTTLENECK",
                    "write_time": disk_write_time,
                    "recommendation": "High disk write time. Investigate disk-intensive processes.",
                }
            )

        return bottlenecks

    def log_performance_metrics(
        self, metrics: Dict[str, Any], bottlenecks: List[Dict[str, Any]]
    ):
        """Log performance metrics and bottlenecks to historical record."""
        self.performance_history["timestamps"].append(metrics["timestamp"])
        self.performance_history["cpu_usage"].append(metrics["cpu"]["percent"])
        self.performance_history["memory_usage"].append(
            metrics["memory"]["percent"]
        )
        self.performance_history["disk_io"].append(metrics["disk_io"])
        self.performance_history["network_io"].append(metrics["network_io"])

        if bottlenecks:
            self.performance_history["bottlenecks"].extend(bottlenecks)

    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        report_path = os.path.join(
            self.project_root,
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(report_path, "w") as f:
            json.dump(self.performance_history, f, indent=2)

        logger.info(f"📊 Performance Report Generated: {report_path}")
        return report_path

    def send_performance_alert(self, bottlenecks: List[Dict[str, Any]]):
        """Send performance alerts via logging and optional external notification."""
        for bottleneck in bottlenecks:
            logger.warning(
                f"🚨 Performance Bottleneck Detected: {bottleneck['type']}"
            )
            logger.warning(f"Recommendation: {bottleneck['recommendation']}")

    def continuous_monitoring_job(self):
        """Periodic job for continuous performance monitoring."""
        try:
            metrics = self.collect_system_metrics()
            bottlenecks = self.detect_performance_bottlenecks(metrics)

            self.log_performance_metrics(metrics, bottlenecks)

            if bottlenecks:
                self.send_performance_alert(bottlenecks)
                self.generate_performance_report()

        except Exception as e:
            logger.error(f"Continuous monitoring job failed: {e}")

    def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous performance monitoring."""
        logger.info(
            f"🚀 Starting Continuous Performance Monitoring (Interval: {interval_minutes} minutes)"
        )

        # Schedule periodic monitoring job
        schedule.every(interval_minutes).minutes.do(
            self.continuous_monitoring_job
        )

        # Run monitoring in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        monitoring_thread = threading.Thread(target=run_scheduler, daemon=True)
        monitoring_thread.start()

        # Keep main thread running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 Performance Monitoring Stopped")


def main():
    performance_monitor = SutazAIPerformanceMonitor()
    performance_monitor.start_monitoring()


if __name__ == "__main__":
    main()
