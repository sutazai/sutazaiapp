#!/usr/bin/env python3
"""
SutazAI Resource Analyzer
-------------------------
Monitors and analyzes system resources, providing insights and optimization
recommendations for the SutazAI platform.
"""

import logging
import os
import platform
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

try:
    import psutil
except ImportError:
    psutil = None
    logging.warning(
        "psutil not installed, resource monitoring will be limited"
    )

try:
    import GPUtil
except ImportError:
    GPUtil = None
    logging.warning("GPUtil not installed, GPU monitoring will be disabled")


class ResourceAnalyzer:
    """Analyzes and monitors system resources for the SutazAI platform."""

    def __init__(self, interval: int = 5, log_level: str = "INFO"):
        """
        Initialize the resource analyzer.

        Args:
            interval: Monitoring interval in seconds
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level))

        self._setup_logging()
        self._check_dependencies()

        # Resource thresholds (percentage)
        self.cpu_threshold = 80
        self.memory_threshold = 85
        self.disk_threshold = 90
        self.gpu_threshold = 80

        # Historical data storage
        self.history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "gpu": [],
            "network": [],
            "timestamp": [],
        }
        self.history_limit = 1000  # Maximum number of data points to store

    def _setup_logging(self) -> None:
        """Set up logging for the resource analyzer."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "../logs")
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(
                log_dir, f"resources_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.warning(f"Failed to set up file logging: {str(e)}")

    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        if psutil is None:
            self.logger.warning(
                "psutil not installed. Install with: pip install psutil"
            )

        if GPUtil is None:
            self.logger.warning(
                "GPUtil not installed. Install with: pip install gputil"
            )

    def get_system_info(self) -> Dict[str, str]:
        """
        Get general system information.

        Returns:
            Dictionary with system information
        """
        info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        if psutil:
            info["boot_time"] = datetime.fromtimestamp(
                psutil.boot_time()
            ).strftime("%Y-%m-%d %H:%M:%S")

        return info

    def get_cpu_info(self) -> Dict[str, Union[float, int, List[float]]]:
        """
        Get CPU usage information.

        Returns:
            Dictionary with CPU metrics
        """
        if not psutil:
            return {"error": "psutil not available"}

        cpu_info = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "per_cpu": psutil.cpu_percent(interval=0.1, percpu=True),
        }

        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["frequency_current"] = cpu_freq.current
                cpu_info["frequency_min"] = cpu_freq.min
                cpu_info["frequency_max"] = cpu_freq.max
        except Exception as e:
            self.logger.debug(f"Could not get CPU frequency: {str(e)}")

        return cpu_info

    def get_memory_info(self) -> Dict[str, Union[float, int]]:
        """
        Get memory usage information.

        Returns:
            Dictionary with memory metrics
        """
        if not psutil:
            return {"error": "psutil not available"}

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_info = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent,
        }

        return memory_info

    def get_disk_info(self) -> List[Dict[str, Union[str, int, float]]]:
        """
        Get disk usage information.

        Returns:
            List of dictionaries with disk metrics for each partition
        """
        if not psutil:
            return [{"error": "psutil not available"}]

        disk_info = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info.append(
                    {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                    }
                )
            except (PermissionError, OSError) as e:
                self.logger.debug(
                    f"Could not check disk {partition.mountpoint}: {str(e)}"
                )

        return disk_info

    def get_network_info(self) -> Dict[str, Dict[str, int]]:
        """
        Get network usage information.

        Returns:
            Dictionary with network metrics for each interface
        """
        if not psutil:
            return {"error": "psutil not available"}

        net_io = psutil.net_io_counters(pernic=True)
        net_info = {}

        for interface, stats in net_io.items():
            net_info[interface] = {
                "bytes_sent": stats.bytes_sent,
                "bytes_recv": stats.bytes_recv,
                "packets_sent": stats.packets_sent,
                "packets_recv": stats.packets_recv,
                "errin": stats.errin if hasattr(stats, "errin") else 0,
                "errout": stats.errout if hasattr(stats, "errout") else 0,
                "dropin": stats.dropin if hasattr(stats, "dropin") else 0,
                "dropout": stats.dropout if hasattr(stats, "dropout") else 0,
            }

        return net_info

    def get_gpu_info(self) -> List[Dict[str, Union[str, float, int]]]:
        """
        Get GPU usage information.

        Returns:
            List of dictionaries with GPU metrics for each GPU
        """
        if not GPUtil:
            return [{"error": "GPUtil not available"}]

        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append(
                    {
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,  # Convert to percentage
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal)
                        * 100,
                        "temperature": gpu.temperature,
                    }
                )
        except Exception as e:
            self.logger.warning(f"Error getting GPU information: {str(e)}")

        return gpu_info

    def get_all_resources(self) -> Dict[str, Any]:
        """
        Get a comprehensive overview of all system resources.

        Returns:
            Dictionary with all resource metrics
        """
        resources = {
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_info(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
            "network": self.get_network_info(),
            "gpu": self.get_gpu_info(),
        }

        return resources

    def check_resource_thresholds(self) -> List[Dict[str, str]]:
        """
        Check if any resources are exceeding defined thresholds.

        Returns:
            List of alerts for resources that exceed thresholds
        """
        alerts = []

        # CPU check
        cpu_info = self.get_cpu_info()
        if not isinstance(cpu_info, dict) or "error" not in cpu_info:
            if cpu_info.get("percent", 0) > self.cpu_threshold:
                alerts.append(
                    {
                        "resource": "CPU",
                        "value": f"{cpu_info.get('percent')}%",
                        "threshold": f"{self.cpu_threshold}%",
                        "severity": (
                            "WARNING"
                            if cpu_info.get("percent", 0) < 95
                            else "CRITICAL"
                        ),
                    }
                )

        # Memory check
        memory_info = self.get_memory_info()
        if not isinstance(memory_info, dict) or "error" not in memory_info:
            if memory_info.get("percent", 0) > self.memory_threshold:
                alerts.append(
                    {
                        "resource": "Memory",
                        "value": f"{memory_info.get('percent')}%",
                        "threshold": f"{self.memory_threshold}%",
                        "severity": (
                            "WARNING"
                            if memory_info.get("percent", 0) < 95
                            else "CRITICAL"
                        ),
                    }
                )

        # Disk check
        disk_info = self.get_disk_info()
        if (
            isinstance(disk_info, list)
            and disk_info
            and "error" not in disk_info[0]
        ):
            for disk in disk_info:
                if disk.get("percent", 0) > self.disk_threshold:
                    alerts.append(
                        {
                            "resource": f"Disk ({disk.get('mountpoint')})",
                            "value": f"{disk.get('percent')}%",
                            "threshold": f"{self.disk_threshold}%",
                            "severity": (
                                "WARNING"
                                if disk.get("percent", 0) < 95
                                else "CRITICAL"
                            ),
                        }
                    )

        # GPU check
        gpu_info = self.get_gpu_info()
        if (
            isinstance(gpu_info, list)
            and gpu_info
            and "error" not in gpu_info[0]
        ):
            for gpu in gpu_info:
                if gpu.get("memory_percent", 0) > self.gpu_threshold:
                    alerts.append(
                        {
                            "resource": f"GPU {gpu.get('id')} ({gpu.get('name')})",
                            "value": f"{gpu.get('memory_percent')}%",
                            "threshold": f"{self.gpu_threshold}%",
                            "severity": (
                                "WARNING"
                                if gpu.get("memory_percent", 0) < 95
                                else "CRITICAL"
                            ),
                        }
                    )

        return alerts

    def update_history(self) -> None:
        """Update historical resource data."""
        timestamp = datetime.now()

        # CPU
        cpu_info = self.get_cpu_info()
        if not isinstance(cpu_info, dict) or "error" not in cpu_info:
            self.history["cpu"].append(cpu_info.get("percent", 0))
        else:
            self.history["cpu"].append(0)

        # Memory
        memory_info = self.get_memory_info()
        if not isinstance(memory_info, dict) or "error" not in memory_info:
            self.history["memory"].append(memory_info.get("percent", 0))
        else:
            self.history["memory"].append(0)

        # Disk - average of all disks
        disk_info = self.get_disk_info()
        if (
            isinstance(disk_info, list)
            and disk_info
            and "error" not in disk_info[0]
        ):
            disk_percent = sum(
                disk.get("percent", 0) for disk in disk_info
            ) / len(disk_info)
            self.history["disk"].append(disk_percent)
        else:
            self.history["disk"].append(0)

        # GPU - average of all GPUs
        gpu_info = self.get_gpu_info()
        if (
            isinstance(gpu_info, list)
            and gpu_info
            and "error" not in gpu_info[0]
        ):
            gpu_percent = sum(
                gpu.get("memory_percent", 0) for gpu in gpu_info
            ) / len(gpu_info)
            self.history["gpu"].append(gpu_percent)
        else:
            self.history["gpu"].append(0)

        # Network - total bytes (sent + received) across all interfaces
        net_info = self.get_network_info()
        if not isinstance(net_info, dict) or "error" not in net_info:
            total_bytes = sum(
                stats.get("bytes_sent", 0) + stats.get("bytes_recv", 0)
                for interface, stats in net_info.items()
            )
            self.history["network"].append(total_bytes)
        else:
            self.history["network"].append(0)

        self.history["timestamp"].append(timestamp)

        # Trim history if it exceeds the limit
        if len(self.history["timestamp"]) > self.history_limit:
            for key in self.history:
                self.history[key] = self.history[key][-self.history_limit :]

    def monitor_resources(self, duration: Optional[int] = None) -> None:
        """
        Continuously monitor system resources.

        Args:
            duration: Monitoring duration in seconds (None for indefinite)
        """
        self.logger.info("Starting resource monitoring")
        start_time = time.time()

        try:
            while True:
                # Check if monitoring should end
                if duration and time.time() - start_time > duration:
                    break

                # Get resource data
                resources = self.get_all_resources()
                self.update_history()

                # Check for alerts
                alerts = self.check_resource_thresholds()
                for alert in alerts:
                    log_method = getattr(
                        self.logger,
                        alert["severity"].lower(),
                        self.logger.warning,
                    )
                    log_method(
                        f"{alert['resource']} usage at {alert['value']} "
                        f"(threshold: {alert['threshold']})"
                    )

                # Log basic info
                if not alerts:
                    self.logger.debug("Resources within normal parameters")

                # Sleep for the interval
                time.sleep(self.interval)

        except KeyboardInterrupt:
            self.logger.info("Resource monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error during resource monitoring: {str(e)}")
        finally:
            self.logger.info("Resource monitoring ended")

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive resource report.

        Returns:
            Dictionary with current and historical resource metrics
        """
        current = self.get_all_resources()
        report = {
            "timestamp": datetime.now().isoformat(),
            "current": current,
            "history": self.history,
            "alerts": self.check_resource_thresholds(),
            "summary": {
                "cpu_avg": (
                    sum(self.history["cpu"]) / len(self.history["cpu"])
                    if self.history["cpu"]
                    else 0
                ),
                "memory_avg": (
                    sum(self.history["memory"]) / len(self.history["memory"])
                    if self.history["memory"]
                    else 0
                ),
                "disk_avg": (
                    sum(self.history["disk"]) / len(self.history["disk"])
                    if self.history["disk"]
                    else 0
                ),
                "gpu_avg": (
                    sum(self.history["gpu"]) / len(self.history["gpu"])
                    if self.history["gpu"]
                    else 0
                ),
            },
        }

        return report


def get_analyzer() -> ResourceAnalyzer:
    """
    Factory function to create a ResourceAnalyzer instance.

    Returns:
        Configured ResourceAnalyzer instance
    """
    return ResourceAnalyzer()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create analyzer and start monitoring
    analyzer = ResourceAnalyzer(interval=2)
    analyzer.monitor_resources(duration=60)  # Monitor for 60 seconds

    # Generate and print report summary
    report = analyzer.generate_report()
    print("\nResource Report Summary:")
    print(f"CPU avg: {report['summary']['cpu_avg']:.2f}%")
    print(f"Memory avg: {report['summary']['memory_avg']:.2f}%")
    print(f"Disk avg: {report['summary']['disk_avg']:.2f}%")
    print(f"GPU avg: {report['summary']['gpu_avg']:.2f}%")

    # Print alerts
    if report["alerts"]:
        print("\nResource Alerts:")
        for alert in report["alerts"]:
            print(
                f"- {alert['resource']}: {alert['value']} (threshold: {alert['threshold']})"
            )
    else:
        print("\nNo resource alerts detected.")
