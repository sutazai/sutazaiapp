#!/usr/bin/env python3
"""
Ultimate System Audit Script

This script performs a comprehensive audit of the system components:
- CPU usage
- Memory usage
- Disk usage across mounted partitions

It then consolidates these metrics into a summary report and prints the results.
"""

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict

import psutil

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/opt/sutazai/logs/system_audit.log",
)
logger = logging.getLogger("UltimateSystemAudit")


class SystemAuditManager:
    """
    Comprehensive system audit and diagnostic management class.

    Provides in-depth system analysis across multiple dimensions:
    - Hardware resources
    - Software configuration
    - Network connectivity
    """

    def __init__(self, output_dir: str = "/opt/sutazai/audit_reports"):
        """
        Initialize the SystemAuditManager.

        Args:
            output_dir (str): Directory to store audit reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def collect_system_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive system information.

        Returns:
            Dict containing detailed system metadata
        """
        return {
            "os": {
                "platform": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "path": sys.path,
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(logical=False),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "total_memory": psutil.virtual_memory().total / (1024**3),  # GB
                "available_memory": psutil.virtual_memory().available / (1024**3),  # GB
            },
        }

    def network_diagnostics(self) -> Dict[str, Any]:
        """
        Perform comprehensive network diagnostics.

        Returns:
            Dict with network connectivity and interface details
        """
        try:
            # Use socket.gethostname() for hostname retrieval
            hostname = socket.gethostname()

            # Retrieve network interface statistics
            interfaces = psutil.net_if_stats()

            return {
                "hostname": hostname,
                "interfaces": {
                    name: {
                        "is_up": details.isup,  # Network interface status
                        "speed": details.speed,
                    }
                    for name, details in interfaces.items()
                },
                "default_gateway": self._get_default_gateway(),
            }
        except Exception as e:
            logger.error(f"Network diagnostics error: {e}")
            return {"status": "error", "message": str(e)}

    def _get_default_gateway(self) -> str:
        """
        Retrieve the default network gateway.

        Returns:
            str: Default gateway IP address
        """
        try:
            # Platform-specific gateway retrieval
            if platform.system() == "Windows":
                output = subprocess.check_output(["ipconfig"], universal_newlines=True)
                for line in output.split("\n"):
                    if "Default Gateway" in line:
                        return line.split(":")[-1].strip()
            else:
                output = subprocess.check_output(
                    ["ip", "route", "show", "default"], universal_newlines=True
                )
                return output.split()[2]
        except Exception:
            return "Unknown"

    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive system audit report.

        Returns:
            str: Path to the generated report file
        """
        try:
            report_data = {
                "timestamp": self.timestamp,
                "system_info": self.collect_system_info(),
                "network_diagnostics": self.network_diagnostics(),
                "cpu_usage": self._get_cpu_usage(),
                "memory_usage": self._get_memory_usage(),
                "disk_usage": self._get_disk_usage(),
            }

            report_path = os.path.join(
                self.output_dir, f"system_audit_report_{self.timestamp}.json"
            )

            with open(report_path, "w") as report_file:
                json.dump(report_data, report_file, indent=4)

            print(f"🔍 Comprehensive System Audit Complete!")
            print(f"📄 Report Generated: {report_path}")

            logger.info(f"System audit report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"System audit failed: {e}")
            sys.exit(1)

    def _get_cpu_usage(self) -> Dict[str, Any]:
        """
        Retrieve detailed CPU usage statistics.

        Returns:
            Dict with CPU usage metrics
        """
        return {
            "overall_usage": psutil.cpu_percent(),
            "per_cpu_usage": psutil.cpu_percent(percpu=True),
        }

    def _get_memory_usage(self) -> Dict[str, Any]:
        """
        Retrieve detailed memory usage statistics.

        Returns:
            Dict with memory usage metrics
        """
        memory = psutil.virtual_memory()
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "percent_used": memory.percent,
        }

    def _get_disk_usage(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive disk usage statistics.

        Returns:
            Dict with disk usage metrics
        """
        return {
            partition.mountpoint: {
                "total": psutil.disk_usage(partition.mountpoint).total
                / (1024**3),  # GB
                "used": psutil.disk_usage(partition.mountpoint).used / (1024**3),  # GB
                "free": psutil.disk_usage(partition.mountpoint).free / (1024**3),  # GB
                "percent": psutil.disk_usage(partition.mountpoint).percent,
            }
            for partition in psutil.disk_partitions()
        }


def get_system_message() -> str:
    """
    Generate a system health message based on CPU usage.

    Returns:
        str: A message summarizing the system health status.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    # Although psutil.cpu_percent() should always return a float,
    # we explicitly ensure a message is returned on all code paths.
    if cpu_usage is None:
        return "CPU usage data is unavailable."
    else:
        return f"System is running with CPU usage at {cpu_usage}%."


def get_cpu_usage() -> Dict[str, float]:
    """
    Calculate the average CPU usage over all available CPU cores.

    Returns:
        Dict[str, float]: Dictionary containing the average CPU usage as a percentage.
    """
    # Obtain per-core CPU usage percentages; psutil returns a list of floats.
    per_cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    avg_usage = sum(per_cpu_usage) / len(per_cpu_usage) if per_cpu_usage else 0.0
    return {"average_cpu_usage": avg_usage}


def get_memory_usage() -> Dict[str, float]:
    """
    Retrieve statistics on system memory usage.

    Returns:
        Dict[str, float]: Dictionary containing total, available, and used memory (in bytes)
                          and the memory usage percentage.
    """
    mem = psutil.virtual_memory()
    return {
        "total_memory": float(mem.total),
        "available_memory": float(mem.available),
        "used_memory": float(mem.used),
        "memory_percent": float(mem.percent),
    }


def get_disk_usage() -> Dict[str, Dict[str, float]]:
    """
    Aggregate disk usage data from all accessible mounted partitions.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary of mount points, each containing
        total, used, free space (in bytes), and the disk usage percentage.
    """
    partitions = psutil.disk_partitions()
    disk_usage = {}

    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            disk_usage[partition.mountpoint] = {
                "total": float(usage.total),
                "used": float(usage.used),
                "free": float(usage.free),
                "percent": float(usage.percent),
            }
        except (PermissionError, OSError):
            # Skip partitions that cannot be accessed
            continue

    return disk_usage


def main():
    """Main function that runs the system audit."""
    try:
        audit_manager = SystemAuditManager()
        report_path = audit_manager.generate_comprehensive_report()
        print(f"Report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Error running system audit: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
