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
from typing import Any, Dict, List, Union

import psutil

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(level_name)s - %(message)s",
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
    - Security posture
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
                output = subprocess.check_output(["ip", "route", "show", "default"], universal_newlines=True)
                return output.split()[2]
        except Exception:
            return "Unknown"
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
                output = subprocess.check_output(["ip", "route", "show", "default"], universal_newlines=True)
                return output.split()[2]
        except Exception:
            return "Unknown"
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
        report_data = {
            "timestamp": self.timestamp,
            "system_info": self.collect_system_info(),
            "network_diagnostics": self.network_diagnostics(),
            "cpu_usage": self._get_cpu_usage(),
            "memory_usage": self._get_memory_usage(),
    def _get_disk_usage(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive disk usage statistics.

        Returns:
            Dict with disk usage metrics
        """
        return {
            partition.mountpoint: {
                "total": psutil.disk_usage(partition.mountpoint).total / (1024**3),  # GB
                "used": psutil.disk_usage(partition.mountpoint).used / (1024**3),  # GB
                "free": psutil.disk_usage(partition.mountpoint).free / (1024**3),  # GB
                "percent": psutil.disk_usage(partition.mountpoint).percent,
            }
            for partition in psutil.disk_partitions()
        }
        report_path = os.path.join(
            self.output_dir, f"system_audit_report_{self.timestamp}.json"
        )

        with open(report_path, "w") as report_file:
            json.dump(report_data, report_file, indent=4)

        return report_path

    def _get_cpu_usage(self) -> Dict[str, float]:
        """
        Retrieve detailed CPU usage statistics.

        Returns:
            Dict with CPU usage metrics
        """
        return {
            "overall_usage": psutil.cpu_percent(),
            "per_cpu_usage": psutil.cpu_percent(percpu=True),
        }

    def _get_memory_usage(self) -> Dict[str, float]:
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
                "total": psutil.disk_usage(partition.mountpoint).total / (1024**3),  # GB
                "used": psutil.disk_usage(partition.mountpoint).used / (1024**3),  # GB
                "free": psutil.disk_usage(partition.mountpoint).free / (1024**3),  # GB
                "percent": psutil.disk_usage(partition.mountpoint).percent,
            }
            for partition in psutil.disk_partitions()
        }
        Returns:
            Dict with disk usage metrics
        """
        return {
            partition.mountpoint: {
                "total": partition.total / (1024**3),  # GB
                "used": partition.used / (1024**3),  # GB
                "free": partition.free / (1024**3),  # GB
                "percent": partition.percent,
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


def get_disk_usage() -> Dict[str, float]:
    """
    Aggregate disk usage data from all accessible mounted partitions.

    Returns:
        Dict[str, float]: A dictionary including total, used, free space (in bytes), and
                          the average disk usage percentage.
    """
    total_total = 0
    total_used = 0
    total_free = 0
    total_percent = 0.0
    count = 0

    # Iterate over each mounted disk partition.
    for partition in psutil.disk_partitions():
        try:
            # Obtain disk usage statistics using the partition's mountpoint.
            usage = psutil.disk_usage(partition.mountpoint)
            total_total += usage.total
            total_used += usage.used
            total_free += usage.free
            total_percent += usage.percent
            count += 1
        except Exception:
            # If disk usage cannot be accessed for a partition, skip to the next.
            continue

    # If no valid partitions are found, return zeros.
    if count == 0:
        return {"total": 0.0, "used": 0.0, "free": 0.0, "percent": 0.0}

    # Compute the mean disk usage percentage.
    avg_percent = total_percent / count

    return {
        "total": float(total_total),
        "used": float(total_used),
        "free": float(total_free),
        "percent": float(avg_percent),
    }


def get_all_audit() -> Dict[str, Union[float, str]]:
    """
    Consolidate all system audit metrics into a single dictionary.

    Returns:
        Dict[str, Union[float, str]]: A dictionary containing CPU, memory, and disk usage metrics,
                                      along with a system health message.
    """
    audit_data: Dict[str, Union[float, str]] = {}
    audit_data.update(get_cpu_usage())
    audit_data.update(get_memory_usage())
    audit_data.update(get_disk_usage())
    audit_data["system_message"] = get_system_message()
    return audit_data


def generate_audit_report() -> Dict[str, float]:
    """
    Generates a detailed audit report.

    Returns:
        A dictionary with system audit metrics as floats.
    """
    cpu_metric = float(psutil.cpu_percent(interval=1))
    mem_metric = float(psutil.virtual_memory().percent)
    disk_usage = psutil.disk_usage("/")
    disk_metric = float(disk_usage.percent)

    report: Dict[str, float] = {
        "cpu_usage": cpu_metric,
        "memory_usage": mem_metric,
        "disk_usage": disk_metric,
    }
    return report


def ultimate_system_audit() -> str:
    """
    Performs the ultimate system audit and returns a summary string.

    Returns:
        A formatted string summarizing audit metrics.
    """
    report = generate_audit_report()
    report_lines = [f"{key}: {value:.2f}" for key, value in report.items()]
    final_report = "\n".join(report_lines)
    return final_report


def main() -> None:
    try:
        audit_manager = SystemAuditManager()
        report_path = audit_manager.generate_comprehensive_report()

        print(f"🔍 Comprehensive System Audit Complete!")
        print(f"📄 Report Generated: {report_path}")

        logger.info(f"System audit report generated: {report_path}")

    except Exception as e:
        logger.error(f"System audit failed: {e}")
        sys.exit(1)
        print(f"🔍 Comprehensive System Audit Complete!")
        print(f"📄 Report Generated: {report_path}")

        logger.info(f"System audit report generated: {report_path}")

    except Exception as e:
        logger.error(f"System audit failed: {e}")
        sys.exit(1)
        print(f"🔍 Comprehensive System Audit Complete!")
        print(f"📄 Report Generated: {report_path}")

        logger.info(f"System audit report generated: {report_path}")

    except Exception as e:
        logger.error(f"System audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    report_str = ultimate_system_audit()
    print("Ultimate System Audit Report:")
    print(report_str)
