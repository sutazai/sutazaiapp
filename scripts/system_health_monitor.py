import json
import logging
import os
import platform
import socket
import sys
import time
from typing import Any, Dict, List, Optional

import psutil


class SystemHealthMonitor:
    """
    A comprehensive system health monitoring tool for SutazAI project.
    Tracks system resources, network status, and performance metrics.
    """

    def __init__(self, project_root: str = '.'):
        """
        Initialize the system health monitor.

        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = os.path.abspath(project_root)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_system_info(self) -> Dict[str, Any]:
        """
        Collect comprehensive system information.

        Returns:
            Dict[str, Any]: System information details
        """
        return {
            'os': {
                'platform': platform.platform(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'hostname': socket.gethostname(),
            'python_version': platform.python_version()
        }

    def monitor_cpu_usage(self) -> Dict[str, float]:
        """
        Monitor CPU usage and performance.

        Returns:
            Dict[str, float]: CPU usage metrics
        """
        return {
            'percent_used': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count(),
            'logical_cores': psutil.cpu_count(logical=True)
        }

    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor system memory usage.

        Returns:
            Dict[str, float]: Memory usage metrics
        """
        memory = psutil.virtual_memory()
        return {
            'total_memory_mb': memory.total / (1024 * 1024),
            'available_memory_mb': memory.available / (1024 * 1024),
            'used_memory_mb': memory.used / (1024 * 1024),
            'memory_percent_used': memory.percent
        }

    def monitor_disk_usage(self) -> List[Dict[str, Any]]:
        """
        Monitor disk usage across all partitions.

        Returns:
            List[Dict[str, Any]]: Disk usage metrics for each partition
        """
        disk_usage = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'filesystem_type': partition.fstype,
                    'total_size_gb': usage.total / (1024 * 1024 * 1024),
                    'used_size_gb': usage.used / (1024 * 1024 * 1024),
                    'free_size_gb': usage.free / (1024 * 1024 * 1024),
                    'percent_used': usage.percent
                })
            except Exception as e:
                self.logger.warning(f"Could not get disk usage for {partition.mountpoint}: {e}")
        return disk_usage

    def monitor_network_connections(self) -> Dict[str, Any]:
        """
        Monitor network connections and interfaces.

        Returns:
            Dict[str, Any]: Network connection details
        """
        network_info = {
            'interfaces': {},
            'connections': []
        }

        # Network interfaces
        for interface, addresses in psutil.net_if_addrs().items():
            network_info['interfaces'][interface] = [
                addr.address for addr in addresses if addr.family == socket.AF_INET
            ]

        # Active network connections
        for conn in psutil.net_connections():
            network_info['connections'].append({
                'fd': conn.fd,
                'family': conn.family,
                'type': conn.type,
                'laddr': conn.laddr,
                'raddr': conn.raddr,
                'status': conn.status
            })

        return network_info

    def generate_health_report(self) -> None:
        """
        Generate a comprehensive system health report.
        """
        health_report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': self.get_system_info(),
            'cpu_usage': self.monitor_cpu_usage(),
            'memory_usage': self.monitor_memory_usage(),
            'disk_usage': self.monitor_disk_usage(),
            'network_connections': self.monitor_network_connections()
        }

        report_path = os.path.join(
            self.project_root, 
            f'system_health_report_{time.strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(health_report, f, indent=2)
            self.logger.info(f"System health report generated: {report_path}")
        except Exception as e:
            self.logger.error(f"Error generating system health report: {e}")

def main():
    health_monitor = SystemHealthMonitor()
    health_monitor.generate_health_report()

if __name__ == '__main__':
    main() 