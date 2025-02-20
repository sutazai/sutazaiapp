import json
import logging
import os
import platform
import socket
import sys
from typing import Any, Dict

import psutil


class SystemDiagnostic:
    def __init__(self):
        self.diagnostic_report = {
            "system_info": {},
            "performance": {},
            "network": {},
            "potential_issues": [],
        }

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler("/var/log/sutazai/system_diagnostic.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def collect_system_info(self):
        """Collect comprehensive system information."""
        try:
            self.diagnostic_report["system_info"] = {
                "os": {
                    "name": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                },
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                },
                "hardware": {
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
            }
        except Exception as e:
            self.logger.error(f"System info collection error: {e}")
            self.diagnostic_report["potential_issues"].append(
                f"System info collection failed: {e}"
            )

    def analyze_performance(self):
        """Analyze system performance and resource utilization."""
        try:
            self.diagnostic_report["performance"] = {
                "cpu": {
                    "cores": psutil.cpu_count(),
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "load_average": (
                        os.getloadavg() if hasattr(os, "getloadavg") else "N/A"
                    ),
                },
                "memory": {
                    "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "available_gb": round(
                        psutil.virtual_memory().available / (1024**3), 2
                    ),
                    "usage_percent": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total_gb": round(psutil.disk_usage("/").total / (1024**3), 2),
                    "free_gb": round(psutil.disk_usage("/").free / (1024**3), 2),
                    "usage_percent": psutil.disk_usage("/").percent,
                },
            }

            # Check for performance bottlenecks
            if self.diagnostic_report["performance"]["cpu"]["usage_percent"] > 80:
                self.diagnostic_report["potential_issues"].append(
                    f"High CPU usage: {self.diagnostic_report['performance']['cpu']['usage_percent']}%"
                )

            if self.diagnostic_report["performance"]["memory"]["usage_percent"] > 85:
                self.diagnostic_report["potential_issues"].append(
                    f"High memory usage: {self.diagnostic_report['performance']['memory']['usage_percent']}%"
                )

        except Exception as e:
            self.logger.error(f"Performance analysis error: {e}")
            self.diagnostic_report["potential_issues"].append(
                f"Performance analysis failed: {e}"
            )

    def check_network(self):
        """Check network configuration and connectivity."""
        try:
            # Check network interfaces
            network_interfaces = psutil.net_if_addrs()
            self.diagnostic_report["network"]["interfaces"] = {
                name: [addr.address for addr in addrs if addr.family == socket.AF_INET]
                for name, addrs in network_interfaces.items()
            }

            # Check critical ports
            critical_ports = [8000, 5432, 6379]  # API, PostgreSQL, Redis
            port_status = {}
            for port in critical_ports:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(("localhost", port))
                port_status[port] = "open" if result == 0 else "closed"
                sock.close()

            self.diagnostic_report["network"]["port_status"] = port_status

        except Exception as e:
            self.logger.error(f"Network diagnostic error: {e}")
            self.diagnostic_report["potential_issues"].append(
                f"Network diagnostics failed: {e}"
            )

    def generate_report(
        self, output_path="/var/log/sutazai/system_diagnostic_report.json"
    ):
        """Generate a comprehensive diagnostic report."""
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(self.diagnostic_report, f, indent=2)

            self.logger.info(f"Diagnostic report generated: {output_path}")

            # Print summary
            print("\n🔍 System Diagnostic Report 🔍")

            print("\nSystem Information:")
            print(json.dumps(self.diagnostic_report["system_info"], indent=2))

            print("\nPerformance:")
            print(json.dumps(self.diagnostic_report["performance"], indent=2))

            print("\nPotential Issues:")
            for issue in self.diagnostic_report.get("potential_issues", []):
                print(f"  - {issue}")

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")


def main():
    diagnostic = SystemDiagnostic()
    diagnostic.collect_system_info()
    diagnostic.analyze_performance()
    diagnostic.check_network()
    diagnostic.generate_report()


if __name__ == "__main__":
    main()
