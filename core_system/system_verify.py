#!/usr/bin/env python3
"""
SutazAI Lightweight System Verification Script
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys


class SystemVerifier:
    def __init__(self):
        self.results = {
            "os_compatibility": {},
            "hardware_requirements": {},
            "software_dependencies": {},
            "network_checks": {}
        }
        self.project_root = "/media/ai/SutazAI_Storage/SutazAI/v1"

    def _run_command(self, command):
        """Run shell command and return output."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout.strip(), result.returncode
        except Exception as e:
            return str(e), 1

    def verify_os_compatibility(self):
        """Check operating system compatibility."""
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "distribution": platform.platform()
        }
        
        compatible_distros = ["Ubuntu", "Debian", "Linux"]
        is_compatible = (
            os_info["system"] == "Linux" and 
            any(distro in os_info["distribution"] for distro in compatible_distros)
        )
        
        self.results["os_compatibility"] = {
            "status": "PASS" if is_compatible else "FAIL",
            "details": os_info
        }
        return is_compatible

    def check_hardware_requirements(self):
        """Verify hardware requirements using built-in methods."""
        try:
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 * 1024 * 1024)  # GB
            cpu_cores = os.cpu_count()
            disk_space = shutil.disk_usage(self.project_root).free / (1024 * 1024 * 1024)  # GB

            requirements = {
                "min_cpu_cores": 4,
                "min_memory_gb": 8,
                "min_disk_space_gb": 20
            }

            hardware_check = {
                "cpu_cores": cpu_cores >= requirements["min_cpu_cores"],
                "memory": total_memory >= requirements["min_memory_gb"],
                "disk_space": disk_space >= requirements["min_disk_space_gb"]
            }

            self.results["hardware_requirements"] = {
                "status": "PASS" if all(hardware_check.values()) else "WARN",
                "details": {
                    "cpu_cores": f"{cpu_cores} (min {requirements['min_cpu_cores']})",
                    "memory_gb": f"{total_memory:.2f} (min {requirements['min_memory_gb']})",
                    "free_disk_space_gb": f"{disk_space:.2f} (min {requirements['min_disk_space_gb']})"
                }
            }
            return all(hardware_check.values())
        except Exception as e:
            print(f"Hardware check error: {e}")
            return False

    def verify_software_dependencies(self):
        """Check required software dependencies."""
        dependencies = {
            "docker": self._run_command("docker --version")[1] == 0,
            "docker_compose": self._run_command("docker-compose --version")[1] == 0,
            "python3": self._run_command("python3 --version")[1] == 0,
            "git": self._run_command("git --version")[1] == 0
        }

        self.results["software_dependencies"] = {
            "status": "PASS" if all(dependencies.values()) else "FAIL",
            "details": dependencies
        }
        return all(dependencies.values())

    def check_network_connectivity(self):
        """Verify network connectivity and required ports."""
        required_ports = {
            "postgres": 5432,
            "redis": 6379,
            "api": 8000
        }

        network_checks = {}
        for service, port in required_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                # Only check if port is not already in use
                result = sock.connect_ex(('localhost', port))
                network_checks[service] = result == 0 or result == 111  # 111 means address already in use
            except Exception:
                network_checks[service] = False
            finally:
                sock.close()

        self.results["network_checks"] = {
            "status": "PASS" if len([x for x in network_checks.values() if x]) > 0 else "WARN",
            "details": network_checks
        }
        return len([x for x in network_checks.values() if x]) > 0

    def run_verification(self):
        """Run all system verification checks."""
        checks = [
            self.verify_os_compatibility,
            self.check_hardware_requirements,
            self.verify_software_dependencies,
            self.check_network_connectivity
        ]

        overall_status = True
        for check in checks:
            overall_status &= check()

        return overall_status

    def generate_report(self):
        """Generate a detailed verification report."""
        report_path = os.path.join(self.project_root, "system_verification_report.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        return report_path

def main():
    verifier = SystemVerifier()
    
    try:
        verification_result = verifier.run_verification()
        report_path = verifier.generate_report()
        
        print(json.dumps(verifier.results, indent=2))
        
        if verification_result:
            print("\n✅ System Verification Passed!")
            sys.exit(0)
        else:
            print("\n❌ System Verification Failed. Check the report for details.")
            sys.exit(1)
    
    except Exception as e:
        print(f"Verification Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()