#!/usr/bin/env python3
"""
SutazAI Comprehensive System Audit Script

Performs an exhaustive audit of the entire system, including:
- Configuration Validation
- Dependency Checks
- Security Assessment
- Performance Metrics
- Logging and Monitoring Evaluation
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import platform

import pkg_resources
import psutil
import safety

from config.config_manager import ConfigurationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="/opt/sutazai_project/SutazAI/logs/system_audit.log",
    filemode="a",
)
logger = logging.getLogger("SutazAI.SystemAudit")


class SystemAuditor:
    """
    Comprehensive system auditing framework for SutazAI
    """

    def __init__(self, config_env: str = "development"):
        """
        Initialize system auditor with configuration environment

        Args:
            config_env (str): Configuration environment to audit
        """
        self.config_manager = ConfigurationManager(environment=config_env)
        self.audit_report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "environment": config_env,
            "sections": {},
        }

    def audit_configuration(self) -> Dict[str, Any]:
        """
        Perform comprehensive configuration audit

        Returns:
            Configuration audit results
        """
        try:
            config = self.config_manager.load_config()
            profile = self.config_manager.create_profile()

            audit_result = {
                "status": "PASSED",
                "config_profile": json.loads(json.dumps(profile.__dict__)),
                "config_sections": list(config.keys()),
            }

            self.audit_report["sections"]["configuration"] = audit_result
            return audit_result

        except Exception as e:
            error_result = {"status": "FAILED", "error": str(e)}
            self.audit_report["sections"]["configuration"] = error_result
            logger.error(f"Configuration audit failed: {e}")
            return error_result

    def audit_dependencies(self) -> Dict[str, Any]:
        """
        Perform comprehensive dependency audit

        Returns:
            Dependency audit results
        """
        try:
            # Get installed packages
            installed_packages = {
                pkg.key: pkg.version for pkg in pkg_resources.working_set
            }

            # Check vulnerabilities
            vulnerability_check = safety.check()

            audit_result = {
                "status": "PASSED",
                "total_packages": len(installed_packages),
                "vulnerable_packages": [vuln[0] for vuln in vulnerability_check],
                "outdated_packages": [],  # TODO: Implement version comparison
            }

            if audit_result["vulnerable_packages"]:
                audit_result["status"] = "WARNING"

            self.audit_report["sections"]["dependencies"] = audit_result
            return audit_result

        except Exception as e:
            error_result = {"status": "FAILED", "error": str(e)}
            self.audit_report["sections"]["dependencies"] = error_result
            logger.error(f"Dependency audit failed: {e}")
            return error_result

    def audit_system_resources(self) -> Dict[str, Any]:
        """
        Audit system resources and performance metrics

        Returns:
            System resources audit results
        """
        try:
            audit_result = {
                "status": "PASSED",
                "system_info": {
                    "os": platform.platform(),
                    "python_version": platform.python_version(),
                    "machine": platform.machine(),
                },
                "cpu": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "usage_percent": psutil.cpu_percent(interval=1),
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent_used": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": psutil.disk_usage("/").total,
                    "free": psutil.disk_usage("/").free,
                    "percent_used": psutil.disk_usage("/").percent,
                },
            }

            # Add warnings for high resource usage
            if audit_result["cpu"]["usage_percent"] > 70:
                audit_result["status"] = "WARNING"

            if audit_result["memory"]["percent_used"] > 80:
                audit_result["status"] = "WARNING"

            self.audit_report["sections"]["system_resources"] = audit_result
            return audit_result

        except Exception as e:
            error_result = {"status": "FAILED", "error": str(e)}
            self.audit_report["sections"]["system_resources"] = error_result
            logger.error(f"System resources audit failed: {e}")
            return error_result

    def generate_comprehensive_audit(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system audit report

        Returns:
            Complete audit report
        """
        # Run all audit sections
        self.audit_configuration()
        self.audit_dependencies()
        self.audit_system_resources()

        # Determine overall system status
        statuses = [
            section.get("status", "UNKNOWN")
            for section in self.audit_report["sections"].values()
        ]

        if "FAILED" in statuses:
            self.audit_report["overall_status"] = "CRITICAL"
        elif "WARNING" in statuses:
            self.audit_report["overall_status"] = "WARNING"
        else:
            self.audit_report["overall_status"] = "PASSED"

        # Save audit report
        report_path = f'/opt/sutazai_project/SutazAI/logs/system_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        with open(report_path, "w") as f:
            json.dump(self.audit_report, f, indent=2)

        logger.info(
            f"Comprehensive system audit completed. Report saved: {report_path}"
        )

        return self.audit_report


def main():
    """
    Main execution for system audit
    """
    try:
        auditor = SystemAuditor()
        audit_report = auditor.generate_comprehensive_audit()

        print("System Audit Report Summary:")
        print(f"Overall Status: {audit_report['overall_status']}")
        print(f"Timestamp: {audit_report['timestamp']}")
        print("\nDetailed Sections:")
        for section, details in audit_report["sections"].items():
            print(f"{section.upper()}: {details['status']}")

    except Exception as e:
        print(f"System audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
