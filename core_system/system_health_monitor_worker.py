#!/usr/bin/env python3
"""
SutazAI Comprehensive System Health Monitoring Worker

Provides autonomous monitoring, validation, and self-healing
capabilities for critical system jobs and services.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import psutil
import requests
import schedule
import yaml


class SystemHealthMonitorWorker:
    """
    Advanced system health monitoring and job validation system

    Key Responsibilities:
    - Monitor critical system jobs
    - Validate job health and performance
    - Automatic recovery and restart mechanisms
    - Comprehensive logging and alerting
    """

    def __init__(
        self,
        base_dir: str = "/opt/sutazai_project/SutazAI",
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize System Health Monitor Worker

        Args:
            base_dir (str): Base project directory
            config_path (Optional[str]): Path to health monitoring configuration
            log_dir (Optional[str]): Custom log directory
        """
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(
            base_dir, "config", "system_health_config.yml"
        )
        self.log_dir = log_dir or os.path.join(
            base_dir, "logs", "system_health_monitor"
        )

        # Ensure directories exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=os.path.join(self.log_dir, "system_health_monitor.log"),
        )
        self.logger = logging.getLogger("SutazAI.SystemHealthMonitor")

        # Load configuration
        self.load_configuration()

    def load_configuration(self):
        """
        Load system health monitoring configuration
        """
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)

            self.logger.info("System health configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load health configuration: {e}")
            self.config = {
                "critical_jobs": [],
                "monitoring_interval": 300,  # Default 5 minutes
                "recovery_actions": {},
            }

    def check_job_status(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the status of a specific job

        Args:
            job_config (Dict): Configuration for the job to check

        Returns:
            Dictionary with job status details
        """
        job_status = {
            "name": job_config.get("name", "Unknown Job"),
            "status": "UNKNOWN",
            "last_checked": datetime.now().isoformat(),
        }

        try:
            # Check job based on type
            job_type = job_config.get("type", "process")

            if job_type == "process":
                job_status.update(self._check_process_job(job_config))
            elif job_type == "systemd":
                job_status.update(self._check_systemd_job(job_config))
            elif job_type == "http":
                job_status.update(self._check_http_job(job_config))

            return job_status

        except Exception as e:
            job_status["status"] = "ERROR"
            job_status["error"] = str(e)
            self.logger.error(
                f"Job status check failed for {job_config.get('name')}: {e}"
            )
            return job_status

    def _check_process_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check status of a process-based job

        Args:
            job_config (Dict): Job configuration

        Returns:
            Dictionary with process job status
        """
        try:
            # Find process by name or PID
            process_name = job_config.get("process_name")
            pid = job_config.get("pid")

            if pid:
                try:
                    process = psutil.Process(pid)
                    return {
                        "status": "RUNNING",
                        "cpu_usage": process.cpu_percent(),
                        "memory_usage": process.memory_percent(),
                    }
                except psutil.NoSuchProcess:
                    return {"status": "STOPPED"}

            elif process_name:
                for proc in psutil.process_iter(["name"]):
                    if process_name in proc.info["name"]:
                        return {
                            "status": "RUNNING",
                            "cpu_usage": proc.cpu_percent(),
                            "memory_usage": proc.memory_percent(),
                        }

                return {"status": "STOPPED"}

            return {"status": "UNKNOWN"}

        except Exception as e:
            self.logger.error(f"Process job check failed: {e}")
            return {"status": "ERROR", "error": str(e)}

    def _check_systemd_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check status of a systemd service

        Args:
            job_config (Dict): Job configuration

        Returns:
            Dictionary with systemd job status
        """
        try:
            service_name = job_config.get("service_name")

            # Run systemctl command to check service status
            result = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True,
            )

            status = result.stdout.strip()
            return {
                "status": "RUNNING" if status == "active" else "STOPPED",
                "systemd_status": status,
            }

        except Exception as e:
            self.logger.error(f"Systemd job check failed: {e}")
            return {"status": "ERROR", "error": str(e)}

    def _check_http_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check status of an HTTP-based service

        Args:
            job_config (Dict): Job configuration

        Returns:
            Dictionary with HTTP job status
        """
        try:
            url = job_config.get("health_check_url")
            timeout = job_config.get("timeout", 10)

            response = requests.get(url, timeout=timeout)

            return {
                "status": ("RUNNING" if response.status_code == 200 else "DEGRADED"),
                "http_status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
            }

        except requests.RequestException as e:
            self.logger.error(f"HTTP job check failed: {e}")
            return {"status": "STOPPED", "error": str(e)}

    def recover_job(self, job_config: Dict[str, Any], job_status: Dict[str, Any]):
        """
        Attempt to recover a failed job

        Args:
            job_config (Dict): Job configuration
            job_status (Dict): Current job status
        """
        try:
            recovery_actions = job_config.get("recovery_actions", [])

            for action in recovery_actions:
                if action == "restart_process":
                    self._restart_process(job_config)
                elif action == "restart_systemd":
                    self._restart_systemd_service(job_config)
                elif action == "notify_admin":
                    self._notify_admin(job_config, job_status)

            self.logger.info(f"Recovery actions completed for {job_config.get('name')}")

        except Exception as e:
            self.logger.error(f"Job recovery failed: {e}")

    def _restart_process(self, job_config: Dict[str, Any]):
        """
        Restart a specific process

        Args:
            job_config (Dict): Job configuration
        """
        try:
            process_name = job_config.get("process_name")
            pid = job_config.get("pid")

            if pid:
                # Terminate existing process
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass

            # Restart process
            subprocess.Popen(job_config.get("restart_command", "").split())

            self.logger.info(f"Restarted process: {process_name}")

        except Exception as e:
            self.logger.error(f"Process restart failed: {e}")

    def _restart_systemd_service(self, job_config: Dict[str, Any]):
        """
        Restart a systemd service

        Args:
            job_config (Dict): Job configuration
        """
        try:
            service_name = job_config.get("service_name")

            subprocess.run(["systemctl", "restart", service_name], check=True)

            self.logger.info(f"Restarted systemd service: {service_name}")

        except Exception as e:
            self.logger.error(f"Systemd service restart failed: {e}")

    def _notify_admin(self, job_config: Dict[str, Any], job_status: Dict[str, Any]):
        """
        Send notification to system administrators

        Args:
            job_config (Dict): Job configuration
            job_status (Dict): Current job status
        """
        try:
            # Placeholder for admin notification (could be email, Slack, etc.)
            admin_emails = job_config.get("admin_emails", [])

            notification_message = f"""
            Job Failure Detected:
            - Job Name: {job_config.get('name')}
            - Status: {job_status.get('status')}
            - Timestamp: {datetime.now().isoformat()}
            - Error Details: {job_status.get('error', 'No additional details')}
            """

            # Log notification (replace with actual notification mechanism)
            self.logger.warning(f"Admin notification: {notification_message}")

        except Exception as e:
            self.logger.error(f"Admin notification failed: {e}")

    def run_comprehensive_health_check(self):
        """
        Perform comprehensive system health check
        """
        try:
            self.logger.info("Starting comprehensive system health check")

            # Check status of all critical jobs
            critical_jobs = self.config.get("critical_jobs", [])
            job_statuses = []

            for job_config in critical_jobs:
                job_status = self.check_job_status(job_config)
                job_statuses.append(job_status)

                # Trigger recovery if job is not running
                if job_status["status"] not in ["RUNNING", "ACTIVE"]:
                    self.recover_job(job_config, job_status)

            # Persist health check results
            self._persist_health_check_results(job_statuses)

            self.logger.info("Comprehensive health check completed")

        except Exception as e:
            self.logger.error(f"Comprehensive health check failed: {e}")

    def _persist_health_check_results(self, job_statuses: List[Dict[str, Any]]):
        """
        Persist health check results to a log file

        Args:
            job_statuses (List): List of job status dictionaries
        """
        try:
            report_path = os.path.join(
                self.log_dir,
                f'health_check_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            )

            with open(report_path, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "job_statuses": job_statuses,
                    },
                    f,
                    indent=2,
                )

            self.logger.info(f"Health check report generated: {report_path}")

        except Exception as e:
            self.logger.error(f"Health check result persistence failed: {e}")

    def run_continuous_monitoring(self):
        """
        Run continuous system health monitoring
        """
        try:
            # Schedule periodic health checks
            schedule.every(self.config.get("monitoring_interval", 300)).seconds.do(
                self.run_comprehensive_health_check
            )

            # Keep the monitoring process running
            while True:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            self.logger.info("System health monitoring gracefully interrupted")
        except Exception as e:
            self.logger.error(f"Continuous monitoring failed: {e}")
            sys.exit(1)


def main():
    """
    Main entry point for system health monitoring
    """
    try:
        health_monitor = SystemHealthMonitorWorker()
        health_monitor.run_continuous_monitoring()

    except Exception as e:
        print(f"System health monitoring initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
