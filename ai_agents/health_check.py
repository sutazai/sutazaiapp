"""
Agent Health Check Module

This module provides comprehensive health check capabilities for AI agents.
It includes system resource monitoring, dependency checking, and service validation.
"""

import sys
import time
import socket
import logging
import platform
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import psutil


logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status constants."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheck:
    """Health check system for agent monitoring."""

    def __init__(self, check_interval: int = 60):
        """
        Initialize health check system.

        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.checks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 70.0, "critical": 90.0},
            "disk_percent": {"warning": 80.0, "critical": 95.0},
            "response_time": {"warning": 2.0, "critical": 5.0},
        }
        self._stop_monitoring = False
        self._monitor_thread = None
        self._register_default_checks()

    def start(self) -> None:
        """Start the health check monitoring."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("Health check monitoring started")

    def stop(self) -> None:
        """Stop the health check monitoring."""
        self._stop_monitoring = True
        if self._monitor_thread is not None:
            self._monitor_thread.join()
            self._monitor_thread = None
        logger.info("Health check monitoring stopped")

    def register_check(
        self,
        check_id: str,
        check_func: Callable,
        description: str,
        check_interval: Optional[int] = None,
    ) -> None:
        """
        Register a health check function.

        Args:
            check_id: Unique identifier for the check
            check_func: Check function that returns a dict with status and details
            description: Description of the health check
            check_interval: Optional custom check interval in seconds
        """
        self.checks[check_id] = {
            "func": check_func,
            "description": description,
            "interval": check_interval or self.check_interval,
            "last_run": None,
        }
        logger.info(f"Registered health check: {check_id}")

    def unregister_check(self, check_id: str) -> None:
        """
        Unregister a health check function.

        Args:
            check_id: Check identifier to unregister
        """
        if check_id in self.checks:
            del self.checks[check_id]
            if check_id in self.results:
                del self.results[check_id]
            logger.info(f"Unregistered health check: {check_id}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get overall health status.

        Returns:
            Dict[str, Any]: Health status information
        """
        overall_status = HealthStatus.OK
        critical_checks = []
        warning_checks = []

        for check_id, result in self.results.items():
            if result.get("status") == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_checks.append(check_id)
            elif (
                result.get("status") == HealthStatus.WARNING
                and overall_status != HealthStatus.CRITICAL
            ):
                overall_status = HealthStatus.WARNING
                warning_checks.append(check_id)

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "critical_checks": critical_checks,
            "warning_checks": warning_checks,
            "checks": self.results,
        }

    def _monitor(self) -> None:
        """Run continuous health check monitoring."""
        while not self._stop_monitoring:
            # Iterate over a copy to prevent RuntimeError if checks are modified concurrently
            for check_id, check_info in self.checks.copy().items():
                # Check if it's time to run this check
                if (
                    check_info["last_run"] is None
                    or (datetime.utcnow() - check_info["last_run"]).total_seconds()
                    >= check_info["interval"]
                ):
                    try:
                        result = check_info["func"]()
                        self.results[check_id] = {
                            "status": result.get("status", HealthStatus.UNKNOWN),
                            "message": result.get("message", ""),
                            "details": result.get("details", {}),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        check_info["last_run"] = datetime.utcnow()
                    except Exception as e:
                        logger.error(f"Error running health check {check_id}: {str(e)}")
                        self.results[check_id] = {
                            "status": HealthStatus.CRITICAL,
                            "message": f"Check failed: {str(e)}",
                            "details": {},
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        check_info["last_run"] = datetime.utcnow()

            # Sleep for a short interval to avoid high CPU usage
            time.sleep(1)

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check(
            "system_resources",
            self._check_system_resources,
            "Monitor system CPU, memory, and disk usage",
        )

        self.register_check(
            "python_runtime",
            self._check_python_runtime,
            "Check Python runtime version and health",
        )

        self.register_check(
            "network_connectivity",
            self._check_network_connectivity,
            "Verify network connectivity",
        )

    def _check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resources.

        Returns:
            Dict[str, Any]: Check results
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        status = HealthStatus.OK
        message = "System resources are healthy"

        # Check CPU
        if cpu_percent >= self.thresholds["cpu_percent"]["critical"]:
            status = HealthStatus.CRITICAL
            message = f"CPU usage critical: {cpu_percent}%"
        elif (
            cpu_percent >= self.thresholds["cpu_percent"]["warning"]
            and status != HealthStatus.CRITICAL
        ):
            status = HealthStatus.WARNING
            message = f"CPU usage high: {cpu_percent}%"

        # Check memory
        if memory.percent >= self.thresholds["memory_percent"]["critical"]:
            status = HealthStatus.CRITICAL
            message = f"Memory usage critical: {memory.percent}%"
        elif (
            memory.percent >= self.thresholds["memory_percent"]["warning"]
            and status != HealthStatus.CRITICAL
        ):
            status = HealthStatus.WARNING
            message = f"Memory usage high: {memory.percent}%"

        # Check disk
        if disk.percent >= self.thresholds["disk_percent"]["critical"]:
            status = HealthStatus.CRITICAL
            message = f"Disk usage critical: {disk.percent}%"
        elif (
            disk.percent >= self.thresholds["disk_percent"]["warning"]
            and status != HealthStatus.CRITICAL
        ):
            status = HealthStatus.WARNING
            message = f"Disk usage high: {disk.percent}%"

        return {
            "status": status,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            },
        }

    def _check_python_runtime(self) -> Dict[str, Any]:
        """
        Check Python runtime.

        Returns:
            Dict[str, Any]: Check results
        """
        python_version = sys.version

        # Refined GC stats check
        gc_stats = {}
        try:
            import gc

            gc_stats["objects_count"] = len(gc.get_objects())
            # Check if get_thresholds exists before calling
            if hasattr(gc, 'get_thresholds'):
                gc_stats["thresholds"] = gc.get_thresholds()
            else:
                gc_stats["thresholds"] = "Not Available"
            gc_stats["garbage_count"] = len(gc.garbage)
        except ImportError:
            gc_stats = {"error": "GC module not available"}
        except Exception as e:
            logger.error(f"Error getting GC stats: {e}")
            gc_stats = {"error": f"Error getting GC stats: {e}"}

        return {
            "status": HealthStatus.OK,
            "message": "Python runtime is healthy",
            "details": {
                "python_version": python_version,
                "platform": platform.platform(),
                "gc_stats": gc_stats,
            },
        }

    def _check_network_connectivity(self) -> Dict[str, Any]:
        """
        Check network connectivity.

        Returns:
            Dict[str, Any]: Check results
        """
        hosts = ["8.8.8.8", "1.1.1.1"]  # Google DNS and Cloudflare DNS

        results = {}
        successful = 0

        for host in hosts:
            try:
                start_time = time.time()
                socket.create_connection((host, 53), timeout=3)
                response_time = time.time() - start_time
                results[host] = {"status": "reachable", "response_time": response_time}
                successful += 1
            except Exception as e:
                results[host] = {"status": "unreachable", "error": str(e)}

        if successful == 0:
            status = HealthStatus.CRITICAL
            message = "Network connectivity issue: No hosts reachable"
        elif successful < len(hosts):
            status = HealthStatus.WARNING
            message = f"Network connectivity partially available: {successful}/{len(hosts)} hosts reachable"
        else:
            status = HealthStatus.OK
            message = "Network connectivity available"

        return {"status": status, "message": message, "details": results}

    def set_threshold(self, check_type: str, warning: float, critical: float) -> None:
        """
        Set custom thresholds for health checks.

        Args:
            check_type: Type of check (cpu_percent, memory_percent, etc.)
            warning: Warning threshold value
            critical: Critical threshold value
        """
        if check_type in self.thresholds:
            self.thresholds[check_type] = {"warning": warning, "critical": critical}
            logger.info(
                f"Updated thresholds for {check_type}: warning={warning}, critical={critical}"
            )
        else:
            logger.warning(f"Unknown check type: {check_type}")

    def check_agent_health(self, agent_id: str, agent: Any) -> Dict[str, Any]:
        """
        Check health of a specific agent.

        Args:
            agent_id: Agent identifier
            agent: Agent instance

        Returns:
            Dict[str, Any]: Health check results
        """
        status = HealthStatus.OK
        message = f"Agent {agent_id} is healthy"

        # Check if agent is initialized
        if not agent.is_initialized():
            status = HealthStatus.WARNING
            message = f"Agent {agent_id} is not initialized"

        # Check capabilities
        if not agent.get_capabilities():
            status = HealthStatus.WARNING
            message = f"Agent {agent_id} has no capabilities"

        # Check available memory for agent operation
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            status = HealthStatus.CRITICAL
            message = f"Agent {agent_id} may not function correctly: Low system memory"

        return {
            "status": status,
            "message": message,
            "details": {
                "agent_id": agent_id,
                "initialized": agent.is_initialized(),
                "running": agent.is_running(),
                "capabilities": agent.get_capabilities(),
                "model_info": agent.get_model_info(),
            },
        }

    def register_agent_checks(self, agents: Dict[str, Any]) -> None:
        """
        Register health checks for all agents.

        Args:
            agents: Dictionary of agent instances
        """
        for agent_id, agent in agents.items():
            check_id = f"agent_{agent_id}"

            # Create a closure to capture the current agent_id and agent
            def create_check_func(aid, a):
                return lambda: self.check_agent_health(aid, a)

            check_func = create_check_func(agent_id, agent)

            self.register_check(
                check_id,
                check_func,
                f"Health check for agent {agent_id}",
                30,  # Check agent health every 30 seconds
            )
