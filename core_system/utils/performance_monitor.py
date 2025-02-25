import json
import logging
import os
import threading
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import psutil


class PerformanceMonitor:
    """
    Advanced performance monitoring and profiling utility.
    Provides comprehensive system and function-level performance tracking.
    """

    def __init__(
        self,
        log_interval: int = 60,
        logger: Optional[logging.Logger] = None,
        log_dir: str = "/opt/sutazai_project/SutazAI/logs/performance",
    ):
        """
        Initialize advanced performance monitor.

        Args:
            log_interval (int): Interval for system metrics logging (seconds)
            logger (Optional[logging.Logger]): Custom logger
            log_dir (str): Directory for performance logs
        """
        self.logger = logger or logging.getLogger("SutazAI.Performance")
        self._log_interval = log_interval
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        # Create log directory
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def start_system_monitoring(self) -> None:
        """
        Start continuous system performance monitoring.
        Logs system metrics at specified intervals with advanced tracking.
        """

        def monitor_system():
            while not self._stop_monitoring.is_set():
                metrics = self.get_system_metrics()
                self._log_system_metrics(metrics)
                time.sleep(self._log_interval)

        self._monitoring_thread = threading.Thread(
            target=monitor_system, daemon=True, name="SutazAI-SystemMonitor"
        )
        self._monitoring_thread.start()
        self.logger.info("System performance monitoring started")

    def stop_system_monitoring(self) -> None:
        """
        Stop continuous system performance monitoring.
        """
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join()
            self.logger.info("System performance monitoring stopped")

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive system performance metrics.

        Returns:
            Dict[str, Any]: System performance metrics
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": {
                "overall": psutil.cpu_percent(interval=1),
                "per_core": psutil.cpu_percent(interval=1, percpu=True),
            },
            "memory_usage": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used,
            },
            "disk_usage": {
                "total": psutil.disk_usage("/").total,
                "used": psutil.disk_usage("/").used,
                "free": psutil.disk_usage("/").free,
                "percent": psutil.disk_usage("/").percent,
            },
            "network_io": dict(psutil.net_io_counters()._asdict()),
            "processes": self._get_top_processes(),
        }

    def _log_system_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log system metrics to a JSON file.

        Args:
            metrics (Dict[str, Any]): System performance metrics
        """
        log_file = os.path.join(
            self.log_dir,
            f"system_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(log_file, "w") as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(f"System metrics logged: {log_file}")

    def _get_top_processes(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top resource-consuming processes.

        Args:
            top_n (int): Number of top processes to return

        Returns:
            List[Dict[str, Any]]: Top processes by CPU and memory usage
        """
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
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ):
                pass

        # Sort by CPU and memory usage
        processes.sort(
            key=lambda x: x["cpu_percent"] + x["memory_percent"], reverse=True
        )
        return processes[:top_n]

    def function_profiler(
        self,
        log_level: str = "DEBUG",
        threshold_seconds: Optional[float] = None,
    ) -> Callable:
        """
        Decorator for function performance profiling with advanced tracking.

        Args:
            log_level (str): Logging level for performance metrics
            threshold_seconds (Optional[float]): Execution time threshold for logging

        Returns:
            Callable: Decorator function
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss

                result = func(*args, **kwargs)

                execution_time = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss
                memory_diff = end_memory - start_memory

                performance_data = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "execution_time_seconds": execution_time,
                    "memory_usage_bytes": memory_diff,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "timestamp": datetime.now().isoformat(),
                }

                log_method = getattr(self.logger, log_level.lower(), self.logger.debug)

                if threshold_seconds is None or execution_time > threshold_seconds:
                    log_method(f"Function Performance: {performance_data}")

                return result

            return wrapper

        return decorator

    def memory_tracker(self, func: Callable) -> Callable:
        """
        Decorator to track memory usage of a function with detailed reporting.

        Args:
            func (Callable): Function to track

        Returns:
            Callable: Wrapped function with memory tracking
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss

            result = func(*args, **kwargs)

            memory_after = process.memory_info().rss
            memory_diff = memory_after - memory_before

            memory_log = {
                "function": func.__name__,
                "module": func.__module__,
                "memory_increase_bytes": memory_diff,
                "memory_increase_mb": memory_diff / (1024 * 1024),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info(f"Memory Usage for {func.__name__}: {memory_log}")

            return result

        return wrapper
