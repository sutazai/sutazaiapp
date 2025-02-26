#!/usr/bin/env python3
"""
SutazAI Advanced Performance Profiler

Provides comprehensive performance analysis,
bottleneck detection, and optimization recommendations.
"""

import cProfile
import io
import json
import logging
import os
import pstats
import sys
import threading
import time
import tracemalloc
from datetime import datetime

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    filename="/opt/sutazaiapp/logs/performance_profiler.log",
)
logger = logging.getLogger(__name__)


class AdvancedPerformanceProfiler:
    def __init__(self, project_root: str = "/opt/sutazaiapp"):
        """
        Initialize advanced performance profiler.

        Args:
            project_root (str): Root directory of the project
        """
        self.project_root = project_root
        self.profile_dir = "/opt/sutazaiapp/logs/performance_profiles"
        os.makedirs(self.profile_dir, exist_ok=True)

        # Performance metrics storage
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
        }

        # Profiling flags
        self.is_profiling = False
        self.profile_thread = None

    def start_system_monitoring(self, duration: int = 300, interval: int = 5):
        """
        Start comprehensive system performance monitoring.

        Args:
            duration (int): Total monitoring duration in seconds
            interval (int): Sampling interval in seconds
        """
        logger.info(
            f"Starting system performance monitoring for {duration} seconds"
        )

        self.is_profiling = True
        self.profile_thread = threading.Thread(
            target=self._monitoring_loop, args=(duration, interval)
        )
        self.profile_thread.start()

    def _monitoring_loop(self, duration: int, interval: int):
        """
        Continuous monitoring loop for system metrics.

        Args:
            duration (int): Total monitoring duration
            interval (int): Sampling interval
        """
        start_time = time.time()

        while self.is_profiling and time.time() - start_time < duration:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break

    def _collect_system_metrics(self):
        """
        Collect comprehensive system performance metrics.
        """
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["cpu_usage"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "percent": cpu_percent,
                }
            )

            # Memory Usage
            memory = psutil.virtual_memory()
            self.metrics["memory_usage"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                }
            )

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            self.metrics["disk_io"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes,
                }
            )

            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics["network_io"].append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                }
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def profile_python_code(self, target_script: str):
        """
        Profile a specific Python script.

        Args:
            target_script (str): Path to the Python script to profile
        """
        logger.info(f"Profiling Python script: {target_script}")

        # Memory profiling
        tracemalloc.start()

        # CPU profiling
        profiler = cProfile.Profile()

        try:
            # Run the script with profiling
            profiler.enable()
            exec(open(target_script).read())
            profiler.disable()

            # Capture memory snapshot
            snapshot = tracemalloc.take_snapshot()

            # Generate profiling report
            profile_report = io.StringIO()
            stats = pstats.Stats(profiler, stream=profile_report)
            stats.sort_stats("cumulative").print_stats(20)

            # Save memory snapshot
            top_stats = snapshot.statistics("lineno")

            # Generate comprehensive profile report
            profile_report_path = os.path.join(
                self.profile_dir,
                f'profile_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            )

            with open(profile_report_path, "w") as f:
                f.write("CPU Profiling:\n")
                f.write(profile_report.getvalue())

                f.write("\n\nMemory Profiling (Top 10 Memory Allocations):\n")
                for stat in top_stats[:10]:
                    f.write(f"{stat}\n")

            logger.info(f"Profiling report saved: {profile_report_path}")

        except Exception as e:
            logger.error(f"Script profiling failed: {e}")

        finally:
            tracemalloc.stop()

    def analyze_performance_metrics(self):
        """
        Analyze collected performance metrics and generate insights.

        Returns:
            Dict containing performance analysis insights
        """
        insights = {
            "cpu_usage": {
                "max": max(
                    metric["percent"] for metric in self.metrics["cpu_usage"]
                ),
                "avg": sum(
                    metric["percent"] for metric in self.metrics["cpu_usage"]
                )
                / len(self.metrics["cpu_usage"]),
            },
            "memory_usage": {
                "max_percent": max(
                    metric["percent"]
                    for metric in self.metrics["memory_usage"]
                ),
                "avg_percent": sum(
                    metric["percent"]
                    for metric in self.metrics["memory_usage"]
                )
                / len(self.metrics["memory_usage"]),
            },
        }

        # Save insights
        insights_path = os.path.join(
            self.profile_dir,
            f'performance_insights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )

        with open(insights_path, "w") as f:
            json.dump(insights, f, indent=4)

        logger.info(f"Performance insights saved: {insights_path}")
        return insights

    def stop_monitoring(self):
        """
        Stop system performance monitoring.
        """
        self.is_profiling = False
        if self.profile_thread:
            self.profile_thread.join()

        # Analyze and save metrics
        self.analyze_performance_metrics()


def main():
    """
    Main execution function for performance profiling.
    """
    try:
        profiler = AdvancedPerformanceProfiler()

        # Start system monitoring
        profiler.start_system_monitoring(duration=300, interval=5)

        # Profile a specific script (replace with your target script)
        target_script = os.path.join(
            profiler.project_root,
            "scripts",
            "performance_profiler.py",  # Self-profiling for demonstration
        )
        profiler.profile_python_code(target_script)

        # Stop monitoring
        profiler.stop_monitoring()

    except Exception as e:
        logger.error(f"Performance profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
