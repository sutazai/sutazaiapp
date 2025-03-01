#!/usr/bin/env python3
"""
Lightweight Monitoring System for SutazAI
"""

import logging
import os
import signal
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import psutil
import yaml

# Configure logging
log_dir = Path("/opt/sutazaiapp/logs/autonomous_monitor")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "monitor.log"

logger = logging.getLogger("SutazAIMonitor")
logger.setLevel(logging.INFO)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Add file handler for persistent logs
file_handler = RotatingFileHandler(
log_file,
maxBytes=5 * 1024 * 1024,  # 5MB
backupCount=3,
)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

def load_config() -> dict:
    """Load monitoring configuration."""
    default_config = {
    "monitoring_interval": 300,
    "performance_thresholds": {
    "cpu_max": 50.0,
    "memory_max": 70.0,
    "process_cpu_max": 80.0,
    "max_process_duration": 3600,
    },
    "logging_config": {
    "level": "INFO",
    "max_bytes": 5242880,
    "backup_count": 3,
    },
    "critical_scripts": [],
    "optimization": {
    "enabled": False,
    "max_concurrent_tasks": 2,
    },
    }

    try:
        config_path = Path("/opt/sutazaiapp/config/monitor_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
            user_config = yaml.safe_load(f)
            # Deep merge user config with defaults
            if user_config:
                for key, value in user_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                    else:
                        default_config[key] = value
                        return default_config
                    except Exception as e:
                        logger.error(f"Failed to load config: {e}")
                        return default_config

                    def check_system_resources():
                        """Check system resource usage."""
                        try:
                            cpu_percent = psutil.cpu_percent(interval=1)
                            memory = psutil.virtual_memory()
                            disk = psutil.disk_usage("/")

                            logger.info(
                            "System Status - CPU: %.1f%%, Memory: %.1f%% (Used: %.1f GB/Total: %.1f GB), Disk: %.1f%%",
                            cpu_percent,
                            memory.percent,
                            memory.used / (1024**3),
                            memory.total / (1024**3),
                            disk.percent,
                            )

                            return cpu_percent, memory.percent, disk.percent
                        except Exception as e:
                            logger.error(f"Error checking resources: {e}")
                            return 0, 0, 0

                        def check_problematic_processes(config):
                            """Check for and handle problematic processes."""
                            try:
                                thresholds = config.get("performance_thresholds", {})
                                process_cpu_max = thresholds.get("process_cpu_max", 80.0)
                                max_process_duration = thresholds.get("max_process_duration", 3600)
                                current_time = time.time()

                                for proc in psutil.process_iter(["pid", "name", "cpu_percent", "create_time", "cmdline"]):
                                try:
                                    # Skip our own process and system processes
                                    if proc.pid == os.getpid() or proc.pid <= 4:
                                        continue

                                    # Calculate process runtime
                                    runtime = current_time - proc.create_time()
                                    cpu_percent = proc.cpu_percent(interval=0.1)

                                    # Check for runaway bash processes
                                    if (proc.name() == "bash" and
                                        runtime > max_process_duration and
                                        cpu_percent > process_cpu_max):
                                        logger.warning(
                                        f"Found runaway bash process - PID: {proc.pid}, "
                                        f"CPU: {cpu_percent:.1f}%, Runtime: {runtime/3600:.1f}h",
                                        )
                                        try:
                                            os.kill(proc.pid, signal.SIGTERM)
                                            logger.info(f"Terminated runaway process {proc.pid}")
                                            except Exception as e:
                                                logger.error(f"Failed to terminate process {proc.pid}: {e}")

                                                # Check for any process using excessive CPU
                                                elif cpu_percent > process_cpu_max:
                                                    logger.warning(
                                                    f"High CPU process - PID: {proc.pid}, Name: {proc.name()}, "
                                                    f"CPU: {cpu_percent:.1f}%",
                                                    )

                                                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                                                        continue

                                                    except Exception as e:
                                                        logger.error(f"Error checking processes: {e}")

                                                        def main():
                                                            """Main monitoring loop."""
                                                            logger.info("Starting lightweight monitoring system")
                                                            config = load_config()

                                                            interval = config.get("monitoring_interval", 300)
                                                            thresholds = config.get("performance_thresholds", {})
                                                            cpu_max = thresholds.get("cpu_max", 50.0)
                                                            memory_max = thresholds.get("memory_max", 70.0)

                                                            logger.info(
                                                            "Configured with: interval=%ds, cpu_max=%.1f%%, memory_max=%.1f%%",
                                                            interval, cpu_max, memory_max,
                                                            )

                                                            while True:
                                                            try:
                                                                cpu_percent, memory_percent, disk_percent = check_system_resources()

                                                                if cpu_percent > cpu_max:
                                                                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                                                                    check_problematic_processes(config)

                                                                    if memory_percent > memory_max:
                                                                        logger.warning(f"High memory usage: {memory_percent:.1f}%")

                                                                        time.sleep(interval)

                                                                        except KeyboardInterrupt:
                                                                            logger.info("Monitoring stopped by user")
                                                                            break
                                                                        except Exception as e:
                                                                            logger.error(f"Monitoring error: {e}")
                                                                            time.sleep(interval)

                                                                            if __name__ == "__main__":
                                                                                main()
