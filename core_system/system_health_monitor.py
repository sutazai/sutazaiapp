#!/usr/bin/env python3
"""
SutazAI Comprehensive System Health Monitor

Advanced autonomous system health tracking and optimization framework
providing real-time monitoring, predictive analysis, and self-healing capabilities.
"""

import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Union

import psutil
from loguru import logger

# Remove default logger and configure Loguru
logger.remove()
logger.add(
    "/opt/sutazai_project/SutazAI/logs/system_health_monitor.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

class SystemHealthMonitor:
    """
    Comprehensive autonomous system health monitoring framework

    Provides:
    - Real-time resource tracking
    - Performance bottleneck detection
    - Predictive system health assessment
    - Autonomous optimization mechanisms
    - Proactive issue detection
    - System stability maintenance
    """

    def __init__(
        self,
        monitoring_interval: int = 60,  # 1 minute
        base_dir: str = "/opt/sutazai_project/SutazAI",
        critical_cpu_threshold: float = 80.0,
        critical_memory_threshold: float = 85.0,
        critical_disk_threshold: float = 90.0,
    ):
        """
        Initialize system health monitor

        Args:
            monitoring_interval (int): Interval between health checks in seconds
            base_dir (str): Base project directory
            critical_cpu_threshold (float): Critical CPU usage threshold percentage
            critical_memory_threshold (float): Critical memory usage threshold percentage
            critical_disk_threshold (float): Critical disk usage threshold percentage
        """
        self.monitoring_interval = monitoring_interval
        self.base_dir = base_dir
        self.critical_cpu_threshold = critical_cpu_threshold
        self.critical_memory_threshold = critical_memory_threshold
        self.critical_disk_threshold = critical_disk_threshold
        self.health_metrics_lock = threading.Lock()
        self.health_metrics: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.optimization_lock = threading.Lock()

    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive system performance and resource metrics

        Returns:
            Dictionary of system metrics
        """
        try:
            # Get CPU frequencies with error handling
            cpu_freqs = []
            try:
                cpu_freq_info = psutil.cpu_freq(percpu=True)
                if cpu_freq_info:
                    cpu_freqs = [
                        getattr(freq, 'current', 0.0) 
                        for freq in cpu_freq_info
                    ]
            except Exception as e:
                logger.warning(f"Failed to get CPU frequencies: {e}")
                try:
                    # Fallback to overall CPU frequency
                    overall_freq = psutil.cpu_freq()
                    if overall_freq:
                        cpu_freqs = [getattr(overall_freq, 'current', 0.0)]
                except Exception as e:
                    logger.error(f"Failed to get overall CPU frequency: {e}")

            # Get detailed memory information
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "os": platform.platform(),
                    "python_version": platform.python_version(),
                    "machine": platform.machine(),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                },
                "cpu_metrics": {
                    "physical_cores": psutil.cpu_count(logical=False),
                    "total_cores": psutil.cpu_count(logical=True),
                    "cpu_frequencies": cpu_freqs,
                    "cpu_usage_percent": psutil.cpu_percent(interval=1, percpu=True),
                    "cpu_times": psutil.cpu_times()._asdict(),
                    "load_average": os.getloadavg(),
                },
                "memory_metrics": {
                    "total_memory": vm.total,
                    "available_memory": vm.available,
                    "used_memory": vm.used,
                    "memory_usage_percent": vm.percent,
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_free": swap.free,
                    "swap_percent": swap.percent,
                    "cached": vm.cached,
                    "buffers": vm.buffers,
                },
                "disk_metrics": self._get_disk_metrics(),
                "network_metrics": self._get_network_metrics(),
                "process_metrics": self._get_process_metrics(),
                "system_temperature": self._get_system_temperature(),
            }
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def _get_disk_metrics(self) -> Dict[str, Any]:
        """Get detailed disk metrics for all mounted partitions"""
        disk_metrics = {}
        try:
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_metrics[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                        "fstype": partition.fstype,
                        "device": partition.device,
                    }
                except (PermissionError, OSError):
                    continue
            
            # Add disk I/O statistics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_metrics["io_stats"] = {
                    "read_bytes": disk_io.read_bytes,
                    "write_bytes": disk_io.write_bytes,
                    "read_count": disk_io.read_count,
                    "write_count": disk_io.write_count,
                    "read_time": disk_io.read_time,
                    "write_time": disk_io.write_time,
                }
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
        
        return disk_metrics

    def _get_network_metrics(self) -> Dict[str, Any]:
        """Get detailed network metrics for all interfaces"""
        network_metrics = {}
        try:
            # Get network interface statistics
            net_io = psutil.net_io_counters(pernic=True)
            net_if_addrs = psutil.net_if_addrs()
            net_if_stats = psutil.net_if_stats()

            for interface, stats in net_io.items():
                network_metrics[interface] = {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errin": stats.errin,
                    "errout": stats.errout,
                    "dropin": stats.dropin,
                    "dropout": stats.dropout,
                }

                # Add interface addresses if available
                if interface in net_if_addrs:
                    network_metrics[interface]["addresses"] = [
                        addr._asdict() for addr in net_if_addrs[interface]
                    ]

                # Add interface status if available
                if interface in net_if_stats:
                    network_metrics[interface]["status"] = {
                        "isup": net_if_stats[interface].isup,
                        "speed": net_if_stats[interface].speed,
                        "mtu": net_if_stats[interface].mtu,
                    }
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")

        return network_metrics

    def _get_process_metrics(self) -> Dict[str, Any]:
        """Get detailed process metrics"""
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 
                                          'memory_percent', 'status', 'create_time']):
                try:
                    pinfo = proc.info
                    pinfo['create_time'] = datetime.fromtimestamp(pinfo['create_time']).isoformat()
                    processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            return {
                "total_processes": len(processes),
                "running_processes": len([p for p in processes if p['status'] == 'running']),
                "sleeping_processes": len([p for p in processes if p['status'] == 'sleeping']),
                "zombie_processes": len([p for p in processes if p['status'] == 'zombie']),
                "top_cpu_processes": sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:5],
                "top_memory_processes": sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:5],
            }
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {}

    def _get_system_temperature(self) -> Dict[str, Any]:
        """Get system temperature information if available"""
        try:
            temperatures = {}
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        temperatures[name] = [entry._asdict() for entry in entries]
            return temperatures
        except Exception as e:
            logger.error(f"Error collecting temperature metrics: {e}")
            return {}

    def analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze system performance and identify potential bottlenecks

        Args:
            metrics (Dict): Collected system metrics

        Returns:
            List of performance issues with severity and recommendations
        """
        issues = []
        try:
            # CPU Analysis
            cpu_usage = metrics.get("cpu_metrics", {}).get("cpu_usage_percent", [])
            if cpu_usage:
                max_cpu = max(cpu_usage)
                if max_cpu > self.critical_cpu_threshold:
                    issues.append({
                        "component": "CPU",
                        "severity": "CRITICAL",
                        "message": f"Critical CPU usage: {max_cpu}%",
                        "recommendation": "Identify and optimize CPU-intensive processes",
                        "metrics": {
                            "current_usage": max_cpu,
                            "threshold": self.critical_cpu_threshold
                        }
                    })
                elif max_cpu > 70:
                    issues.append({
                        "component": "CPU",
                        "severity": "WARNING",
                        "message": f"High CPU usage: {max_cpu}%",
                        "recommendation": "Monitor CPU-intensive processes",
                        "metrics": {
                            "current_usage": max_cpu,
                            "threshold": 70
                        }
                    })

            # Memory Analysis
            memory_metrics = metrics.get("memory_metrics", {})
            memory_usage = memory_metrics.get("memory_usage_percent", 0)
            if memory_usage > self.critical_memory_threshold:
                issues.append({
                    "component": "Memory",
                    "severity": "CRITICAL",
                    "message": f"Critical memory usage: {memory_usage}%",
                    "recommendation": "Implement memory optimization or increase capacity",
                    "metrics": {
                        "current_usage": memory_usage,
                        "threshold": self.critical_memory_threshold,
                        "available": memory_metrics.get("available_memory", 0)
                    }
                })
            elif memory_usage > 75:
                issues.append({
                    "component": "Memory",
                    "severity": "WARNING",
                    "message": f"High memory usage: {memory_usage}%",
                    "recommendation": "Monitor memory usage trends",
                    "metrics": {
                        "current_usage": memory_usage,
                        "threshold": 75
                    }
                })

            # Disk Analysis
            disk_metrics = metrics.get("disk_metrics", {})
            for mount_point, disk_data in disk_metrics.items():
                if isinstance(disk_data, dict) and "percent" in disk_data:
                    disk_usage = disk_data["percent"]
                    if disk_usage > self.critical_disk_threshold:
                        issues.append({
                            "component": "Disk",
                            "severity": "CRITICAL",
                            "message": f"Critical disk usage on {mount_point}: {disk_usage}%",
                            "recommendation": "Clean up disk space or expand storage",
                            "metrics": {
                                "mount_point": mount_point,
                                "current_usage": disk_usage,
                                "threshold": self.critical_disk_threshold
                            }
                        })
                    elif disk_usage > 80:
                        issues.append({
                            "component": "Disk",
                            "severity": "WARNING",
                            "message": f"High disk usage on {mount_point}: {disk_usage}%",
                            "recommendation": "Monitor disk usage trends",
                            "metrics": {
                                "mount_point": mount_point,
                                "current_usage": disk_usage,
                                "threshold": 80
                            }
                        })

            # Process Analysis
            process_metrics = metrics.get("process_metrics", {})
            zombie_count = process_metrics.get("zombie_processes", 0)
            if zombie_count > 0:
                issues.append({
                    "component": "Processes",
                    "severity": "WARNING",
                    "message": f"Detected {zombie_count} zombie processes",
                    "recommendation": "Clean up zombie processes",
                    "metrics": {
                        "zombie_count": zombie_count
                    }
                })

            # Network Analysis
            network_metrics = metrics.get("network_metrics", {})
            for interface, data in network_metrics.items():
                if isinstance(data, dict) and "errin" in data and "errout" in data:
                    total_errors = data["errin"] + data["errout"]
                    if total_errors > 1000:
                        issues.append({
                            "component": "Network",
                            "severity": "WARNING",
                            "message": f"High network errors on {interface}",
                            "recommendation": "Investigate network connectivity issues",
                            "metrics": {
                                "interface": interface,
                                "errors_in": data["errin"],
                                "errors_out": data["errout"]
                            }
                        })

        except Exception as e:
            logger.error(f"Error analyzing performance bottlenecks: {e}")

        return issues

    def autonomous_optimization(self, issues: List[Dict[str, Any]]):
        """
        Apply autonomous system optimizations based on identified issues

        Args:
            issues (List[Dict[str, Any]]): List of identified performance issues
        """
        with self.optimization_lock:
            try:
                for issue in issues:
                    severity = issue.get("severity", "")
                    component = issue.get("component", "")
                    
                    logger.info(f"Attempting optimization for {severity} {component} issue")

                    if component == "CPU" and severity == "CRITICAL":
                        self._optimize_cpu_usage(issue)
                    elif component == "Memory" and severity == "CRITICAL":
                        self._optimize_memory_usage(issue)
                    elif component == "Disk" and severity == "CRITICAL":
                        self._optimize_disk_space(issue)
                    elif component == "Processes" and "zombie" in issue.get("message", "").lower():
                        self._optimize_process_management(issue)
                    elif component == "Network":
                        self._optimize_network(issue)

            except Exception as e:
                logger.error(f"Autonomous optimization failed: {e}")

    def _optimize_cpu_usage(self, issue: Dict[str, Any]):
        """Optimize CPU usage by managing resource-intensive processes"""
        try:
            # Get top CPU-consuming processes
            high_cpu_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'username'])
                if p.info['cpu_percent'] > 70
            ]

            for process in high_cpu_processes:
                try:
                    # Log high CPU process
                    logger.warning(
                        f"High CPU process: {process.info['name']} "
                        f"(PID: {process.info['pid']}, "
                        f"CPU: {process.info['cpu_percent']}%, "
                        f"User: {process.info['username']})"
                    )

                    # Reduce process priority if not a critical system process
                    if process.info['username'] != 'root':
                        try:
                            process.nice(10)  # Lower priority
                            logger.info(f"Reduced priority for PID {process.info['pid']}")
                        except psutil.AccessDenied:
                            logger.warning(f"Cannot modify priority for PID {process.info['pid']}")

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"CPU optimization failed: {e}")

    def _optimize_memory_usage(self, issue: Dict[str, Any]):
        """Optimize memory usage through intelligent memory management"""
        try:
            # Clear system caches
            self.safe_run_command("sync")
            self.safe_run_command("echo 3 > /proc/sys/vm/drop_caches")

            # Optimize swap usage
            self.safe_run_command("swapoff -a && swapon -a")

            # Find memory-hungry processes
            high_memory_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'memory_percent'])
                if p.info['memory_percent'] > 10
            ]

            for process in high_memory_processes:
                try:
                    logger.warning(
                        f"High memory process: {process.info['name']} "
                        f"(PID: {process.info['pid']}, "
                        f"Memory: {process.info['memory_percent']}%)"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def _optimize_disk_space(self, issue: Dict[str, Any]):
        """Optimize disk space through intelligent cleanup"""
        try:
            mount_point = issue.get("metrics", {}).get("mount_point", "/")
            
            # Clean old log files
            log_paths = [
                "/var/log",
                os.path.join(self.base_dir, "logs"),
            ]

            for log_path in log_paths:
                if os.path.exists(log_path):
                    for root, _, files in os.walk(log_path):
                        for file in files:
                            if file.endswith(('.log', '.gz', '.zip')):
                                file_path = os.path.join(root, file)
                                try:
                                    if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
                                        os.remove(file_path)
                                        logger.info(f"Removed large log file: {file_path}")
                                except OSError as e:
                                    logger.error(f"Error removing log file {file_path}: {e}")

            # Clean temporary files
            temp_paths = ["/tmp", "/var/tmp"]
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    self.safe_run_command(f"find {temp_path} -type f -atime +7 -delete")

            # Clean package caches if on a supported system
            if os.path.exists("/var/cache/apt"):
                self.safe_run_command("apt-get clean")
            elif os.path.exists("/var/cache/pacman"):
                self.safe_run_command("pacman -Sc --noconfirm")

        except Exception as e:
            logger.error(f"Disk space optimization failed: {e}")

    def _optimize_process_management(self, issue: Dict[str, Any]):
        """Optimize process management by handling problematic processes"""
        try:
            # Handle zombie processes
            zombie_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'status'])
                if p.info['status'] == psutil.STATUS_ZOMBIE
            ]

            for process in zombie_processes:
                try:
                    # Try to terminate zombie process
                    parent = psutil.Process(process.ppid())
                    logger.warning(
                        f"Terminating zombie process: {process.info['name']} "
                        f"(PID: {process.info['pid']}, "
                        f"PPID: {process.ppid()})"
                    )
                    parent.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Check for and handle hung processes
            hung_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'status', 'cpu_times'])
                if p.info['status'] == psutil.STATUS_DISK_SLEEP
            ]

            for process in hung_processes:
                try:
                    logger.warning(
                        f"Detected hung process: {process.info['name']} "
                        f"(PID: {process.info['pid']})"
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            logger.error(f"Process management optimization failed: {e}")

    def _optimize_network(self, issue: Dict[str, Any]):
        """Optimize network performance"""
        try:
            interface = issue.get("metrics", {}).get("interface")
            if interface:
                # Log network interface status
                logger.info(f"Optimizing network interface: {interface}")
                
                # Restart problematic network interface
                self.safe_run_command(f"ip link set {interface} down")
                time.sleep(1)
                self.safe_run_command(f"ip link set {interface} up")

                # Optimize network parameters
                self.safe_run_command("sysctl -w net.ipv4.tcp_fin_timeout=30")
                self.safe_run_command("sysctl -w net.ipv4.tcp_keepalive_time=1200")
                self.safe_run_command("sysctl -w net.ipv4.tcp_max_syn_backlog=2048")

        except Exception as e:
            logger.error(f"Network optimization failed: {e}")

    def monitor_system_health(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Collect metrics
                metrics = self.collect_system_metrics()
                
                # Store metrics with thread safety
                with self.health_metrics_lock:
                    self.health_metrics.append(metrics)
                    # Keep only last 24 hours of metrics
                    cutoff_time = datetime.now().timestamp() - (24 * 60 * 60)
                    self.health_metrics = [
                        m for m in self.health_metrics
                        if datetime.fromisoformat(m["timestamp"]).timestamp() > cutoff_time
                    ]

                # Analyze and optimize
                issues = self.analyze_performance_bottlenecks(metrics)
                if issues:
                    self.autonomous_optimization(issues)
                    
                    # Log issues
                    for issue in issues:
                        log_level = "ERROR" if issue["severity"] == "CRITICAL" else "WARNING"
                        logger.log(
                            log_level,
                            f"{issue['component']} Issue: {issue['message']} - {issue['recommendation']}"
                        )

                # Generate and save health report
                self.save_health_report()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            finally:
                time.sleep(self.monitoring_interval)

    def save_health_report(self):
        """Generate and save health report to file"""
        try:
            report = self.generate_health_report()
            report_path = os.path.join(self.base_dir, "logs", "health_report.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Save report
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Health report saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving health report: {e}")

    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            with self.health_metrics_lock:
                if not self.health_metrics:
                    return {}

                latest_metrics = self.health_metrics[-1]
                historical_analysis = self._analyze_performance_trends()

                report = {
                    "timestamp": datetime.now().isoformat(),
                    "system_status": {
                        "cpu_status": self._get_component_status(
                            latest_metrics.get("cpu_metrics", {}).get("cpu_usage_percent", [0])[0],
                            self.critical_cpu_threshold
                        ),
                        "memory_status": self._get_component_status(
                            latest_metrics.get("memory_metrics", {}).get("memory_usage_percent", 0),
                            self.critical_memory_threshold
                        ),
                        "disk_status": self._get_component_status(
                            latest_metrics.get("disk_metrics", {}).get("/", {}).get("percent", 0),
                            self.critical_disk_threshold
                        ),
                    },
                    "current_metrics": latest_metrics,
                    "historical_analysis": historical_analysis,
                    "recommendations": self.analyze_performance_bottlenecks(latest_metrics)
                }

                return report

        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            return {}

    def _get_component_status(self, current_value: float, threshold: float) -> str:
        """Determine component status based on current value and threshold"""
        if current_value >= threshold:
            return "CRITICAL"
        elif current_value >= threshold * 0.8:
            return "WARNING"
        return "HEALTHY"

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze historical performance trends"""
        try:
            with self.health_metrics_lock:
                if not self.health_metrics:
                    return {}

                # Calculate trends
                cpu_trend = self._calculate_trend([
                    max(m.get("cpu_metrics", {}).get("cpu_usage_percent", [0]))
                    for m in self.health_metrics
                ])

                memory_trend = self._calculate_trend([
                    m.get("memory_metrics", {}).get("memory_usage_percent", 0)
                    for m in self.health_metrics
                ])

                disk_trend = self._calculate_trend([
                    m.get("disk_metrics", {}).get("/", {}).get("percent", 0)
                    for m in self.health_metrics
                ])

                return {
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend,
                    "disk_trend": disk_trend,
                    "analysis_period": {
                        "start": self.health_metrics[0]["timestamp"],
                        "end": self.health_metrics[-1]["timestamp"]
                    }
                }

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if not values or len(values) < 2:
            return "STABLE"

        # Calculate simple linear regression
        x = list(range(len(values)))
        slope = (
            (len(values) * sum(i * j for i, j in zip(x, values)) - sum(x) * sum(values))
            / (len(values) * sum(i * i for i in x) - sum(x) ** 2)
        )

        if slope > 0.1:
            return "INCREASING"
        elif slope < -0.1:
            return "DECREASING"
        return "STABLE"

    def start_monitoring(self):
        """Start the monitoring thread"""
        logger.info("Starting system health monitoring")
        self.stop_event.clear()
        threading.Thread(target=self.monitor_system_health, daemon=True).start()

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        logger.info("Stopping system health monitoring")
        self.stop_event.set()

    def safe_run_command(
        self,
        command: Union[str, List[str]],
        capture_output: bool = True,
        text: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Safely execute a system command

        Args:
            command: Command to execute (string or list of strings)
            capture_output: Whether to capture command output
            text: Whether to return string output

        Returns:
            CompletedProcess instance with command result
        """
        try:
            # Convert string command to list if necessary
            cmd_list: List[str] = (
                shlex.split(command) if isinstance(command, str)
                else command
            )
            
            result = subprocess.run(
                cmd_list,
                capture_output=capture_output,
                text=text,
                check=True
            )
            
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error running command {command}: {e}")
            raise


def main():
    """Main entry point"""
    try:
        monitor = SystemHealthMonitor()
        monitor.start_monitoring()

        # Keep the main thread running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down system health monitor")
        monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"System health monitor failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
