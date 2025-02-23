#!/usr/bin/env python3
"""
SutazAI Comprehensive System Health Monitor

Advanced autonomous system health tracking and optimization framework
providing real-time monitoring, predictive analysis, and self-healing capabilities.
"""

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/system_health_monitor.log'
)
logger = logging.getLogger('SutazAI.SystemHealthMonitor')

class SystemHealthMonitor:
    """
    Comprehensive autonomous system health monitoring framework
    
    Provides:
    - Real-time resource tracking
    - Performance bottleneck detection
    - Predictive system health assessment
    - Autonomous optimization mechanisms
    """
    
    def __init__(
        self, 
        monitoring_interval: int = 300,  # 5 minutes
        base_dir: str = '/opt/sutazai_project/SutazAI'
    ):
        """
        Initialize system health monitor
        
        Args:
            monitoring_interval (int): Interval between health checks
            base_dir (str): Base project directory
        """
        self.monitoring_interval = monitoring_interval
        self.base_dir = base_dir
        self.health_metrics_lock = threading.Lock()
        self.health_metrics: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect comprehensive system performance and resource metrics
        
        Returns:
            Dictionary of system metrics
        """
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'system_info': {
                    'os': platform.platform(),
                    'python_version': platform.python_version(),
                    'machine': platform.machine()
                },
                'cpu_metrics': {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'total_cores': psutil.cpu_count(logical=True),
                    'cpu_frequencies': [freq.current for freq in psutil.cpu_freq(percpu=True)],
                    'cpu_usage_percent': psutil.cpu_percent(interval=1, percpu=True)
                },
                'memory_metrics': {
                    'total_memory': psutil.virtual_memory().total,
                    'available_memory': psutil.virtual_memory().available,
                    'memory_usage_percent': psutil.virtual_memory().percent
                },
                'disk_metrics': {
                    'total_disk_space': psutil.disk_usage('/').total,
                    'free_disk_space': psutil.disk_usage('/').free,
                    'disk_usage_percent': psutil.disk_usage('/').percent
                },
                'network_metrics': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                },
                'process_metrics': {
                    'total_processes': len(psutil.process_iter()),
                    'running_processes': len([p for p in psutil.process_iter() if p.status() == psutil.STATUS_RUNNING])
                }
            }
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def analyze_performance_bottlenecks(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Analyze system performance and identify potential bottlenecks
        
        Args:
            metrics (Dict): Collected system metrics
        
        Returns:
            List of performance optimization recommendations
        """
        recommendations = []
        
        try:
            # CPU optimization recommendations
            if max(metrics.get('cpu_metrics', {}).get('cpu_usage_percent', [0])) > 70:
                recommendations.append(
                    f"High CPU usage detected. Consider optimizing resource-intensive processes. "
                    f"Peak CPU usage: {max(metrics['cpu_metrics']['cpu_usage_percent'])}%"
                )
            
            # Memory optimization recommendations
            if metrics.get('memory_metrics', {}).get('memory_usage_percent', 0) > 80:
                recommendations.append(
                    f"High memory usage detected. Implement memory optimization strategies. "
                    f"Memory usage: {metrics['memory_metrics']['memory_usage_percent']}%"
                )
            
            # Disk space recommendations
            if metrics.get('disk_metrics', {}).get('disk_usage_percent', 0) > 85:
                recommendations.append(
                    f"Low disk space. Consider cleaning up or expanding storage. "
                    f"Disk usage: {metrics['disk_metrics']['disk_usage_percent']}%"
                )
            
            # Process count recommendations
            if metrics.get('process_metrics', {}).get('total_processes', 0) > psutil.cpu_count() * 10:
                recommendations.append(
                    f"High number of running processes. Review and optimize process management. "
                    f"Total processes: {metrics['process_metrics']['total_processes']}"
                )
        
        except Exception as e:
            logger.error(f"Error analyzing performance bottlenecks: {e}")
        
        return recommendations
    
    def autonomous_optimization(self, recommendations: List[str]):
        """
        Apply autonomous system optimizations based on recommendations
        
        Args:
            recommendations (List[str]): Performance optimization recommendations
        """
        try:
            for recommendation in recommendations:
                logger.info(f"Applying optimization: {recommendation}")
                
                # Example optimization strategies
                if "CPU usage" in recommendation:
                    self._optimize_cpu_usage()
                
                if "memory usage" in recommendation:
                    self._optimize_memory_usage()
                
                if "disk space" in recommendation:
                    self._optimize_disk_space()
                
                if "running processes" in recommendation:
                    self._optimize_process_management()
        
        except Exception as e:
            logger.error(f"Autonomous optimization failed: {e}")
    
    def _optimize_cpu_usage(self):
        """Optimize CPU usage by managing resource-intensive processes"""
        try:
            # Identify and potentially terminate high CPU usage processes
            high_cpu_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'cpu_percent']) 
                if p.info['cpu_percent'] > 70
            ]
            
            for process in high_cpu_processes:
                logger.warning(f"High CPU process: {process.info['name']} (PID: {process.info['pid']})")
                # Optionally terminate or reduce process priority
        except Exception as e:
            logger.error(f"CPU optimization failed: {e}")
    
    def _optimize_memory_usage(self):
        """Optimize memory usage through intelligent memory management"""
        try:
            # Implement memory-freeing strategies
            subprocess.run(['sync'], check=True)  # Sync filesystem buffers
            subprocess.run(['echo', '3', '>', '/proc/sys/vm/drop_caches'], shell=True, check=True)
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
    
    def _optimize_disk_space(self):
        """Optimize disk space by cleaning temporary and log files"""
        try:
            # Remove old log files
            log_dir = os.path.join(self.base_dir, 'logs')
            for filename in os.listdir(log_dir):
                file_path = os.path.join(log_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if (datetime.now() - file_modified).days > 30:
                            os.unlink(file_path)
                            logger.info(f"Removed old log file: {filename}")
                except Exception as e:
                    logger.error(f"Error removing log file {filename}: {e}")
        except Exception as e:
            logger.error(f"Disk space optimization failed: {e}")
    
    def _optimize_process_management(self):
        """Optimize process management by terminating zombie or unnecessary processes"""
        try:
            # Identify and terminate zombie processes
            zombie_processes = [
                p for p in psutil.process_iter(['pid', 'name', 'status']) 
                if p.info['status'] == psutil.STATUS_ZOMBIE
            ]
            
            for process in zombie_processes:
                try:
                    process.terminate()
                    logger.warning(f"Terminated zombie process: {process.info['name']} (PID: {process.info['pid']})")
                except Exception as e:
                    logger.error(f"Failed to terminate zombie process: {e}")
        except Exception as e:
            logger.error(f"Process management optimization failed: {e}")
    
    def monitor_system_health(self):
        """
        Continuously monitor system health and apply autonomous optimizations
        """
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self.collect_system_metrics()
                
                # Analyze performance bottlenecks
                recommendations = self.analyze_performance_bottlenecks(metrics)
                
                # Apply autonomous optimizations
                if recommendations:
                    self.autonomous_optimization(recommendations)
                
                # Store health metrics
                with self.health_metrics_lock:
                    self.health_metrics.append(metrics)
                    
                    # Limit historical metrics to prevent excessive memory usage
                    if len(self.health_metrics) > 100:
                        self.health_metrics.pop(0)
                
                # Wait for next monitoring interval
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def generate_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive system health report
        
        Returns:
            Detailed system health report
        """
        try:
            with self.health_metrics_lock:
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'total_metrics_collected': len(self.health_metrics),
                    'latest_metrics': self.health_metrics[-1] if self.health_metrics else {},
                    'performance_trends': self._analyze_performance_trends()
                }
            
            # Persist report
            report_path = os.path.join(
                self.base_dir, 
                f'logs/system_health_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"System health report generated: {report_path}")
            return report
        
        except Exception as e:
            logger.error(f"Health report generation failed: {e}")
            return {}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Analyze performance trends from historical metrics
        
        Returns:
            Performance trend analysis
        """
        trends = {}
        
        try:
            with self.health_metrics_lock:
                if len(self.health_metrics) > 1:
                    # CPU usage trend
                    cpu_usage_trend = [
                        max(metrics.get('cpu_metrics', {}).get('cpu_usage_percent', [0])) 
                        for metrics in self.health_metrics
                    ]
                    trends['cpu_usage_trend'] = {
                        'average': sum(cpu_usage_trend) / len(cpu_usage_trend),
                        'max': max(cpu_usage_trend),
                        'min': min(cpu_usage_trend)
                    }
                    
                    # Memory usage trend
                    memory_usage_trend = [
                        metrics.get('memory_metrics', {}).get('memory_usage_percent', 0) 
                        for metrics in self.health_metrics
                    ]
                    trends['memory_usage_trend'] = {
                        'average': sum(memory_usage_trend) / len(memory_usage_trend),
                        'max': max(memory_usage_trend),
                        'min': min(memory_usage_trend)
                    }
        
        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
        
        return trends
    
    def start_monitoring(self):
        """
        Start system health monitoring in a separate thread
        """
        monitoring_thread = threading.Thread(target=self.monitor_system_health)
        monitoring_thread.daemon = True
        monitoring_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """
        Stop system health monitoring
        """
        self.stop_event.set()
        logger.info("System health monitoring stopped")

def main():
    """
    Main execution for system health monitoring
    """
    health_monitor = SystemHealthMonitor()
    
    try:
        # Start monitoring
        health_monitor.start_monitoring()
        
        # Keep main thread alive
        while True:
            time.sleep(3600)  # Generate report every hour
            health_monitor.generate_health_report()
    
    except KeyboardInterrupt:
        health_monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"System health monitoring failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 