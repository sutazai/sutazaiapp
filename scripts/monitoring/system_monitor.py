#!/usr/bin/env python3
"""

logger = logging.getLogger(__name__)
System Monitor for SutazAI
==========================

Consolidated system monitoring module that replaces 218+ individual monitoring scripts.
Provides comprehensive monitoring of system resources, services, and health status.
"""

import psutil
import time
import json
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import signal
import sys

from ..utils.common_utils import setup_logging, format_size, health_check_url
from ..utils.docker_utils import DockerManager, get_system_containers_overview
from ..utils.network_utils import validate_sutazai_services, NetworkValidator

logger = setup_logging('system_monitor')

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring"""
    refresh_interval: float = 5.0
    history_size: int = 100
    thresholds: Dict[str, float] = None
    enable_logging: bool = True
    log_file: Optional[str] = None
    enable_alerts: bool = False
    alert_thresholds: Dict[str, float] = None
    services_to_monitor: List[str] = None
    
    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0,
                'temperature': 70.0  # Celsius
            }
        
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_percent': 90.0,
                'memory_percent': 95.0,
                'disk_percent': 95.0,
                'service_down_minutes': 5.0
            }
        
        if self.services_to_monitor is None:
            self.services_to_monitor = [
                'backend', 'frontend', 'ollama', 'postgres', 
                'redis', 'neo4j_http', 'rabbitmq_management',
                'hardware_optimizer', 'ai_orchestrator'
            ]

@dataclass 
class SystemMetrics:
    """System metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory: Dict[str, Union[int, float]]
    disk: Dict[str, Union[int, float]]
    network: Dict[str, int]
    processes: Dict[str, Any]
    temperature: Optional[float] = None
    load_average: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable types"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: str
    accessible: bool
    response_time: Optional[float]
    last_check: datetime
    error_message: Optional[str] = None
    uptime_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_check'] = self.last_check.isoformat()
        return data

class MetricsCollector:
    """Collects system metrics efficiently"""
    
    def __init__(self):
        self._network_io_last = None
        self._network_io_time = None
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_data = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_data = {
            'total': disk.total,
            'free': disk.free,
            'used': disk.used,
            'percent': (disk.used / disk.total) * 100
        }
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_data = {
            'bytes_sent': network_io.bytes_sent,
            'bytes_recv': network_io.bytes_recv,
            'packets_sent': network_io.packets_sent,
            'packets_recv': network_io.packets_recv
        }
        
        # Calculate network rate if we have previous data
        if self._network_io_last and self._network_io_time:
            time_delta = (timestamp - self._network_io_time).total_seconds()
            if time_delta > 0:
                network_data['bytes_sent_rate'] = (
                    network_io.bytes_sent - self._network_io_last.bytes_sent
                ) / time_delta
                network_data['bytes_recv_rate'] = (
                    network_io.bytes_recv - self._network_io_last.bytes_recv
                ) / time_delta
        
        self._network_io_last = network_io
        self._network_io_time = timestamp
        
        # Process information
        processes_data = {
            'count': len(psutil.pids()),
            'top_cpu': [],
            'top_memory': []
        }
        
        try:
            # Get top processes by CPU and memory
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Sort by CPU usage
            top_cpu = sorted(processes, key=lambda p: p['cpu_percent'] or 0, reverse=True)[:5]
            processes_data['top_cpu'] = [
                {'name': p['name'], 'cpu_percent': p['cpu_percent']} 
                for p in top_cpu if p['cpu_percent']
            ]
            
            # Sort by memory usage
            top_memory = sorted(processes, key=lambda p: p['memory_percent'] or 0, reverse=True)[:5]
            processes_data['top_memory'] = [
                {'name': p['name'], 'memory_percent': p['memory_percent']} 
                for p in top_memory if p['memory_percent']
            ]
        except Exception as e:
            logger.warning(f"Error collecting process data: {e}")
        
        # Temperature (if available)
        temperature = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get first available temperature sensor
                for sensor_name, sensor_list in temps.items():
                    if sensor_list:
                        temperature = sensor_list[0].current
                        break
        except Exception:
            pass  # Temperature sensors not available on all systems
        
        # Load average (Unix systems)
        load_average = None
        try:
            load_average = list(psutil.getloadavg())
        except AttributeError:
            pass  # Not available on Windows
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory=memory_data,
            disk=disk_data,
            network=network_data,
            processes=processes_data,
            temperature=temperature,
            load_average=load_average
        )

class HealthChecker:
    """Health checking for services and containers"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.docker_manager = DockerManager()
        self.network_validator = NetworkValidator()
        self._service_history: Dict[str, List[ServiceHealth]] = {}
    
    def check_service_health(self, service_name: str, endpoint: str) -> ServiceHealth:
        """Check health of a single service"""
        start_time = time.time()
        
        if endpoint.startswith('http'):
            # HTTP health check
            result = self.network_validator.test_http_endpoint(endpoint)
            accessible = result['accessible']
            response_time = result.get('response_time')
            error_message = result.get('error')
            status = 'healthy' if accessible else 'unhealthy'
        else:
            # TCP port check
            host, port = endpoint.split(':')
            is_open, message = self.network_validator.check_port(host, int(port))
            accessible = is_open
            response_time = (time.time() - start_time) * 1000  # ms
            error_message = None if is_open else message
            status = 'healthy' if is_open else 'unhealthy'
        
        health = ServiceHealth(
            name=service_name,
            status=status,
            accessible=accessible,
            response_time=response_time,
            last_check=datetime.now(),
            error_message=error_message
        )
        
        # Update history
        if service_name not in self._service_history:
            self._service_history[service_name] = []
        
        self._service_history[service_name].append(health)
        
        # Keep only recent history
        if len(self._service_history[service_name]) > self.config.history_size:
            self._service_history[service_name] = self._service_history[service_name][-self.config.history_size:]
        
        # Calculate uptime percentage
        if len(self._service_history[service_name]) > 1:
            recent_checks = self._service_history[service_name][-50:]  # Last 50 checks
            successful_checks = sum(1 for h in recent_checks if h.accessible)
            health.uptime_percentage = (successful_checks / len(recent_checks)) * 100
        
        return health
    
    def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all configured services"""
        service_endpoints = {
            'backend': 'http://localhost:10010/health',
            'frontend': 'http://localhost:10011/',
            'ollama': 'http://localhost:10104/api/tags',
            'postgres': 'localhost:10000',
            'redis': 'localhost:10001',
            'neo4j_http': 'http://localhost:10002',
            'neo4j_bolt': 'localhost:10003',
            'rabbitmq_management': 'http://localhost:10008',
            'grafana': 'http://localhost:10201',
            'prometheus': 'http://localhost:10200',
            'hardware_optimizer': 'http://localhost:11110/health',
            'ai_orchestrator': 'http://localhost:8589/health'
        }
        
        results = {}
        
        # Filter to only monitor configured services
        services_to_check = {
            name: endpoint for name, endpoint in service_endpoints.items()
            if name in self.config.services_to_monitor
        }
        
        # Use thread pool for concurrent health checks
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_service = {
                executor.submit(self.check_service_health, name, endpoint): name
                for name, endpoint in services_to_check.items()
            }
            
            for future in as_completed(future_to_service):
                service_name = future_to_service[future]
                try:
                    results[service_name] = future.result()
                except Exception as e:
                    logger.error(f"Error checking {service_name}: {e}")
                    results[service_name] = ServiceHealth(
                        name=service_name,
                        status='error',
                        accessible=False,
                        response_time=None,
                        last_check=datetime.now(),
                        error_message=str(e)
                    )
        
        return results
    
    def get_service_uptime_report(self) -> Dict[str, Dict[str, Any]]:
        """Generate uptime report for all services"""
        report = {}
        
        for service_name, history in self._service_history.items():
            if not history:
                continue
            
            total_checks = len(history)
            successful_checks = sum(1 for h in history if h.accessible)
            uptime_percentage = (successful_checks / total_checks) * 100 if total_checks > 0 else 0
            
            # Calculate average response time
            response_times = [h.response_time for h in history if h.response_time is not None]
            avg_response_time = sum(response_times) / len(response_times) if response_times else None
            
            # Find last downtime
            last_downtime = None
            for health in reversed(history):
                if not health.accessible:
                    last_downtime = health.last_check
                    break
            
            report[service_name] = {
                'uptime_percentage': round(uptime_percentage, 2),
                'total_checks': total_checks,
                'successful_checks': successful_checks,
                'failed_checks': total_checks - successful_checks,
                'average_response_time': round(avg_response_time, 2) if avg_response_time else None,
                'last_downtime': last_downtime.isoformat() if last_downtime else None,
                'current_status': history[-1].status if history else 'unknown'
            }
        
        return report

class SystemMonitor:
    """Main system monitoring class"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.config)
        self.running = False
        self.monitoring_thread = None
        self._metrics_history: List[SystemMetrics] = []
        self._alerts_queue = queue.Queue()
        
        # Setup logging if enabled
        if self.config.enable_logging and self.config.log_file:
            self.logger = setup_logging(
                'system_monitor_file',
                log_file=self.config.log_file
            )
        else:
            self.logger = logger
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring in background thread"""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=self.config.refresh_interval * 2)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                metrics = self.metrics_collector.collect_system_metrics()
                self._metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self._metrics_history) > self.config.history_size:
                    self._metrics_history = self._metrics_history[-self.config.history_size:]
                
                # Check for threshold alerts
                if self.config.enable_alerts:
                    self._check_alerts(metrics)
                
                # Log if enabled
                if self.config.enable_logging:
                    self.logger.info(f"System: CPU {metrics.cpu_percent}%, "
                                   f"Memory {metrics.memory['percent']}%, "
                                   f"Disk {metrics.disk['percent']:.1f}%")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.config.refresh_interval)
    
    def _check_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.config.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent}%")
        
        # Memory alert
        if metrics.memory['percent'] > self.config.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory['percent']}%")
        
        # Disk alert
        if metrics.disk['percent'] > self.config.alert_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics.disk['percent']:.1f}%")
        
        # Temperature alert (if available)
        if metrics.temperature and metrics.temperature > 80:
            alerts.append(f"High temperature: {metrics.temperature}°C")
        
        for alert in alerts:
            self._alerts_queue.put({
                'timestamp': metrics.timestamp,
                'level': 'warning',
                'message': alert
            })
            self.logger.warning(alert)
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics"""
        return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self._metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        current_metrics = self.get_current_metrics()
        service_health = self.health_checker.check_all_services()
        container_overview = get_system_containers_overview()
        
        overview = {
            'timestamp': datetime.now().isoformat(),
            'system_status': 'healthy',  # Will be updated based on checks
            'metrics': current_metrics.to_dict() if current_metrics else None,
            'services': {name: health.to_dict() for name, health in service_health.items()},
            'containers': container_overview,
            'uptime_report': self.health_checker.get_service_uptime_report(),
            'alerts': self._get_recent_alerts()
        }
        
        # Determine overall system status
        unhealthy_services = [name for name, health in service_health.items() if not health.accessible]
        if unhealthy_services:
            overview['system_status'] = 'degraded'
            if len(unhealthy_services) > len(service_health) / 2:
                overview['system_status'] = 'critical'
        
        return overview
    
    def _get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        alerts = []
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        while not self._alerts_queue.empty():
            try:
                alert = self._alerts_queue.get_nowait()
                if alert['timestamp'] > cutoff_time:
                    alert['timestamp'] = alert['timestamp'].isoformat()
                    alerts.append(alert)
            except queue.Empty:
                break
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def export_metrics(self, file_path: str, format: str = 'json') -> bool:
        """Export metrics to file"""
        try:
            data = {
                'export_time': datetime.now().isoformat(),
                'config': asdict(self.config),
                'metrics_history': [m.to_dict() for m in self._metrics_history],
                'uptime_report': self.health_checker.get_service_uptime_report()
            }
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Metrics exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False

# Convenience functions
def create_system_monitor(
    refresh_interval: float = 5.0,
    enable_logging: bool = True,
    log_file: Optional[str] = None
) -> SystemMonitor:
    """Create a configured system monitor"""
    config = MonitoringConfig(
        refresh_interval=refresh_interval,
        enable_logging=enable_logging,
        log_file=log_file
    )
    return SystemMonitor(config)

def quick_system_check() -> Dict[str, Any]:
    """Perform a quick system health check"""
    monitor = SystemMonitor()
    
    # Get current metrics
    metrics = monitor.metrics_collector.collect_system_metrics()
    
    # Check services
    service_health = monitor.health_checker.check_all_services()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cpu_percent': metrics.cpu_percent,
        'memory_percent': metrics.memory['percent'],
        'disk_percent': metrics.disk['percent'],
        'services_healthy': sum(1 for h in service_health.values() if h.accessible),
        'services_total': len(service_health),
        'services': {name: h.status for name, h in service_health.items()}
    }

if __name__ == "__main__":
    # Test/demo the system monitor
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI System Monitor')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--interval', type=float, default=5.0, help='Refresh interval')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--quick', action='store_true', help='Quick system check')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick check mode
        result = quick_system_check()
        logger.info(json.dumps(result, indent=2))
        sys.exit(0)
    
    # Create monitor
    config = MonitoringConfig(
        refresh_interval=args.interval,
        enable_logging=True,
        log_file=args.log_file
    )
    monitor = SystemMonitor(config)
    
    if args.daemon:
        # Daemon mode
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            monitor.stop_monitoring()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        monitor.start_monitoring()
        logger.info("System monitor running as daemon. Press Ctrl+C to stop.")
        
        try:
            while monitor.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        monitor.stop_monitoring()
    else:
        # Interactive mode
        overview = monitor.get_system_overview()
        logger.info(json.dumps(overview, indent=2))