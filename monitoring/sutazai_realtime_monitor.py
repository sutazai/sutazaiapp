#!/usr/bin/env python3
"""
SutazAI Real-Time Deployment Monitor
Purpose: Comprehensive monitoring of all SutazAI services and infrastructure
Usage: python sutazai_realtime_monitor.py [--interval SECONDS] [--alert-threshold PERCENT]
Requirements: docker, psutil, requests, prometheus_client
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
import sys
import os

# Third-party imports
try:
    import docker
    import psutil
    import requests
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install docker psutil requests prometheus-client")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/realtime_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SutazAI-Monitor')

# Prometheus metrics
registry = CollectorRegistry()
metrics = {
    'service_health': Gauge('sutazai_service_health', 'Service health status (1=healthy, 0=unhealthy)', ['service'], registry=registry),
    'container_status': Gauge('sutazai_container_status', 'Container status (1=running, 0=stopped)', ['container'], registry=registry),
    'cpu_usage': Gauge('sutazai_cpu_usage_percent', 'CPU usage percentage', ['service'], registry=registry),
    'memory_usage': Gauge('sutazai_memory_usage_bytes', 'Memory usage in bytes', ['service'], registry=registry),
    'memory_percent': Gauge('sutazai_memory_usage_percent', 'Memory usage percentage', ['service'], registry=registry),
    'network_rx': Gauge('sutazai_network_rx_bytes', 'Network received bytes', ['service'], registry=registry),
    'network_tx': Gauge('sutazai_network_tx_bytes', 'Network transmitted bytes', ['service'], registry=registry),
    'disk_usage': Gauge('sutazai_disk_usage_percent', 'Disk usage percentage', ['mountpoint'], registry=registry),
    'service_response_time': Histogram('sutazai_service_response_time_seconds', 'Service response time', ['service'], registry=registry),
    'deployment_errors': Counter('sutazai_deployment_errors_total', 'Total deployment errors', ['error_type'], registry=registry),
    'startup_time': Gauge('sutazai_startup_time_seconds', 'Service startup time', ['service'], registry=registry),
}

class SutazAIMonitor:
    def __init__(self, interval: int = 10, alert_threshold: float = 80.0):
        self.interval = interval
        self.alert_threshold = alert_threshold
        self.docker_client = docker.from_env()
        self.startup_times = {}
        self.service_configs = self.load_service_configs()
        self.alerts = []
        
        # Ensure logs directory exists
        Path('/opt/sutazaiapp/logs').mkdir(exist_ok=True)
        
    def load_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load service configuration from docker-compose files"""
        configs = {}
        
        # Define expected services and their health check endpoints
        expected_services = {
            # Service Mesh
            'consul': {'port': 10006, 'health_path': '/v1/status/leader', 'type': 'service_mesh'},
            'kong': {'port': 10007, 'health_path': '/status', 'type': 'service_mesh'},
            'rabbitmq': {'port': 10042, 'health_path': '/api/whoami', 'type': 'service_mesh'},
            
            # Core Services
            'sutazai-neo4j': {'port': 10002, 'health_path': '/db/manage/server/version', 'type': 'database'},
            'sutazai-postgres': {'port': 5432, 'health_path': None, 'type': 'database'},
            'hygiene-postgres': {'port': 10020, 'health_path': None, 'type': 'database'},
            'hygiene-redis': {'port': 10021, 'health_path': None, 'type': 'cache'},
            
            # Monitoring Stack
            'sutazai-loki': {'port': 10202, 'health_path': '/ready', 'type': 'monitoring'},
            'sutazai-alertmanager': {'port': 10203, 'health_path': '/-/healthy', 'type': 'monitoring'},
            'sutazai-integration-dashboard': {'port': 10050, 'health_path': '/api/health', 'type': 'monitoring'},
            
            # Application Services
            'sutazai-faiss-vector': {'port': 10103, 'health_path': '/health', 'type': 'ai_service'},
            'hygiene-backend': {'port': 10420, 'health_path': '/health', 'type': 'application'},
            'hygiene-dashboard': {'port': 10422, 'health_path': '/', 'type': 'frontend'},
            'rule-control-api': {'port': 10421, 'health_path': '/health', 'type': 'application'},
            
            # Hardware Optimization
            'sutazai-hardware-resource-optimizer': {'port': 8116, 'health_path': '/health', 'type': 'optimization'},
        }
        
        for service, config in expected_services.items():
            configs[service] = config
            
        return configs
        
    async def check_service_health(self, service: str, config: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Check health of a specific service"""
        start_time = time.time()
        try:
            port = config['port']
            health_path = config.get('health_path', '/')
            
            if health_path:
                url = f"http://localhost:{port}{health_path}"
                response = requests.get(url, timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    return True, response_time, "OK"
                else:
                    return False, response_time, f"HTTP {response.status_code}"
            else:
                # For services without HTTP endpoints (like databases), check port
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                response_time = time.time() - start_time
                
                if result == 0:
                    return True, response_time, "Port accessible"
                else:
                    return False, response_time, "Port not accessible"
                    
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, str(e)
    
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get container resource usage statistics"""
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage with error handling
            try:
                cpu_usage = stats.get('cpu_stats', {}).get('cpu_usage', {})
                precpu_usage = stats.get('precpu_stats', {}).get('cpu_usage', {})
                
                cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
                system_delta = stats.get('cpu_stats', {}).get('system_cpu_usage', 0) - stats.get('precpu_stats', {}).get('system_cpu_usage', 0)
                
                # Handle missing percpu_usage gracefully
                percpu_usage = cpu_usage.get('percpu_usage', [])
                cpu_count = len(percpu_usage) if percpu_usage else 1
                
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0 if system_delta > 0 else 0.0
                cpu_percent = max(0.0, min(100.0, cpu_percent))  # Clamp between 0-100%
            except (KeyError, TypeError, ZeroDivisionError):
                cpu_percent = 0.0
            
            # Memory usage with error handling
            try:
                memory_stats = stats.get('memory_stats', {})
                memory_usage = memory_stats.get('usage', 0)
                memory_limit = memory_stats.get('limit', 0)
                memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            except (KeyError, TypeError, ZeroDivisionError):
                memory_usage = 0
                memory_percent = 0.0
            
            # Network stats
            network_rx = 0
            network_tx = 0
            if 'networks' in stats:
                for interface in stats['networks'].values():
                    network_rx += interface['rx_bytes']
                    network_tx += interface['tx_bytes']
            
            return {
                'cpu_percent': cpu_percent,
                'memory_usage': memory_usage,
                'memory_percent': memory_percent,
                'network_rx': network_rx,
                'network_tx': network_tx,
                'status': container.status,
                'health': getattr(container.attrs.get('State', {}), 'Health', {}).get('Status', 'unknown') if container.attrs.get('State', {}).get('Health') else 'no-healthcheck'
            }
        except Exception as e:
            logger.error(f"Error getting stats for container {container_name}: {e}")
            return {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide resource statistics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        'total': partition_usage.total,
                        'used': partition_usage.used,
                        'free': partition_usage.free,
                        'percent': partition_usage.percent
                    }
                except PermissionError:
                    continue
            
            # Network stats
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'disk_usage': disk_usage,
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def check_docker_containers(self) -> Dict[str, Any]:
        """Check status of all Docker containers"""
        container_stats = {}
        try:
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                name = container.name
                stats = self.get_container_stats(name)
                container_stats[name] = {
                    'status': container.status,
                    'image': container.image.tags[0] if container.image.tags else container.image.id,
                    'created': container.attrs['Created'],
                    'stats': stats
                }
                
                # Update Prometheus metrics
                if stats:
                    metrics['cpu_usage'].labels(service=name).set(stats.get('cpu_percent', 0))
                    metrics['memory_usage'].labels(service=name).set(stats.get('memory_usage', 0))
                    metrics['memory_percent'].labels(service=name).set(stats.get('memory_percent', 0))
                    metrics['network_rx'].labels(service=name).set(stats.get('network_rx', 0))
                    metrics['network_tx'].labels(service=name).set(stats.get('network_tx', 0))
                
                # Container status metric
                metrics['container_status'].labels(container=name).set(1 if container.status == 'running' else 0)
                
        except Exception as e:
            logger.error(f"Error checking Docker containers: {e}")
            metrics['deployment_errors'].labels(error_type='docker_check').inc()
            
        return container_stats
    
    async def monitor_services(self) -> Dict[str, Any]:
        """Monitor all configured services"""
        service_status = {}
        
        for service, config in self.service_configs.items():
            try:
                healthy, response_time, status_msg = await self.check_service_health(service, config)
                
                service_status[service] = {
                    'healthy': healthy,
                    'response_time': response_time,
                    'status_message': status_msg,
                    'type': config['type'],
                    'port': config['port'],
                    'last_check': datetime.now(timezone.utc).isoformat()
                }
                
                # Update Prometheus metrics
                metrics['service_health'].labels(service=service).set(1 if healthy else 0)
                metrics['service_response_time'].labels(service=service).observe(response_time)
                
                # Track startup times for newly healthy services
                if healthy and service not in self.startup_times:
                    self.startup_times[service] = response_time
                    metrics['startup_time'].labels(service=service).set(response_time)
                
                # Generate alerts for unhealthy services
                if not healthy:
                    alert = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'service': service,
                        'severity': 'critical',
                        'message': f"Service {service} is unhealthy: {status_msg}",
                        'response_time': response_time
                    }
                    self.alerts.append(alert)
                    logger.error(f"🚨 ALERT: {alert['message']}")
                    
            except Exception as e:
                logger.error(f"Error monitoring service {service}: {e}")
                metrics['deployment_errors'].labels(error_type='service_check').inc()
                service_status[service] = {
                    'healthy': False,
                    'error': str(e),
                    'last_check': datetime.now(timezone.utc).isoformat()
                }
        
        return service_status
    
    def check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity between services"""
        connectivity_results = {}
        
        # Test key service-to-service connections
        test_connections = [
            ('kong', 'consul', 8500),
            ('kong', 'rabbitmq', 5672),
            ('hygiene-backend', 'hygiene-postgres', 5432),
            ('hygiene-backend', 'hygiene-redis', 6379),
            ('sutazai-integration-dashboard', 'sutazai-loki', 3100),
        ]
        
        for source, target, port in test_connections:
            try:
                # Check if containers exist and are running
                source_container = self.docker_client.containers.get(source)
                target_container = self.docker_client.containers.get(target)
                
                if source_container.status == 'running' and target_container.status == 'running':
                    # Get target container IP
                    target_ip = target_container.attrs['NetworkSettings']['Networks'][list(target_container.attrs['NetworkSettings']['Networks'].keys())[0]]['IPAddress']
                    
                    # Test connection from source to target
                    exec_result = source_container.exec_run(f"timeout 5 nc -z {target_ip} {port}")
                    
                    connectivity_results[f"{source}->{target}"] = {
                        'connected': exec_result.exit_code == 0,
                        'target_ip': target_ip,
                        'port': port,
                        'output': exec_result.output.decode() if exec_result.output else ""
                    }
                else:
                    connectivity_results[f"{source}->{target}"] = {
                        'connected': False,
                        'error': f"Source status: {source_container.status}, Target status: {target_container.status}"
                    }
                    
            except Exception as e:
                connectivity_results[f"{source}->{target}"] = {
                    'connected': False,
                    'error': str(e)
                }
        
        return connectivity_results
    
    def generate_report(self, service_status: Dict, container_stats: Dict, system_stats: Dict, connectivity: Dict) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Count healthy/unhealthy services
        healthy_services = sum(1 for s in service_status.values() if s.get('healthy', False))
        total_services = len(service_status)
        
        # Count running containers
        running_containers = sum(1 for c in container_stats.values() if c.get('status') == 'running')
        total_containers = len(container_stats)
        
        # Calculate overall health score
        health_score = (healthy_services / total_services * 100) if total_services > 0 else 0
        
        # Check for resource alerts
        resource_alerts = []
        if system_stats.get('cpu_percent', 0) > self.alert_threshold:
            resource_alerts.append(f"High CPU usage: {system_stats['cpu_percent']:.1f}%")
        
        if system_stats.get('memory', {}).get('percent', 0) > self.alert_threshold:
            resource_alerts.append(f"High memory usage: {system_stats['memory']['percent']:.1f}%")
        
        for mountpoint, disk_info in system_stats.get('disk_usage', {}).items():
            if disk_info['percent'] > self.alert_threshold:
                resource_alerts.append(f"High disk usage on {mountpoint}: {disk_info['percent']:.1f}%")
        
        report = {
            'timestamp': timestamp,
            'monitoring_interval': self.interval,
            'overall_health': {
                'score': health_score,
                'healthy_services': healthy_services,
                'total_services': total_services,
                'running_containers': running_containers,
                'total_containers': total_containers
            },
            'service_mesh': {
                'consul': service_status.get('consul', {}),
                'kong': service_status.get('kong', {}),
                'rabbitmq': service_status.get('rabbitmq', {})
            },
            'core_services': {k: v for k, v in service_status.items() if v.get('type') in ['ai_service', 'application', 'database', 'cache']},
            'monitoring_stack': {k: v for k, v in service_status.items() if v.get('type') == 'monitoring'},
            'system_resources': system_stats,
            'container_stats': container_stats,
            'network_connectivity': connectivity,
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'resource_alerts': resource_alerts,
            'startup_times': self.startup_times
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]):
        """Save monitoring report to file"""
        try:
            report_file = Path('/opt/sutazaiapp/logs/monitoring_report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            # Also save latest metrics in Prometheus format
            metrics_file = Path('/opt/sutazaiapp/logs/prometheus_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(generate_latest(registry).decode())
                
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print monitoring summary to console"""
        print("\n" + "="*80)
        print(f"🔍 SutazAI Deployment Monitor - {report['timestamp']}")
        print("="*80)
        
        # Overall health
        health_score = report['overall_health']['score']
        health_icon = "🟢" if health_score >= 90 else "🟡" if health_score >= 70 else "🔴"
        print(f"{health_icon} Overall Health Score: {health_score:.1f}%")
        print(f"   Services: {report['overall_health']['healthy_services']}/{report['overall_health']['total_services']} healthy")
        print(f"   Containers: {report['overall_health']['running_containers']}/{report['overall_health']['total_containers']} running")
        
        # Service Mesh Status
        print(f"\n🕸️  Service Mesh Components:")
        for service, status in report['service_mesh'].items():
            icon = "✅" if status.get('healthy') else "❌"
            rt = status.get('response_time', 0) * 1000
            print(f"   {icon} {service:<20} ({rt:.1f}ms) - {status.get('status_message', 'Unknown')}")
        
        # Core Services Status
        print(f"\n⚙️  Core Services:")
        for service, status in report['core_services'].items():
            if status:  # Only show if we have status data
                icon = "✅" if status.get('healthy') else "❌"
                rt = status.get('response_time', 0) * 1000
                stype = status.get('type', 'unknown')
                print(f"   {icon} {service:<30} ({stype}) - {rt:.1f}ms")
        
        # System Resources
        sys_stats = report['system_resources']
        if sys_stats:
            print(f"\n📊 System Resources:")
            print(f"   CPU: {sys_stats.get('cpu_percent', 0):.1f}%")
            if 'memory' in sys_stats:
                mem = sys_stats['memory']
                print(f"   Memory: {mem.get('percent', 0):.1f}% ({mem.get('used', 0) / 1024**3:.1f}GB / {mem.get('total', 0) / 1024**3:.1f}GB)")
            
            # Show disk usage for main partitions
            for mount, disk in sys_stats.get('disk_usage', {}).items():
                if mount in ['/', '/var', '/opt']:
                    print(f"   Disk {mount}: {disk['percent']:.1f}% ({disk['used'] / 1024**3:.1f}GB / {disk['total'] / 1024**3:.1f}GB)")
        
        # Alerts
        if report['resource_alerts'] or report['alerts']:
            print(f"\n🚨 Active Alerts:")
            for alert in report['resource_alerts']:
                print(f"   ⚠️  {alert}")
            for alert in report['alerts'][-3:]:  # Last 3 service alerts
                print(f"   🔥 {alert.get('service', 'Unknown')}: {alert.get('message', 'No message')}")
        
        print("="*80)
    
    async def run_monitoring_cycle(self):
        """Run one complete monitoring cycle"""
        try:
            logger.info("Starting monitoring cycle...")
            
            # Monitor services
            service_status = await self.monitor_services()
            
            # Check Docker containers
            container_stats = self.check_docker_containers()
            
            # Get system stats
            system_stats = self.get_system_stats()
            
            # Check network connectivity
            connectivity = self.check_network_connectivity()
            
            # Update system metrics
            if system_stats:
                for mount, disk in system_stats.get('disk_usage', {}).items():
                    metrics['disk_usage'].labels(mountpoint=mount).set(disk['percent'])
            
            # Generate and save report
            report = self.generate_report(service_status, container_stats, system_stats, connectivity)
            self.save_report(report)
            self.print_summary(report)
            
            # Trim alerts list to prevent memory growth
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-50:]
            
            logger.info("Monitoring cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            logger.error(traceback.format_exc())
            metrics['deployment_errors'].labels(error_type='monitoring_cycle').inc()
    
    async def run(self):
        """Main monitoring loop"""
        logger.info(f"🔍 Starting SutazAI Real-Time Monitor (interval: {self.interval}s, alert threshold: {self.alert_threshold}%)")
        
        try:
            while True:
                await self.run_monitoring_cycle()
                await asyncio.sleep(self.interval)
                
        except KeyboardInterrupt:
            logger.info("👋 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
            logger.error(traceback.format_exc())

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SutazAI Real-Time Deployment Monitor')
    parser.add_argument('--interval', type=int, default=10, help='Monitoring interval in seconds (default: 10)')
    parser.add_argument('--alert-threshold', type=float, default=80.0, help='Resource usage alert threshold percentage (default: 80.0)')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output')
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    monitor = SutazAIMonitor(interval=args.interval, alert_threshold=args.alert_threshold)
    await monitor.run()

if __name__ == "__main__":
    asyncio.run(main())