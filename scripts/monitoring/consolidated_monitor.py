#!/usr/bin/env python3
"""
CONSOLIDATED MONITORING SYSTEM
==============================
Replaces 26 scattered monitoring files with one unified, professional implementation.
Compliant with all 20 rules from Enforcement_Rules.

This is the SINGLE SOURCE OF TRUTH for all monitoring operations.
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import aiohttp
import docker
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
health_check_counter = Counter('health_checks_total', 'Total health checks', ['service', 'status'])
service_uptime = Gauge('service_uptime_seconds', 'Service uptime in seconds', ['service'])
response_time_histogram = Histogram('response_time_seconds', 'Response time', ['service', 'endpoint'])
error_counter = Counter('errors_total', 'Total errors', ['service', 'error_type'])
resource_usage = Gauge('resource_usage_percent', 'Resource usage percentage', ['service', 'resource'])

class UnifiedMonitor:
    """
    Consolidated monitoring system for all SutazAI services.
    Replaces: health_monitor.py, system_monitor.py, service_monitor.py, 
              performance_monitor.py, api_health_monitor.py, and 21 others.
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.services = self._get_service_config()
        self.health_status = {}
        self.start_time = datetime.utcnow()
        
    def _get_service_config(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for all services to monitor"""
        return {
            'backend': {
                'url': 'http://localhost:10010/health',
                'container': 'sutazai-backend',
                'critical': True,
                'timeout': 5
            },
            'frontend': {
                'url': 'http://localhost:10011/',
                'container': 'sutazai-frontend',
                'critical': True,
                'timeout': 5
            },
            'postgres': {
                'container': 'sutazai-postgres',
                'critical': True,
                'check_cmd': ['pg_isready', '-U', 'sutazai']
            },
            'redis': {
                'container': 'sutazai-redis',
                'critical': True,
                'check_cmd': ['redis-cli', 'ping']
            },
            'ollama': {
                'url': 'http://localhost:10104/api/version',
                'container': 'sutazai-ollama',
                'critical': True,
                'timeout': 10
            },
            'neo4j': {
                'url': 'http://localhost:10002',
                'container': 'sutazai-neo4j',
                'critical': False,
                'timeout': 5
            },
            'chromadb': {
                'url': 'http://localhost:10100/api/v1/heartbeat',
                'container': 'sutazai-chromadb',
                'critical': False,
                'timeout': 5
            },
            'qdrant': {
                'url': 'http://localhost:10101/health',
                'container': 'sutazai-qdrant',
                'critical': False,
                'timeout': 5
            },
            'faiss': {
                'url': 'http://localhost:10103/health',
                'container': 'sutazai-faiss',
                'critical': False,
                'timeout': 5
            },
            'kong': {
                'url': 'http://localhost:10005/status',
                'container': 'sutazai-kong',
                'critical': False,
                'timeout': 5
            },
            'prometheus': {
                'url': 'http://localhost:10200/-/healthy',
                'container': 'sutazai-prometheus',
                'critical': False,
                'timeout': 5
            },
            'grafana': {
                'url': 'http://localhost:10201/api/health',
                'container': 'sutazai-grafana', 
                'critical': False,
                'timeout': 5
            }
        }
    
    async def check_service_health(self, name: str, config: Dict[str, Any]) -> Tuple[str, bool, str]:
        """Check health of a single service"""
        try:
            # Check container status first
            try:
                container = self.docker_client.containers.get(config['container'])
                if container.status != 'running':
                    health_check_counter.labels(service=name, status='down').inc()
                    return name, False, f"Container not running: {container.status}"
            except docker.errors.NotFound:
                health_check_counter.labels(service=name, status='not_found').inc()
                return name, False, "Container not found"
            
            # Check HTTP endpoint if configured
            if 'url' in config:
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        config['url'], 
                        timeout=aiohttp.ClientTimeout(total=config.get('timeout', 5))
                    ) as response:
                        response_time = time.time() - start_time
                        response_time_histogram.labels(service=name, endpoint='health').observe(response_time)
                        
                        if response.status == 200:
                            health_check_counter.labels(service=name, status='healthy').inc()
                            return name, True, f"Healthy (response time: {response_time:.2f}s)"
                        else:
                            health_check_counter.labels(service=name, status='unhealthy').inc()
                            return name, False, f"HTTP {response.status}"
            
            # Check with container command if configured
            if 'check_cmd' in config:
                result = container.exec_run(config['check_cmd'])
                if result.exit_code == 0:
                    health_check_counter.labels(service=name, status='healthy').inc()
                    return name, True, "Health check passed"
                else:
                    health_check_counter.labels(service=name, status='unhealthy').inc()
                    return name, False, f"Health check failed: {result.output.decode()}"
            
            # Default: just check if container is running
            health_check_counter.labels(service=name, status='running').inc()
            return name, True, "Container running"
            
        except asyncio.TimeoutError:
            error_counter.labels(service=name, error_type='timeout').inc()
            return name, False, "Health check timeout"
        except Exception as e:
            error_counter.labels(service=name, error_type='exception').inc()
            logger.error(f"Error checking {name}: {e}")
            return name, False, str(e)
    
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services concurrently"""
        tasks = [
            self.check_service_health(name, config) 
            for name, config in self.services.items()
        ]
        results = await asyncio.gather(*tasks)
        
        health_report = {}
        for name, healthy, message in results:
            health_report[name] = {
                'healthy': healthy,
                'message': message,
                'critical': self.services[name].get('critical', False),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Update uptime metric for healthy services
            if healthy:
                uptime = (datetime.utcnow() - self.start_time).total_seconds()
                service_uptime.labels(service=name).set(uptime)
        
        self.health_status = health_report
        return health_report
    
    def check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        resource_usage.labels(service='system', resource='cpu').set(cpu_percent)
        resource_usage.labels(service='system', resource='memory').set(memory.percent)
        resource_usage.labels(service='system', resource='disk').set(disk.percent)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def check_container_resources(self) -> Dict[str, Dict[str, Any]]:
        """Check resource usage for all containers"""
        container_stats = {}
        
        for container in self.docker_client.containers.list():
            try:
                stats = container.stats(stream=False)
                
                # Calculate CPU percentage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                
                # Calculate memory usage
                memory_usage = stats['memory_stats'].get('usage', 0)
                memory_limit = stats['memory_stats'].get('limit', 1)
                memory_percent = (memory_usage / memory_limit) * 100
                
                container_stats[container.name] = {
                    'cpu_percent': round(cpu_percent, 2),
                    'memory_mb': round(memory_usage / (1024**2), 2),
                    'memory_percent': round(memory_percent, 2),
                    'status': container.status
                }
                
                resource_usage.labels(service=container.name, resource='cpu').set(cpu_percent)
                resource_usage.labels(service=container.name, resource='memory').set(memory_percent)
                
            except Exception as e:
                logger.error(f"Error getting stats for {container.name}: {e}")
                container_stats[container.name] = {'error': str(e)}
        
        return container_stats
    
    def generate_report(self, health_status: Dict, system_resources: Dict, container_resources: Dict) -> str:
        """Generate comprehensive monitoring report"""
        report = []
        report.append("=" * 80)
        report.append("SUTAZAI SYSTEM MONITORING REPORT")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("=" * 80)
        
        # Service Health
        report.append("\n📊 SERVICE HEALTH STATUS")
        report.append("-" * 40)
        
        critical_down = []
        non_critical_down = []
        
        for name, status in health_status.items():
            icon = "✅" if status['healthy'] else "❌"
            critical_marker = " [CRITICAL]" if status['critical'] else ""
            report.append(f"{icon} {name.upper()}: {status['message']}{critical_marker}")
            
            if not status['healthy']:
                if status['critical']:
                    critical_down.append(name)
                else:
                    non_critical_down.append(name)
        
        # System Resources
        report.append("\n💻 SYSTEM RESOURCES")
        report.append("-" * 40)
        report.append(f"CPU Usage: {system_resources['cpu_percent']:.1f}%")
        report.append(f"Memory: {system_resources['memory_percent']:.1f}% used, "
                     f"{system_resources['memory_available_gb']:.1f} GB available")
        report.append(f"Disk: {system_resources['disk_percent']:.1f}% used, "
                     f"{system_resources['disk_free_gb']:.1f} GB free")
        
        # Container Resources (top 5 by CPU)
        report.append("\n📦 TOP CONTAINERS BY CPU USAGE")
        report.append("-" * 40)
        
        sorted_containers = sorted(
            [(k, v) for k, v in container_resources.items() if 'cpu_percent' in v],
            key=lambda x: x[1]['cpu_percent'],
            reverse=True
        )[:5]
        
        for name, stats in sorted_containers:
            report.append(f"{name}: CPU {stats['cpu_percent']:.1f}%, "
                         f"Memory {stats['memory_mb']:.1f} MB ({stats['memory_percent']:.1f}%)")
        
        # Summary
        report.append("\n📈 SUMMARY")
        report.append("-" * 40)
        
        healthy_count = sum(1 for s in health_status.values() if s['healthy'])
        total_count = len(health_status)
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        report.append(f"Overall Health: {health_percentage:.1f}% ({healthy_count}/{total_count} services healthy)")
        
        if critical_down:
            report.append(f"⚠️  CRITICAL SERVICES DOWN: {', '.join(critical_down)}")
        if non_critical_down:
            report.append(f"⚠️  Non-critical services down: {', '.join(non_critical_down)}")
        
        if health_percentage == 100:
            report.append("✅ All systems operational!")
        elif critical_down:
            report.append("🚨 CRITICAL: System degraded - immediate action required!")
        else:
            report.append("⚠️  WARNING: Some services are down")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def continuous_monitoring(self, interval: int = 30):
        """Run continuous monitoring loop"""
        logger.info(f"Starting continuous monitoring with {interval}s interval")
        
        while True:
            try:
                # Run all checks
                health_status = await self.check_all_services()
                system_resources = self.check_system_resources()
                container_resources = self.check_container_resources()
                
                # Generate and display report
                report = self.generate_report(health_status, system_resources, container_resources)
                print("\n" + report)
                
                # Save report to file
                report_dir = Path('/opt/sutazaiapp/logs/monitoring')
                report_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                report_file = report_dir / f'monitor_report_{timestamp}.txt'
                report_file.write_text(report)
                
                # Save JSON for programmatic access
                json_file = report_dir / 'latest_status.json'
                json_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'health_status': health_status,
                    'system_resources': system_resources,
                    'container_resources': container_resources
                }
                json_file.write_text(json.dumps(json_data, indent=2))
                
                # Sleep until next check
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                error_counter.labels(service='monitor', error_type='loop_error').inc()
                await asyncio.sleep(interval)
    
    def get_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest()


async def main():
    """Main entry point for monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Unified Monitoring System')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--prometheus', action='store_true', help='Export Prometheus metrics')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    args = parser.parse_args()
    
    monitor = UnifiedMonitor()
    
    if args.prometheus:
        # Export Prometheus metrics
        metrics = monitor.get_prometheus_metrics()
        print(metrics.decode())
    elif args.once:
        # Run once
        health_status = await monitor.check_all_services()
        system_resources = monitor.check_system_resources()
        container_resources = monitor.check_container_resources()
        
        if args.json:
            # Output JSON
            output = {
                'timestamp': datetime.utcnow().isoformat(),
                'health_status': health_status,
                'system_resources': system_resources,
                'container_resources': container_resources
            }
            print(json.dumps(output, indent=2))
        else:
            # Output human-readable report
            report = monitor.generate_report(health_status, system_resources, container_resources)
            print(report)
    else:
        # Run continuous monitoring
        await monitor.continuous_monitoring(interval=args.interval)


if __name__ == '__main__':
    asyncio.run(main())