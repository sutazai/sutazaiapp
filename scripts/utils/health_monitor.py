#\!/usr/bin/env python3
"""
SutazAI Health Monitor - Continuous system health monitoring
"""

import docker
import psutil
import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/health_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SutazAIHealthMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.backend_url = "http://localhost:8000"
        self.alert_threshold = {
            'cpu': 80.0,  # CPU usage %
            'memory': 85.0,  # Memory usage %
            'disk': 90.0,  # Disk usage %
            'unhealthy_agents': 5  # Number of unhealthy agents
        }
        
    def check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'swap_percent': psutil.swap_memory().percent
        }
    
    def check_docker_health(self) -> Dict[str, Any]:
        """Check Docker containers health"""
        containers = self.docker_client.containers.list()
        
        agent_containers = [c for c in containers if any(
            keyword in c.name for keyword in 
            ['agent', 'developer', 'engineer', 'specialist', 'coordinator', 
             'manager', 'optimizer', 'architect', 'gpt', 'ai']
        )]
        
        healthy = 0
        unhealthy = 0
        stopped = 0
        
        unhealthy_list = []
        
        for container in agent_containers:
            try:
                if container.status == 'running':
                    # Check if container has health check
                    if 'Health' in container.attrs['State']:
                        health_status = container.attrs['State']['Health']['Status']
                        if health_status == 'healthy':
                            healthy += 1
                        else:
                            unhealthy += 1
                            unhealthy_list.append(container.name)
                    else:
                        # No health check defined, assume healthy if running
                        healthy += 1
                else:
                    stopped += 1
            except Exception as e:
                logger.error(f"Error checking container {container.name}: {e}")
                unhealthy += 1
        
        return {
            'total_agents': len(agent_containers),
            'healthy': healthy,
            'unhealthy': unhealthy,
            'stopped': stopped,
            'unhealthy_list': unhealthy_list
        }
    
    def check_api_health(self) -> Dict[str, Any]:
        """Check API endpoints health"""
        endpoints = [
            {'name': 'Backend Health', 'url': f'{self.backend_url}/health'},
            {'name': 'Agents Endpoint', 'url': f'{self.backend_url}/agents'},
            {'name': 'Metrics Endpoint', 'url': f'{self.backend_url}/metrics'},
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint['url'], timeout=5)
                results[endpoint['name']] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_time': response.elapsed.total_seconds()
                }
            except Exception as e:
                results[endpoint['name']] = {
                    'status': 'unreachable',
                    'error': str(e)
                }
        
        return results
    
    def check_critical_services(self) -> Dict[str, bool]:
        """Check if critical services are running"""
        critical_services = {
            'postgres': 'sutazai-postgres',
            'redis': 'sutazai-redis',
            'ollama': 'sutazai-ollama',
            'backend': 'sutazai-backend',
            'frontend': 'sutazai-frontend'
        }
        
        status = {}
        for service, container_name in critical_services.items():
            try:
                container = self.docker_client.containers.get(container_name)
                status[service] = container.status == 'running'
            except docker.errors.NotFound:
                status[service] = False
            except Exception as e:
                logger.error(f"Error checking {service}: {e}")
                status[service] = False
        
        return status
    
    def generate_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate alerts based on metrics"""
        alerts = []
        
        # Resource alerts
        resources = metrics['system_resources']
        if resources['cpu_percent'] > self.alert_threshold['cpu']:
            alerts.append(f"⚠️ HIGH CPU USAGE: {resources['cpu_percent']:.1f}%")
        
        if resources['memory_percent'] > self.alert_threshold['memory']:
            alerts.append(f"⚠️ HIGH MEMORY USAGE: {resources['memory_percent']:.1f}%")
        
        if resources['disk_percent'] > self.alert_threshold['disk']:
            alerts.append(f"⚠️ HIGH DISK USAGE: {resources['disk_percent']:.1f}%")
        
        # Docker health alerts
        docker_health = metrics['docker_health']
        if docker_health['unhealthy'] > self.alert_threshold['unhealthy_agents']:
            alerts.append(f"⚠️ {docker_health['unhealthy']} UNHEALTHY AGENTS")
        
        # Critical services alerts
        critical = metrics['critical_services']
        for service, is_running in critical.items():
            if not is_running:
                alerts.append(f"🔴 CRITICAL: {service.upper()} is DOWN!")
        
        return alerts
    
    def save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to file for historical analysis"""
        metrics_file = '/opt/sutazaiapp/logs/health_metrics.json'
        
        # Load existing metrics
        try:
            with open(metrics_file, 'r') as f:
                historical = json.load(f)
        except:
            historical = []
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        # Keep only last 24 hours of data (288 entries at 5-minute intervals)
        historical.append(metrics)
        if len(historical) > 288:
            historical = historical[-288:]
        
        # Save updated metrics
        with open(metrics_file, 'w') as f:
            json.dump(historical, f, indent=2)
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a health report"""
        report = []
        report.append("=" * 60)
        report.append(f"SutazAI Health Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # System Resources
        resources = metrics['system_resources']
        report.append("\n📊 System Resources:")
        report.append(f"  CPU Usage: {resources['cpu_percent']:.1f}%")
        report.append(f"  Memory Usage: {resources['memory_percent']:.1f}%")
        report.append(f"  Disk Usage: {resources['disk_percent']:.1f}%")
        report.append(f"  Swap Usage: {resources['swap_percent']:.1f}%")
        
        # Docker Health
        docker = metrics['docker_health']
        report.append(f"\n🐳 Docker Health:")
        report.append(f"  Total Agents: {docker['total_agents']}")
        report.append(f"  Healthy: {docker['healthy']}")
        report.append(f"  Unhealthy: {docker['unhealthy']}")
        report.append(f"  Stopped: {docker['stopped']}")
        
        if docker['unhealthy_list']:
            report.append(f"  Unhealthy Agents: {', '.join(docker['unhealthy_list'][:5])}")
        
        # API Health
        api = metrics['api_health']
        report.append(f"\n🌐 API Health:")
        for endpoint, status in api.items():
            if status['status'] == 'healthy':
                report.append(f"  {endpoint}: ✅ ({status['response_time']:.2f}s)")
            else:
                report.append(f"  {endpoint}: ❌ ({status.get('error', 'unhealthy')})")
        
        # Critical Services
        critical = metrics['critical_services']
        report.append(f"\n🔧 Critical Services:")
        for service, is_running in critical.items():
            status_icon = "✅" if is_running else "❌"
            report.append(f"  {service.capitalize()}: {status_icon}")
        
        # Alerts
        if metrics['alerts']:
            report.append(f"\n⚠️  Active Alerts:")
            for alert in metrics['alerts']:
                report.append(f"  {alert}")
        else:
            report.append(f"\n✅ No active alerts")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run complete health check"""
        logger.info("Running health check...")
        
        metrics = {
            'system_resources': self.check_system_resources(),
            'docker_health': self.check_docker_health(),
            'api_health': self.check_api_health(),
            'critical_services': self.check_critical_services()
        }
        
        # Generate alerts
        metrics['alerts'] = self.generate_alerts(metrics)
        
        # Save metrics
        self.save_metrics(metrics)
        
        # Generate and log report
        report = self.generate_report(metrics)
        logger.info("\n" + report)
        
        # Save report to file
        with open('/opt/sutazaiapp/logs/latest_health_report.txt', 'w') as f:
            f.write(report)
        
        return metrics
    
    def continuous_monitoring(self, interval: int = 300):
        """Run continuous monitoring (default: every 5 minutes)"""
        logger.info(f"Starting continuous health monitoring (interval: {interval}s)")
        
        while True:
            try:
                metrics = self.run_health_check()
                
                # If there are critical alerts, run recovery actions
                if any('CRITICAL' in alert for alert in metrics['alerts']):
                    self.run_recovery_actions()
                
            except Exception as e:
                logger.error(f"Error during health check: {e}")
            
            time.sleep(interval)
    
    def run_recovery_actions(self):
        """Run automatic recovery actions for critical issues"""
        logger.warning("Running automatic recovery actions...")
        
        # Restart stopped critical services
        critical_services = {
            'postgres': 'sutazai-postgres',
            'redis': 'sutazai-redis',
            'ollama': 'sutazai-ollama',
            'backend': 'sutazai-backend',
            'frontend': 'sutazai-frontend'
        }
        
        for service, container_name in critical_services.items():
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status != 'running':
                    logger.info(f"Restarting {service}...")
                    container.restart()
                    time.sleep(5)  # Wait for container to start
            except Exception as e:
                logger.error(f"Failed to restart {service}: {e}")

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
    
    # Initialize and run monitor
    monitor = SutazAIHealthMonitor()
    
    # Check if running in continuous mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        monitor.continuous_monitoring()
    else:
        # Run single health check
        monitor.run_health_check()
