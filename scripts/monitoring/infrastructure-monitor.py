#!/usr/bin/env python3
"""
Infrastructure Monitoring and Resource Control Script
Monitors system resources and manages container health
"""

import os
import sys
import time
import json
import logging
import psutil
import docker
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/infrastructure-monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class InfrastructureMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.running = True
        self.monitoring_interval = 30  # seconds
        self.resource_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 80.0,
            'disk_percent': 85.0
        }
        self.critical_services = [
            'consul', 'kong', 'rabbitmq', 'kong-database',
            'sutazai-postgres', 'sutazai-redis', 'sutazai-backend'
        ]
        
        # Create logs directory if it doesn't exist
        os.makedirs('/opt/sutazaiapp/logs', exist_ok=True)
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        
    def get_system_metrics(self) -> Dict:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_container_metrics(self) -> List[Dict]:
        """Get metrics for all running containers"""
        containers = []
        
        try:
            for container in self.docker_client.containers.list():
                try:
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    
                    cpu_percent = 0.0
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * \
                                     len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
                    
                    # Calculate memory usage percentage
                    memory_usage = stats['memory_stats']['usage']
                    memory_limit = stats['memory_stats']['limit']
                    memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                    
                    container_info = {
                        'name': container.name,
                        'id': container.id[:12],
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown',
                        'cpu_percent': round(cpu_percent, 2),
                        'memory_usage': memory_usage,
                        'memory_limit': memory_limit,
                        'memory_percent': round(memory_percent, 2),
                        'restart_count': container.attrs['RestartCount']
                    }
                    
                    containers.append(container_info)
                    
                except Exception as e:
                    logger.warning(f"Error getting stats for container {container.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            
        return containers
    
    def check_critical_services(self) -> Dict:
        """Check if critical services are running and healthy"""
        service_status = {}
        
        for service in self.critical_services:
            try:
                container = self.docker_client.containers.get(service)
                health = 'unknown'
                
                if hasattr(container.attrs, 'State') and 'Health' in container.attrs['State']:
                    health = container.attrs['State']['Health']['Status']
                
                service_status[service] = {
                    'status': container.status,
                    'health': health,
                    'restart_count': container.attrs['RestartCount']
                }
                
            except docker.errors.NotFound:
                service_status[service] = {
                    'status': 'not_found',
                    'health': 'unknown',
                    'restart_count': 0
                }
            except Exception as e:
                logger.error(f"Error checking service {service}: {e}")
                service_status[service] = {
                    'status': 'error',
                    'health': 'unknown',
                    'restart_count': 0
                }
                
        return service_status
    
    def handle_resource_alerts(self, metrics: Dict):
        """Handle resource threshold alerts"""
        alerts = []
        
        if metrics.get('cpu', {}).get('percent', 0) > self.resource_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics['cpu']['percent']:.1f}%")
            
        if metrics.get('memory', {}).get('percent', 0) > self.resource_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics['memory']['percent']:.1f}%")
            
        if metrics.get('disk', {}).get('percent', 0) > self.resource_thresholds['disk_percent']:
            alerts.append(f"High disk usage: {metrics['disk']['percent']:.1f}%")
        
        if alerts:
            logger.warning("Resource Alerts: " + "; ".join(alerts))
            self.take_corrective_action(alerts)
    
    def take_corrective_action(self, alerts: List[str]):
        """Take corrective action based on alerts"""
        try:
            # Stop non-critical containers if resources are critical
            if any("High" in alert for alert in alerts):
                logger.info("Taking corrective action for high resource usage")
                
                # Get containers sorted by resource usage
                containers = self.get_container_metrics()
                high_usage_containers = [
                    c for c in containers 
                    if c['cpu_percent'] > 50 or c['memory_percent'] > 70
                    if c['name'] not in self.critical_services
                ]
                
                # Stop highest resource consuming non-critical containers
                for container in sorted(high_usage_containers, 
                                      key=lambda x: x['cpu_percent'] + x['memory_percent'], 
                                      reverse=True)[:3]:
                    try:
                        docker_container = self.docker_client.containers.get(container['name'])
                        logger.info(f"Stopping high-resource container: {container['name']}")
                        docker_container.stop(timeout=10)
                    except Exception as e:
                        logger.error(f"Error stopping container {container['name']}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in corrective action: {e}")
    
    def restart_failed_services(self, service_status: Dict):
        """Restart failed critical services"""
        for service, status in service_status.items():
            if status['status'] in ['exited', 'dead'] or status['health'] in ['unhealthy']:
                try:
                    logger.info(f"Attempting to restart failed service: {service}")
                    container = self.docker_client.containers.get(service)
                    container.restart(timeout=30)
                    logger.info(f"Successfully restarted service: {service}")
                except Exception as e:
                    logger.error(f"Failed to restart service {service}: {e}")
    
    def save_metrics(self, system_metrics: Dict, container_metrics: List[Dict], service_status: Dict):
        """Save metrics to file for historical analysis"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = f"/opt/sutazaiapp/logs/metrics_{timestamp}.json"
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'system': system_metrics,
                'containers': container_metrics,
                'services': service_status
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # Keep only last 24 hours of metrics files
            self.cleanup_old_metrics()
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def cleanup_old_metrics(self):
        """Remove metrics files older than 24 hours"""
        try:
            logs_dir = "/opt/sutazaiapp/logs"
            current_time = time.time()
            
            for filename in os.listdir(logs_dir):
                if filename.startswith("metrics_") and filename.endswith(".json"):
                    file_path = os.path.join(logs_dir, filename)
                    if current_time - os.path.getmtime(file_path) > 86400:  # 24 hours
                        os.remove(file_path)
                        
        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {e}")
    
    def monitor(self):
        """Main monitoring loop"""
        logger.info("Starting infrastructure monitoring...")
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        while self.running:
            try:
                # Get system metrics
                system_metrics = self.get_system_metrics()
                
                # Get container metrics
                container_metrics = self.get_container_metrics()
                
                # Check critical services
                service_status = self.check_critical_services()
                
                # Log summary
                logger.info(f"System CPU: {system_metrics.get('cpu', {}).get('percent', 0):.1f}%, "
                           f"Memory: {system_metrics.get('memory', {}).get('percent', 0):.1f}%, "
                           f"Containers: {len(container_metrics)}")
                
                # Handle alerts
                self.handle_resource_alerts(system_metrics)
                
                # Restart failed services
                self.restart_failed_services(service_status)
                
                # Save metrics
                self.save_metrics(system_metrics, container_metrics, service_status)
                
                # Wait for next iteration
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
        
        logger.info("Infrastructure monitoring stopped.")

def main():
    """Main function"""
    if os.geteuid() != 0:
        logger.warning("Running without root privileges - some features may not work")
    
    monitor = InfrastructureMonitor()
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in monitoring: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()