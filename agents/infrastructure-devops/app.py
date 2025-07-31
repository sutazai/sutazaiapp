#!/usr/bin/env python3
"""
Infrastructure DevOps Manager Agent
Manages containers, monitoring, and system health
"""

import os
import time
import logging
import requests
import docker
from flask import Flask, jsonify
from threading import Thread
import schedule
import psutil
from prometheus_client import Counter, Gauge, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics
container_restarts = Counter('sutazai_container_restarts_total', 'Total container restarts', ['container'])
system_memory_usage = Gauge('sutazai_system_memory_usage_percent', 'System memory usage percentage')
system_cpu_usage = Gauge('sutazai_system_cpu_usage_percent', 'System CPU usage percentage')

class InfrastructureDevOpsManager:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.grafana_url = os.getenv('GRAFANA_URL', 'http://grafana:3000')
        
    def check_container_health(self):
        """Check health of all SutazAI containers"""
        unhealthy_containers = []
        
        try:
            containers = self.docker_client.containers.list(all=True)
            sutazai_containers = [c for c in containers if 'sutazai' in c.name.lower()]
            
            for container in sutazai_containers:
                if container.status != 'running':
                    unhealthy_containers.append({
                        'name': container.name,
                        'status': container.status,
                        'image': container.image.tags[0] if container.image.tags else 'unknown'
                    })
                    logger.warning(f"Container {container.name} is {container.status}")
                    
        except Exception as e:
            logger.error(f"Failed to check container health: {e}")
            
        return unhealthy_containers
    
    def restart_unhealthy_containers(self):
        """Restart containers that are unhealthy"""
        unhealthy = self.check_container_health()
        
        for container_info in unhealthy:
            try:
                container = self.docker_client.containers.get(container_info['name'])
                if container.status in ['exited', 'dead']:
                    logger.info(f"Restarting container: {container_info['name']}")
                    container.restart()
                    container_restarts.labels(container=container_info['name']).inc()
                    
            except Exception as e:
                logger.error(f"Failed to restart {container_info['name']}: {e}")
    
    def check_monitoring_stack(self):
        """Check if monitoring stack is healthy"""
        prometheus_healthy = False
        grafana_healthy = False
        
        try:
            response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            prometheus_healthy = response.status_code == 200
        except Exception as e:
            logger.error(f"Prometheus health check failed: {e}")
            
        try:
            response = requests.get(f"{self.grafana_url}/api/health", timeout=5)
            grafana_healthy = response.status_code == 200
        except Exception as e:
            logger.error(f"Grafana health check failed: {e}")
            
        return prometheus_healthy, grafana_healthy
    
    def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage.set(cpu_percent)
            
            logger.info(f"System metrics - Memory: {memory.percent:.1f}%, CPU: {cpu_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def cleanup_resources(self):
        """Clean up unused Docker resources"""
        try:
            # Remove unused images
            self.docker_client.images.prune()
            
            # Remove unused volumes
            self.docker_client.volumes.prune()
            
            # Remove unused networks
            self.docker_client.networks.prune()
            
            logger.info("Completed resource cleanup")
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
    
    def monitor_infrastructure(self):
        """Main monitoring routine"""
        self.collect_system_metrics()
        unhealthy = self.check_container_health()
        prometheus_healthy, grafana_healthy = self.check_monitoring_stack()
        
        if unhealthy:
            self.restart_unhealthy_containers()
            
        logger.info(f"Infrastructure check - Unhealthy containers: {len(unhealthy)}, "
                   f"Prometheus: {'OK' if prometheus_healthy else 'FAIL'}, "
                   f"Grafana: {'OK' if grafana_healthy else 'FAIL'}")

# Global instance
devops_manager = InfrastructureDevOpsManager()

@app.route('/health')
def health():
    """Health check endpoint"""
    unhealthy_containers = devops_manager.check_container_health()
    prometheus_healthy, grafana_healthy = devops_manager.check_monitoring_stack()
    
    return jsonify({
        'status': 'healthy',
        'unhealthy_containers': len(unhealthy_containers),
        'prometheus_healthy': prometheus_healthy,
        'grafana_healthy': grafana_healthy,
        'timestamp': time.time()
    })

@app.route('/containers')
def containers():
    """Get container status"""
    return jsonify({
        'unhealthy_containers': devops_manager.check_container_health()
    })

@app.route('/restart-unhealthy')
def restart_unhealthy():
    """Restart unhealthy containers"""
    try:
        devops_manager.restart_unhealthy_containers()
        return jsonify({'status': 'success', 'message': 'Unhealthy containers restart initiated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/cleanup')
def cleanup():
    """Trigger resource cleanup"""
    try:
        devops_manager.cleanup_resources()
        return jsonify({'status': 'success', 'message': 'Resource cleanup completed'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

def run_scheduler():
    """Run scheduled tasks"""
    schedule.every(1).minutes.do(devops_manager.monitor_infrastructure)
    schedule.every(10).minutes.do(devops_manager.cleanup_resources)
    
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == '__main__':
    # Start scheduler in background
    scheduler_thread = Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    app.run(host='0.0.0.0', port=8522, debug=False)