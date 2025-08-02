#!/usr/bin/env python3
"""
SutazAI Health Check Service
Monitors all services and provides health status
"""

import os
import time
import logging
import requests
import docker
from typing import Dict, List, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Health checker for SutazAI services"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '30'))
        self.services_to_check = os.getenv('SERVICES_TO_CHECK', '').split(',')
        
        # Service health endpoints
        self.health_endpoints = {
            'sutazai-backend': 'http://sutazai-backend:8000/health',
            'sutazai-frontend': 'http://sutazai-frontend:8501/healthz',
            'sutazai-qdrant': 'http://sutazai-qdrant:6333/healthz',
            'sutazai-chromadb': 'http://sutazai-chromadb:8000/api/v1/heartbeat',
            'sutazai-ollama': 'http://sutazai-ollama:11434/api/tags',
            'sutazai-prometheus': 'http://sutazai-prometheus:9090/-/healthy',
            'sutazai-grafana': 'http://sutazai-grafana:3000/api/health',
            'sutazai-neo4j': 'http://sutazai-neo4j:7474',
            'sutazai-langflow': 'http://sutazai-langflow:7860/health',
            'sutazai-flowise': 'http://sutazai-flowise:3000/api/v1/ping',
            'sutazai-dify': 'http://sutazai-dify:5000',
            'sutazai-n8n': 'http://sutazai-n8n:5678/healthz',
        }
        
        # Services that don't have HTTP endpoints - rely on container status only
        self.non_http_services = {
            'sutazai-postgres', 
            'sutazai-redis',
            'sutazai-dify',
        }
    
    def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            # Check if container is running
            container = self.docker_client.containers.get(service_name)
            container_status = container.status
            
            # Check HTTP endpoint if available
            endpoint_status = None
            if service_name in self.non_http_services:
                # For non-HTTP services, we only check container status
                endpoint_status = None
            elif service_name in self.health_endpoints:
                try:
                    response = requests.get(
                        self.health_endpoints[service_name],
                        timeout=5
                    )
                    endpoint_status = response.status_code == 200
                except requests.RequestException:
                    endpoint_status = False
            
            return {
                'service': service_name,
                'container_status': container_status,
                'endpoint_healthy': endpoint_status,
                'healthy': container_status == 'running' and (endpoint_status is None or endpoint_status),
                'timestamp': datetime.now().isoformat()
            }
            
        except docker.errors.NotFound:
            return {
                'service': service_name,
                'container_status': 'not_found',
                'endpoint_healthy': False,
                'healthy': False,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking {service_name}: {e}")
            return {
                'service': service_name,
                'container_status': 'error',
                'endpoint_healthy': False,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services"""
        results = {}
        
        for service in self.services_to_check:
            if service.strip():
                results[service] = self.check_service_health(service.strip())
        
        # Calculate overall health
        healthy_services = sum(1 for result in results.values() if result['healthy'])
        total_services = len(results)
        
        overall_health = {
            'healthy_services': healthy_services,
            'total_services': total_services,
            'health_percentage': (healthy_services / total_services * 100) if total_services > 0 else 0,
            'overall_healthy': healthy_services == total_services,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'overall': overall_health,
            'services': results
        }
    
    def restart_unhealthy_services(self, health_status: Dict[str, Any]):
        """Restart services that are unhealthy"""
        for service_name, status in health_status['services'].items():
            if not status['healthy'] and status['container_status'] == 'running':
                try:
                    logger.info(f"Restarting unhealthy service: {service_name}")
                    container = self.docker_client.containers.get(service_name)
                    container.restart()
                    logger.info(f"Service {service_name} restarted successfully")
                except Exception as e:
                    logger.error(f"Failed to restart {service_name}: {e}")
    
    def run(self):
        """Main health check loop"""
        logger.info("Starting SutazAI Health Check Service")
        logger.info(f"Checking services: {self.services_to_check}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        
        while True:
            try:
                # Check all services
                health_status = self.check_all_services()
                
                # Log results
                overall = health_status['overall']
                logger.info(f"Health Check - {overall['healthy_services']}/{overall['total_services']} services healthy ({overall['health_percentage']:.1f}%)")
                
                # Log unhealthy services
                for service_name, status in health_status['services'].items():
                    if not status['healthy']:
                        logger.warning(f"Service {service_name} is unhealthy: {status['container_status']}")
                
                # Restart unhealthy services (optional)
                # self.restart_unhealthy_services(health_status)
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Health check service stopped")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(self.check_interval)

if __name__ == "__main__":
    health_checker = HealthChecker()
    health_checker.run()