#!/usr/bin/env python3
"""
Purpose: Dynamic service scaler for AI services based on resource usage and queue depth
Usage: python service_scaler.py
Requirements: docker, consul, prometheus_client, redis
"""

import asyncio
import docker
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import consul
import redis
from prometheus_client.parser import text_string_to_metric_families
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceScaler:
    """Manages dynamic scaling of AI services based on demand"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.consul_client = consul.Consul(host=os.getenv('CONSUL_HTTP_ADDR', 'consul:8500').split(':')[0])
        self.redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        
        # Service configurations
        self.service_configs = {
            'langchain': {
                'image': 'python:3.11-slim',
                'min_replicas': 0,
                'max_replicas': 3,
                'scale_up_threshold': 5,  # Queue depth
                'scale_down_threshold': 0,
                'idle_timeout': 300,  # 5 minutes
                'memory_limit': '2g',
                'cpu_limit': 1.0
            },
            'autogpt': {
                'image': 'significantgravitas/auto-gpt:latest',
                'min_replicas': 0,
                'max_replicas': 2,
                'scale_up_threshold': 3,
                'scale_down_threshold': 0,
                'idle_timeout': 600,  # 10 minutes
                'memory_limit': '2g',
                'cpu_limit': 1.0
            },
            'letta': {
                'image': 'ghcr.io/letta-ai/letta:latest',
                'min_replicas': 0,
                'max_replicas': 2,
                'scale_up_threshold': 3,
                'scale_down_threshold': 0,
                'idle_timeout': 300,
                'memory_limit': '1g',
                'cpu_limit': 0.5
            },
            'tabbyml': {
                'image': 'tabbyml/tabby:latest',
                'min_replicas': 0,
                'max_replicas': 1,
                'scale_up_threshold': 2,
                'scale_down_threshold': 0,
                'idle_timeout': 900,  # 15 minutes
                'memory_limit': '2g',
                'cpu_limit': 1.0
            }
        }
        
        # Track service states
        self.service_states = {}
        
    async def get_queue_depth(self, service_name: str) -> int:
        """Get the current queue depth for a service from RabbitMQ"""
        try:
            response = requests.get(
                f'http://rabbitmq:15672/api/queues/%2F/{service_name}_queue',
                auth=('admin', 'admin')
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('messages', 0)
        except Exception as e:
            logger.error(f"Failed to get queue depth for {service_name}: {e}")
        return 0
    
    async def get_service_metrics(self, service_name: str) -> Dict:
        """Get service metrics from Prometheus"""
        metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'request_rate': 0,
            'error_rate': 0
        }
        
        try:
            # Query Prometheus for metrics
            queries = {
                'cpu_usage': f'rate(container_cpu_usage_seconds_total{{name=~"{service_name}.*"}}[5m])',
                'memory_usage': f'container_memory_usage_bytes{{name=~"{service_name}.*"}}',
                'request_rate': f'rate(http_requests_total{{service="{service_name}"}}[5m])',
                'error_rate': f'rate(http_requests_total{{service="{service_name}",status=~"5.."}}[5m])'
            }
            
            for metric, query in queries.items():
                response = requests.get(
                    f'{self.prometheus_url}/api/v1/query',
                    params={'query': query}
                )
                if response.status_code == 200:
                    data = response.json()
                    if data['data']['result']:
                        metrics[metric] = float(data['data']['result'][0]['value'][1])
        
        except Exception as e:
            logger.error(f"Failed to get metrics for {service_name}: {e}")
        
        return metrics
    
    async def get_running_containers(self, service_name: str) -> List:
        """Get running containers for a service"""
        try:
            containers = self.docker_client.containers.list(
                filters={
                    'label': f'com.sutazai.service={service_name}',
                    'status': 'running'
                }
            )
            return containers
        except Exception as e:
            logger.error(f"Failed to get containers for {service_name}: {e}")
            return []
    
    async def scale_service(self, service_name: str, target_replicas: int):
        """Scale a service to the target number of replicas"""
        config = self.service_configs.get(service_name)
        if not config:
            return
        
        current_containers = await self.get_running_containers(service_name)
        current_replicas = len(current_containers)
        
        if current_replicas == target_replicas:
            return
        
        logger.info(f"Scaling {service_name} from {current_replicas} to {target_replicas} replicas")
        
        if target_replicas > current_replicas:
            # Scale up
            for i in range(target_replicas - current_replicas):
                await self.start_container(service_name, i + current_replicas)
        else:
            # Scale down
            containers_to_stop = current_containers[target_replicas:]
            for container in containers_to_stop:
                await self.stop_container(container)
    
    async def start_container(self, service_name: str, instance_id: int):
        """Start a new container instance"""
        config = self.service_configs[service_name]
        
        container_name = f"{service_name}-{instance_id}"
        
        try:
            container = self.docker_client.containers.run(
                image=config['image'],
                name=container_name,
                detach=True,
                network='sutazaiapp_ai-mesh',
                labels={
                    'com.sutazai.service': service_name,
                    'com.sutazai.instance': str(instance_id)
                },
                environment={
                    'SERVICE_NAME': service_name,
                    'INSTANCE_ID': str(instance_id),
                    'CONSUL_HTTP_ADDR': 'consul:8500',
                    'REDIS_URL': 'redis://redis:6379',
                    'RABBITMQ_URL': 'amqp://guest:guest@rabbitmq:5672/'
                },
                volumes={
                    'sutazaiapp_shared-models': {'bind': '/models', 'mode': 'ro'},
                    'sutazaiapp_shared-cache': {'bind': '/cache', 'mode': 'rw'},
                    'sutazaiapp_shared-libs': {'bind': '/shared-libs', 'mode': 'ro'}
                },
                mem_limit=config['memory_limit'],
                nano_cpus=int(config['cpu_limit'] * 1e9),
                restart_policy={'Name': 'unless-stopped'}
            )
            
            # Register with Consul
            self.consul_client.agent.service.register(
                name=service_name,
                service_id=f"{service_name}-{instance_id}",
                address=container_name,
                port=8080,
                tags=['ai', 'scaled'],
                check=consul.Check.http(f"http://{container_name}:8080/health", interval="30s")
            )
            
            logger.info(f"Started container {container_name}")
            
        except Exception as e:
            logger.error(f"Failed to start container {container_name}: {e}")
    
    async def stop_container(self, container):
        """Stop and remove a container"""
        try:
            container_name = container.name
            service_name = container.labels.get('com.sutazai.service')
            instance_id = container.labels.get('com.sutazai.instance')
            
            # Deregister from Consul
            self.consul_client.agent.service.deregister(f"{service_name}-{instance_id}")
            
            # Stop and remove container
            container.stop(timeout=30)
            container.remove()
            
            logger.info(f"Stopped container {container_name}")
            
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
    
    async def check_idle_timeout(self, service_name: str) -> bool:
        """Check if a service has been idle beyond its timeout"""
        last_activity = self.redis_client.get(f"service:activity:{service_name}")
        if not last_activity:
            return True
        
        last_activity_time = datetime.fromisoformat(last_activity)
        idle_time = datetime.now() - last_activity_time
        config = self.service_configs[service_name]
        
        return idle_time.total_seconds() > config['idle_timeout']
    
    async def update_service_activity(self, service_name: str):
        """Update the last activity timestamp for a service"""
        self.redis_client.set(
            f"service:activity:{service_name}",
            datetime.now().isoformat(),
            ex=3600  # Expire after 1 hour
        )
    
    async def scale_decision(self, service_name: str):
        """Make scaling decision for a service"""
        config = self.service_configs[service_name]
        
        # Get current state
        queue_depth = await self.get_queue_depth(service_name)
        metrics = await self.get_service_metrics(service_name)
        current_containers = await self.get_running_containers(service_name)
        current_replicas = len(current_containers)
        
        # Update activity if there's work
        if queue_depth > 0 or metrics['request_rate'] > 0:
            await self.update_service_activity(service_name)
        
        # Determine target replicas
        target_replicas = current_replicas
        
        # Scale up if queue is building up
        if queue_depth >= config['scale_up_threshold']:
            target_replicas = min(current_replicas + 1, config['max_replicas'])
        
        # Scale down if idle
        elif queue_depth <= config['scale_down_threshold'] and current_replicas > config['min_replicas']:
            if await self.check_idle_timeout(service_name):
                target_replicas = max(current_replicas - 1, config['min_replicas'])
        
        # Apply scaling decision
        if target_replicas != current_replicas:
            await self.scale_service(service_name, target_replicas)
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                for service_name in self.service_configs:
                    await self.scale_decision(service_name)
                
                # Check system resources
                await self.check_system_resources()
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def check_system_resources(self):
        """Monitor overall system resource usage"""
        try:
            # Get Docker stats
            stats = {}
            for container in self.docker_client.containers.list():
                container_stats = container.stats(stream=False)
                stats[container.name] = {
                    'cpu_percent': self.calculate_cpu_percent(container_stats),
                    'memory_usage': container_stats['memory_stats']['usage'],
                    'memory_limit': container_stats['memory_stats']['limit']
                }
            
            # Store in Redis for monitoring
            self.redis_client.set('system:stats', json.dumps(stats), ex=60)
            
            # Check if we're approaching resource limits
            total_memory = sum(s['memory_usage'] for s in stats.values())
            total_limit = 8 * 1024 * 1024 * 1024  # 8GB system limit
            
            if total_memory > total_limit * 0.8:
                logger.warning(f"System memory usage high: {total_memory / 1024 / 1024 / 1024:.2f}GB")
                # Could trigger scale-down of less critical services
        
        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
    
    def calculate_cpu_percent(self, stats):
        """Calculate CPU percentage from Docker stats"""
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                       stats['precpu_stats']['system_cpu_usage']
        
        if system_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * 100.0
            return round(cpu_percent, 2)
        return 0.0
    
    async def run(self):
        """Run the service scaler"""
        logger.info("Starting service scaler...")
        await self.monitor_loop()

if __name__ == "__main__":
    scaler = ServiceScaler()
    asyncio.run(scaler.run())