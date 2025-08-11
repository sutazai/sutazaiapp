#!/usr/bin/env python3
"""
Docker Swarm Auto-scaler for SutazAI
Automatically scales services based on CPU, memory, and custom metrics
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp
import docker
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SwarmAutoscaler:
    """Docker Swarm service autoscaler with AI workload awareness"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        self.scale_check_interval = int(os.getenv('SCALE_CHECK_INTERVAL', '30'))
        
        # Scaling thresholds
        self.cpu_threshold_up = float(os.getenv('CPU_THRESHOLD_UP', '70'))
        self.cpu_threshold_down = float(os.getenv('CPU_THRESHOLD_DOWN', '30'))
        self.memory_threshold_up = float(os.getenv('MEMORY_THRESHOLD_UP', '80'))
        self.memory_threshold_down = float(os.getenv('MEMORY_THRESHOLD_DOWN', '50'))
        
        # Replica limits
        self.min_replicas = int(os.getenv('MIN_REPLICAS', '1'))
        self.max_replicas = int(os.getenv('MAX_REPLICAS', '10'))
        
        # Service configurations
        self.service_configs = {
            'sutazai-backend': {
                'min_replicas': 2,
                'max_replicas': 10,
                'cpu_threshold_up': 70,
                'cpu_threshold_down': 30,
                'memory_threshold_up': 80,
                'memory_threshold_down': 50,
                'scale_up_cooldown': 60,
                'scale_down_cooldown': 300,
                'custom_metrics': ['http_requests_per_second', 'response_time_p95']
            },
            'sutazai-ollama': {
                'min_replicas': 2,
                'max_replicas': 8,
                'cpu_threshold_up': 75,
                'cpu_threshold_down': 35,
                'memory_threshold_up': 85,
                'memory_threshold_down': 55,
                'scale_up_cooldown': 120,
                'scale_down_cooldown': 600,
                'custom_metrics': ['inference_queue_depth', 'model_memory_usage']
            },
            'sutazai-frontend': {
                'min_replicas': 1,
                'max_replicas': 5,
                'cpu_threshold_up': 70,
                'cpu_threshold_down': 25,
                'memory_threshold_up': 75,
                'memory_threshold_down': 40,
                'scale_up_cooldown': 60,
                'scale_down_cooldown': 300,
                'custom_metrics': ['active_sessions']
            },
            'sutazai-autogpt': {
                'min_replicas': 1,
                'max_replicas': 6,
                'cpu_threshold_up': 70,
                'cpu_threshold_down': 30,
                'memory_threshold_up': 80,
                'memory_threshold_down': 50,
                'scale_up_cooldown': 90,
                'scale_down_cooldown': 300,
                'custom_metrics': ['agent_task_queue_depth']
            },
            'sutazai-crewai': {
                'min_replicas': 1,
                'max_replicas': 4,
                'cpu_threshold_up': 70,
                'cpu_threshold_down': 30,
                'memory_threshold_up': 80,
                'memory_threshold_down': 50,
                'scale_up_cooldown': 120,
                'scale_down_cooldown': 300,
                'custom_metrics': ['crew_active_tasks']
            },
            'sutazai-chromadb': {
                'min_replicas': 1,
                'max_replicas': 3,
                'cpu_threshold_up': 75,
                'cpu_threshold_down': 35,
                'memory_threshold_up': 85,
                'memory_threshold_down': 55,
                'scale_up_cooldown': 180,
                'scale_down_cooldown': 600,
                'custom_metrics': ['vector_search_latency_p95']
            },
            'sutazai-qdrant': {
                'min_replicas': 1,
                'max_replicas': 3,
                'cpu_threshold_up': 75,
                'cpu_threshold_down': 35,
                'memory_threshold_up': 85,
                'memory_threshold_down': 55,
                'scale_up_cooldown': 180,
                'scale_down_cooldown': 600,
                'custom_metrics': ['vector_collection_size']
            }
        }
        
        # Track last scaling actions to prevent oscillation
        self.last_scale_actions = {}
        
        # HTTP session for metrics collection
        self.session = None
        
    async def start(self):
        """Start the autoscaler"""
        logger.info("Starting Docker Swarm Autoscaler for SutazAI")
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        try:
            while True:
                await self.autoscale_services()
                await asyncio.sleep(self.scale_check_interval)
                
        except KeyboardInterrupt:
            logger.info("Shutting down autoscaler...")
        finally:
            if self.session:
                await self.session.close()
    
    async def autoscale_services(self):
        """Check and scale all configured services"""
        logger.debug("Starting autoscaling check...")
        
        # Get current swarm services
        services = self.docker_client.services.list()
        
        for service in services:
            service_name = service.name
            
            if service_name in self.service_configs:
                try:
                    await self.check_and_scale_service(service, service_name)
                except Exception as e:
                    logger.error(f"Error scaling service {service_name}: {e}")
                    
        logger.debug("Autoscaling check completed")
    
    async def check_and_scale_service(self, service, service_name: str):
        """Check metrics and scale a specific service if needed"""
        config = self.service_configs[service_name]
        
        # Get current replica count
        current_replicas = self.get_service_replicas(service)
        if current_replicas is None:
            logger.warning(f"Could not get replica count for {service_name}")
            return
            
        # Check cooldown period
        if not self.can_scale_service(service_name):
            logger.debug(f"Service {service_name} in cooldown period")
            return
            
        # Collect metrics
        cpu_usage = await self.get_cpu_usage(service_name)
        memory_usage = await self.get_memory_usage(service_name)
        custom_metrics = await self.get_custom_metrics(service_name, config['custom_metrics'])
        
        logger.debug(f"{service_name}: CPU={cpu_usage:.1f}%, Memory={memory_usage:.1f}%, Replicas={current_replicas}")
        
        # Determine scaling action
        scale_action = self.determine_scale_action(
            service_name, current_replicas, cpu_usage, memory_usage, custom_metrics
        )
        
        if scale_action != 0:
            new_replicas = max(
                config['min_replicas'],
                min(config['max_replicas'], current_replicas + scale_action)
            )
            
            if new_replicas != current_replicas:
                await self.scale_service(service, service_name, new_replicas, scale_action > 0)
    
    def get_service_replicas(self, service) -> Optional[int]:
        """Get current number of replicas for a service"""
        try:
            service.reload()
            spec = service.attrs['Spec']
            
            if 'Replicated' in spec['Mode']:
                return spec['Mode']['Replicated']['Replicas']
            elif 'Global' in spec['Mode']:
                # Global services have one replica per node
                nodes = self.docker_client.nodes.list()
                return len([n for n in nodes if n.attrs['Spec']['Availability'] == 'active'])
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting replica count: {e}")
            return None
    
    async def get_cpu_usage(self, service_name: str) -> float:
        """Get CPU usage percentage for a service"""
        query = f'avg(rate(container_cpu_usage_seconds_total{{container_label_com_docker_swarm_service_name="{service_name}"}}[5m])) * 100'
        
        try:
            result = await self.query_prometheus(query)
            if result and result[0]['value']:
                return float(result[0]['value'][1])
        except Exception as e:
            logger.error(f"Error getting CPU usage for {service_name}: {e}")
            
        return 0.0
    
    async def get_memory_usage(self, service_name: str) -> float:
        """Get memory usage percentage for a service"""
        query = f'avg(container_memory_usage_bytes{{container_label_com_docker_swarm_service_name="{service_name}"}} / container_spec_memory_limit_bytes{{container_label_com_docker_swarm_service_name="{service_name}"}}) * 100'
        
        try:
            result = await self.query_prometheus(query)
            if result and result[0]['value']:
                return float(result[0]['value'][1])
        except Exception as e:
            logger.error(f"Error getting memory usage for {service_name}: {e}")
            
        return 0.0
    
    async def get_custom_metrics(self, service_name: str, metric_names: List[str]) -> Dict[str, float]:
        """Get custom metrics for a service"""
        metrics = {}
        
        for metric_name in metric_names:
            try:
                if metric_name == 'http_requests_per_second':
                    query = f'rate(http_requests_total{{service="{service_name}"}}[5m])'
                elif metric_name == 'response_time_p95':
                    query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{service="{service_name}"}}[5m]))'
                elif metric_name == 'inference_queue_depth':
                    query = f'ollama_inference_queue_depth{{service="{service_name}"}}'
                elif metric_name == 'agent_task_queue_depth':
                    query = f'sutazai_agent_task_queue_depth{{agent_name="{service_name}"}}'
                elif metric_name == 'active_sessions':
                    query = f'sutazai_user_sessions_active{{interface="streamlit"}}'
                elif metric_name == 'vector_search_latency_p95':
                    query = f'histogram_quantile(0.95, rate(vector_search_duration_seconds_bucket{{service="{service_name}"}}[5m]))'
                else:
                    continue
                    
                result = await self.query_prometheus(query)
                if result and result[0]['value']:
                    metrics[metric_name] = float(result[0]['value'][1])
                    
            except Exception as e:
                logger.error(f"Error getting custom metric {metric_name} for {service_name}: {e}")
                
        return metrics
    
    async def query_prometheus(self, query: str) -> Optional[List[Dict]]:
        """Query Prometheus for metrics"""
        if not self.session:
            return None
            
        try:
            url = f"{self.prometheus_url}/api/v1/query"
            params = {'query': query}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', {}).get('result', [])
                    
        except Exception as e:
            logger.error(f"Error querying Prometheus: {e}")
            
        return None
    
    def determine_scale_action(
        self, 
        service_name: str, 
        current_replicas: int, 
        cpu_usage: float, 
        memory_usage: float, 
        custom_metrics: Dict[str, float]
    ) -> int:
        """Determine if service should be scaled up (+1), down (-1), or not at all (0)"""
        config = self.service_configs[service_name]
        
        # Check for scale up conditions
        scale_up_reasons = []
        
        if cpu_usage > config['cpu_threshold_up']:
            scale_up_reasons.append(f"CPU usage {cpu_usage:.1f}% > {config['cpu_threshold_up']}%")
            
        if memory_usage > config['memory_threshold_up']:
            scale_up_reasons.append(f"Memory usage {memory_usage:.1f}% > {config['memory_threshold_up']}%")
        
        # Check custom metrics for scale up
        if 'http_requests_per_second' in custom_metrics and custom_metrics['http_requests_per_second'] > 100:
            scale_up_reasons.append(f"High request rate: {custom_metrics['http_requests_per_second']:.1f} RPS")
            
        if 'inference_queue_depth' in custom_metrics and custom_metrics['inference_queue_depth'] > 10:
            scale_up_reasons.append(f"High inference queue: {custom_metrics['inference_queue_depth']}")
            
        if 'agent_task_queue_depth' in custom_metrics and custom_metrics['agent_task_queue_depth'] > 5:
            scale_up_reasons.append(f"High task queue: {custom_metrics['agent_task_queue_depth']}")
        
        # Scale up if we have reasons and haven't reached max replicas
        if scale_up_reasons and current_replicas < config['max_replicas']:
            logger.info(f"Scaling up {service_name}: {', '.join(scale_up_reasons)}")
            return 1
        
        # Check for scale down conditions
        scale_down_reasons = []
        
        if (cpu_usage < config['cpu_threshold_down'] and 
            memory_usage < config['memory_threshold_down'] and
            current_replicas > config['min_replicas']):
            
            # Additional checks for custom metrics
            can_scale_down = True
            
            if 'http_requests_per_second' in custom_metrics and custom_metrics['http_requests_per_second'] > 50:
                can_scale_down = False
                
            if 'inference_queue_depth' in custom_metrics and custom_metrics['inference_queue_depth'] > 5:
                can_scale_down = False
                
            if 'agent_task_queue_depth' in custom_metrics and custom_metrics['agent_task_queue_depth'] > 2:
                can_scale_down = False
            
            if can_scale_down:
                scale_down_reasons.append(f"Low resource usage: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%")
                logger.info(f"Scaling down {service_name}: {', '.join(scale_down_reasons)}")
                return -1
        
        return 0
    
    def can_scale_service(self, service_name: str) -> bool:
        """Check if service can be scaled based on cooldown period"""
        if service_name not in self.last_scale_actions:
            return True
            
        last_action = self.last_scale_actions[service_name]
        config = self.service_configs[service_name]
        
        if last_action['action'] == 'up':
            cooldown = config['scale_up_cooldown']
        else:
            cooldown = config['scale_down_cooldown']
            
        time_since_last = time.time() - last_action['timestamp']
        return time_since_last >= cooldown
    
    async def scale_service(self, service, service_name: str, new_replicas: int, scale_up: bool):
        """Scale a service to the specified number of replicas"""
        try:
            # Update service spec
            service.reload()
            spec = service.attrs['Spec']
            
            if 'Replicated' in spec['Mode']:
                spec['Mode']['Replicated']['Replicas'] = new_replicas
                
                # Update the service
                service.update(**spec)
                
                action = 'up' if scale_up else 'down'
                logger.info(f"Scaled {service_name} {action} to {new_replicas} replicas")
                
                # Record the scaling action
                self.last_scale_actions[service_name] = {
                    'action': action,
                    'timestamp': time.time(),
                    'replicas': new_replicas
                }
                
            else:
                logger.warning(f"Cannot scale {service_name}: not a replicated service")
                
        except Exception as e:
            logger.error(f"Error scaling service {service_name}: {e}")
    
    def get_service_health(self, service_name: str) -> Dict[str, bool]:
        """Get health status of service containers"""
        try:
            containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.swarm.service.name={service_name}'}
            )
            
            health_status = {'healthy': 0, 'unhealthy': 0, 'unknown': 0}
            
            for container in containers:
                health = container.attrs.get('State', {}).get('Health', {})
                status = health.get('Status', 'unknown')
                
                if status == 'healthy':
                    health_status['healthy'] += 1
                elif status == 'unhealthy':
                    health_status['unhealthy'] += 1
                else:
                    health_status['unknown'] += 1
                    
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting health status for {service_name}: {e}")
            return {'healthy': 0, 'unhealthy': 0, 'unknown': 0}

async def main():
    """Main entry point"""
    autoscaler = SwarmAutoscaler()
    await autoscaler.start()

if __name__ == '__main__':
    asyncio.run(main())