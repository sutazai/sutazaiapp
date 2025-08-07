#!/usr/bin/env python3
"""
Purpose: Distributed auto-scaler for SutazAI services
Usage: python distributed-autoscaler.py [--config /path/to/config.yaml]
Requirements: docker, prometheus-api-client, consul, pyyaml
"""

import os
import sys
import yaml
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import docker
import consul
from prometheus_api_client import PrometheusConnect
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics for auto-scaler
scaling_events = Counter('autoscaler_scaling_events_total', 'Total scaling events', ['service', 'action'])
current_replicas = Gauge('autoscaler_current_replicas', 'Current number of replicas', ['service'])
target_replicas = Gauge('autoscaler_target_replicas', 'Target number of replicas', ['service'])
scaling_duration = Histogram('autoscaler_scaling_duration_seconds', 'Time taken to scale services')

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

@dataclass
class ScalingRule:
    """Scaling rule configuration"""
    metric: str
    target_value: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_change: int = 1
    scale_down_change: int = 1
    
@dataclass
class ServiceConfig:
    """Service scaling configuration"""
    name: str
    min_replicas: int
    max_replicas: int
    rules: List[ScalingRule]
    cooldown_scale_up: int = 60
    cooldown_scale_down: int = 300
    
class DistributedAutoScaler:
    """Distributed auto-scaler for container services"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.docker_client = docker.from_env()
        self.prometheus = PrometheusConnect(
            url=os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')
        )
        self.consul_client = consul.Consul(
            host=os.getenv('CONSUL_HOST', 'consul-server-1'),
            port=int(os.getenv('CONSUL_PORT', 8500))
        )
        self.scaling_history: Dict[str, datetime] = {}
        self.running = False
        
    def _load_config(self, config_path: str) -> Dict[str, ServiceConfig]:
        """Load scaling configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        services = {}
        for service_name, service_data in config_data['services'].items():
            rules = []
            for rule_data in service_data['rules']:
                rule = ScalingRule(
                    metric=rule_data['metric'],
                    target_value=rule_data['target'],
                    scale_up_threshold=rule_data.get('scale_up_threshold', rule_data['target'] * 1.2),
                    scale_down_threshold=rule_data.get('scale_down_threshold', rule_data['target'] * 0.8),
                    scale_up_change=rule_data.get('scale_up_change', 1),
                    scale_down_change=rule_data.get('scale_down_change', 1)
                )
                rules.append(rule)
            
            service = ServiceConfig(
                name=service_name,
                min_replicas=service_data['min_replicas'],
                max_replicas=service_data['max_replicas'],
                rules=rules,
                cooldown_scale_up=service_data.get('cooldown_scale_up', 60),
                cooldown_scale_down=service_data.get('cooldown_scale_down', 300)
            )
            services[service_name] = service
            
        return services
    
    def get_service_replicas(self, service_name: str) -> int:
        """Get current number of replicas for a service"""
        try:
            # Try Docker Swarm mode first
            service = self.docker_client.services.get(service_name)
            replicas = service.attrs['Spec']['Mode']['Replicated']['Replicas']
            current_replicas.labels(service=service_name).set(replicas)
            return replicas
        except docker.errors.NotFound:
            # Try Docker Compose
            containers = self.docker_client.containers.list(
                filters={'label': f'com.docker.compose.service={service_name}'}
            )
            replicas = len(containers)
            current_replicas.labels(service=service_name).set(replicas)
            return replicas
        except Exception as e:
            logger.error(f"Failed to get replicas for {service_name}: {str(e)}")
            return 0
    
    def scale_service(self, service_name: str, new_replicas: int) -> bool:
        """Scale a service to the specified number of replicas"""
        try:
            start_time = time.time()
            
            # Try Docker Swarm mode first
            try:
                service = self.docker_client.services.get(service_name)
                service.update(mode={'Replicated': {'Replicas': new_replicas}})
                logger.info(f"Scaled {service_name} to {new_replicas} replicas (Swarm mode)")
                
            except docker.errors.NotFound:
                # Try Docker Compose scaling
                os.system(f"docker-compose up -d --scale {service_name}={new_replicas}")
                logger.info(f"Scaled {service_name} to {new_replicas} replicas (Compose mode)")
            
            # Update Consul with new replica count
            self.consul_client.kv.put(
                f"scaling/{service_name}/replicas",
                str(new_replicas)
            )
            
            # Update metrics
            target_replicas.labels(service=service_name).set(new_replicas)
            scaling_duration.observe(time.time() - start_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale {service_name}: {str(e)}")
            return False
    
    def get_metric_value(self, metric_query: str, service_name: str) -> Optional[float]:
        """Get metric value from Prometheus"""
        try:
            # Replace placeholders in query
            query = metric_query.replace('${service}', service_name)
            
            # Query Prometheus
            result = self.prometheus.custom_query(query)
            
            if result and len(result) > 0:
                # Get the latest value
                value = float(result[0]['value'][1])
                return value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metric for {service_name}: {str(e)}")
            return None
    
    def evaluate_scaling_rules(self, service: ServiceConfig, current_replicas: int) -> tuple[ScalingAction, int]:
        """Evaluate scaling rules for a service"""
        scale_up_votes = 0
        scale_down_votes = 0
        total_scale_change = 0
        
        for rule in service.rules:
            metric_value = self.get_metric_value(rule.metric, service.name)
            
            if metric_value is None:
                continue
            
            logger.debug(f"{service.name} - {rule.metric}: {metric_value} (target: {rule.target_value})")
            
            # Check if scale up is needed
            if metric_value > rule.scale_up_threshold:
                scale_up_votes += 1
                total_scale_change = max(total_scale_change, rule.scale_up_change)
                
            # Check if scale down is needed
            elif metric_value < rule.scale_down_threshold:
                scale_down_votes += 1
                total_scale_change = max(total_scale_change, rule.scale_down_change)
        
        # Determine action based on votes
        if scale_up_votes > scale_down_votes and scale_up_votes > 0:
            new_replicas = min(
                current_replicas + total_scale_change,
                service.max_replicas
            )
            return ScalingAction.SCALE_UP, new_replicas
            
        elif scale_down_votes > scale_up_votes and scale_down_votes > 0:
            new_replicas = max(
                current_replicas - total_scale_change,
                service.min_replicas
            )
            return ScalingAction.SCALE_DOWN, new_replicas
            
        return ScalingAction.NO_ACTION, current_replicas
    
    def check_cooldown(self, service_name: str, action: ScalingAction, cooldown_period: int) -> bool:
        """Check if service is in cooldown period"""
        if service_name not in self.scaling_history:
            return True
        
        last_scaling = self.scaling_history[service_name]
        time_since_scaling = (datetime.now() - last_scaling).total_seconds()
        
        return time_since_scaling >= cooldown_period
    
    def scale_service_if_needed(self, service: ServiceConfig):
        """Check and scale a service if needed"""
        try:
            # Get current replicas
            current = self.get_service_replicas(service.name)
            if current == 0:
                logger.warning(f"Service {service.name} not found or has 0 replicas")
                return
            
            # Evaluate scaling rules
            action, new_replicas = self.evaluate_scaling_rules(service, current)
            
            if action == ScalingAction.NO_ACTION or new_replicas == current:
                logger.debug(f"No scaling needed for {service.name}")
                return
            
            # Check cooldown
            cooldown = service.cooldown_scale_up if action == ScalingAction.SCALE_UP else service.cooldown_scale_down
            if not self.check_cooldown(service.name, action, cooldown):
                logger.debug(f"Service {service.name} in cooldown period")
                return
            
            # Perform scaling
            logger.info(f"Scaling {service.name} from {current} to {new_replicas} replicas")
            if self.scale_service(service.name, new_replicas):
                self.scaling_history[service.name] = datetime.now()
                scaling_events.labels(service=service.name, action=action.value).inc()
                
                # Send notification to Consul
                self.consul_client.event.fire(
                    name='service-scaled',
                    body=f"{service.name} scaled from {current} to {new_replicas}"
                )
                
        except Exception as e:
            logger.error(f"Error scaling service {service.name}: {str(e)}")
    
    def run_scaling_loop(self):
        """Main scaling loop"""
        logger.info("Starting auto-scaler loop")
        
        while self.running:
            try:
                # Check each service
                for service_name, service_config in self.config.items():
                    if not self.running:
                        break
                    
                    self.scale_service_if_needed(service_config)
                
                # Sleep before next iteration
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in scaling loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def register_with_consul(self):
        """Register auto-scaler with Consul"""
        try:
            import socket
            hostname = socket.gethostname()
            
            self.consul_client.agent.service.register(
                name='autoscaler',
                service_id=f'autoscaler-{hostname}',
                address=hostname,
                port=9100,
                tags=['autoscaler', 'monitoring'],
                check=consul.Check.http(
                    url='http://localhost:9100/metrics',
                    interval='30s',
                    timeout='10s'
                )
            )
            logger.info("Registered with Consul")
            
        except Exception as e:
            logger.error(f"Failed to register with Consul: {str(e)}")
    
    def start(self):
        """Start the auto-scaler"""
        self.running = True
        
        # Start Prometheus metrics server
        start_http_server(9100)
        logger.info("Started Prometheus metrics server on port 9100")
        
        # Register with Consul
        self.register_with_consul()
        
        # Start scaling loop
        self.run_scaling_loop()
    
    def stop(self):
        """Stop the auto-scaler"""
        self.running = False
        logger.info("Stopping auto-scaler")

def create_default_config() -> Dict[str, Any]:
    """Create default scaling configuration"""
    return {
        'services': {
            'ai-agent': {
                'min_replicas': 3,
                'max_replicas': 50,
                'cooldown_scale_up': 60,
                'cooldown_scale_down': 300,
                'rules': [
                    {
                        'metric': 'avg(rate(container_cpu_usage_seconds_total{service="${service}"}[1m])) * 100',
                        'target': 70,
                        'scale_up_threshold': 80,
                        'scale_down_threshold': 30,
                        'scale_up_change': 2,
                        'scale_down_change': 1
                    },
                    {
                        'metric': 'avg(container_memory_usage_bytes{service="${service}"} / container_spec_memory_limit_bytes{service="${service}"}) * 100',
                        'target': 70,
                        'scale_up_threshold': 85,
                        'scale_down_threshold': 40
                    },
                    {
                        'metric': 'sum(rate(celery_tasks_total{task_name=~"ai.inference.*"}[1m]))',
                        'target': 10,
                        'scale_up_threshold': 20,
                        'scale_down_threshold': 5,
                        'scale_up_change': 3,
                        'scale_down_change': 1
                    }
                ]
            },
            'celery-worker': {
                'min_replicas': 2,
                'max_replicas': 20,
                'cooldown_scale_up': 30,
                'cooldown_scale_down': 180,
                'rules': [
                    {
                        'metric': 'sum(celery_queue_length{queue_name="celery"})',
                        'target': 50,
                        'scale_up_threshold': 100,
                        'scale_down_threshold': 10,
                        'scale_up_change': 2
                    },
                    {
                        'metric': 'avg(celery_active_tasks{task_name=~".*"})',
                        'target': 20,
                        'scale_up_threshold': 30,
                        'scale_down_threshold': 5
                    }
                ]
            },
            'ollama': {
                'min_replicas': 2,
                'max_replicas': 10,
                'cooldown_scale_up': 120,
                'cooldown_scale_down': 600,
                'rules': [
                    {
                        'metric': 'avg(rate(ollama_request_duration_seconds_sum[1m]) / rate(ollama_request_duration_seconds_count[1m]))',
                        'target': 5,  # Target 5 second response time
                        'scale_up_threshold': 10,
                        'scale_down_threshold': 2
                    },
                    {
                        'metric': 'sum(rate(ollama_requests_total[1m]))',
                        'target': 20,
                        'scale_up_threshold': 40,
                        'scale_down_threshold': 10
                    }
                ]
            }
        }
    }

if __name__ == '__main__':
    import argparse
    import signal
    
    parser = argparse.ArgumentParser(description='Distributed auto-scaler for SutazAI')
    parser.add_argument('--config', default='/config/scaling-rules.yaml',
                        help='Path to scaling configuration file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create default configuration file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run in dry-run mode (no actual scaling)')
    
    args = parser.parse_args()
    
    if args.create_config:
        # Create default configuration
        config = create_default_config()
        with open('scaling-rules.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("Created default configuration file: scaling-rules.yaml")
        sys.exit(0)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default configuration...")
        config = create_default_config()
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Created configuration at: {args.config}")
    
    # Create and start auto-scaler
    scaler = DistributedAutoScaler(args.config)
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        scaler.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start auto-scaler
    try:
        scaler.start()
    except Exception as e:
        logger.error(f"Auto-scaler failed: {str(e)}")
        sys.exit(1)