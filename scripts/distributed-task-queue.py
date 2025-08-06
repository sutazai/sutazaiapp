#!/usr/bin/env python3
"""
Purpose: Distributed task queue implementation for SutazAI agents
Usage: python distributed-task-queue.py [--worker] [--beat] [--flower]
Requirements: celery, redis, rabbitmq, prometheus-client
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from celery import Celery, Task, group, chain, chord
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import consul

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
task_counter = Counter('celery_tasks_total', 'Total number of tasks', ['task_name', 'status'])
task_duration = Histogram('celery_task_duration_seconds', 'Task execution duration', ['task_name'])
active_tasks = Gauge('celery_active_tasks', 'Number of active tasks', ['task_name'])
queue_length = Gauge('celery_queue_length', 'Length of task queues', ['queue_name'])

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 10
    HIGH = 7
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1

@dataclass
class TaskConfig:
    """Task configuration"""
    name: str
    queue: str
    priority: TaskPriority
    max_retries: int = 3
    timeout: int = 300
    rate_limit: Optional[str] = None

# Celery configuration
class CeleryConfig:
    # Broker settings
    broker_url = os.getenv('CELERY_BROKER_URL', 'amqp://admin:admin@rabbitmq-1:5672//')
    broker_connection_retry_on_startup = True
    broker_connection_max_retries = 10
    
    # Result backend
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis-master-1:6379/0')
    result_expires = 3600
    
    # Task execution
    task_serializer = 'json'
    accept_content = ['json']
    result_serializer = 'json'
    timezone = 'UTC'
    enable_utc = True
    
    # Task routing
    task_routes = {
        'ai.inference.*': {'queue': 'inference', 'routing_key': 'ai.inference'},
        'ai.training.*': {'queue': 'training', 'routing_key': 'ai.training'},
        'ai.embedding.*': {'queue': 'embedding', 'routing_key': 'ai.embedding'},
        'data.processing.*': {'queue': 'processing', 'routing_key': 'data.processing'},
        'system.health.*': {'queue': 'health', 'routing_key': 'system.health'},
        'system.scaling.*': {'queue': 'scaling', 'routing_key': 'system.scaling'},
    }
    
    # Queue configuration
    task_queue_max_priority = 10
    task_default_priority = 5
    task_inherit_parent_priority = True
    
    # Worker configuration
    worker_prefetch_multiplier = 4
    worker_max_tasks_per_child = 1000
    worker_disable_rate_limits = False
    
    # Task behavior
    task_acks_late = True
    task_reject_on_worker_lost = True
    task_ignore_result = False
    
    # Monitoring
    worker_send_task_events = True
    task_send_sent_event = True
    
    # Beat scheduler
    beat_schedule = {
        'health-check': {
            'task': 'system.health.check_all_services',
            'schedule': timedelta(seconds=30),
            'options': {'priority': TaskPriority.HIGH.value}
        },
        'queue-metrics': {
            'task': 'system.health.collect_queue_metrics',
            'schedule': timedelta(seconds=15),
            'options': {'priority': TaskPriority.NORMAL.value}
        },
        'auto-scale': {
            'task': 'system.scaling.evaluate_scaling',
            'schedule': timedelta(seconds=60),
            'options': {'priority': TaskPriority.HIGH.value}
        },
        'cleanup-old-tasks': {
            'task': 'system.health.cleanup_old_results',
            'schedule': timedelta(hours=1),
            'options': {'priority': TaskPriority.LOW.value}
        },
    }

# Create Celery app
app = Celery('sutazai')
app.config_from_object(CeleryConfig)

# Redis client for state management
redis_client = redis.RedisCluster(
    startup_nodes=[
        {"host": "redis-master-1", "port": 6379},
        {"host": "redis-master-2", "port": 6379},
        {"host": "redis-master-3", "port": 6379},
    ],
    decode_responses=True,
    skip_full_coverage_check=True
)

# Consul client for service discovery
consul_client = consul.Consul(host='consul-server-1', port=8500)

class DistributedTask(Task):
    """Base class for distributed tasks with monitoring"""
    
    def __call__(self, *args, **kwargs):
        """Execute task with monitoring"""
        task_name = self.name
        active_tasks.labels(task_name=task_name).inc()
        
        start_time = time.time()
        try:
            result = super().__call__(*args, **kwargs)
            task_counter.labels(task_name=task_name, status='success').inc()
            return result
        except Exception as e:
            task_counter.labels(task_name=task_name, status='failure').inc()
            raise
        finally:
            duration = time.time() - start_time
            task_duration.labels(task_name=task_name).observe(duration)
            active_tasks.labels(task_name=task_name).dec()

# AI Inference Tasks
@app.task(base=DistributedTask, bind=True, name='ai.inference.process_prompt')
def process_prompt(self, prompt: str, model: str = 'gpt-oss', **kwargs) -> Dict[str, Any]:
    """Process AI prompt using distributed Ollama instances"""
    try:
        # Get available Ollama instances from Consul
        _, services = consul_client.health.service('ollama', passing=True)
        if not services:
            raise Exception("No healthy Ollama instances available")
        
        # Select instance using round-robin
        instance_key = f"ollama:round_robin:{model}"
        instance_idx = redis_client.hincrby(instance_key, "counter", 1) % len(services)
        selected_service = services[instance_idx]
        
        host = selected_service['Service']['Address']
        port = selected_service['Service']['Port']
        
        # Call Ollama API
        import requests
        response = requests.post(
            f"http://{host}:{port}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            },
            timeout=300
        )
        response.raise_for_status()
        
        return {
            "status": "success",
            "response": response.json(),
            "instance": f"{host}:{port}",
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

@app.task(base=DistributedTask, bind=True, name='ai.embedding.generate')
def generate_embedding(self, text: str, model: str = 'all-minilm') -> List[float]:
    """Generate text embeddings"""
    try:
        # Implementation for embedding generation
        # This would connect to your embedding service
        pass
    except Exception as e:
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

# Data Processing Tasks
@app.task(base=DistributedTask, bind=True, name='data.processing.batch_process')
def batch_process(self, data_batch: List[Dict], operation: str) -> Dict[str, Any]:
    """Process data batch with specified operation"""
    try:
        results = []
        for item in data_batch:
            # Process each item
            processed = {
                "id": item.get("id"),
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat(),
                "result": f"Processed with {operation}"
            }
            results.append(processed)
        
        # Store results in Redis
        batch_id = f"batch:{self.request.id}"
        redis_client.hset(batch_id, mapping={
            "status": "completed",
            "count": len(results),
            "timestamp": datetime.utcnow().isoformat()
        })
        redis_client.expire(batch_id, 3600)
        
        return {
            "status": "success",
            "batch_id": batch_id,
            "processed_count": len(results)
        }
        
    except Exception as e:
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

# System Health Tasks
@app.task(base=DistributedTask, name='system.health.check_all_services')
def check_all_services() -> Dict[str, Any]:
    """Check health of all distributed services"""
    health_status = {}
    
    # Check Consul cluster
    try:
        leader = consul_client.status.leader()
        health_status['consul'] = {'status': 'healthy', 'leader': leader}
    except Exception as e:
        health_status['consul'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check Redis cluster
    try:
        redis_client.ping()
        cluster_info = redis_client.cluster_info()
        health_status['redis'] = {
            'status': 'healthy',
            'cluster_state': cluster_info.get('cluster_state', 'unknown')
        }
    except Exception as e:
        health_status['redis'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Check RabbitMQ
    try:
        from kombu import Connection
        with Connection(CeleryConfig.broker_url) as conn:
            conn.ensure_connection(max_retries=3)
        health_status['rabbitmq'] = {'status': 'healthy'}
    except Exception as e:
        health_status['rabbitmq'] = {'status': 'unhealthy', 'error': str(e)}
    
    # Store health status
    redis_client.hset('system:health', mapping={
        'status': json.dumps(health_status),
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return health_status

@app.task(base=DistributedTask, name='system.health.collect_queue_metrics')
def collect_queue_metrics() -> Dict[str, int]:
    """Collect metrics from all queues"""
    queue_metrics = {}
    
    try:
        # Get queue lengths from RabbitMQ
        import requests
        response = requests.get(
            'http://rabbitmq-1:15672/api/queues',
            auth=('admin', os.getenv('RABBITMQ_PASSWORD', 'admin'))
        )
        
        if response.status_code == 200:
            queues = response.json()
            for queue in queues:
                queue_name = queue['name']
                messages = queue.get('messages', 0)
                queue_metrics[queue_name] = messages
                queue_length.labels(queue_name=queue_name).set(messages)
        
        # Store in Redis
        redis_client.hset('system:queue_metrics', mapping={
            'metrics': json.dumps(queue_metrics),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to collect queue metrics: {str(e)}")
    
    return queue_metrics

# Auto-scaling Tasks
@app.task(base=DistributedTask, name='system.scaling.evaluate_scaling')
def evaluate_scaling() -> Dict[str, Any]:
    """Evaluate and trigger auto-scaling based on metrics"""
    scaling_decisions = []
    
    try:
        # Get current metrics
        queue_metrics = json.loads(
            redis_client.hget('system:queue_metrics', 'metrics') or '{}'
        )
        
        # Define scaling rules
        scaling_rules = {
            'inference': {
                'min_workers': 3,
                'max_workers': 20,
                'scale_up_threshold': 100,
                'scale_down_threshold': 10
            },
            'processing': {
                'min_workers': 2,
                'max_workers': 10,
                'scale_up_threshold': 50,
                'scale_down_threshold': 5
            }
        }
        
        for queue_name, rules in scaling_rules.items():
            current_length = queue_metrics.get(queue_name, 0)
            
            if current_length > rules['scale_up_threshold']:
                decision = {
                    'queue': queue_name,
                    'action': 'scale_up',
                    'reason': f'Queue length {current_length} > {rules["scale_up_threshold"]}'
                }
                scaling_decisions.append(decision)
                
                # Trigger scaling via Docker API
                scale_service(f'celery-worker-{queue_name}', 'up')
                
            elif current_length < rules['scale_down_threshold']:
                decision = {
                    'queue': queue_name,
                    'action': 'scale_down',
                    'reason': f'Queue length {current_length} < {rules["scale_down_threshold"]}'
                }
                scaling_decisions.append(decision)
                
                # Trigger scaling via Docker API
                scale_service(f'celery-worker-{queue_name}', 'down')
        
        # Store scaling decisions
        redis_client.hset('system:scaling_decisions', mapping={
            'decisions': json.dumps(scaling_decisions),
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Scaling evaluation error: {str(e)}")
    
    return {'decisions': scaling_decisions}

def scale_service(service_name: str, direction: str):
    """Scale a Docker service up or down"""
    try:
        import docker
        client = docker.from_env()
        
        service = client.services.get(service_name)
        current_replicas = service.attrs['Spec']['Mode']['Replicated']['Replicas']
        
        if direction == 'up':
            new_replicas = min(current_replicas + 1, 20)  # Max 20 replicas
        else:
            new_replicas = max(current_replicas - 1, 1)   # Min 1 replica
        
        if new_replicas != current_replicas:
            service.update(mode={'Replicated': {'Replicas': new_replicas}})
            logger.info(f"Scaled {service_name} from {current_replicas} to {new_replicas} replicas")
            
    except Exception as e:
        logger.error(f"Failed to scale service {service_name}: {str(e)}")

# Task orchestration examples
@app.task(base=DistributedTask, name='ai.pipeline.complete_analysis')
def complete_analysis(data: Dict) -> Dict[str, Any]:
    """Orchestrate a complete AI analysis pipeline"""
    
    # Create a workflow using Celery primitives
    workflow = chain(
        # Step 1: Preprocess data
        batch_process.s([data], 'preprocess'),
        
        # Step 2: Generate embeddings in parallel
        group(
            generate_embedding.s(data.get('text', '')),
            generate_embedding.s(data.get('title', ''))
        ),
        
        # Step 3: Process with AI
        process_prompt.s(
            prompt=f"Analyze the following: {data.get('text', '')}",
            model='gpt-oss'
        ),
        
        # Step 4: Post-process results
        batch_process.s([data], 'postprocess')
    )
    
    # Execute workflow
    result = workflow.apply_async(
        priority=TaskPriority.HIGH.value,
        expires=3600
    )
    
    return {
        'workflow_id': result.id,
        'status': 'initiated',
        'steps': 4
    }

# Signal handlers for monitoring
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    """Handle task pre-run events"""
    logger.debug(f"Task {task.name} starting with ID {task_id}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, result=None, **kwargs):
    """Handle task post-run events"""
    logger.debug(f"Task {task.name} completed with ID {task_id}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure events"""
    logger.error(f"Task {sender.name} failed with ID {task_id}: {str(exception)}")

def start_prometheus_metrics_server(port: int = 9100):
    """Start Prometheus metrics server"""
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed task queue for SutazAI')
    parser.add_argument('--worker', action='store_true', help='Start as Celery worker')
    parser.add_argument('--beat', action='store_true', help='Start as Celery beat scheduler')
    parser.add_argument('--flower', action='store_true', help='Start Flower monitoring')
    parser.add_argument('--queue', default='celery', help='Queue name for worker')
    parser.add_argument('--concurrency', type=int, default=4, help='Worker concurrency')
    parser.add_argument('--loglevel', default='info', help='Log level')
    
    args = parser.parse_args()
    
    if args.worker:
        # Start Prometheus metrics server
        start_prometheus_metrics_server()
        
        # Start Celery worker
        worker = app.Worker(
            queues=[args.queue],
            concurrency=args.concurrency,
            loglevel=args.loglevel,
            optimization='fair',
            task_events=True
        )
        worker.start()
        
    elif args.beat:
        # Start Celery beat scheduler
        from celery.bin import beat
        beat = beat.beat(app=app)
        beat.run(loglevel=args.loglevel)
        
    elif args.flower:
        # Start Flower monitoring
        from flower.command import FlowerCommand
        flower = FlowerCommand()
        flower.execute_from_commandline([
            'flower',
            '--broker=' + CeleryConfig.broker_url,
            '--port=5555',
            '--basic_auth=admin:admin'
        ])
        
    else:
        # Run a test task
        result = check_all_services.delay()
        print(f"Test task submitted: {result.id}")
        print(f"Result: {result.get(timeout=30)}")