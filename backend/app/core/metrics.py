"""
Prometheus metrics for SutazAI system monitoring
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable

# System metrics
active_agents = Gauge('sutazai_active_agents', 'Number of currently active agents')
total_agents = Gauge('sutazai_total_agents', 'Total number of registered agents')

# Agent metrics
agent_tasks_total = Counter(
    'sutazai_agent_tasks_total',
    'Total number of agent tasks',
    ['agent_name', 'task_type']
)
agent_tasks_successful = Counter(
    'sutazai_agent_tasks_successful_total',
    'Total number of successful agent tasks',
    ['agent_name', 'task_type']
)
agent_tasks_failed = Counter(
    'sutazai_agent_tasks_failed_total',
    'Total number of failed agent tasks',
    ['agent_name', 'task_type', 'error_type']
)
agent_task_duration = Histogram(
    'sutazai_agent_task_duration_seconds',
    'Agent task execution duration',
    ['agent_name', 'task_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
)

# Model metrics
model_inference_requests = Counter(
    'sutazai_model_inference_requests_total',
    'Total number of model inference requests',
    ['model_name', 'model_type']
)
model_inference_duration = Histogram(
    'sutazai_model_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'model_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)
model_tokens_processed = Counter(
    'sutazai_model_tokens_processed_total',
    'Total number of tokens processed',
    ['model_name', 'operation']
)

# Vector database metrics
vector_db_operations = Counter(
    'sutazai_vector_db_operations_total',
    'Total number of vector database operations',
    ['db_type', 'operation', 'collection']
)
vector_db_query_duration = Histogram(
    'sutazai_vector_db_query_duration_seconds',
    'Vector database query duration',
    ['db_type', 'operation', 'collection'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)
vector_db_documents = Gauge(
    'sutazai_vector_db_documents_total',
    'Total number of documents in vector database',
    ['db_type', 'collection']
)

# API metrics
api_requests = Counter(
    'sutazai_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)
api_request_duration = Histogram(
    'sutazai_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)
api_active_requests = Gauge(
    'sutazai_api_active_requests',
    'Number of active API requests'
)

# System info
system_info = Info('sutazai_system_info', 'SutazAI system information')
system_info.info({
    'version': '11.0.0',
    'environment': 'production',
    'deployment': 'docker'
})

# Memory metrics
memory_usage = Gauge(
    'sutazai_memory_usage_bytes',
    'Memory usage by component',
    ['component']
)

# Task queue metrics
task_queue_size = Gauge(
    'sutazai_task_queue_size',
    'Number of tasks in queue',
    ['queue_name', 'priority']
)
task_processing_time = Histogram(
    'sutazai_task_processing_time_seconds',
    'Task processing time',
    ['task_type', 'priority'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
)

# Decorator for tracking function execution time
def track_time(metric: Histogram, labels: dict = None):
    """Decorator to track function execution time"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Decorator for tracking request count
def track_request(counter: Counter, labels: dict = None):
    """Decorator to track request count"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

import asyncio