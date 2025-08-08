"""
Comprehensive Prometheus Metrics Instrumentation for SutazAI System

This module provides standardized metrics collection for all Python microservices
following Prometheus best practices and JARVIS observability requirements.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server, REGISTRY
)
import time
import functools
import asyncio
import psutil
import logging
from typing import Callable, Dict, List, Optional, Any
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
logger = logging.getLogger("sutazai.metrics")

# =============================================================================
# JARVIS METRICS COLLECTION (as per requirements)
# =============================================================================

# Request metrics by service and endpoint
jarvis_requests_total = Counter(
    'jarvis_requests_total',
    'Total number of requests to JARVIS services',
    ['service', 'endpoint', 'method', 'status_code']
)

# Latency histogram with appropriate buckets  
jarvis_latency_seconds_bucket = Histogram(
    'jarvis_latency_seconds',
    'Request latency in seconds for JARVIS services',
    ['service', 'endpoint', 'method'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Error counter by service and error type
jarvis_errors_total = Counter(
    'jarvis_errors_total', 
    'Total number of errors in JARVIS services',
    ['service', 'endpoint', 'error_type', 'error_code']
)

# =============================================================================
# ENHANCED SYSTEM METRICS
# =============================================================================

# System metrics
active_agents = Gauge('sutazai_active_agents', 'Number of currently active agents')
total_agents = Gauge('sutazai_total_agents', 'Total number of registered agents')

# System resource metrics
system_cpu_usage = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
system_memory_usage = Gauge('system_memory_usage_bytes', 'System memory usage in bytes')
system_memory_total = Gauge('system_memory_total_bytes', 'Total system memory in bytes')
system_disk_usage = Gauge('system_disk_usage_bytes', 'System disk usage in bytes')
system_disk_total = Gauge('system_disk_total_bytes', 'Total system disk in bytes')

# Application metrics
app_uptime_seconds = Gauge('app_uptime_seconds', 'Application uptime in seconds')
app_info = Info('app_info', 'Application information')

# Service health metrics
service_health_status = Gauge(
    'service_health_status',
    'Service health status (1=healthy, 0=unhealthy)',
    ['service_name', 'service_type']
)

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
        
# =============================================================================
# METRICS COLLECTION UTILITIES
# =============================================================================

class MetricsCollector:
    """Central metrics collection and management for JARVIS system"""
    
    def __init__(self, service_name: str = "backend", app_version: str = "17.0.0"):
        self.service_name = service_name
        self.app_version = app_version
        self.start_time = time.time()
        
        # Initialize app info
        app_info.info({
            'version': app_version,
            'service_name': service_name,
            'environment': 'production'
        })
        
        logger.info(f"JARVIS Metrics collector initialized for service: {service_name}")
    
    def record_request(self, service: str, endpoint: str, method: str, status_code: int, duration: float):
        """Record HTTP request metrics (JARVIS requirement)"""
        # Record total requests
        jarvis_requests_total.labels(
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
        
        # Record latency histogram (JARVIS requirement)
        jarvis_latency_seconds_bucket.labels(
            service=service,
            endpoint=endpoint,
            method=method
        ).observe(duration)
        
        # Record errors if status code indicates error (JARVIS requirement)
        if status_code >= 400:
            error_type = "client_error" if status_code < 500 else "server_error"
            jarvis_errors_total.labels(
                service=service,
                endpoint=endpoint,
                error_type=error_type,
                error_code=str(status_code)
            ).inc()
    
    def record_model_inference(self, model_name: str, operation: str, duration: float, status: str = "success"):
        """Record model inference metrics"""
        model_inference_duration.labels(
            model_name=model_name,
            model_type=operation
        ).observe(duration)
        
        model_inference_requests.labels(
            model_name=model_name,
            model_type=operation
        ).inc()
    
    def record_agent_task(self, agent_name: str, task_type: str, duration: float, status: str = "success"):
        """Record agent task metrics"""
        agent_task_duration.labels(
            agent_name=agent_name,
            task_type=task_type
        ).observe(duration)
        
        if status == "success":
            agent_tasks_successful.labels(
                agent_name=agent_name,
                task_type=task_type
            ).inc()
        else:
            agent_tasks_failed.labels(
                agent_name=agent_name,
                task_type=task_type,
                error_type=status
            ).inc()
        
        agent_tasks_total.labels(
            agent_name=agent_name,
            task_type=task_type
        ).inc()
    
    def record_vector_db_operation(self, db_type: str, operation: str, collection: str, duration: float):
        """Record vector database operation metrics"""
        vector_db_operations.labels(
            db_type=db_type,
            operation=operation,
            collection=collection
        ).inc()
        
        vector_db_query_duration.labels(
            db_type=db_type,
            operation=operation,
            collection=collection
        ).observe(duration)
    
    def update_system_metrics(self):
        """Update system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage.set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)
            system_memory_total.set(memory.total)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            system_disk_usage.set(disk.used)
            system_disk_total.set(disk.total)
            
            # Uptime
            uptime = time.time() - self.start_time
            app_uptime_seconds.set(uptime)
            
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def update_service_health(self, service_name: str, service_type: str, is_healthy: bool):
        """Update service health status"""
        service_health_status.labels(
            service_name=service_name,
            service_type=service_type
        ).set(1 if is_healthy else 0)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        self.update_system_metrics()
        return generate_latest(REGISTRY)

# Global metrics collector instance
metrics_collector = MetricsCollector()

# =============================================================================
# MIDDLEWARE FOR AUTOMATIC INSTRUMENTATION
# =============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware to automatically instrument HTTP requests"""
    
    def __init__(self, app, service_name: str = "backend"):
        super().__init__(app)
        self.service_name = service_name
    
    async def dispatch(self, request: Request, call_next):
        """Process request and record JARVIS metrics"""
        start_time = time.time()
        
        # Extract endpoint and method
        endpoint = request.url.path
        method = request.method
        
        # Increment active requests
        api_active_requests.inc()
        
        try:
            # Call the actual request handler
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record JARVIS metrics
            metrics_collector.record_request(
                service=self.service_name,
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                duration=duration
            )
            
            # Record API metrics
            api_requests.labels(
                method=method,
                endpoint=endpoint,
                status=str(response.status_code)
            ).inc()
            
            api_request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            return response
        
        finally:
            # Decrement active requests
            api_active_requests.dec()

# =============================================================================
# DECORATORS FOR FUNCTION-LEVEL INSTRUMENTATION
# =============================================================================

def track_model_inference(model_name: str, operation: str = "inference"):
    """Decorator to track model inference metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_model_inference(model_name, operation, duration, status)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_model_inference(model_name, operation, duration, status)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def track_agent_task(agent_name: str, task_type: str = "general"):
    """Decorator to track agent task metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_agent_task(agent_name, task_type, duration, status)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                metrics_collector.record_agent_task(agent_name, task_type, duration, status)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# =============================================================================
# INITIALIZATION AND SETUP
# =============================================================================

def setup_metrics_endpoint(app):
    """Setup /metrics endpoint for Prometheus scraping"""
    @app.get("/metrics", response_class=Response)
    async def prometheus_metrics():
        """Prometheus metrics endpoint with JARVIS metrics"""
        metrics_data = metrics_collector.get_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    
    logger.info("JARVIS metrics endpoint configured at /metrics")

def initialize_metrics(app, service_name: str = "backend"):
    """Initialize JARVIS metrics collection for a FastAPI app"""
    # Add middleware for automatic request instrumentation
    app.add_middleware(PrometheusMiddleware, service_name=service_name)
    
    # Setup metrics endpoint
    setup_metrics_endpoint(app)
    
    # Update service name in global collector
    global metrics_collector
    metrics_collector.service_name = service_name
    
    logger.info(f"JARVIS metrics initialization completed for service: {service_name}")

import asyncio