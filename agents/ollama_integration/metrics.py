"""
Agent Metrics Module for Prometheus Integration.
Provides standardized metrics collection for all SutazAI agents.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from functools import wraps
import time
import logging
from typing import Optional, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentMetrics:
    """
    Standardized metrics collection for SutazAI agents.
    
    Metrics:
    - request_count: Total number of requests processed
    - error_count: Total number of errors
    - queue_latency: Time spent waiting in queue
    - db_query_duration: Database query execution time
    - active_requests: Currently processing requests
    - last_request_timestamp: Unix timestamp of last request
    """
    
    def __init__(self, agent_name: str, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics for an agent.
        
        Args:
            agent_name: Name of the agent for metric labeling
            registry: Optional prometheus registry (defaults to global)
        """
        self.agent_name = agent_name
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'agent_request_total',
            'Total number of requests processed',
            ['agent', 'method', 'status'],
            registry=self.registry
        )
        
        self.error_count = Counter(
            'agent_error_total',
            'Total number of errors',
            ['agent', 'error_type'],
            registry=self.registry
        )
        
        # Latency metrics
        self.queue_latency = Histogram(
            'agent_queue_latency_seconds',
            'Time spent waiting in queue',
            ['agent'],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'agent_db_query_duration_seconds',
            'Database query execution time',
            ['agent', 'query_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'agent_processing_duration_seconds',
            'Request processing duration',
            ['agent', 'method'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self.registry
        )
        
        # Gauge metrics
        self.active_requests = Gauge(
            'agent_active_requests',
            'Number of requests currently being processed',
            ['agent'],
            registry=self.registry
        )
        
        self.last_request_timestamp = Gauge(
            'agent_last_request_timestamp',
            'Unix timestamp of last request',
            ['agent'],
            registry=self.registry
        )
        
        # Health metrics
        self.health_status = Gauge(
            'agent_health_status',
            'Agent health status (1=healthy, 0=unhealthy)',
            ['agent'],
            registry=self.registry
        )
        
        # Initialize gauges
        self.active_requests.labels(agent=self.agent_name).set(0)
        self.health_status.labels(agent=self.agent_name).set(1)
    
    def track_request(self, method: str = "unknown"):
        """
        Decorator to track request metrics.
        
        Args:
            method: Name of the method being tracked
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                # Record request start
                self.active_requests.labels(agent=self.agent_name).inc()
                self.last_request_timestamp.labels(agent=self.agent_name).set(time.time())
                
                start_time = time.time()
                status = "success"
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    error_type = type(e).__name__
                    self.error_count.labels(
                        agent=self.agent_name,
                        error_type=error_type
                    ).inc()
                    logger.error(f"Error in {method}: {e}")
                    raise
                finally:
                    # Record request completion
                    duration = time.time() - start_time
                    self.request_count.labels(
                        agent=self.agent_name,
                        method=method,
                        status=status
                    ).inc()
                    self.processing_duration.labels(
                        agent=self.agent_name,
                        method=method
                    ).observe(duration)
                    self.active_requests.labels(agent=self.agent_name).dec()
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                # Record request start
                self.active_requests.labels(agent=self.agent_name).inc()
                self.last_request_timestamp.labels(agent=self.agent_name).set(time.time())
                
                start_time = time.time()
                status = "success"
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "error"
                    error_type = type(e).__name__
                    self.error_count.labels(
                        agent=self.agent_name,
                        error_type=error_type
                    ).inc()
                    logger.error(f"Error in {method}: {e}")
                    raise
                finally:
                    # Record request completion
                    duration = time.time() - start_time
                    self.request_count.labels(
                        agent=self.agent_name,
                        method=method,
                        status=status
                    ).inc()
                    self.processing_duration.labels(
                        agent=self.agent_name,
                        method=method
                    ).observe(duration)
                    self.active_requests.labels(agent=self.agent_name).dec()
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def record_queue_latency(self, latency: float):
        """Record time spent waiting in queue."""
        self.queue_latency.labels(agent=self.agent_name).observe(latency)
    
    def record_db_query(self, query_type: str, duration: float):
        """Record database query duration."""
        self.db_query_duration.labels(
            agent=self.agent_name,
            query_type=query_type
        ).observe(duration)
    
    def set_health_status(self, healthy: bool):
        """Update agent health status."""
        self.health_status.labels(agent=self.agent_name).set(1 if healthy else 0)
    
    def increment_error(self, error_type: str = "unknown"):
        """Increment error counter."""
        self.error_count.labels(
            agent=self.agent_name,
            error_type=error_type
        ).inc()
    
    def generate_metrics(self) -> bytes:
        """Generate metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get the content type for metrics endpoint."""
        return CONTENT_TYPE_LATEST


# FastAPI integration helper
def setup_metrics_endpoint(app, metrics: AgentMetrics):
    """
    Add /metrics endpoint to FastAPI app.
    
    Args:
        app: FastAPI application instance
        metrics: AgentMetrics instance
    """
    from fastapi import Response
    
    @app.get("/metrics", include_in_schema=False)
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=metrics.generate_metrics(),
            media_type=metrics.get_content_type()
        )
    
    logger.info(f"Metrics endpoint configured for {metrics.agent_name}")


# Context manager for timing operations
class MetricsTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: AgentMetrics, operation: str, labels: dict = None):
        self.metrics = metrics
        self.operation = operation
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if self.operation == "queue":
            self.metrics.record_queue_latency(duration)
        elif self.operation == "db_query":
            query_type = self.labels.get("query_type", "unknown")
            self.metrics.record_db_query(query_type, duration)