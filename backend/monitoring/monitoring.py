"""
Monitoring Service Module for SutazAI AGI System

This module provides the MonitoringService expected by the main application,
wrapping the existing Prometheus metrics from backend.app.core.metrics.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from functools import wraps

# Import existing metrics
try:
    from backend.app.core.metrics import (
        active_agents,
        total_agents,
        agent_tasks_total,
        agent_tasks_successful,
        agent_tasks_failed,
        agent_task_duration,
        model_inference_requests,
        model_inference_duration,
        model_tokens_processed,
        vector_db_operations,
        vector_db_query_duration,
        vector_db_documents,
        api_requests,
        api_request_duration,
        api_active_requests,
        memory_usage,
        task_queue_size,
        task_processing_time,
        track_time,
        track_request
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    
    # Create dummy implementations if metrics not available
    class DummyMetric:
        def labels(self, **kwargs):
            return self
        def inc(self, amount=1):
            pass
        def dec(self, amount=1):
            pass
        def set(self, value):
            pass
        def observe(self, value):
            pass
    
    active_agents = DummyMetric()
    total_agents = DummyMetric()
    agent_tasks_total = DummyMetric()
    agent_tasks_successful = DummyMetric()
    agent_tasks_failed = DummyMetric()
    agent_task_duration = DummyMetric()
    model_inference_requests = DummyMetric()
    model_inference_duration = DummyMetric()
    model_tokens_processed = DummyMetric()
    api_requests = DummyMetric()
    api_request_duration = DummyMetric()
    api_active_requests = DummyMetric()
    memory_usage = DummyMetric()


class MonitoringService:
    """
    Monitoring service that provides system monitoring capabilities
    using Prometheus metrics.
    """
    
    def __init__(self, app_name: str = "sutazai-agi"):
        """Initialize the monitoring service."""
        self.app_name = app_name
        self._started = False
        self._events = []  # Store events for retrieval
        self._max_events = 10000  # Limit stored events
        
    def setup_app(self, app):
        """
        Set up monitoring for a FastAPI application.
        
        Args:
            app: FastAPI application instance
        """
        # Add Prometheus metrics endpoint if available
        if METRICS_AVAILABLE:
            try:
                from prometheus_client import make_asgi_app
                metrics_app = make_asgi_app()
                app.mount("/metrics", metrics_app)
            except ImportError:
                pass
        
        # Add monitoring endpoints
        from fastapi import APIRouter
        router = APIRouter(prefix="/monitoring")
        
        @router.get("/status")
        async def get_monitoring_status():
            return self.get_status()
        
        @router.get("/metrics/agents")
        async def get_agent_metrics():
            return {
                "active_agents": active_agents._value if hasattr(active_agents, '_value') else 0,
                "total_agents": total_agents._value if hasattr(total_agents, '_value') else 0
            }
        
        @router.get("/events")
        async def get_recent_events(limit: int = 100):
            return {"events": self._events[-limit:]}
        
        app.include_router(router)
        
        # Add middleware for request tracking
        @app.middleware("http")
        async def monitor_requests(request, call_next):
            start_time = time.time()
            api_active_requests.inc()
            
            try:
                response = await call_next(request)
                duration = time.time() - start_time
                
                api_requests.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                api_request_duration.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
                
                return response
            finally:
                api_active_requests.dec()
    
    def start(self):
        """Start monitoring service."""
        self._started = True
        
    def stop(self):
        """Stop monitoring service."""
        self._started = False
    
    def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log a monitoring event.
        
        Args:
            event_type: Type of event
            message: Event message
            severity: Severity level
            details: Additional details
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "message": message,
            "severity": severity,
            "details": details or {}
        }
        
        self._events.append(event)
        
        # Trim events if too many
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
        return {"status": "logged", "event_id": len(self._events)}
    
    def track_agent_task(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        duration: float,
        error_type: Optional[str] = None
    ):
        """
        Track an agent task execution.
        
        Args:
            agent_name: Name of the agent
            task_type: Type of task
            success: Whether task succeeded
            duration: Task duration in seconds
            error_type: Type of error if failed
        """
        agent_tasks_total.labels(agent_name=agent_name, task_type=task_type).inc()
        
        if success:
            agent_tasks_successful.labels(agent_name=agent_name, task_type=task_type).inc()
        else:
            agent_tasks_failed.labels(
                agent_name=agent_name,
                task_type=task_type,
                error_type=error_type or "unknown"
            ).inc()
        
        agent_task_duration.labels(agent_name=agent_name, task_type=task_type).observe(duration)
    
    def track_model_inference(
        self,
        model_name: str,
        model_type: str,
        duration: float,
        tokens: Optional[int] = None
    ):
        """
        Track a model inference.
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            duration: Inference duration in seconds
            tokens: Number of tokens processed
        """
        model_inference_requests.labels(model_name=model_name, model_type=model_type).inc()
        model_inference_duration.labels(model_name=model_name, model_type=model_type).observe(duration)
        
        if tokens:
            model_tokens_processed.labels(model_name=model_name, operation="inference").inc(tokens)
    
    def update_agent_count(self, active: int, total: int):
        """
        Update agent count metrics.
        
        Args:
            active: Number of active agents
            total: Total number of agents
        """
        active_agents.set(active)
        total_agents.set(total)
    
    def update_memory_usage(self, component: str, bytes_used: int):
        """
        Update memory usage for a component.
        
        Args:
            component: Component name
            bytes_used: Bytes of memory used
        """
        memory_usage.labels(component=component).set(bytes_used)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get monitoring service status.
        
        Returns:
            Status dictionary
        """
        return {
            "service": "monitoring",
            "status": "active" if self._started else "inactive",
            "app_name": self.app_name,
            "metrics_available": METRICS_AVAILABLE,
            "events_stored": len(self._events),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def monitor_model_inference(self, model_id: str, endpoint: str = "unknown"):
        """
        Decorator for monitoring model inferences.
        
        Args:
            model_id: Model identifier
            endpoint: API endpoint
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.track_model_inference(
                        model_name=model_id,
                        model_type="inference",
                        duration=duration
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_event(
                        event_type="model_inference_error",
                        message=f"Model inference failed: {str(e)}",
                        severity="error",
                        details={"model_id": model_id, "endpoint": endpoint, "duration": duration}
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self.track_model_inference(
                        model_name=model_id,
                        model_type="inference",
                        duration=duration
                    )
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self.log_event(
                        event_type="model_inference_error",
                        message=f"Model inference failed: {str(e)}",
                        severity="error",
                        details={"model_id": model_id, "endpoint": endpoint, "duration": duration}
                    )
                    raise
            
            import asyncio
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Export main components
__all__ = [
    'MonitoringService',
    'track_time',
    'track_request',
    'METRICS_AVAILABLE'
]