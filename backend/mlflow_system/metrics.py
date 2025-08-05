"""
Prometheus metrics for MLflow tracking system
Monitors experiment tracking performance and system health
"""

from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry
import time
from functools import wraps
from typing import Callable, Dict, Any
import asyncio


class MLflowMetrics:
    """Prometheus metrics for MLflow system"""
    
    def __init__(self, registry: CollectorRegistry = None):
        self.registry = registry or CollectorRegistry()
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'mlflow_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'mlflow_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'mlflow_disk_usage_percent',
            'Disk usage percentage for artifacts',
            registry=self.registry
        )
        
        # Experiment metrics
        self.active_experiments = Gauge(
            'mlflow_active_experiments_total',
            'Number of active experiments',
            registry=self.registry
        )
        
        self.total_experiments = Gauge(
            'mlflow_total_experiments',
            'Total number of experiments created',
            registry=self.registry
        )
        
        self.experiments_created = Counter(
            'mlflow_experiments_created_total',
            'Total number of experiments created',
            ['agent_type', 'experiment_type'],
            registry=self.registry
        )
        
        # Run metrics
        self.active_runs = Gauge(
            'mlflow_active_runs_total',
            'Number of currently active runs',
            registry=self.registry
        )
        
        self.runs_started = Counter(
            'mlflow_runs_started_total',
            'Total number of runs started',
            ['experiment_name', 'agent_id'],
            registry=self.registry
        )
        
        self.runs_completed = Counter(
            'mlflow_runs_completed_total',
            'Total number of runs completed',
            ['experiment_name', 'agent_id', 'status'],
            registry=self.registry
        )
        
        self.run_duration = Histogram(
            'mlflow_run_duration_seconds',
            'Duration of completed runs',
            ['experiment_name', 'agent_id'],
            buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200),
            registry=self.registry
        )
        
        # Logging metrics
        self.metrics_logged = Counter(
            'mlflow_metrics_logged_total',
            'Total number of metrics logged',
            ['experiment_name', 'metric_name'],
            registry=self.registry
        )
        
        self.params_logged = Counter(
            'mlflow_params_logged_total',
            'Total number of parameters logged',
            ['experiment_name'],
            registry=self.registry
        )
        
        self.artifacts_logged = Counter(
            'mlflow_artifacts_logged_total',
            'Total number of artifacts logged',
            ['experiment_name', 'artifact_type'],
            registry=self.registry
        )
        
        self.logging_duration = Histogram(
            'mlflow_logging_duration_seconds',
            'Duration of logging operations',
            ['operation_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self.logging_batch_size = Histogram(
            'mlflow_logging_batch_size',
            'Size of batch logging operations',
            ['operation_type'],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
            registry=self.registry
        )
        
        # Model registry metrics
        self.models_registered = Counter(
            'mlflow_models_registered_total',
            'Total number of models registered',
            ['model_name', 'agent_type'],
            registry=self.registry
        )
        
        self.model_versions_created = Counter(
            'mlflow_model_versions_created_total',
            'Total number of model versions created',
            ['model_name', 'stage'],
            registry=self.registry
        )
        
        self.model_transitions = Counter(
            'mlflow_model_transitions_total',
            'Total number of model stage transitions',
            ['model_name', 'from_stage', 'to_stage'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections_active = Gauge(
            'mlflow_database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.database_operations = Counter(
            'mlflow_database_operations_total',
            'Total number of database operations',
            ['operation_type', 'table_name'],
            registry=self.registry
        )
        
        self.database_operation_duration = Histogram(
            'mlflow_database_operation_duration_seconds',
            'Duration of database operations',
            ['operation_type', 'table_name'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self.registry
        )
        
        self.database_errors = Counter(
            'mlflow_database_errors_total',
            'Total number of database errors',
            ['error_type'],
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'mlflow_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'mlflow_api_request_duration_seconds',
            'Duration of API requests',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.api_concurrent_requests = Gauge(
            'mlflow_api_concurrent_requests',
            'Number of concurrent API requests',
            registry=self.registry
        )
        
        # Agent-specific metrics
        self.agent_experiments = Gauge(
            'mlflow_agent_experiments_active',
            'Number of active experiments per agent',
            ['agent_id', 'agent_type'],
            registry=self.registry
        )
        
        self.agent_logging_rate = Gauge(
            'mlflow_agent_logging_rate_per_second',
            'Rate of logging operations per agent',
            ['agent_id', 'operation_type'],
            registry=self.registry
        )
        
        # Artifact metrics
        self.artifacts_size_bytes = Histogram(
            'mlflow_artifacts_size_bytes',
            'Size of logged artifacts in bytes',
            ['experiment_name', 'artifact_type'],
            buckets=(1024, 10240, 102400, 1048576, 10485760, 104857600, 1073741824),
            registry=self.registry
        )
        
        self.artifacts_cleaned = Counter(
            'mlflow_artifacts_cleaned_total',
            'Total number of artifacts cleaned up',
            registry=self.registry
        )
        
        # Performance metrics
        self.tracking_server_uptime = Gauge(
            'mlflow_tracking_server_uptime_seconds',
            'Tracking server uptime in seconds',
            registry=self.registry
        )
        
        self.system_info = Info(
            'mlflow_system_info',
            'MLflow system information',
            registry=self.registry
        )
        
        # Initialize system info
        self.system_info.info({
            'version': '2.9.2',
            'backend_store': 'postgresql',
            'artifact_store': 'local',
            'system': 'sutazai'
        })
    
    def track_experiment_operation(self, operation_type: str, experiment_name: str = "", agent_id: str = ""):
        """Decorator to track experiment operations"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success metrics
                    if operation_type == "create_experiment":
                        self.experiments_created.labels(
                            agent_type=kwargs.get('agent_type', 'unknown'),
                            experiment_type=kwargs.get('experiment_type', 'default')
                        ).inc()
                    elif operation_type == "start_run":
                        self.runs_started.labels(
                            experiment_name=experiment_name,
                            agent_id=agent_id
                        ).inc()
                    elif operation_type == "end_run":
                        self.runs_completed.labels(
                            experiment_name=experiment_name,
                            agent_id=agent_id,
                            status=kwargs.get('status', 'FINISHED')
                        ).inc()
                    
                    return result
                    
                finally:
                    duration = time.time() - start_time
                    self.logging_duration.labels(operation_type=operation_type).observe(duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    # Record success metrics (same logic as async)
                    if operation_type == "create_experiment":
                        self.experiments_created.labels(
                            agent_type=kwargs.get('agent_type', 'unknown'),
                            experiment_type=kwargs.get('experiment_type', 'default')
                        ).inc()
                    
                    return result
                    
                finally:
                    duration = time.time() - start_time
                    self.logging_duration.labels(operation_type=operation_type).observe(duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def track_logging_operation(self, operation_type: str):
        """Decorator to track logging operations"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                batch_size = 1
                
                try:
                    # Try to determine batch size
                    if 'metrics' in kwargs and isinstance(kwargs['metrics'], list):
                        batch_size = len(kwargs['metrics'])
                    elif 'params' in kwargs and isinstance(kwargs['params'], dict):
                        batch_size = len(kwargs['params'])
                    elif args and hasattr(args[0], '__len__'):
                        batch_size = len(args[0])
                    
                    result = await func(*args, **kwargs)
                    
                    # Record metrics
                    if operation_type == "log_metrics":
                        experiment_name = kwargs.get('experiment_name', 'unknown')
                        metric_name = kwargs.get('metric_name', 'unknown')
                        self.metrics_logged.labels(
                            experiment_name=experiment_name,
                            metric_name=metric_name
                        ).inc(batch_size)
                    elif operation_type == "log_params":
                        experiment_name = kwargs.get('experiment_name', 'unknown')
                        self.params_logged.labels(experiment_name=experiment_name).inc(batch_size)
                    elif operation_type == "log_artifacts":
                        experiment_name = kwargs.get('experiment_name', 'unknown')
                        artifact_type = kwargs.get('artifact_type', 'unknown')
                        self.artifacts_logged.labels(
                            experiment_name=experiment_name,
                            artifact_type=artifact_type
                        ).inc(batch_size)
                    
                    return result
                    
                finally:
                    duration = time.time() - start_time
                    self.logging_duration.labels(operation_type=operation_type).observe(duration)
                    self.logging_batch_size.labels(operation_type=operation_type).observe(batch_size)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                batch_size = 1
                
                try:
                    # Try to determine batch size
                    if 'metrics' in kwargs and isinstance(kwargs['metrics'], list):
                        batch_size = len(kwargs['metrics'])
                    elif 'params' in kwargs and isinstance(kwargs['params'], dict):
                        batch_size = len(kwargs['params'])
                    
                    result = func(*args, **kwargs)
                    return result
                    
                finally:
                    duration = time.time() - start_time
                    self.logging_duration.labels(operation_type=operation_type).observe(duration)
                    self.logging_batch_size.labels(operation_type=operation_type).observe(batch_size)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def track_database_operation(self, operation_type: str, table_name: str = ""):
        """Decorator to track database operations"""
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    self.database_operations.labels(
                        operation_type=operation_type,
                        table_name=table_name
                    ).inc()
                    return result
                except Exception as e:
                    self.database_errors.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.database_operation_duration.labels(
                        operation_type=operation_type,
                        table_name=table_name
                    ).observe(duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.database_operations.labels(
                        operation_type=operation_type,
                        table_name=table_name
                    ).inc()
                    return result
                except Exception as e:
                    self.database_errors.labels(error_type=type(e).__name__).inc()
                    raise
                finally:
                    duration = time.time() - start_time
                    self.database_operation_duration.labels(
                        operation_type=operation_type,
                        table_name=table_name
                    ).observe(duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            'experiments': {
                'active': self.active_experiments._value._value if hasattr(self.active_experiments._value, '_value') else 0,
                'total_created': sum(self.experiments_created._value.values()) if hasattr(self.experiments_created, '_value') else 0
            },
            'runs': {
                'active': self.active_runs._value._value if hasattr(self.active_runs._value, '_value') else 0,
                'total_started': sum(self.runs_started._value.values()) if hasattr(self.runs_started, '_value') else 0
            },
            'system': {
                'cpu_usage': self.system_cpu_usage._value._value if hasattr(self.system_cpu_usage._value, '_value') else 0,
                'memory_usage': self.system_memory_usage._value._value if hasattr(self.system_memory_usage._value, '_value') else 0
            }
        }


# Global metrics instance
mlflow_metrics = MLflowMetrics()