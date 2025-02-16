import time
import psutil
import logging
from typing import Dict, Any
from functools import wraps

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'execution_times': {}
        }
    
    def track_performance(self, metric_type: str, value: float):
        """Track performance metrics"""
        self.metrics[metric_type].append(value)
    
    def measure_execution_time(self, func):
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            self.metrics['execution_times'][func.__name__] = execution_time
            logging.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            return result
        return wrapper
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Collect current system resource usage"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        } 