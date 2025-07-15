"""
Performance Profiler
Basic performance profiling functionality
"""

import time
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Basic performance profiler"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "cpu_usage": 25.0,  # Mock data
            "memory_usage": 45.0,  # Mock data
            "uptime": time.time() - self.start_time,
            "timestamp": datetime.now().isoformat()
        }
    
    def profile_function(self, func_name: str, duration: float):
        """Profile a function"""
        self.metrics[func_name] = {
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_profile_data(self) -> Dict[str, Any]:
        """Get profiling data"""
        return self.metrics.copy()