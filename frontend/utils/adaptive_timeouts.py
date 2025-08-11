"""
Adaptive Timeout Management for SutazAI Frontend
Dynamic timeout adjustment based on system state and performance
"""

import time
import logging
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System state enumeration"""
    STARTUP = "startup"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class TimeoutConfig:
    """Timeout configuration for different operations"""
    base_timeout: float
    max_timeout: float
    multiplier: float = 1.5
    
class AdaptiveTimeoutManager:
    """Manages adaptive timeouts based on system state and performance"""
    
    def __init__(self):
        self.current_state = SystemState.UNKNOWN
        self.state_start_time = time.time()
        self.response_times = {}  # operation -> list of recent response times
        self.max_history = 10
        
        # Default timeout configurations
        self.configs = {
            "health_check": TimeoutConfig(2.0, 15.0),
            "api_call": TimeoutConfig(5.0, 30.0),
            "chat_request": TimeoutConfig(10.0, 45.0),
            "model_load": TimeoutConfig(30.0, 180.0)
        }
    
    def set_system_state(self, state: SystemState) -> None:
        """Update current system state"""
        if self.current_state != state:
            logger.info(f"System state changed: {self.current_state} -> {state}")
            self.current_state = state
            self.state_start_time = time.time()
    
    def get_timeout(self, operation: str) -> float:
        """Get adaptive timeout for operation"""
        config = self.configs.get(operation, TimeoutConfig(5.0, 30.0))
        base_timeout = config.base_timeout
        
        # Adjust based on system state
        if self.current_state == SystemState.STARTUP:
            # Longer timeouts during startup
            timeout = base_timeout * 3
        elif self.current_state == SystemState.DEGRADED:
            # Moderate increase for degraded performance
            timeout = base_timeout * 2
        elif self.current_state == SystemState.FAILED:
            # Short timeouts to fail fast
            timeout = base_timeout * 0.5
        else:
            # Normal timeout
            timeout = base_timeout
        
        # Adjust based on recent performance
        recent_times = self.response_times.get(operation, [])
        if recent_times:
            avg_response = sum(recent_times) / len(recent_times)
            # Add buffer based on average response time
            timeout = max(timeout, avg_response * 1.5)
        
        return min(timeout, config.max_timeout)
    
    def record_response_time(self, operation: str, response_time: float) -> None:
        """Record response time for adaptive learning"""
        if operation not in self.response_times:
            self.response_times[operation] = []
        
        times = self.response_times[operation]
        times.append(response_time)
        
        # Keep only recent history
        if len(times) > self.max_history:
            times.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timeout manager statistics"""
        return {
            "current_state": self.current_state.value,
            "state_duration": time.time() - self.state_start_time,
            "timeouts": {
                op: self.get_timeout(op)
                for op in self.configs.keys()
            },
            "avg_response_times": {
                op: sum(times) / len(times) if times else 0
                for op, times in self.response_times.items()
            }
        }

# Global timeout manager instance
timeout_manager = AdaptiveTimeoutManager()