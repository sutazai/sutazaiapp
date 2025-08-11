"""
Adaptive Timeout Manager for SutazAI Frontend
Handles backend startup delays and varying response times
"""

import time
import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class SystemState(Enum):
    STARTUP = "startup"      # System is starting up (3+ minute timeouts)
    WARMUP = "warmup"        # Models warming up (60+ second timeouts) 
    HEALTHY = "healthy"      # Normal operation (10-30 second timeouts)
    DEGRADED = "degraded"    # Partial failures (45+ second timeouts)
    FAILED = "failed"        # System down (quick timeouts, circuit open)

class AdaptiveTimeoutManager:
    """Intelligent timeout management based on system state"""
    
    def __init__(self):
        self.current_state = SystemState.STARTUP
        self.state_start_time = time.time()
        self.last_successful_call = 0
        self.consecutive_failures = 0
        self.system_start_time = time.time()
        
        # Timeout configurations by system state
        self.timeout_configs = {
            SystemState.STARTUP: {
                "health_check": 5.0,        # Short health checks during startup
                "api_call": 180.0,          # Long API timeouts for startup
                "model_request": 240.0,     # Very long for model loading
                "retry_interval": 10.0      # Retry every 10 seconds
            },
            SystemState.WARMUP: {
                "health_check": 3.0,
                "api_call": 60.0,           # Medium timeouts during warmup
                "model_request": 120.0,     # Long for model warmup
                "retry_interval": 5.0
            },
            SystemState.HEALTHY: {
                "health_check": 2.0,
                "api_call": 10.0,           # Standard timeouts
                "model_request": 30.0,      # Normal model response time
                "retry_interval": 30.0      # Less frequent health checks
            },
            SystemState.DEGRADED: {
                "health_check": 5.0,        # Longer timeouts for degraded system
                "api_call": 45.0,
                "model_request": 90.0,
                "retry_interval": 15.0
            },
            SystemState.FAILED: {
                "health_check": 1.0,        # Quick failure detection
                "api_call": 2.0,            # Fail fast
                "model_request": 2.0,
                "retry_interval": 60.0      # Less frequent retries
            }
        }
    
    def get_timeout(self, operation_type: str) -> float:
        """Get appropriate timeout for operation based on current system state"""
        config = self.timeout_configs[self.current_state]
        return config.get(operation_type, config["api_call"])
    
    def record_success(self, operation_type: str, response_time: float):
        """Record successful operation and update system state"""
        self.last_successful_call = time.time()
        self.consecutive_failures = 0
        
        # Auto-transition to healthier states based on success patterns
        if self.current_state == SystemState.STARTUP:
            # After first success, move to warmup
            if time.time() - self.system_start_time > 30:  # At least 30s uptime
                self._transition_state(SystemState.WARMUP)
        
        elif self.current_state == SystemState.WARMUP:
            # After consistent fast responses, move to healthy
            if response_time < 5.0 and time.time() - self.state_start_time > 60:
                self._transition_state(SystemState.HEALTHY)
        
        elif self.current_state == SystemState.DEGRADED:
            # Recovery to healthy after good performance
            if response_time < 10.0 and time.time() - self.state_start_time > 120:
                self._transition_state(SystemState.HEALTHY)
        
        elif self.current_state == SystemState.FAILED:
            # Recovery from failure
            self._transition_state(SystemState.DEGRADED)
        
        logger.debug(f"Recorded success for {operation_type}: {response_time:.2f}s")
    
    def record_failure(self, operation_type: str, error_type: str):
        """Record failed operation and update system state"""
        self.consecutive_failures += 1
        
        # State transitions based on failure patterns
        if self.consecutive_failures >= 5:
            if self.current_state in [SystemState.HEALTHY, SystemState.WARMUP]:
                self._transition_state(SystemState.DEGRADED)
            elif self.current_state == SystemState.DEGRADED:
                self._transition_state(SystemState.FAILED)
        
        elif self.consecutive_failures >= 10:
            self._transition_state(SystemState.FAILED)
        
        logger.warning(f"Recorded failure for {operation_type}: {error_type} (consecutive: {self.consecutive_failures})")
    
    def _transition_state(self, new_state: SystemState):
        """Transition to new system state"""
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            self.state_start_time = time.time()
            
            logger.info(f"System state transition: {old_state.value} → {new_state.value}")
    
    def get_retry_interval(self) -> float:
        """Get appropriate retry interval for current state"""
        return self.timeout_configs[self.current_state]["retry_interval"]
    
    def should_attempt_request(self) -> bool:
        """Check if we should attempt a request based on current state"""
        if self.current_state == SystemState.FAILED:
            # Only retry periodically when in failed state
            time_since_last_attempt = time.time() - getattr(self, '_last_attempt', 0)
            return time_since_last_attempt > self.get_retry_interval()
        
        return True
    
    def get_state_info(self) -> Dict:
        """Get current state information for UI display"""
        return {
            "state": self.current_state.value,
            "state_duration": time.time() - self.state_start_time,
            "consecutive_failures": self.consecutive_failures,
            "last_success": time.time() - self.last_successful_call if self.last_successful_call else None,
            "timeouts": self.timeout_configs[self.current_state]
        }

# Global timeout manager instance
timeout_manager = AdaptiveTimeoutManager()

def get_timeout_for_operation(operation_type: str) -> float:
    """Convenience function to get timeout for operation"""
    return timeout_manager.get_timeout(operation_type)

def record_operation_result(operation_type: str, success: bool, 
                          response_time: float = None, error_type: str = None):
    """Convenience function to record operation result"""
    if success:
        timeout_manager.record_success(operation_type, response_time or 0)
    else:
        timeout_manager.record_failure(operation_type, error_type or "unknown")