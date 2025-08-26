"""
Circuit Breaker Pattern Implementation for Resilient Service Communication
Provides fault tolerance for external service calls with automatic recovery
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Circuit tripped, requests fail immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    def __init__(self, service_name: str, message: str = None):
        self.service_name = service_name
        super().__init__(message or f"Circuit breaker is OPEN for service: {service_name}")


class CircuitBreakerMetrics:
    """Tracks circuit breaker metrics for monitoring"""
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.circuit_opens = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        
    def record_success(self):
        """Record a successful call"""
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        
    def record_failure(self):
        """Record a failed call"""
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.last_failure_time = datetime.now()
        
    def record_circuit_open(self):
        """Record when circuit opens"""
        self.circuit_opens += 1
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics as dictionary"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'success_rate': f"{success_rate:.1f}%",
            'circuit_opens': self.circuit_opens,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success': self.last_success_time.isoformat() if self.last_success_time else None
        }


class CircuitBreaker:
    """
    Circuit Breaker implementation for fault tolerance
    
    Configuration:
    - failure_threshold: Number of consecutive failures before opening circuit (default: 5)
    - recovery_timeout: Seconds to wait before attempting recovery (default: 30)
    - expected_exception: Exception types to count as failures
    - exclude_exceptions: Exception types to propagate without counting
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: tuple = (Exception,),
        exclude_exceptions: tuple = (),
        success_threshold: int = 1,  # Successes needed to close from half-open
        on_state_change: Optional[Callable] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.exclude_exceptions = exclude_exceptions
        self.success_threshold = success_threshold
        self.on_state_change = on_state_change
        
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None
        self.metrics = CircuitBreakerMetrics()
        self._lock = asyncio.Lock()
        self._half_open_request_in_progress = False
        
        logger.info(f"Circuit breaker '{name}' initialized: "
                   f"failure_threshold={failure_threshold}, "
                   f"recovery_timeout={recovery_timeout}s")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection
        
        Args:
            func: The function to call (can be async or sync)
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the function call
        """
        async with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to_half_open()
                else:
                    raise CircuitBreakerError(
                        self.name, 
                        f"Circuit breaker is OPEN. Will retry in "
                        f"{self.recovery_timeout - (time.time() - self.last_failure_time):.1f}s"
                    )
            
            # If HALF_OPEN and another request is in progress, fail fast
            if self.state == CircuitState.HALF_OPEN and self._half_open_request_in_progress:
                raise CircuitBreakerError(
                    self.name,
                    "Circuit breaker is HALF_OPEN and testing another request"
                )
        
        # Mark that we're testing in HALF_OPEN state
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_request_in_progress = True
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            await self._on_success()
            return result
            
        except self.exclude_exceptions:
            # Don't count these exceptions as failures
            raise
            
        except self.expected_exception as e:
            # Record failure
            await self._on_failure()
            raise
            
        finally:
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_request_in_progress = False
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to HALF_OPEN")
        
        if self.on_state_change:
            try:
                self.on_state_change(self.name, old_state, self.state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.metrics.record_success()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.consecutive_successes >= self.success_threshold:
                    # Close the circuit
                    old_state = self.state
                    self.state = CircuitState.CLOSED
                    self.metrics.consecutive_failures = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to CLOSED "
                              f"after {self.success_threshold} successful call(s)")
                    
                    if self.on_state_change:
                        try:
                            self.on_state_change(self.name, old_state, self.state)
                        except Exception as e:
                            logger.error(f"Error in state change callback: {e}")
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.metrics.record_failure()
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failed in HALF_OPEN, go back to OPEN
                old_state = self.state
                self.state = CircuitState.OPEN
                self.metrics.record_circuit_open()
                logger.warning(f"Circuit breaker '{self.name}' transitioned from HALF_OPEN to OPEN "
                             f"after failure in recovery attempt")
                
                if self.on_state_change:
                    try:
                        self.on_state_change(self.name, old_state, self.state)
                    except Exception as e:
                        logger.error(f"Error in state change callback: {e}")
                        
            elif self.state == CircuitState.CLOSED:
                if self.metrics.consecutive_failures >= self.failure_threshold:
                    # Open the circuit
                    old_state = self.state
                    self.state = CircuitState.OPEN
                    self.metrics.record_circuit_open()
                    logger.warning(f"Circuit breaker '{self.name}' transitioned from CLOSED to OPEN "
                                 f"after {self.failure_threshold} consecutive failures")
                    
                    if self.on_state_change:
                        try:
                            self.on_state_change(self.name, old_state, self.state)
                        except Exception as e:
                            logger.error(f"Error in state change callback: {e}")
    
    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'metrics': self.metrics.get_stats()
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.metrics = CircuitBreakerMetrics()
        logger.info(f"Circuit breaker '{self.name}' has been reset")


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._breakers: Dict[str, CircuitBreaker] = {}
            self._global_metrics = {
                'total_circuits': 0,
                'open_circuits': 0,
                'half_open_circuits': 0,
                'closed_circuits': 0
            }
    
    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        **kwargs
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                on_state_change=self._on_breaker_state_change,
                **kwargs
            )
            self._update_global_metrics()
        return self._breakers[name]
    
    def _on_breaker_state_change(self, name: str, old_state: CircuitState, new_state: CircuitState):
        """Callback for circuit breaker state changes"""
        self._update_global_metrics()
        logger.info(f"Circuit breaker '{name}' state changed: {old_state.value} -> {new_state.value}")
    
    def _update_global_metrics(self):
        """Update global circuit breaker metrics"""
        self._global_metrics['total_circuits'] = len(self._breakers)
        self._global_metrics['open_circuits'] = sum(
            1 for b in self._breakers.values() if b.state == CircuitState.OPEN
        )
        self._global_metrics['half_open_circuits'] = sum(
            1 for b in self._breakers.values() if b.state == CircuitState.HALF_OPEN
        )
        self._global_metrics['closed_circuits'] = sum(
            1 for b in self._breakers.values() if b.state == CircuitState.CLOSED
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        return {
            'global': self._global_metrics,
            'breakers': {
                name: breaker.get_metrics()
                for name, breaker in self._breakers.items()
            }
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers have been reset")
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a specific circuit breaker by name"""
        return self._breakers.get(name)


# Global singleton instance
_breaker_manager: Optional[CircuitBreakerManager] = None


def get_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager"""
    global _breaker_manager
    if _breaker_manager is None:
        _breaker_manager = CircuitBreakerManager()
    return _breaker_manager


# Decorator for easy circuit breaker application
def with_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    **kwargs
):
    """
    Decorator to apply circuit breaker pattern to a function
    
    Usage:
        @with_circuit_breaker("external_api", failure_threshold=3)
        async def call_external_api():
            # Your code here
            pass
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **func_kwargs):
            manager = get_breaker_manager()
            breaker = manager.get_or_create(name, failure_threshold, recovery_timeout, **kwargs)
            return await breaker.call(func, *args, **func_kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **func_kwargs):
            manager = get_breaker_manager()
            breaker = manager.get_or_create(name, failure_threshold, recovery_timeout, **kwargs)
            return asyncio.run(breaker.call(func, *args, **func_kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator