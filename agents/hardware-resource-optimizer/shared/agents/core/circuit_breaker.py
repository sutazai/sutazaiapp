#!/usr/bin/env python3
"""
Circuit Breaker Pattern for SutazAI System
Provides resilience and fault tolerance for Ollama interactions

Features:
- Configurable failure thresholds and recovery timeouts
- Automatic state transitions (CLOSED -> OPEN -> HALF_OPEN)
- Request metrics and failure tracking
- Async-first design with proper exception handling
- Graceful degradation for limited hardware environments
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import functools
import inspect


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changed_at: datetime = field(default_factory=datetime.utcnow)
    trip_count: int = 0
    average_response_time: float = 0.0


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, failure_count: int):
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        super().__init__(f"Circuit breaker '{circuit_name}' is OPEN after {failure_count} failures")


class CircuitBreakerTimeoutException(Exception):
    """Exception raised when circuit breaker times out"""
    def __init__(self, circuit_name: str, timeout: float):
        self.circuit_name = circuit_name
        self.timeout = timeout
        super().__init__(f"Circuit breaker '{circuit_name}' timed out after {timeout}s")


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    
    Designed for resource-constrained environments with focus on:
    - Fast failure detection
    - Automatic recovery testing
    -   resource overhead
    - Comprehensive metrics tracking
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Union[Type[Exception], tuple] = Exception,
                 timeout: Optional[float] = None,
                 name: Optional[str] = None):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before testing recovery (seconds)
            expected_exception: Exception types that count as failures
            timeout: Request timeout (optional)
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.timeout = timeout
        self.name = name or "circuit_breaker"
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        
        # Async synchronization
        self._lock = asyncio.Lock()
        self._state_change_callbacks = []
        
        logger.info(f"Circuit breaker '{self.name}' initialized: threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Add callback for state changes"""
        self._state_change_callbacks.append(callback)
    
    async def _change_state(self, new_state: CircuitState):
        """Change circuit breaker state"""
        old_state = self.state
        if old_state != new_state:
            self.state = new_state
            self.metrics.state_changed_at = datetime.utcnow()
            
            if new_state == CircuitState.OPEN:
                self.metrics.trip_count += 1
            
            logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    if inspect.iscoroutinefunction(callback):
                        await callback(old_state, new_state)
                    else:
                        callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    async def _record_success(self, response_time: float):
        """Record successful request"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = datetime.utcnow()
            
            # Update average response time
            if self.metrics.successful_requests == 1:
                self.metrics.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1  # Smoothing factor
                self.metrics.average_response_time = (
                    alpha * response_time + 
                    (1 - alpha) * self.metrics.average_response_time
                )
            
            # If we were in HALF_OPEN and got a success, close the circuit
            if self.state == CircuitState.HALF_OPEN:
                await self._change_state(CircuitState.CLOSED)
    
    async def _record_failure(self, exception: Exception):
        """Record failed request"""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = datetime.utcnow()
            
            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {type(exception).__name__}: {exception}")
            
            # Check if we should open the circuit
            if (self.state == CircuitState.CLOSED and 
                self.metrics.consecutive_failures >= self.failure_threshold):
                await self._change_state(CircuitState.OPEN)
            
            # If we were in HALF_OPEN and got a failure, go back to OPEN
            elif self.state == CircuitState.HALF_OPEN:
                await self._change_state(CircuitState.OPEN)
    
    async def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.state != CircuitState.OPEN:
            return False
        
        if not self.metrics.last_failure_time:
            return False
        
        time_since_failure = datetime.utcnow() - self.metrics.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute (can be async or sync)
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result of func execution
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            CircuitBreakerTimeoutException: When request times out
            Exception: Original exception from func
        """
        # Check if circuit should be reset
        if await self._should_attempt_reset():
            async with self._lock:
                if self.state == CircuitState.OPEN:
                    await self._change_state(CircuitState.HALF_OPEN)
        
        # Reject request if circuit is open
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpenException(self.name, self.metrics.consecutive_failures)
        
        start_time = time.time()
        
        try:
            # Execute the function with optional timeout
            if self.timeout:
                if inspect.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
                else:
                    # For sync functions, run in executor with timeout
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, functools.partial(func, *args, **kwargs)),
                        timeout=self.timeout
                    )
            else:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # For sync functions, run in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
            
            response_time = time.time() - start_time
            await self._record_success(response_time)
            return result
            
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            timeout_exception = CircuitBreakerTimeoutException(self.name, self.timeout)
            await self._record_failure(timeout_exception)
            raise timeout_exception
            
        except Exception as e:
            # Only count expected exceptions as failures
            if isinstance(e, self.expected_exception):
                await self._record_failure(e)
            
            raise e
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator version of circuit breaker
        
        Usage:
            @circuit_breaker
            async def my_function():
                pass
        """
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            async def sync_wrapper(*args, **kwargs):
                return await self.call(func, *args, **kwargs)
            return sync_wrapper
    
    async def reset(self):
        """Manually reset the circuit breaker"""
        async with self._lock:
            old_state = self.state
            await self._change_state(CircuitState.CLOSED)
            self.metrics.consecutive_failures = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset from {old_state.value}")
    
    async def force_open(self):
        """Manually open the circuit breaker"""
        async with self._lock:
            await self._change_state(CircuitState.OPEN)
            logger.info(f"Circuit breaker '{self.name}' manually opened")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.successful_requests / self.metrics.total_requests
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "consecutive_failures": self.metrics.consecutive_failures,
            "success_rate": success_rate,
            "trip_count": self.metrics.trip_count,
            "average_response_time": self.metrics.average_response_time,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "state_changed_at": self.metrics.state_changed_at.isoformat(),
            "time_until_retry": self._get_time_until_retry()
        }
    
    def _get_time_until_retry(self) -> Optional[float]:
        """Get time remaining until retry attempt"""
        if self.state != CircuitState.OPEN or not self.metrics.last_failure_time:
            return None
        
        time_since_failure = datetime.utcnow() - self.metrics.last_failure_time
        time_remaining = self.recovery_timeout - time_since_failure.total_seconds()
        
        return max(0.0, time_remaining)
    
    @property
    def is_healthy(self) -> bool:
        """Check if circuit breaker is in a healthy state"""
        return self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]
    
    @property
    def failure_rate(self) -> float:
        """Get current failure rate"""
        if self.metrics.total_requests == 0:
            return 0.0
        return self.metrics.failed_requests / self.metrics.total_requests
    
    def __repr__(self) -> str:
        return (f"CircuitBreaker(name='{self.name}', state={self.state.value}, "
                f"failures={self.metrics.consecutive_failures}/{self.failure_threshold})")


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers
    
    Useful for managing circuit breakers for different services or endpoints
    """
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_breaker(self, 
                         name: str,
                         failure_threshold: int = 5,
                         recovery_timeout: float = 60.0,
                         **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    name=name,
                    **kwargs
                )
            return self._breakers[name]
    
    async def call(self, breaker_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with named circuit breaker"""
        breaker = await self.get_breaker(breaker_name)
        return await breaker.call(func, *args, **kwargs)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            await breaker.reset()
    
    def __len__(self) -> int:
        return len(self._breakers)
    
    def __contains__(self, name: str) -> bool:
        return name in self._breakers


# Factory functions for common use cases
def create_ollama_circuit_breaker(failure_threshold: int = 3,
                                 recovery_timeout: float = 30.0,
                                 timeout: float = 60.0) -> CircuitBreaker:
    """Create circuit breaker optimized for Ollama requests"""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        timeout=timeout,
        expected_exception=(Exception,),  # Count all exceptions as failures
        name="ollama_circuit_breaker"
    )


def create_api_circuit_breaker(failure_threshold: int = 5,
                              recovery_timeout: float = 60.0,
                              timeout: float = 30.0) -> CircuitBreaker:
    """Create circuit breaker optimized for API requests"""
    import httpx
    
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        timeout=timeout,
        expected_exception=(httpx.RequestError, httpx.HTTPStatusError),
        name="api_circuit_breaker"
    )


# Global circuit breaker manager instance
_global_manager = CircuitBreakerManager()


async def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get a circuit breaker from the global manager"""
    return await _global_manager.get_breaker(name, **kwargs)


async def call_with_circuit_breaker(breaker_name: str, func: Callable, *args, **kwargs) -> Any:
    """Execute function with named circuit breaker from global manager"""
    return await _global_manager.call(breaker_name, func, *args, **kwargs)