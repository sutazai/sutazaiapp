"""Circuit Breaker Pattern Implementation
Prevents cascading failures by stopping calls to failing services
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is in OPEN state"""
    pass


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker for {func.__name__} entering HALF_OPEN state")
                else:
                    logger.warning(f"Circuit breaker OPEN for {func.__name__}, blocking call")
                    raise CircuitBreakerOpen(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit breaker for {func.__name__} recovered, CLOSED")
                
                return result
            
            except self.expected_exception as e:
                self._failure_count += 1
                self._last_failure_time = time.time()
                
                logger.error(f"Circuit breaker failure {self._failure_count}/{self.failure_threshold} for {func.__name__}: {e}")
                
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(f"Circuit breaker OPEN for {func.__name__} after {self._failure_count} failures")
                
                raise
        
        return wrapper
    
    async def call(self, func: Callable) -> Any:
        """
        Call a function through the circuit breaker
        
        Args:
            func: Async callable to execute
            
        Returns:
            Result from the function
            
        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                logger.warning("Circuit breaker OPEN, blocking call")
                raise CircuitBreakerOpen("Circuit breaker is OPEN")
        
        try:
            result = await func()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker recovered, CLOSED")
            
            return result
        
        except self.expected_exception as e:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.error(f"Circuit breaker failure {self._failure_count}/{self.failure_threshold}: {e}")
            
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.error(f"Circuit breaker OPEN after {self._failure_count} failures")
            
            raise
