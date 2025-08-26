#!/usr/bin/env python3
"""
Circuit Breaker Integration for Backend Health Monitoring
Simplified integration to avoid import conflicts
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class SimpleCircuitBreaker:
    """
    Simplified circuit breaker for backend integration
    Designed to work without complex dependencies
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 timeout: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        
        # Metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Simple circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        # Check if circuit should be reset
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")
        
        # Reject request if circuit is open
        if self.state == CircuitState.OPEN:
            raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        self.total_requests += 1
        start_time = time.time()
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=self.timeout
                )
            
            # Record success
            self.successful_requests += 1
            self.consecutive_failures = 0
            self.last_success_time = datetime.now()
            
            # If we were in HALF_OPEN and got a success, close the circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED")
            
            return result
            
        except Exception as e:
            # Record failure
            self.failed_requests += 1
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()
            
            logger.warning(f"Circuit breaker '{self.name}' recorded failure: {e}")
            
            # Check if we should open the circuit
            if self.consecutive_failures >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN after {self.consecutive_failures} failures")
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if not self.last_failure_time:
            return False
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.total_requests > 0:
            success_rate = self.successful_requests / self.total_requests
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "consecutive_failures": self.consecutive_failures,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
        }
    
    @property
    def is_healthy(self) -> bool:
        """Check if circuit breaker is in a healthy state"""
        return self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]


class CircuitBreakerManager:
    """Simple circuit breaker manager for backend services"""
    
    def __init__(self):
        self._breakers: Dict[str, SimpleCircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_create_breaker(self, 
                                  name: str,
                                  failure_threshold: int = 5,
                                  recovery_timeout: float = 60.0,
                                  timeout: float = 30.0) -> SimpleCircuitBreaker:
        """Get or create a circuit breaker"""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = SimpleCircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    timeout=timeout
                )
            return self._breakers[name]
    
    def get_breaker(self, name: str) -> Optional[SimpleCircuitBreaker]:
        """Get existing circuit breaker"""
        return self._breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self._breakers.values():
            breaker.state = CircuitState.CLOSED
            breaker.consecutive_failures = 0
            logger.info(f"Reset circuit breaker '{breaker.name}'")


# Global circuit breaker manager
_circuit_breaker_manager = None
_manager_lock = asyncio.Lock()


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance"""
    global _circuit_breaker_manager
    
    async with _manager_lock:
        if _circuit_breaker_manager is None:
            _circuit_breaker_manager = CircuitBreakerManager()
        return _circuit_breaker_manager


# Convenience functions for common service circuit breakers
async def get_redis_circuit_breaker() -> SimpleCircuitBreaker:
    """Get circuit breaker for Redis service"""
    manager = await get_circuit_breaker_manager()
    return await manager.get_or_create_breaker(
        "redis",
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout=5.0
    )


async def get_database_circuit_breaker() -> SimpleCircuitBreaker:
    """Get circuit breaker for database service"""
    manager = await get_circuit_breaker_manager()
    return await manager.get_or_create_breaker(
        "database",
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout=10.0
    )


async def get_ollama_circuit_breaker() -> SimpleCircuitBreaker:
    """Get circuit breaker for Ollama service"""
    manager = await get_circuit_breaker_manager()
    return await manager.get_or_create_breaker(
        "ollama",
        failure_threshold=5,
        recovery_timeout=60.0,
        timeout=30.0
    )