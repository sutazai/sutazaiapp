"""
Circuit Breaker Pattern Implementation for SutazAI Frontend
Prevents cascade failures and provides intelligent recovery
"""

import time
import logging
from typing import Dict, Callable, Any, Optional
from enum import Enum
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation, all requests allowed
    OPEN = "open"           # Circuit tripped, requests blocked
    HALF_OPEN = "half_open" # Testing recovery, limited requests allowed

class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass

class CircuitBreaker:
    """Circuit breaker for protecting against cascade failures"""
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,      # Failures before opening
                 recovery_timeout: float = 60.0,  # Seconds before trying recovery
                 success_threshold: int = 3,      # Successes needed to close
                 timeout: float = 10.0):          # Default operation timeout
        
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout = timeout
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.blocked_requests = 0
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        self.total_requests += 1
        
        # Check if circuit should allow request
        if not self._should_allow_request():
            self.blocked_requests += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}. "
                f"Last failure: {time.time() - self.last_failure_time:.1f}s ago"
            )
        
        # Execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            self._on_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker"""
        self.total_requests += 1
        
        if not self._should_allow_request():
            self.blocked_requests += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is {self.state.value}"
            )
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            self._on_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_allow_request(self) -> bool:
        """Determine if request should be allowed based on circuit state"""
        current_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            return True
            
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN for recovery test")
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited requests for testing
            return True
        
        return False
    
    def _on_success(self, response_time: float):
        """Handle successful request"""
        self.total_successes += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
        
        logger.debug(f"Circuit breaker '{self.name}' recorded success ({response_time:.2f}s)")
    
    def _on_failure(self, exception: Exception):
        """Handle failed request"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Reset success count on any failure
        self.success_count = 0
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure during recovery testing opens the circuit
            self._open_circuit()
        
        logger.warning(f"Circuit breaker '{self.name}' recorded failure: {exception}")
    
    def _open_circuit(self):
        """Open the circuit (block all requests)"""
        self.state = CircuitState.OPEN
        self.failure_count = 0
        logger.error(f"Circuit breaker '{self.name}' OPENED - blocking requests")
    
    def _close_circuit(self):
        """Close the circuit (allow all requests)"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' CLOSED - normal operation resumed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "blocked_requests": self.blocked_requests,
            "success_rate": (self.total_successes / self.total_requests * 100) if self.total_requests > 0 else 0,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time else None,
            "recovery_timeout": self.recovery_timeout,
            "failure_threshold": self.failure_threshold
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

class CircuitBreakerRegistry:
    """Central registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker by name"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name=name, **kwargs)
        return self.breakers[name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self.breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

# Global circuit breaker registry
circuit_registry = CircuitBreakerRegistry()

def with_circuit_breaker(name: str, **breaker_kwargs):
    """Decorator to protect function with circuit breaker"""
    def decorator(func: Callable):
        breaker = circuit_registry.get_breaker(name, **breaker_kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.acall(func, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Pre-configured circuit breakers for common SutazAI services
def get_health_check_breaker() -> CircuitBreaker:
    """Get circuit breaker for health checks"""
    return circuit_registry.get_breaker(
        "health_check",
        failure_threshold=3,     # Open after 3 health check failures
        recovery_timeout=30.0,   # Try recovery every 30 seconds
        success_threshold=2,     # Close after 2 successful health checks
        timeout=5.0
    )

def get_api_breaker() -> CircuitBreaker:
    """Get circuit breaker for API calls"""
    return circuit_registry.get_breaker(
        "api_calls",
        failure_threshold=5,     # Open after 5 API failures
        recovery_timeout=60.0,   # Try recovery every minute
        success_threshold=3,     # Close after 3 successful calls
        timeout=30.0
    )

def get_ollama_breaker() -> CircuitBreaker:
    """Get circuit breaker for Ollama model requests"""
    return circuit_registry.get_breaker(
        "ollama_requests", 
        failure_threshold=3,     # Open after 3 Ollama failures
        recovery_timeout=120.0,  # Try recovery every 2 minutes (model warmup)
        success_threshold=2,     # Close after 2 successful requests
        timeout=60.0
    )