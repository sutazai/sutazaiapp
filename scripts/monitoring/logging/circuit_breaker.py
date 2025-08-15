#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation for SutazAI Self-Healing Architecture

This module implements the circuit breaker pattern to prevent cascading failures
by monitoring service calls and automatically switching to failure mode when
thresholds are exceeded.

Author: SutazAI Infrastructure Team
Version: 1.0.0
"""

import asyncio
import time
import logging
from typing import Callable, Any, Optional, Dict, Union
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import json
import threading
from contextlib import asynccontextmanager
import aiohttp
import redis
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0       # Request timeout
    monitoring_window: int = 300  # Monitoring window in seconds
    
class CircuitBreakerMetrics:
    """Prometheus metrics for circuit breaker"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # Counters
        self.requests_total = Counter(
            'circuit_breaker_requests_total',
            'Total requests through circuit breaker',
            ['service', 'state', 'result']
        )
        
        self.state_changes = Counter(
            'circuit_breaker_state_changes_total',
            'Circuit breaker state changes',
            ['service', 'from_state', 'to_state']
        )
        
        # Histograms
        self.request_duration = Histogram(
            'circuit_breaker_request_duration_seconds',
            'Request duration through circuit breaker',
            ['service', 'result']
        )
        
        # Gauges
        self.failure_rate = Gauge(
            'circuit_breaker_failure_rate',
            'Current failure rate',
            ['service']
        )
        
        self.current_state = Gauge(
            'circuit_breaker_state',
            'Current circuit breaker state (0=closed, 1=open, 2=half_open)',
            ['service']
        )

class CircuitBreaker:
    """
    Circuit breaker implementation with multiple backends and monitoring
    """
    
    def __init__(self, 
                 service_name: str,
                 config: CircuitConfig = None,
                 redis_client: Optional[redis.Redis] = None):
        self.service_name = service_name
        self.config = config or CircuitConfig()
        self.redis_client = redis_client
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._success_count = 0
        
        # Monitoring
        self._call_history = deque(maxlen=100)
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics(service_name)
        
        # Initialize state in Redis if available
        if self.redis_client:
            self._load_state_from_redis()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state"""
        with self._lock:
            return self._state
    
    def _load_state_from_redis(self):
        """Load circuit breaker state from Redis"""
        try:
            key = f"circuit_breaker:{self.service_name}"
            state_data = self.redis_client.get(key)
            
            if state_data:
                data = json.loads(state_data)
                self._state = CircuitState(data.get('state', 'closed'))
                self._failure_count = data.get('failure_count', 0)
                self._last_failure_time = data.get('last_failure_time', 0)
                self._success_count = data.get('success_count', 0)
                
                logger.info(f"Loaded circuit breaker state for {self.service_name}: {self._state}")
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state from Redis: {e}")
    
    def _save_state_to_redis(self):
        """Save circuit breaker state to Redis"""
        if not self.redis_client:
            return
            
        try:
            key = f"circuit_breaker:{self.service_name}"
            state_data = {
                'state': self._state.value,
                'failure_count': self._failure_count,
                'last_failure_time': self._last_failure_time,
                'success_count': self._success_count,
                'updated_at': time.time()
            }
            
            self.redis_client.setex(
                key, 
                3600,  # 1 hour TTL
                json.dumps(state_data)
            )
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state to Redis: {e}")
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state with metrics"""
        old_state = self._state
        
        if old_state != new_state:
            logger.info(f"Circuit breaker {self.service_name}: {old_state.value} -> {new_state.value}")
            
            # Update metrics
            self.metrics.state_changes.labels(
                service=self.service_name,
                from_state=old_state.value,
                to_state=new_state.value
            ).inc()
            
            # Update state gauge
            state_values = {
                CircuitState.CLOSED: 0,
                CircuitState.OPEN: 1,
                CircuitState.HALF_OPEN: 2
            }
            self.metrics.current_state.labels(service=self.service_name).set(
                state_values[new_state]
            )
            
            self._state = new_state
            self._save_state_to_redis()
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed based on current state"""
        current_time = time.time()
        
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if current_time - self._last_failure_time >= self.config.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False
            
            elif self._state == CircuitState.HALF_OPEN:
                return True
            
        return False
    
    def _record_success(self):
        """Record successful call"""
        current_time = time.time()
        
        with self._lock:
            self._call_history.append({
                'timestamp': current_time,
                'success': True
            })
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._failure_count = 0
                    self._success_count = 0
                    self._transition_to(CircuitState.CLOSED)
            
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    def _record_failure(self):
        """Record failed call"""
        current_time = time.time()
        
        with self._lock:
            self._call_history.append({
                'timestamp': current_time,
                'success': False
            })
            
            self._failure_count += 1
            self._last_failure_time = current_time
            
            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
            
            elif self._state == CircuitState.HALF_OPEN:
                # Single failure in half-open state opens circuit
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0
    
    def _calculate_failure_rate(self) -> float:
        """Calculate current failure rate"""
        current_time = time.time()
        window_start = current_time - self.config.monitoring_window
        
        recent_calls = [
            call for call in self._call_history
            if call['timestamp'] >= window_start
        ]
        
        if not recent_calls:
            return 0.0
        
        failures = sum(1 for call in recent_calls if not call['success'])
        return failures / len(recent_calls)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function call through circuit breaker
        
        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenError: When circuit is open
            Exception: Original function exceptions
        """
        if not self._should_allow_request():
            self.metrics.requests_total.labels(
                service=self.service_name,
                state=self._state.value,
                result='blocked'
            ).inc()
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for service: {self.service_name}"
            )
        
        start_time = time.time()
        
        try:
            # Execute the function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            else:
                result = func(*args, **kwargs)
            
            # Record success
            self._record_success()
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.request_duration.labels(
                service=self.service_name,
                result='success'
            ).observe(duration)
            
            self.metrics.requests_total.labels(
                service=self.service_name,
                state=self._state.value,
                result='success'
            ).inc()
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure()
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.request_duration.labels(
                service=self.service_name,
                result='failure'
            ).observe(duration)
            
            self.metrics.requests_total.labels(
                service=self.service_name,
                state=self._state.value,
                result='failure'
            ).inc()
            
            # Update failure rate
            failure_rate = self._calculate_failure_rate()
            self.metrics.failure_rate.labels(service=self.service_name).set(failure_rate)
            
            raise e
    
    @asynccontextmanager
    async def context(self):
        """Context manager for circuit breaker"""
        if not self._should_allow_request():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for service: {self.service_name}"
            )
        
        start_time = time.time()
        
        try:
            yield self
            
            # Record success
            self._record_success()
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.request_duration.labels(
                service=self.service_name,
                result='success'
            ).observe(duration)
            
        except Exception as e:
            # Record failure
            self._record_failure()
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.request_duration.labels(
                service=self.service_name,
                result='failure'
            ).observe(duration)
            
            raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        failure_rate = self._calculate_failure_rate()
        
        return {
            'service_name': self.service_name,
            'state': self._state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'failure_rate': failure_rate,
            'last_failure_time': self._last_failure_time,
            'call_history_size': len(self._call_history),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class HttpCircuitBreaker(CircuitBreaker):
    """Circuit breaker specifically for HTTP calls"""
    
    def __init__(self, service_name: str, base_url: str, config: CircuitConfig = None):
        super().__init__(service_name, config)
        self.base_url = base_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP GET with circuit breaker"""
        url = f"{self.base_url}{endpoint}"
        
        async def _make_request():
            if not self.session:
                raise RuntimeError("HttpCircuitBreaker must be used as async context manager")
            return await self.session.get(url, **kwargs)
        
        return await self.call(_make_request)
    
    async def post(self, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """HTTP POST with circuit breaker"""
        url = f"{self.base_url}{endpoint}"
        
        async def _make_request():
            if not self.session:
                raise RuntimeError("HttpCircuitBreaker must be used as async context manager")
            return await self.session.post(url, **kwargs)
        
        return await self.call(_make_request)

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_circuit_breaker(self, 
                           service_name: str, 
                           config: CircuitConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        with self._lock:
            if service_name not in self._circuit_breakers:
                self._circuit_breakers[service_name] = CircuitBreaker(
                    service_name, config, self.redis_client
                )
            return self._circuit_breakers[service_name]
    
    def get_http_circuit_breaker(self, 
                                service_name: str, 
                                base_url: str,
                                config: CircuitConfig = None) -> HttpCircuitBreaker:
        """Get or create HTTP circuit breaker for service"""
        with self._lock:
            key = f"{service_name}_http"
            if key not in self._circuit_breakers:
                self._circuit_breakers[key] = HttpCircuitBreaker(
                    service_name, base_url, config
                )
            return self._circuit_breakers[key]
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self._lock:
            return {
                name: breaker.get_stats() 
                for name, breaker in self._circuit_breakers.items()
            }
    
    def reset_all(self):
        """Reset all circuit breakers to closed state"""
        with self._lock:
            for breaker in self._circuit_breakers.values():
                with breaker._lock:
                    breaker._transition_to(CircuitState.CLOSED)
                    breaker._failure_count = 0
                    breaker._success_count = 0

# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()

def get_circuit_breaker(service_name: str, config: CircuitConfig = None) -> CircuitBreaker:
    """Convenience function to get circuit breaker"""
    return circuit_breaker_registry.get_circuit_breaker(service_name, config)

# Example usage and decorators
def circuit_breaker(service_name: str, config: CircuitConfig = None):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                breaker = get_circuit_breaker(service_name, config)
                return await breaker.call(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                breaker = get_circuit_breaker(service_name, config)
                return asyncio.run(breaker.call(func, *args, **kwargs))
            return sync_wrapper
    return decorator

if __name__ == "__main__":
    # Example usage
    async def example_usage():
        # Basic circuit breaker
        config = CircuitConfig(failure_threshold=3, recovery_timeout=30)
        breaker = CircuitBreaker("example_service", config)
        
        # HTTP circuit breaker
        async with HttpCircuitBreaker("api_service", "http://api.example.com") as http_breaker:
            try:
                response = await http_breaker.get("/health")
                logger.info(f"Health check status: {response.status}")
            except CircuitBreakerOpenError:
                logger.info("Circuit breaker is open - service unavailable")
        
        # Using decorator
        @circuit_breaker("decorated_service")
        async def external_api_call():
            # Simulate API call
            await asyncio.sleep(0.1)
            return {"status": "ok"}
        
        try:
            result = await external_api_call()
            logger.info(f"API result: {result}")
        except CircuitBreakerOpenError:
            logger.info("Service unavailable due to circuit breaker")
        
        # Get statistics
        stats = breaker.get_stats()
        logger.info(f"Circuit breaker stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(example_usage())