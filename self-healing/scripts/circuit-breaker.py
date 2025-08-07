#!/usr/bin/env python3
"""
Circuit Breaker Implementation for SutazAI
Provides automatic service protection and recovery
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import redis
import yaml
import json
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Implements the Circuit Breaker pattern for service protection
    """
    
    def __init__(self, 
                 name: str,
                 failure_threshold: int = 5,
                 success_threshold: int = 2,
                 timeout: int = 30,
                 half_open_requests: int = 3,
                 fallback_function: Optional[Callable] = None,
                 redis_client: Optional[redis.Redis] = None):
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests
        self.fallback_function = fallback_function
        self.redis_client = redis_client
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_open_count = 0
        
    def _get_state_key(self) -> str:
        """Get Redis key for circuit state"""
        return f"circuit_breaker:{self.name}:state"
    
    def _get_metrics_key(self) -> str:
        """Get Redis key for circuit metrics"""
        return f"circuit_breaker:{self.name}:metrics"
    
    def _save_state(self):
        """Save circuit state to Redis"""
        if self.redis_client:
            state_data = {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
                "half_open_attempts": self.half_open_attempts
            }
            self.redis_client.setex(
                self._get_state_key(),
                300,  # 5 minute TTL
                json.dumps(state_data)
            )
            
    def _load_state(self):
        """Load circuit state from Redis"""
        if self.redis_client:
            state_data = self.redis_client.get(self._get_state_key())
            if state_data:
                data = json.loads(state_data)
                self.state = CircuitState(data["state"])
                self.failure_count = data["failure_count"]
                self.success_count = data["success_count"]
                self.last_failure_time = datetime.fromisoformat(data["last_failure_time"]) if data["last_failure_time"] else None
                self.half_open_attempts = data["half_open_attempts"]
    
    def _update_metrics(self, success: bool):
        """Update circuit metrics"""
        self.total_requests += 1
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
            
        if self.redis_client:
            metrics = {
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "circuit_open_count": self.circuit_open_count,
                "current_state": self.state.value,
                "last_updated": datetime.now().isoformat()
            }
            self.redis_client.setex(
                self._get_metrics_key(),
                3600,  # 1 hour TTL
                json.dumps(metrics)
            )
    
    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        """
        self._load_state()
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               datetime.now() > self.last_failure_time + timedelta(seconds=self.timeout):
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                self._save_state()
        
        # Handle different circuit states
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit {self.name} is OPEN, using fallback")
            self._update_metrics(False)
            if self.fallback_function:
                return self.fallback_function(*args, **kwargs)
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
                
        elif self.state == CircuitState.HALF_OPEN:
            if self.half_open_attempts >= self.half_open_requests:
                logger.info(f"Circuit {self.name} half-open limit reached, opening circuit")
                self._open_circuit()
                self._update_metrics(False)
                if self.fallback_function:
                    return self.fallback_function(*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            self.half_open_attempts += 1
        
        # Try to execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    async def async_call(self, func: Callable, *args, **kwargs):
        """
        Execute async function with circuit breaker protection
        """
        self._load_state()
        
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and \
               datetime.now() > self.last_failure_time + timedelta(seconds=self.timeout):
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                self._save_state()
        
        # Handle different circuit states
        if self.state == CircuitState.OPEN:
            logger.warning(f"Circuit {self.name} is OPEN, using fallback")
            self._update_metrics(False)
            if self.fallback_function:
                return await self.fallback_function(*args, **kwargs)
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
                
        elif self.state == CircuitState.HALF_OPEN:
            if self.half_open_attempts >= self.half_open_requests:
                logger.info(f"Circuit {self.name} half-open limit reached, opening circuit")
                self._open_circuit()
                self._update_metrics(False)
                if self.fallback_function:
                    return await self.fallback_function(*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            self.half_open_attempts += 1
        
        # Try to execute the function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution"""
        self._update_metrics(True)
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"Circuit {self.name} half-open success: {self.success_count}/{self.success_threshold}")
            
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit {self.name} closing after successful recovery")
                self._close_circuit()
        else:
            self.failure_count = 0  # Reset failure count on success
            
        self._save_state()
    
    def _on_failure(self):
        """Handle failed execution"""
        self._update_metrics(False)
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        logger.warning(f"Circuit {self.name} failure: {self.failure_count}/{self.failure_threshold}")
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name} failure in HALF_OPEN state, opening circuit")
            self._open_circuit()
        elif self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit {self.name} failure threshold reached, opening circuit")
            self._open_circuit()
            
        self._save_state()
    
    def _open_circuit(self):
        """Open the circuit"""
        self.state = CircuitState.OPEN
        self.circuit_open_count += 1
        self.half_open_attempts = 0
        logger.error(f"Circuit {self.name} is now OPEN")
        
    def _close_circuit(self):
        """Close the circuit"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_attempts = 0
        logger.info(f"Circuit {self.name} is now CLOSED")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit status"""
        self._load_state()
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "circuit_open_count": self.circuit_open_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitBreakerManager:
    """
    Manages all circuit breakers in the system
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/self-healing/config/self-healing-config.yaml"):
        self.config_path = config_path
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.redis_client = None
        self._load_config()
        self._init_redis()
        self._init_circuit_breakers()
        
    def _load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    def _init_circuit_breakers(self):
        """Initialize circuit breakers from config"""
        cb_config = self.config.get('circuit_breakers', {})
        default_config = cb_config.get('default', {})
        
        for service_name, service_config in cb_config.get('services', {}).items():
            # Merge with default config
            config = {**default_config, **service_config}
            
            # Create fallback function if specified
            fallback = None
            if 'fallback_response' in config:
                fallback_response = config['fallback_response']
                fallback = lambda *args, **kwargs: fallback_response
            
            # Create circuit breaker
            cb = CircuitBreaker(
                name=service_name,
                failure_threshold=config.get('failure_threshold', 5),
                success_threshold=config.get('success_threshold', 2),
                timeout=self._parse_duration(config.get('timeout', '30s')),
                half_open_requests=config.get('half_open_requests', 3),
                fallback_function=fallback,
                redis_client=self.redis_client
            )
            
            self.circuit_breakers[service_name] = cb
            logger.info(f"Initialized circuit breaker for {service_name}")
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to seconds"""
        if duration_str.endswith('s'):
            return int(duration_str[:-1])
        elif duration_str.endswith('m'):
            return int(duration_str[:-1]) * 60
        elif duration_str.endswith('h'):
            return int(duration_str[:-1]) * 3600
        else:
            return int(duration_str)
    
    def get_circuit_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service"""
        return self.circuit_breakers.get(service_name)
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            name: cb.get_status() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_circuit(self, service_name: str):
        """Reset a circuit breaker"""
        cb = self.circuit_breakers.get(service_name)
        if cb:
            cb._close_circuit()
            cb._save_state()
            logger.info(f"Reset circuit breaker for {service_name}")


# Decorator for easy circuit breaker usage
def circuit_breaker(service_name: str):
    """
    Decorator to apply circuit breaker protection to a function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = CircuitBreakerManager()
            cb = manager.get_circuit_breaker(service_name)
            if cb:
                return cb.call(func, *args, **kwargs)
            else:
                # No circuit breaker configured, execute normally
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = CircuitBreakerManager()
            cb = manager.get_circuit_breaker(service_name)
            if cb:
                return await cb.async_call(func, *args, **kwargs)
            else:
                # No circuit breaker configured, execute normally
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    manager = CircuitBreakerManager()
    
    # Get status of all circuit breakers
    print("Circuit Breaker Status:")
    for name, status in manager.get_all_status().items():
        print(f"\n{name}:")
        for key, value in status.items():
            print(f"  {key}: {value}")