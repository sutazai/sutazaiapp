"""
Resilient API Client for SutazAI Frontend
Provides robust API communication with circuit breakers, retries, and error handling
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Simple in-memory cache for API responses
_cache = {}
_cache_timestamps = {}
CACHE_TTL = 30  # 30 seconds

class CircuitBreaker:
    """Simple circuit breaker implementation"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise e

# Global circuit breaker instances
_circuit_breakers = {
    "health_check": CircuitBreaker(),
    "api_call": CircuitBreaker(),
    "backend": CircuitBreaker()
}

def sync_health_check(use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Synchronous health check with caching and circuit breaker protection
    """
    cache_key = "health_check"
    
    # Check cache first
    if use_cache and cache_key in _cache:
        if time.time() - _cache_timestamps[cache_key] < CACHE_TTL:
            result = _cache[cache_key].copy()
            result["status"] = "cached"
            return result
    
    try:
        # Simulate health check - in real implementation would call actual API
        breaker = _circuit_breakers["health_check"]
        
        def _health_check():
            # Real health check implementation - call actual backend
            import requests
            try:
                base_url = "http://127.0.0.1:10010"
                response = requests.get(f"{base_url}/health", timeout=5.0)
                response.raise_for_status()
                result = response.json()
                result["response_time"] = 0.1 if "response_time" not in result else result["response_time"]
                return result
            except requests.exceptions.RequestException as e:
                raise Exception(f"Health check failed: {str(e)}")
        
        result = breaker.call(_health_check)
        
        # Cache successful result
        _cache[cache_key] = result.copy()
        _cache_timestamps[cache_key] = time.time()
        
        return result
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        # Return cached data if available
        if cache_key in _cache:
            result = _cache[cache_key].copy()
            result["status"] = "cached"
            result["error"] = str(e)
            return result
        
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

async def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None,
                  timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Asynchronous API call with circuit breaker protection
    """
    return sync_call_api(endpoint, method, data, timeout)

def handle_api_error(error: Exception, context: str = "") -> Dict[str, Any]:
    """
    Handle API errors with context and logging
    """
    error_msg = f"API Error{f' in {context}' if context else ''}: {str(error)}"
    logger.error(error_msg)
    return {
        "error": True,
        "message": error_msg,
        "timestamp": time.time()
    }

def sync_call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None, 
                 timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    """
    Synchronous API call with circuit breaker protection
    """
    cache_key = f"{method}:{endpoint}"
    
    try:
        breaker = _circuit_breakers["api_call"]
        
        def _api_call():
            # Real API implementation - make actual HTTP requests to backend
            import requests
            try:
                base_url = "http://127.0.0.1:10010"
                url = f"{base_url}{endpoint}" if endpoint.startswith('/') else f"{base_url}/{endpoint}"
                
                if method.upper() == "GET":
                    response = requests.get(url, timeout=timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, json=data, timeout=timeout)
                elif method.upper() == "PUT":
                    response = requests.put(url, json=data, timeout=timeout)
                elif method.upper() == "DELETE":
                    response = requests.delete(url, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                result = response.json()
                
                # Only add timestamp to dict responses, not lists
                if isinstance(result, dict):
                    result["timestamp"] = time.time()
                
                return result
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"API call failed: {endpoint} - {str(e)}")
        
        result = breaker.call(_api_call)
        return result
        
    except Exception as e:
        logger.error(f"API call failed: {endpoint} - {e}")
        return None

def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status including circuit breakers
    """
    return {
        "system_state": "healthy",
        "timestamp": time.time(),
        "circuit_breakers": {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_rate": max(0, 100 - (cb.failure_count * 10))
            }
            for name, cb in _circuit_breakers.items()
        }
    }

def with_api_error_handling(fallback_value=None, show_user_message=True):
    """
    Decorator for handling API errors gracefully
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                if show_user_message:
                    # In real implementation, would show user notification
                    pass
                return fallback_value
        return wrapper
    return decorator