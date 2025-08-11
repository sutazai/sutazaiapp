"""
Resilient API Client for SutazAI Frontend
Integrates adaptive timeouts, circuit breaker, and intelligent error recovery
"""

import asyncio
import httpx
import streamlit as st
import logging
import time
from typing import Dict, Any, Optional, List
from functools import wraps

from .adaptive_timeouts import timeout_manager, SystemState
from .circuit_breaker import get_health_check_breaker, get_api_breaker, get_ollama_breaker, CircuitBreakerError
from .performance_cache import cache

logger = logging.getLogger(__name__)

class ResilientAPIClient:
    """Ultra-resilient API client with comprehensive failure handling"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:10010"
        self._client = None
        self._request_deduplication = {}
        
        # Circuit breakers for different service types
        self.health_breaker = get_health_check_breaker()
        self.api_breaker = get_api_breaker()  
        self.ollama_breaker = get_ollama_breaker()
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get persistent HTTP client with optimized settings"""
        if self._client is None or self._client.is_closed:
            # Use adaptive timeouts based on system state
            connect_timeout = timeout_manager.get_timeout("health_check")
            read_timeout = timeout_manager.get_timeout("api_call")
            
            timeout = httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=5.0,
                pool=10.0
            )
            
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
                timeout=timeout,
                http2=True,
                verify=False,  # Local development
                follow_redirects=True,
                headers={
                    "User-Agent": "SutazAI-Resilient-Client/1.0",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate"
                }
            )
        
        return self._client
    
    async def health_check(self, use_cache: bool = True) -> Dict[str, Any]:
        """Resilient health check with circuit breaker protection"""
        
        # Check cache first to reduce backend load
        if use_cache:
            cached_health = cache.get("health_check_data")
            if cached_health is not None:
                logger.debug("Using cached health check data")
                return cached_health
        
        # Use circuit breaker to prevent hammering failing endpoint
        try:
            result = await self.health_breaker.acall(self._perform_health_check)
            
            # Cache successful health check
            if result.get("status") == "healthy":
                cache.set("health_check_data", result, ttl=30)
                timeout_manager.record_success("health_check", 1.0)
            
            return result
            
        except CircuitBreakerError as e:
            logger.warning(f"Health check blocked by circuit breaker: {e}")
            timeout_manager.record_failure("health_check", "circuit_breaker_open")
            
            # Return cached data if available, otherwise degraded status
            cached_health = cache.get("health_check_data") 
            if cached_health:
                cached_health["status"] = "cached"
                return cached_health
            
            return {
                "status": "circuit_open",
                "message": "Health check temporarily disabled due to repeated failures",
                "circuit_state": self.health_breaker.state.value,
                "retry_in": self.health_breaker.recovery_timeout
            }
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            timeout_manager.record_failure("health_check", str(type(e).__name__))
            
            return {
                "status": "error", 
                "message": str(e),
                "error_type": type(e).__name__
            }
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform actual health check request"""
        client = await self._get_client()
        timeout = timeout_manager.get_timeout("health_check")
        
        start_time = time.time()
        response = await client.get(f"{self.base_url}/health", timeout=timeout)
        response_time = time.time() - start_time
        
        response.raise_for_status()
        result = response.json()
        result["response_time"] = response_time
        
        return result
    
    async def call_api(self, endpoint: str, method: str = "GET", 
                      data: Dict = None, use_cache: bool = True,
                      service_type: str = "api") -> Optional[Dict[str, Any]]:
        """Resilient API call with comprehensive error handling"""
        
        # Request deduplication for identical concurrent requests
        request_key = f"{method}:{endpoint}:{hash(str(data))}"
        if request_key in self._request_deduplication:
            logger.debug(f"Deduplicating concurrent request: {endpoint}")
            return await self._request_deduplication[request_key]
        
        # Create request future for deduplication
        request_future = self._make_api_request(endpoint, method, data, use_cache, service_type)
        self._request_deduplication[request_key] = request_future
        
        try:
            result = await request_future
            return result
        finally:
            # Clean up deduplication entry
            self._request_deduplication.pop(request_key, None)
    
    async def _make_api_request(self, endpoint: str, method: str, data: Dict, 
                               use_cache: bool, service_type: str) -> Optional[Dict[str, Any]]:
        """Make the actual API request with circuit breaker protection"""
        
        # Check cache for GET requests
        if method.upper() == "GET" and use_cache:
            cache_key = f"api:{endpoint}:{hash(str(data))}"
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached API response: {endpoint}")
                return cached_result
        
        # Select appropriate circuit breaker
        if "ollama" in endpoint.lower() or "chat" in endpoint.lower():
            breaker = self.ollama_breaker
            operation_type = "model_request"
        else:
            breaker = self.api_breaker
            operation_type = "api_call"
        
        try:
            result = await breaker.acall(
                self._perform_api_request,
                endpoint, method, data, operation_type
            )
            
            # Cache successful GET requests
            if method.upper() == "GET" and use_cache and result is not None:
                cache_key = f"api:{endpoint}:{hash(str(data))}"
                ttl = 60 if "health" not in endpoint else 30
                cache.set(cache_key, result, ttl)
            
            return result
            
        except CircuitBreakerError as e:
            logger.warning(f"API call blocked by circuit breaker: {endpoint}")
            timeout_manager.record_failure(operation_type, "circuit_breaker_open")
            
            # Try to return cached data for GET requests
            if method.upper() == "GET" and use_cache:
                cache_key = f"api:{endpoint}:{hash(str(data))}"
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    cached_result["cached"] = True
                    cached_result["warning"] = "Service temporarily unavailable, showing cached data"
                    return cached_result
            
            return {
                "error": "Service temporarily unavailable",
                "circuit_state": breaker.state.value,
                "retry_in": breaker.recovery_timeout,
                "endpoint": endpoint
            }
        
        except Exception as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            timeout_manager.record_failure(operation_type, str(type(e).__name__))
            return None
    
    async def _perform_api_request(self, endpoint: str, method: str, 
                                  data: Dict, operation_type: str) -> Dict[str, Any]:
        """Perform actual API request"""
        client = await self._get_client()
        url = f"{self.base_url}{endpoint}"
        timeout = timeout_manager.get_timeout(operation_type)
        
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = await client.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, timeout=timeout)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = await client.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            timeout_manager.record_success(operation_type, response_time)
            
            response.raise_for_status()
            result = response.json()
            result["response_time"] = response_time
            
            return result
            
        except httpx.TimeoutException:
            raise Exception(f"Request timeout after {timeout:.1f}s")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise Exception(f"Endpoint not found: {endpoint}")
            elif e.response.status_code >= 500:
                raise Exception(f"Server error ({e.response.status_code})")
            elif e.response.status_code in [403, 405]:
                # Potential CORS issues
                raise Exception(f"Access denied ({e.response.status_code}) - possible CORS issue")
            else:
                raise Exception(f"HTTP error {e.response.status_code}")
        except httpx.RequestError as e:
            raise Exception(f"Connection error: {str(e)}")
    
    async def batch_health_check(self, services: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch health check for multiple services to reduce backend load"""
        
        # Check if we should batch or individual calls based on system state
        if timeout_manager.current_state in [SystemState.STARTUP, SystemState.FAILED]:
            # During startup or failure, avoid batch calls that might overwhelm backend
            return {"batch_disabled": True, "reason": "System in startup/failed state"}
        
        results = {}
        
        # Execute health checks concurrently but with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent health checks
        
        async def check_service(service_name: str, port: int):
            async with semaphore:
                try:
                    url = f"http://127.0.0.1:{port}/health"
                    client = await self._get_client()
                    response = await client.get(url, timeout=2.0)
                    return service_name, {"status": "healthy", "port": port}
                except Exception as e:
                    return service_name, {"status": "unhealthy", "error": str(e), "port": port}
        
        # Known service ports
        service_ports = {
            "backend": 10010,
            "ollama": 10104,  
            "redis": 10001,
            "postgres": 10000
        }
        
        tasks = []
        for service in services:
            if service in service_ports:
                tasks.append(check_service(service, service_ports[service]))
        
        if tasks:
            service_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in service_results:
                if isinstance(result, tuple):
                    service_name, status = result
                    results[service_name] = status
                else:
                    logger.error(f"Batch health check error: {result}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including circuit breaker states"""
        timeout_info = timeout_manager.get_state_info()
        
        return {
            "system_state": timeout_info["state"],
            "state_duration": timeout_info["state_duration"],
            "adaptive_timeouts": timeout_info["timeouts"],
            "circuit_breakers": {
                "health_check": self.health_breaker.get_stats(),
                "api_calls": self.api_breaker.get_stats(), 
                "ollama_requests": self.ollama_breaker.get_stats()
            },
            "last_success": timeout_info["last_success"],
            "consecutive_failures": timeout_info["consecutive_failures"]
        }
    
    async def close(self):
        """Clean up HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

# Global resilient client instance
resilient_client = ResilientAPIClient()

# Streamlit-compatible synchronous wrappers
def sync_health_check(use_cache: bool = True) -> Dict[str, Any]:
    """Synchronous health check wrapper"""
    try:
        return asyncio.run(resilient_client.health_check(use_cache))
    except Exception as e:
        logger.error(f"Sync health check failed: {e}")
        return {"status": "error", "message": str(e)}

def sync_call_api(endpoint: str, method: str = "GET", 
                 data: Dict = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Synchronous API call wrapper"""
    try:
        return asyncio.run(resilient_client.call_api(endpoint, method, data, use_cache))
    except Exception as e:
        logger.error(f"Sync API call failed: {e}")
        return {"error": str(e), "endpoint": endpoint}

def get_system_status() -> Dict[str, Any]:
    """Get system status information"""
    return resilient_client.get_system_status()

# Error handling decorator for UI components
def with_api_error_handling(fallback_value=None, show_user_message=True):
    """Decorator to add comprehensive error handling to API-dependent functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                
                if show_user_message:
                    # Provide context-specific error messages
                    if "circuit_breaker_open" in str(e):
                        st.error("🔌 Service temporarily unavailable. We're working to restore it.")
                        st.info("The system is protecting itself from cascade failures. Please wait a moment.")
                    elif "timeout" in str(e).lower():
                        st.warning("⏱️ Request is taking longer than expected. The system may be starting up.")
                        st.info("Backend services may be warming up. This can take a few minutes on first start.")
                    elif "connection" in str(e).lower():
                        st.error("🚫 Unable to connect to backend services.")
                        st.info("Please check if the backend is running and try refreshing the page.")
                    elif "cors" in str(e).lower():
                        st.error("🔒 Cross-origin request blocked.")
                        st.info("This may be a security configuration issue. Please contact support.")
                    else:
                        st.error(f"🔧 Service error: {str(e)}")
                        st.info("Please try again in a moment. If the problem persists, refresh the page.")
                
                return fallback_value
        return wrapper
    return decorator