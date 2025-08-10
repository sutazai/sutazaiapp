"""
Optimized API Client Utilities - Performance Enhanced Version
Centralized API communication with caching, connection pooling, and monitoring
"""

import asyncio
import httpx
import streamlit as st
import logging
import time
from typing import Dict, Any, Optional, List
from functools import lru_cache
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://127.0.0.1:10010"
DEFAULT_TIMEOUT = 10.0

# Global HTTP client for connection pooling
_client_instance = None
_client_lock = asyncio.Lock()

# Performance monitoring
_api_metrics = {
    "total_requests": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "total_response_time": 0.0,
    "error_count": 0
}

async def get_http_client() -> httpx.AsyncClient:
    """Get shared HTTP client instance for connection pooling"""
    global _client_instance, _client_lock
    
    async with _client_lock:
        if _client_instance is None or _client_instance.is_closed:
            _client_instance = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                http2=True,  # Enable HTTP/2 for better performance
                headers={
                    "User-Agent": "SutazAI-Frontend/2.0",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate"
                }
            )
    
    return _client_instance

def _generate_cache_key(endpoint: str, method: str, data: Dict = None) -> str:
    """Generate cache key for API requests"""
    cache_data = {
        "endpoint": endpoint,
        "method": method.upper(),
        "data": data or {}
    }
    return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

@st.cache_data(ttl=30, max_entries=100, show_spinner=False)
def _cached_get_request(endpoint: str, cache_key: str) -> Optional[Dict]:
    """Cached GET requests (30 second TTL)"""
    return asyncio.run(_make_uncached_request(endpoint, "GET", None))

@st.cache_data(ttl=10, max_entries=50, show_spinner=False)  
def _cached_health_check(url: str, cache_key: str) -> bool:
    """Cached health checks (10 second TTL)"""
    return asyncio.run(_check_service_health_uncached(url))

async def _make_uncached_request(endpoint: str, method: str, data: Dict = None) -> Optional[Dict]:
    """Make uncached API request with performance monitoring"""
    start_time = time.time()
    _api_metrics["total_requests"] += 1
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        client = await get_http_client()
        
        if method.upper() == "GET":
            response = await client.get(url)
        elif method.upper() == "POST":
            response = await client.post(url, json=data)
        elif method.upper() == "PUT":
            response = await client.put(url, json=data)
        elif method.upper() == "DELETE":
            response = await client.delete(url)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            _api_metrics["error_count"] += 1
            return None
            
        response.raise_for_status()
        result = response.json()
        
        # Update performance metrics
        response_time = (time.time() - start_time) * 1000
        _api_metrics["total_response_time"] += response_time
        
        return result
            
    except httpx.TimeoutException:
        _api_metrics["error_count"] += 1
        logger.warning(f"API call timeout: {endpoint}")
        return None
        
    except httpx.HTTPStatusError as e:
        _api_metrics["error_count"] += 1
        logger.error(f"HTTP error {e.response.status_code} for {endpoint}: {e.response.text}")
        return None
        
    except httpx.RequestError as e:
        _api_metrics["error_count"] += 1
        logger.error(f"Request error for {endpoint}: {str(e)}")
        return None
        
    except Exception as e:
        _api_metrics["error_count"] += 1
        logger.error(f"Unexpected error for {endpoint}: {str(e)}")
        return None

async def call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: float = None, use_cache: bool = True) -> Optional[Dict]:
    """
    Make async API call to backend with caching and performance optimization
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, PUT, DELETE)
        data: Request data for POST/PUT requests
        timeout: Request timeout in seconds
        use_cache: Whether to use caching for GET requests
        
    Returns:
        Response data dict or None if error
    """
    # Use caching for GET requests by default
    if method.upper() == "GET" and use_cache:
        cache_key = _generate_cache_key(endpoint, method, data)
        _api_metrics["cache_hits"] += 1
        return _cached_get_request(endpoint, cache_key)
    else:
        _api_metrics["cache_misses"] += 1
        return await _make_uncached_request(endpoint, method, data)

def handle_api_error(response: dict, context: str = "operation") -> bool:
    """
    Handle API response and show appropriate error messages
    
    Args:
        response: API response dictionary
        context: Context description for error messages
        
    Returns:
        True if response is valid, False if error
    """
    if not response:
        st.error(f"No response received for {context}")
        return False
        
    if "error" in response:
        st.error(f"API Error ({context}): {response['error']}")
        return False
        
    if "status" in response and response["status"] == "error":
        error_msg = response.get("message", "Unknown error")
        st.error(f"Error ({context}): {error_msg}")
        return False
        
    return True

async def _check_service_health_uncached(url: str, timeout: float = 2.0) -> bool:
    """Uncached health check"""
    try:
        client = await get_http_client()
        response = await client.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Health check failed for {url}: {e}")
        return False

async def check_service_health(url: str, timeout: float = 2.0, use_cache: bool = True) -> bool:
    """
    Check if a service is healthy with caching
    
    Args:
        url: Service health check URL
        timeout: Request timeout
        use_cache: Whether to use caching
        
    Returns:
        True if service is healthy
    """
    if use_cache:
        cache_key = hashlib.md5(f"{url}_{timeout}".encode()).hexdigest()
        return _cached_health_check(url, cache_key)
    else:
        return await _check_service_health_uncached(url, timeout)

# Synchronous wrappers for Streamlit compatibility
def sync_call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: float = None, use_cache: bool = True) -> Optional[Dict]:
    """Synchronous wrapper for API calls with caching"""
    try:
        return asyncio.run(call_api(endpoint, method, data, timeout, use_cache))
    except Exception as e:
        logger.error(f"Sync API call failed: {e}")
        return None

def sync_check_service_health(url: str, timeout: float = 2.0, use_cache: bool = True) -> bool:
    """Synchronous wrapper for health checks with caching"""
    try:
        return asyncio.run(check_service_health(url, timeout, use_cache))
    except Exception as e:
        logger.error(f"Sync health check failed: {e}")
        return False

# Performance monitoring functions
def get_api_metrics() -> Dict[str, Any]:
    """Get API performance metrics"""
    metrics = _api_metrics.copy()
    if metrics["total_requests"] > 0:
        metrics["average_response_time"] = metrics["total_response_time"] / metrics["total_requests"]
        metrics["cache_hit_rate"] = metrics["cache_hits"] / (metrics["cache_hits"] + metrics["cache_misses"]) if (metrics["cache_hits"] + metrics["cache_misses"]) > 0 else 0
        metrics["error_rate"] = metrics["error_count"] / metrics["total_requests"]
    return metrics

def reset_api_metrics():
    """Reset API metrics"""
    global _api_metrics
    _api_metrics = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "total_response_time": 0.0,
        "error_count": 0
    }

# Batch API requests for efficiency
async def batch_api_calls(requests: List[Dict]) -> List:
    """Execute multiple API calls concurrently"""
    tasks = []
    for req in requests:
        endpoint = req.get("endpoint")
        method = req.get("method", "GET")
        data = req.get("data")
        use_cache = req.get("use_cache", True)
        
        task = call_api(endpoint, method, data, use_cache=use_cache)
        tasks.append(task)
    
    return await asyncio.gather(*tasks, return_exceptions=True)

def sync_batch_api_calls(requests: List[Dict]) -> List:
    """Synchronous wrapper for batch API calls"""
    try:
        return asyncio.run(batch_api_calls(requests))
    except Exception as e:
        logger.error(f"Batch API calls failed: {e}")
        return [None] * len(requests)

# Lazy loading helper
@st.cache_data(ttl=60, show_spinner=False)
def load_paginated_data(endpoint: str, page: int = 1, page_size: int = 20, filters: Dict = None) -> Dict:
    """Load paginated data with caching"""
    params = {
        "page": page,
        "page_size": page_size,
        **(filters or {})
    }
    
    # Convert params to query string
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_endpoint = f"{endpoint}?{query_string}"
    
    return sync_call_api(full_endpoint, use_cache=True)

# Error boundary decorator
def with_error_boundary(fallback_value=None, show_error=True):
    """Decorator to add error boundaries to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                if show_error:
                    st.error(f"An error occurred in {func.__name__}. Please try again.")
                return fallback_value
        return wrapper
    return decorator