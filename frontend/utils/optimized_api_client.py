"""
Ultra-Optimized API Client for SutazAI Frontend
Implements request batching, connection pooling, and intelligent caching
"""

import asyncio
import httpx
import streamlit as st
from typing import Dict, Any, Optional, List, Union
import logging
import time
from contextlib import asynccontextmanager
from .performance_cache import cache, cached_api_call, SmartRefresh

logger = logging.getLogger(__name__)

class OptimizedAPIClient:
    """High-performance API client with connection pooling and batching"""
    
    def __init__(self):
        self.base_url = "http://127.0.0.1:10010"
        self.default_timeout = 10.0
        self.batch_size = 10
        self.batch_timeout = 0.1  # 100ms batch window
        self._client = None
        self._batch_queue = []
        self._batch_timer = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client with optimized settings"""
        if self._client is None or self._client.is_closed:
            # Optimized client configuration
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
            
            timeout = httpx.Timeout(
                connect=5.0,
                read=self.default_timeout,
                write=5.0,
                pool=10.0
            )
            
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                http2=True,  # Enable HTTP/2 for multiplexing
                verify=False,  # Skip SSL verification for local dev
                follow_redirects=True
            )
        
        return self._client
    
    async def close(self):
        """Close the HTTP client connection"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    @cached_api_call(ttl=60, key_prefix="health")
    async def health_check(self) -> Dict[str, Any]:
        """Optimized health check with caching"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=2.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def batch_request(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple requests in parallel with optimal batching"""
        client = await self._get_client()
        
        async def single_request(req: Dict[str, Any]) -> Dict[str, Any]:
            try:
                method = req.get("method", "GET").upper()
                endpoint = req["endpoint"]
                url = f"{self.base_url}{endpoint}"
                data = req.get("data")
                timeout = req.get("timeout", self.default_timeout)
                
                if method == "GET":
                    response = await client.get(url, timeout=timeout)
                elif method == "POST":
                    response = await client.post(url, json=data, timeout=timeout)
                elif method == "PUT":
                    response = await client.put(url, json=data, timeout=timeout)
                elif method == "DELETE":
                    response = await client.delete(url, timeout=timeout)
                else:
                    return {"error": f"Unsupported method: {method}"}
                
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                logger.error(f"Request failed {req.get('endpoint', 'unknown')}: {e}")
                return {"error": str(e), "endpoint": req.get("endpoint")}
        
        # Execute requests concurrently with semaphore for rate limiting
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def limited_request(req):
            async with semaphore:
                return await single_request(req)
        
        results = await asyncio.gather(
            *[limited_request(req) for req in requests],
            return_exceptions=True
        )
        
        return results
    
    @cached_api_call(ttl=300, key_prefix="system")
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview with intelligent caching"""
        requests = [
            {"endpoint": "/health", "method": "GET"},
            {"endpoint": "/metrics", "method": "GET"},
            {"endpoint": "/api/v1/agents/status", "method": "GET"},
            {"endpoint": "/api/v1/models", "method": "GET"}
        ]
        
        results = await self.batch_request(requests)
        
        # Combine results into structured response
        overview = {
            "health": results[0] if len(results) > 0 else {},
            "metrics": results[1] if len(results) > 1 else {},
            "agents": results[2] if len(results) > 2 else {},
            "models": results[3] if len(results) > 3 else {},
            "timestamp": time.time()
        }
        
        return overview
    
    async def call_api(self, endpoint: str, method: str = "GET", 
                      data: Dict = None, timeout: float = None,
                      use_cache: bool = True) -> Optional[Dict]:
        """
        Optimized single API call with caching and error handling
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            timeout: Request timeout
            use_cache: Whether to use caching
        """
        if use_cache and method.upper() == "GET":
            # Use cache for GET requests
            cache_key = cache._get_cache_key(f"api:{endpoint}", method, data or {})
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        try:
            client = await self._get_client()
            url = f"{self.base_url}{endpoint}"
            
            if timeout is None:
                timeout = self.default_timeout
            
            if method.upper() == "GET":
                response = await client.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = await client.post(url, json=data, timeout=timeout)
            elif method.upper() == "PUT":
                response = await client.put(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = await client.delete(url, timeout=timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful GET requests
            if use_cache and method.upper() == "GET" and result is not None:
                cache_key = cache._get_cache_key(f"api:{endpoint}", method, data or {})
                ttl = 300 if "health" not in endpoint else 60  # Different TTLs
                cache.set(cache_key, result, ttl)
            
            return result
            
        except httpx.TimeoutException:
            logger.warning(f"API timeout: {endpoint}")
            st.warning(f"⏱️ Request timeout for {endpoint}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {endpoint}")
            if e.response.status_code >= 500:
                st.error(f"🔥 Server error ({e.response.status_code}): {endpoint}")
            else:
                st.warning(f"⚠️ API error ({e.response.status_code}): {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Request error {endpoint}: {e}")
            st.error(f"🔌 Connection error: {endpoint}")
            return None

class StreamlitAPIIntegration:
    """Streamlit-optimized API integration with smart refresh"""
    
    def __init__(self):
        self.client = OptimizedAPIClient()
        self.refresh_intervals = {
            "health": 30,      # 30 seconds
            "metrics": 60,     # 1 minute
            "agents": 120,     # 2 minutes
            "models": 300      # 5 minutes
        }
    
    def should_refresh_data(self, data_type: str) -> bool:
        """Determine if data should be refreshed"""
        interval = self.refresh_intervals.get(data_type, 60)
        return SmartRefresh.should_refresh(data_type, interval)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data with smart refresh logic"""
        
        # Check what needs refreshing
        refresh_health = self.should_refresh_data("health")
        refresh_metrics = self.should_refresh_data("metrics")
        
        if refresh_health or refresh_metrics:
            # Fetch fresh data
            overview = await self.client.get_system_overview()
            
            # Mark as refreshed
            SmartRefresh.mark_refreshed("health")
            SmartRefresh.mark_refreshed("metrics")
            
            return overview
        else:
            # Use cached data
            cached_overview = cache.get("system:get_system_overview")
            if cached_overview:
                return cached_overview
            
            # Fallback to fresh fetch if no cache
            return await self.client.get_system_overview()
    
    def sync_call_api(self, endpoint: str, method: str = "GET", 
                     data: Dict = None, timeout: float = None) -> Optional[Dict]:
        """Synchronous wrapper for API calls"""
        return asyncio.run(
            self.client.call_api(endpoint, method, data, timeout)
        )
    
    def sync_health_check(self) -> Dict[str, Any]:
        """Synchronous health check"""
        return asyncio.run(self.client.health_check())
    
    def sync_get_dashboard_data(self) -> Dict[str, Any]:
        """Synchronous dashboard data fetch"""
        return asyncio.run(self.get_dashboard_data())

# Global optimized client instance
optimized_client = StreamlitAPIIntegration()

# Backward compatibility functions
def sync_call_api(endpoint: str, method: str = "GET", data: Dict = None, timeout: float = None) -> Optional[Dict]:
    """Backward compatible API call function"""
    return optimized_client.sync_call_api(endpoint, method, data, timeout)

def sync_check_service_health(url: str = None, timeout: float = 2.0) -> bool:
    """Backward compatible health check"""
    try:
        result = optimized_client.sync_health_check()
        return result.get("status") == "healthy"
    except:
        return False

# Cleanup function for app shutdown
def cleanup_client():
    """Clean up HTTP client connections"""
    asyncio.run(optimized_client.client.close())

# Register cleanup with Streamlit
if hasattr(st, 'session_state'):
    if 'api_client_registered' not in st.session_state:
        import atexit
        atexit.register(cleanup_client)
        st.session_state.api_client_registered = True

__all__ = ['optimized_client', 'sync_call_api', 'sync_check_service_health', 'StreamlitAPIIntegration']