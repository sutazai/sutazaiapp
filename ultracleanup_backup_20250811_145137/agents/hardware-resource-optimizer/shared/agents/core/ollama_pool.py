#!/usr/bin/env python3
"""
Ollama Connection Pool for SutazAI System
Efficient connection management for limited hardware resources

Features:
- Connection pooling with configurable limits
- Automatic connection reuse and cleanup
- Model warming and caching
- Request queuing with priorities
- Health monitoring and automatic recovery
- Resource-efficient for CPU-only hardware
"""

import asyncio
import httpx
import time
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import weakref


logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state tracking"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class PoolConnection:
    """Individual connection in the pool"""
    client: httpx.AsyncClient
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    error_count: int = 0
    current_model: Optional[str] = None


@dataclass
class ModelCache:
    """Model warming cache entry"""
    model_name: str
    last_used: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    average_response_time: float = 0.0
    is_warmed: bool = False


@dataclass
class PoolStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    error_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    models_cached: int = 0
    pool_hits: int = 0
    pool_misses: int = 0


class OllamaConnectionPool:
    """
    Efficient connection pool for Ollama API with resource optimization
    
    Designed for limited hardware environments with focus on:
    - Minimal memory footprint
    - Connection reuse
    - Model warming and caching
    - Graceful degradation under load
    """
    
    def __init__(self,
                 base_url: str = "http://localhost:10104",
                 max_connections: int = 2,  # Conservative for limited hardware
                 min_connections: int = 1,
                 default_model: str = "tinyllama",
                 connection_timeout: int = 30,
                 request_timeout: int = 300,
                 max_retries: int = 3,
                 cleanup_interval: int = 300,  # 5 minutes
                 model_cache_size: int = 5):
        
        self.base_url = base_url.rstrip('/')
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.default_model = default_model
        self.connection_timeout = connection_timeout
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.cleanup_interval = cleanup_interval
        self.model_cache_size = model_cache_size
        
        # Connection pool
        self._connections: List[PoolConnection] = []
        self._connection_lock = asyncio.Lock()
        self._request_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        
        # Model caching
        self._model_cache: Dict[str, ModelCache] = {}
        self._available_models: Optional[Set[str]] = None
        self._models_last_checked: Optional[datetime] = None
        
        # Statistics
        self.stats = PoolStats()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        
        # Finalizer for cleanup
        self._finalizer = weakref.finalize(self, self._cleanup_sync)
        
        logger.info(f"Initialized Ollama pool: {max_connections} max connections, model: {default_model}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _initialize(self):
        """Initialize the connection pool"""
        try:
            # Start with minimum connections
            for _ in range(self.min_connections):
                await self._create_connection()
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._health_task = asyncio.create_task(self._health_monitor())
            
            # Warm up default model
            await self._warm_model(self.default_model)
            
            logger.info(f"Connection pool initialized with {len(self._connections)} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def _create_connection(self) -> PoolConnection:
        """Create a new connection"""
        try:
            timeout = httpx.Timeout(
                connect=self.connection_timeout,
                read=self.request_timeout,
                write=30.0,
                pool=60.0
            )
            
            # Configure client for efficiency
            limits = httpx.Limits(
                max_keepalive_connections=1,
                max_connections=1,
                keepalive_expiry=300.0
            )
            
            client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                follow_redirects=True
            )
            
            connection = PoolConnection(client=client)
            
            # Test the connection
            await self._test_connection(connection)
            
            self._connections.append(connection)
            self.stats.total_connections += 1
            
            return connection
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            raise
    
    async def _test_connection(self, connection: PoolConnection):
        """Test if a connection is working"""
        try:
            response = await connection.client.get(f"{self.base_url}/api/version")
            if response.status_code != 200:
                raise Exception(f"Health check failed: {response.status_code}")
            
            connection.last_used = datetime.utcnow()
            connection.state = ConnectionState.IDLE
            
        except Exception as e:
            connection.state = ConnectionState.ERROR
            connection.error_count += 1
            raise Exception(f"Connection test failed: {e}")
    
    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool"""
        connection = None
        
        try:
            async with self._connection_lock:
                # Find idle connection
                for conn in self._connections:
                    if conn.state == ConnectionState.IDLE:
                        conn.state = ConnectionState.BUSY
                        conn.last_used = datetime.utcnow()
                        connection = conn
                        self.stats.pool_hits += 1
                        break
                
                # No idle connection, create new if under limit
                if not connection and len(self._connections) < self.max_connections:
                    try:
                        connection = await self._create_connection()
                        connection.state = ConnectionState.BUSY
                        self.stats.pool_misses += 1
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
                
                # Still no connection, wait for one to become available
                if not connection:
                    self.stats.pool_misses += 1
                    # Find the least recently used busy connection and wait
                    await asyncio.sleep(0.1)  # Small delay before retry
                    raise Exception("No connections available and at max capacity")
            
            if connection:
                self.stats.active_connections += 1
                yield connection
            else:
                raise Exception("Could not acquire connection")
                
        except Exception as e:
            if connection:
                connection.state = ConnectionState.ERROR
                connection.error_count += 1
            raise
            
        finally:
            if connection:
                connection.state = ConnectionState.IDLE
                connection.request_count += 1
                self.stats.active_connections = max(0, self.stats.active_connections - 1)
    
    async def _ensure_model_available(self, model: str) -> bool:
        """Ensure model is available, pull if necessary"""
        try:
            # Check cache first
            if model in self._model_cache:
                cache_entry = self._model_cache[model]
                cache_entry.last_used = datetime.utcnow()
                if cache_entry.is_warmed:
                    return True
            
            # Check available models (cached for 5 minutes)
            if (not self._available_models or 
                not self._models_last_checked or 
                datetime.utcnow() - self._models_last_checked > timedelta(minutes=5)):
                
                await self._refresh_available_models()
            
            if model in self._available_models:
                # Model is available, add to cache
                if model not in self._model_cache:
                    self._model_cache[model] = ModelCache(model_name=model)
                return True
            
            # Model not available, try to pull it
            logger.info(f"Model {model} not available, attempting to pull...")
            return await self._pull_model(model)
            
        except Exception as e:
            logger.error(f"Error ensuring model {model} is available: {e}")
            return False
    
    async def _refresh_available_models(self):
        """Refresh the list of available models"""
        try:
            async with self._get_connection() as connection:
                response = await connection.client.get(f"{self.base_url}/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    self._available_models = {
                        model.get("name", "").split(":")[0] 
                        for model in models 
                        if model.get("name")
                    }
                    self._models_last_checked = datetime.utcnow()
                    self.stats.models_cached = len(self._available_models)
                    
                    logger.debug(f"Refreshed model list: {self._available_models}")
                else:
                    logger.error(f"Failed to get model list: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error refreshing available models: {e}")
    
    async def _pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            async with self._get_connection() as connection:
                logger.info(f"Pulling model {model}...")
                
                response = await connection.client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model},
                    timeout=httpx.Timeout(600.0)  # Extended timeout for model pull
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model {model}")
                    
                    # Add to available models and cache
                    if not self._available_models:
                        self._available_models = set()
                    self._available_models.add(model)
                    
                    self._model_cache[model] = ModelCache(model_name=model)
                    return True
                else:
                    logger.error(f"Failed to pull model {model}: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model}: {e}")
            return False
    
    async def _warm_model(self, model: str):
        """Warm up a model with a simple request"""
        try:
            if not await self._ensure_model_available(model):
                return
            
            logger.debug(f"Warming up model {model}")
            
            # Simple warmup request
            await self.generate(
                prompt="Hello",
                model=model,
                temperature=0.1,
                max_tokens=10
            )
            
            # Mark as warmed
            if model in self._model_cache:
                self._model_cache[model].is_warmed = True
            
            logger.debug(f"Model {model} warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm up model {model}: {e}")
    
    async def generate(self,
                      prompt: str,
                      model: str = None,
                      system: str = None,
                      temperature: float = 0.7,
                      max_tokens: int = 2048,
                      **kwargs) -> Optional[str]:
        """Generate response using Ollama"""
        model = model or self.default_model
        start_time = time.time()
        
        try:
            # Ensure model is available
            if not await self._ensure_model_available(model):
                logger.error(f"Model {model} not available")
                return None
            
            # Build request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }
            
            if system:
                payload["system"] = system
            
            # Make request using connection pool
            async with self._get_connection() as connection:
                response = await connection.client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    
                    # Update statistics
                    response_time = time.time() - start_time
                    self.stats.total_requests += 1
                    self._update_average_response_time(response_time)
                    
                    # Update model cache
                    if model in self._model_cache:
                        cache_entry = self._model_cache[model]
                        cache_entry.request_count += 1
                        cache_entry.last_used = datetime.utcnow()
                        
                        # Update average response time
                        if cache_entry.request_count == 1:
                            cache_entry.average_response_time = response_time
                        else:
                            cache_entry.average_response_time = (
                                (cache_entry.average_response_time * (cache_entry.request_count - 1) + response_time) /
                                cache_entry.request_count
                            )
                    
                    return response_text
                else:
                    logger.error(f"Ollama generate error: {response.status_code} - {response.text}")
                    self.stats.failed_requests += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error in generate: {e}")
            self.stats.failed_requests += 1
            return None
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str = None,
                  temperature: float = 0.7,
                  max_tokens: int = 2048,
                  **kwargs) -> Optional[str]:
        """Chat completion using Ollama"""
        model = model or self.default_model
        start_time = time.time()
        
        try:
            # Ensure model is available
            if not await self._ensure_model_available(model):
                logger.error(f"Model {model} not available")
                return None
            
            # Build request payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    **kwargs
                }
            }
            
            # Make request using connection pool
            async with self._get_connection() as connection:
                response = await connection.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("message", {}).get("content", "")
                    
                    # Update statistics
                    response_time = time.time() - start_time
                    self.stats.total_requests += 1
                    self._update_average_response_time(response_time)
                    
                    # Update model cache
                    if model in self._model_cache:
                        cache_entry = self._model_cache[model]
                        cache_entry.request_count += 1
                        cache_entry.last_used = datetime.utcnow()
                    
                    return response_text
                else:
                    logger.error(f"Ollama chat error: {response.status_code} - {response.text}")
                    self.stats.failed_requests += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            self.stats.failed_requests += 1
            return None
    
    async def embeddings(self,
                        text: str,
                        model: str = None) -> Optional[List[float]]:
        """Generate embeddings using Ollama"""
        model = model or self.default_model
        
        try:
            # Ensure model is available
            if not await self._ensure_model_available(model):
                logger.error(f"Model {model} not available")
                return None
            
            payload = {
                "model": model,
                "prompt": text
            }
            
            async with self._get_connection() as connection:
                response = await connection.client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.stats.total_requests += 1
                    return result.get("embedding", [])
                else:
                    logger.error(f"Ollama embeddings error: {response.status_code}")
                    self.stats.failed_requests += 1
                    return None
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            self.stats.failed_requests += 1
            return None
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time"""
        if self.stats.total_requests == 1:
            self.stats.average_response_time = response_time
        else:
            self.stats.average_response_time = (
                (self.stats.average_response_time * (self.stats.total_requests - 1) + response_time) /
                self.stats.total_requests
            )
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_connections()
                await self._cleanup_model_cache()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.cleanup_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
    async def _cleanup_connections(self):
        """Clean up old or error connections"""
        current_time = datetime.utcnow()
        connections_to_remove = []
        
        async with self._connection_lock:
            for i, connection in enumerate(self._connections):
                # Remove error connections
                if connection.state == ConnectionState.ERROR:
                    connections_to_remove.append(i)
                    continue
                
                # Remove idle connections that are too old (keep minimum)
                if (connection.state == ConnectionState.IDLE and
                    len(self._connections) > self.min_connections and
                    current_time - connection.last_used > timedelta(minutes=30)):
                    connections_to_remove.append(i)
                    continue
            
            # Remove connections (reverse order to maintain indices)
            for i in reversed(connections_to_remove):
                connection = self._connections.pop(i)
                await connection.client.aclose()
                self.stats.total_connections -= 1
                logger.debug(f"Cleaned up connection {i}")
    
    async def _cleanup_model_cache(self):
        """Clean up model cache"""
        if len(self._model_cache) <= self.model_cache_size:
            return
        
        # Sort by last used time and keep only the most recent
        sorted_models = sorted(
            self._model_cache.items(),
            key=lambda x: x[1].last_used,
            reverse=True
        )
        
        # Keep only the most recently used models
        models_to_keep = dict(sorted_models[:self.model_cache_size])
        removed_count = len(self._model_cache) - len(models_to_keep)
        
        self._model_cache = models_to_keep
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} model cache entries")
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                # Update connection statistics
                async with self._connection_lock:
                    self.stats.idle_connections = sum(
                        1 for conn in self._connections 
                        if conn.state == ConnectionState.IDLE
                    )
                    self.stats.error_connections = sum(
                        1 for conn in self._connections 
                        if conn.state == ConnectionState.ERROR
                    )
                
                # Ensure minimum connections
                if len(self._connections) < self.min_connections:
                    try:
                        await self._create_connection()
                        logger.info("Created connection to maintain minimum pool size")
                    except Exception as e:
                        logger.error(f"Failed to maintain minimum connections: {e}")
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60  # Health check every minute
                )
                break
            except asyncio.TimeoutError:
                continue
    
    async def health_check(self) -> bool:
        """Check if the pool is healthy"""
        try:
            async with self._get_connection() as connection:
                await self._test_connection(connection)
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "total_connections": self.stats.total_connections,
            "active_connections": self.stats.active_connections,
            "idle_connections": self.stats.idle_connections,
            "error_connections": self.stats.error_connections,
            "total_requests": self.stats.total_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": (
                (self.stats.total_requests - self.stats.failed_requests) / 
                max(self.stats.total_requests, 1)
            ),
            "average_response_time": self.stats.average_response_time,
            "models_cached": len(self._model_cache),
            "pool_hits": self.stats.pool_hits,
            "pool_misses": self.stats.pool_misses,
            "hit_rate": self.stats.pool_hits / max(self.stats.pool_hits + self.stats.pool_misses, 1),
            "available_models": list(self._available_models) if self._available_models else []
        }
    
    @staticmethod
    def _cleanup_sync():
        """Synchronous cleanup for finalizer"""
        # This is called by weakref finalizer, no async operations allowed
        pass
    
    async def close(self):
        """Close the connection pool"""
        logger.info("Closing Ollama connection pool...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._health_task:
            self._health_task.cancel()
        
        # Close all connections
        async with self._connection_lock:
            for connection in self._connections:
                try:
                    await connection.client.aclose()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._connections.clear()
        
        logger.info("Connection pool closed")


# Factory function for easy instantiation
def create_ollama_pool(max_connections: int = 2, 
                      default_model: str = "tinyllama",
                      **kwargs) -> OllamaConnectionPool:
    """Factory function to create an Ollama connection pool"""
    return OllamaConnectionPool(
        max_connections=max_connections,
        default_model=default_model,
        **kwargs
    )