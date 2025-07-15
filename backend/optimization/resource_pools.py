"""
Resource Pool Management for SutazAI
Advanced resource pooling and lifecycle management
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import weakref
from contextlib import asynccontextmanager
import queue

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class PoolConfig:
    """Pool configuration"""
    min_size: int = 5
    max_size: int = 20
    max_idle_time: float = 300.0  # 5 minutes
    cleanup_interval: float = 60.0  # 1 minute
    acquisition_timeout: float = 30.0
    validation_interval: float = 120.0  # 2 minutes

class PooledResource(ABC):
    """Base class for pooled resources"""
    
    def __init__(self):
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0
        self.is_valid = True
    
    @abstractmethod
    async def initialize(self):
        """Initialize the resource"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup the resource"""
        pass
    
    @abstractmethod
    async def validate(self) -> bool:
        """Validate that the resource is still usable"""
        pass
    
    def touch(self):
        """Update last used timestamp"""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_expired(self, max_idle_time: float) -> bool:
        """Check if resource has expired"""
        return time.time() - self.last_used > max_idle_time

class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management"""
    
    def __init__(self, 
                 resource_factory: Callable[[], T], 
                 config: PoolConfig = None):
        self.resource_factory = resource_factory
        self.config = config or PoolConfig()
        
        # Pool state
        self.available_resources = asyncio.Queue()
        self.all_resources = set()
        self.resource_lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "created": 0,
            "destroyed": 0,
            "acquisitions": 0,
            "releases": 0,
            "timeouts": 0,
            "validation_failures": 0
        }
        
        # Background tasks
        self.cleanup_task = None
        self.validation_task = None
        self.pool_active = False
    
    async def initialize(self):
        """Initialize the resource pool"""
        logger.info(f"🔄 Initializing resource pool (min: {self.config.min_size}, max: {self.config.max_size})")
        
        self.pool_active = True
        
        # Create minimum number of resources
        for _ in range(self.config.min_size):
            resource = await self._create_resource()
            if resource:
                await self.available_resources.put(resource)
        
        # Start background tasks
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.validation_task = asyncio.create_task(self._validation_loop())
        
        logger.info("✅ Resource pool initialized")
    
    async def _create_resource(self) -> Optional[T]:
        """Create a new resource"""
        try:
            resource = self.resource_factory()
            
            if hasattr(resource, 'initialize'):
                await resource.initialize()
            
            async with self.resource_lock:
                self.all_resources.add(resource)
                self.stats["created"] += 1
            
            logger.debug(f"Created new resource (total: {len(self.all_resources)})")
            return resource
            
        except Exception as e:
            logger.error(f"Failed to create resource: {e}")
            return None
    
    async def _destroy_resource(self, resource: T):
        """Destroy a resource"""
        try:
            if hasattr(resource, 'cleanup'):
                await resource.cleanup()
            
            async with self.resource_lock:
                self.all_resources.discard(resource)
                self.stats["destroyed"] += 1
            
            logger.debug(f"Destroyed resource (remaining: {len(self.all_resources)})")
            
        except Exception as e:
            logger.error(f"Failed to destroy resource: {e}")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a resource from the pool"""
        resource = None
        try:
            resource = await self._acquire_resource()
            if resource:
                if hasattr(resource, 'touch'):
                    resource.touch()
                yield resource
            else:
                raise RuntimeError("Failed to acquire resource")
        finally:
            if resource:
                await self._release_resource(resource)
    
    async def _acquire_resource(self) -> Optional[T]:
        """Acquire a resource from the pool"""
        self.stats["acquisitions"] += 1
        
        try:
            # Try to get from available resources
            try:
                resource = await asyncio.wait_for(
                    self.available_resources.get(),
                    timeout=0.1
                )
                
                # Validate resource before returning
                if hasattr(resource, 'validate'):
                    if not await resource.validate():
                        await self._destroy_resource(resource)
                        return await self._acquire_resource()  # Retry
                
                return resource
                
            except asyncio.TimeoutError:
                # No available resources, create new one if under limit
                async with self.resource_lock:
                    if len(self.all_resources) < self.config.max_size:
                        return await self._create_resource()
                
                # Wait for resource with timeout
                try:
                    resource = await asyncio.wait_for(
                        self.available_resources.get(),
                        timeout=self.config.acquisition_timeout
                    )
                    
                    # Validate resource
                    if hasattr(resource, 'validate'):
                        if not await resource.validate():
                            await self._destroy_resource(resource)
                            return await self._acquire_resource()  # Retry
                    
                    return resource
                    
                except asyncio.TimeoutError:
                    self.stats["timeouts"] += 1
                    raise RuntimeError("Resource acquisition timeout")
        
        except Exception as e:
            logger.error(f"Resource acquisition failed: {e}")
            return None
    
    async def _release_resource(self, resource: T):
        """Release a resource back to the pool"""
        self.stats["releases"] += 1
        
        try:
            # Validate resource before returning to pool
            if hasattr(resource, 'validate'):
                if not await resource.validate():
                    await self._destroy_resource(resource)
                    return
            
            # Return to pool
            await self.available_resources.put(resource)
            
        except Exception as e:
            logger.error(f"Resource release failed: {e}")
            await self._destroy_resource(resource)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.pool_active:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired_resources()
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_expired_resources(self):
        """Clean up expired resources"""
        try:
            expired_resources = []
            
            # Check all resources for expiration
            async with self.resource_lock:
                for resource in list(self.all_resources):
                    if (hasattr(resource, 'is_expired') and 
                        resource.is_expired(self.config.max_idle_time)):
                        expired_resources.append(resource)
            
            # Destroy expired resources
            for resource in expired_resources:
                await self._destroy_resource(resource)
            
            # Ensure minimum pool size
            async with self.resource_lock:
                current_size = len(self.all_resources)
                if current_size < self.config.min_size:
                    for _ in range(self.config.min_size - current_size):
                        new_resource = await self._create_resource()
                        if new_resource:
                            await self.available_resources.put(new_resource)
            
            if expired_resources:
                logger.info(f"Cleaned up {len(expired_resources)} expired resources")
                
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
    
    async def _validation_loop(self):
        """Background validation loop"""
        while self.pool_active:
            try:
                await asyncio.sleep(self.config.validation_interval)
                await self._validate_all_resources()
            except Exception as e:
                logger.error(f"Validation loop error: {e}")
    
    async def _validate_all_resources(self):
        """Validate all resources in the pool"""
        try:
            invalid_resources = []
            
            async with self.resource_lock:
                for resource in list(self.all_resources):
                    if hasattr(resource, 'validate'):
                        try:
                            if not await resource.validate():
                                invalid_resources.append(resource)
                                self.stats["validation_failures"] += 1
                        except Exception as e:
                            logger.warning(f"Resource validation error: {e}")
                            invalid_resources.append(resource)
            
            # Remove invalid resources
            for resource in invalid_resources:
                await self._destroy_resource(resource)
            
            if invalid_resources:
                logger.info(f"Removed {len(invalid_resources)} invalid resources")
                
        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            **self.stats,
            "total_resources": len(self.all_resources),
            "available_resources": self.available_resources.qsize(),
            "utilization": (
                (len(self.all_resources) - self.available_resources.qsize()) / 
                max(1, len(self.all_resources))
            ) * 100,
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_idle_time": self.config.max_idle_time
            }
        }
    
    async def shutdown(self):
        """Shutdown the resource pool"""
        logger.info("🛑 Shutting down resource pool")
        
        self.pool_active = False
        
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.validation_task:
            self.validation_task.cancel()
        
        # Destroy all resources
        async with self.resource_lock:
            for resource in list(self.all_resources):
                await self._destroy_resource(resource)
        
        logger.info("✅ Resource pool shutdown complete")

class ConnectionPooledResource(PooledResource):
    """Example pooled resource for database connections"""
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.connection_string = connection_string
        self.connection = None
    
    async def initialize(self):
        """Initialize database connection"""
        # Simulate connection initialization
        await asyncio.sleep(0.1)
        self.connection = f"connection-{id(self)}"
        logger.debug(f"Initialized connection: {self.connection}")
    
    async def cleanup(self):
        """Cleanup database connection"""
        if self.connection:
            # Simulate connection cleanup
            await asyncio.sleep(0.05)
            self.connection = None
            logger.debug("Connection cleaned up")
    
    async def validate(self) -> bool:
        """Validate database connection"""
        # Simulate connection validation
        return self.connection is not None and self.is_valid

# Global resource pools
class ResourcePoolManager:
    """Manages multiple resource pools"""
    
    def __init__(self):
        self.pools = {}
    
    def register_pool(self, name: str, pool: ResourcePool):
        """Register a resource pool"""
        self.pools[name] = pool
    
    async def initialize_all(self):
        """Initialize all registered pools"""
        for name, pool in self.pools.items():
            try:
                await pool.initialize()
                logger.info(f"✅ Initialized pool: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize pool {name}: {e}")
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """Get a resource pool by name"""
        return self.pools.get(name)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools"""
        return {
            name: pool.get_pool_stats()
            for name, pool in self.pools.items()
        }
    
    async def shutdown_all(self):
        """Shutdown all pools"""
        for name, pool in self.pools.items():
            try:
                await pool.shutdown()
                logger.info(f"✅ Shutdown pool: {name}")
            except Exception as e:
                logger.error(f"Failed to shutdown pool {name}: {e}")

# Global pool manager
pool_manager = ResourcePoolManager()
