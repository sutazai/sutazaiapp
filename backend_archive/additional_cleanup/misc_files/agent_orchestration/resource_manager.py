#!/usr/bin/env python3
"""
Resource Manager for Agent Orchestration
Manages system resources, allocation, and optimization
"""

import asyncio
import logging
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class ResourceConfig:
    """Configuration for resource manager"""
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    max_concurrent_tasks: int = 50
    resource_check_interval: float = 5.0
    enable_gpu: bool = False

@dataclass
class ResourceAllocation:
    """Resource allocation for an agent or task"""
    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_memory_mb: int = 0
    priority: int = 3  # 1-5, 1 is highest
    timeout: Optional[float] = None

@dataclass
class SystemResources:
    """Current system resource status"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    available_memory_gb: float = 0.0
    total_memory_gb: float = 0.0
    disk_percent: float = 0.0
    gpu_available: bool = False
    gpu_memory_free: float = 0.0

class ResourceManager:
    """Manages system resources for agent orchestration"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_locks = asyncio.Lock()
        self.current_resources = SystemResources()
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
    async def initialize(self) -> bool:
        """Initialize the resource manager"""
        try:
            logger.info("Initializing resource manager...")
            await self._update_system_resources()
            self._monitor_task = asyncio.create_task(self._monitor_resources())
            logger.info("Resource manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Resource manager initialization failed: {e}")
            return False
    
    async def start(self):
        """Start resource monitoring"""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_resources())
        logger.info("Resource manager started")
    
    async def stop(self):
        """Stop resource monitoring"""
        self._shutdown = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource manager stopped")
    
    async def allocate_resources(self, 
                                agent_id: str, 
                                allocation: ResourceAllocation) -> bool:
        """Allocate resources for an agent"""
        async with self.resource_locks:
            try:
                # Check if resources are available
                if not await self._can_allocate(allocation):
                    logger.warning(f"Insufficient resources for agent {agent_id}")
                    return False
                
                # Allocate resources
                self.allocations[agent_id] = allocation
                logger.info(f"Resources allocated for agent {agent_id}: {allocation}")
                return True
                
            except Exception as e:
                logger.error(f"Resource allocation failed for {agent_id}: {e}")
                return False
    
    async def release_resources(self, agent_id: str) -> bool:
        """Release resources for an agent"""
        async with self.resource_locks:
            try:
                if agent_id in self.allocations:
                    allocation = self.allocations.pop(agent_id)
                    logger.info(f"Resources released for agent {agent_id}: {allocation}")
                    return True
                return False
            except Exception as e:
                logger.error(f"Resource release failed for {agent_id}: {e}")
                return False
    
    async def get_resource_status(self) -> SystemResources:
        """Get current system resource status"""
        await self._update_system_resources()
        return self.current_resources
    
    async def get_allocation_status(self) -> Dict[str, Any]:
        """Get current resource allocations"""
        return {
            "total_allocations": len(self.allocations),
            "allocations": dict(self.allocations),
            "system_resources": self.current_resources,
            "resource_limits": {
                "max_cpu_percent": self.config.max_cpu_percent,
                "max_memory_percent": self.config.max_memory_percent,
                "max_concurrent_tasks": self.config.max_concurrent_tasks
            }
        }
    
    def health_check(self) -> bool:
        """Check resource manager health"""
        try:
            # Check if system resources are within limits
            if self.current_resources.cpu_percent > self.config.max_cpu_percent:
                logger.warning(f"CPU usage high: {self.current_resources.cpu_percent}%")
                return False
            
            if self.current_resources.memory_percent > self.config.max_memory_percent:
                logger.warning(f"Memory usage high: {self.current_resources.memory_percent}%")
                return False
            
            if len(self.allocations) > self.config.max_concurrent_tasks:
                logger.warning(f"Too many concurrent tasks: {len(self.allocations)}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Resource manager health check failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status"""
        return {
            "healthy": self.health_check(),
            "current_allocations": len(self.allocations),
            "max_allocations": self.config.max_concurrent_tasks,
            "system_resources": {
                "cpu_percent": self.current_resources.cpu_percent,
                "memory_percent": self.current_resources.memory_percent,
                "disk_percent": self.current_resources.disk_percent
            }
        }
    
    async def _can_allocate(self, allocation: ResourceAllocation) -> bool:
        """Check if resources can be allocated"""
        try:
            # Check current system usage
            await self._update_system_resources()
            
            # Calculate projected usage
            current_allocations = len(self.allocations)
            if current_allocations >= self.config.max_concurrent_tasks:
                return False
            
            # Check CPU and memory availability (simplified check)
            if (self.current_resources.cpu_percent > self.config.max_cpu_percent or
                self.current_resources.memory_percent > self.config.max_memory_percent):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Resource availability check failed: {e}")
            return False
    
    async def _update_system_resources(self):
        """Update current system resource information"""
        try:
            # Get CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.current_resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                available_memory_gb=memory.available / (1024**3),
                total_memory_gb=memory.total / (1024**3),
                disk_percent=disk.percent,
                gpu_available=False,  # TODO: Add GPU detection
                gpu_memory_free=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to update system resources: {e}")
    
    async def _monitor_resources(self):
        """Monitor system resources continuously"""
        while not self._shutdown:
            try:
                await self._update_system_resources()
                
                # Log warnings for high resource usage
                if self.current_resources.cpu_percent > self.config.max_cpu_percent:
                    logger.warning(f"High CPU usage: {self.current_resources.cpu_percent}%")
                
                if self.current_resources.memory_percent > self.config.max_memory_percent:
                    logger.warning(f"High memory usage: {self.current_resources.memory_percent}%")
                
                await asyncio.sleep(self.config.resource_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)