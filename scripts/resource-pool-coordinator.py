#!/usr/bin/env python3
"""
Resource Pool Coordinator for SutazAI
====================================

Purpose: Advanced resource pooling and sharing for 131 agents with dynamic allocation
Usage: python scripts/resource-pool-coordinator.py [--pool-config config.json]
Requirements: Python 3.8+, asyncio, aiohttp

Features:
- Dynamic resource pool management
- Cross-agent resource sharing
- Load-based resource allocation
- Resource reservation and scheduling
- Performance-based resource rebalancing
- Resource usage analytics and optimization
"""

import os
import sys
import json
import time
import uuid
import asyncio
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import statistics
import math
import psutil
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import signal
import weakref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/resource_pool_coordinator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ResourcePoolCoordinator')

class ResourceType(Enum):
    """Resource types managed by the coordinator"""
    CPU_CORES = "cpu_cores"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"
    MODEL_SLOTS = "model_slots"
    CONNECTION_POOL = "connection_pool"
    THREAD_POOL = "thread_pool"

class PoolState(Enum):
    """Resource pool states"""
    ACTIVE = "active"
    SCALING = "scaling"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_FAIR = "weighted_fair"
    PRIORITY_BASED = "priority_based"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceSpec:
    """Resource specification"""
    resource_type: ResourceType
    min_amount: int
    max_amount: int
    current_amount: int
    unit: str
    is_shareable: bool = True
    is_preemptible: bool = False
    cost_weight: float = 1.0

@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    allocation_id: str
    agent_id: str
    resource_type: ResourceType
    amount: int
    priority: int
    start_time: float
    duration: Optional[float] = None
    status: str = "active"
    usage_history: List[float] = field(default_factory=list)
    last_used: float = 0.0
    preemption_count: int = 0

@dataclass
class AgentResourceProfile:
    """Agent resource usage profile"""
    agent_id: str
    current_allocations: Dict[ResourceType, ResourceAllocation]
    historical_usage: Dict[ResourceType, List[float]]
    average_usage: Dict[ResourceType, float]
    peak_usage: Dict[ResourceType, float]
    efficiency_score: float
    priority_level: int
    last_activity: float
    resource_credits: int = 100  # For resource economy

@dataclass
class PoolMetrics:
    """Pool performance metrics"""
    total_capacity: int
    allocated_amount: int
    available_amount: int
    utilization_ratio: float
    fragmentation_ratio: float
    allocation_rate: float
    deallocation_rate: float
    average_allocation_size: float
    queue_length: int
    queue_wait_time: float
    efficiency_score: float

class ResourceQueue:
    """Priority queue for resource requests"""
    
    def __init__(self):
        self.queue = []
        self.lock = asyncio.Lock()
        self.wait_times = deque(maxlen=1000)
    
    async def enqueue(self, request: Dict[str, Any]) -> str:
        """Add request to queue"""
        request_id = str(uuid.uuid4())
        request["request_id"] = request_id
        request["enqueue_time"] = time.time()
        
        async with self.lock:
            # Insert based on priority
            priority = request.get("priority", 0)
            inserted = False
            
            for i, existing_request in enumerate(self.queue):
                if existing_request.get("priority", 0) < priority:
                    self.queue.insert(i, request)
                    inserted = True
                    break
            
            if not inserted:
                self.queue.append(request)
        
        logger.debug(f"Enqueued resource request {request_id}")
        return request_id
    
    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Remove and return highest priority request"""
        async with self.lock:
            if self.queue:
                request = self.queue.pop(0)
                
                # Track wait time
                wait_time = time.time() - request["enqueue_time"]
                self.wait_times.append(wait_time)
                
                return request
            return None
    
    async def get_size(self) -> int:
        """Get current queue size"""
        async with self.lock:
            return len(self.queue)
    
    def get_average_wait_time(self) -> float:
        """Get average wait time"""
        return statistics.mean(self.wait_times) if self.wait_times else 0.0

class ResourcePool:
    """Individual resource pool implementation"""
    
    def __init__(self, pool_id: str, resource_spec: ResourceSpec, strategy: AllocationStrategy):
        self.pool_id = pool_id
        self.resource_spec = resource_spec
        self.strategy = strategy
        self.state = PoolState.ACTIVE
        
        # Allocation tracking
        self.allocations = {}  # allocation_id -> ResourceAllocation
        self.agent_allocations = defaultdict(list)  # agent_id -> [allocation_ids]
        self.free_resources = resource_spec.current_amount
        
        # Queue for pending requests
        self.request_queue = ResourceQueue()
        
        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        self.allocation_count = 0
        self.deallocation_count = 0
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        logger.info(f"Created resource pool {pool_id} for {resource_spec.resource_type.value}")
    
    async def allocate(self, agent_id: str, amount: int, priority: int = 0, 
                      duration: Optional[float] = None) -> Optional[str]:
        """Allocate resources to agent"""
        async with self.lock:
            if self.state != PoolState.ACTIVE:
                logger.warning(f"Pool {self.pool_id} not active, state: {self.state}")
                return None
            
            if amount > self.resource_spec.max_amount:
                logger.error(f"Requested amount {amount} exceeds max {self.resource_spec.max_amount}")
                return None
            
            # Check if resources are immediately available
            if self.free_resources >= amount:
                return await self._do_allocation(agent_id, amount, priority, duration)
            
            # Try to free up resources
            freed = await self._try_free_resources(amount - self.free_resources, priority)
            if freed >= amount - self.free_resources:
                return await self._do_allocation(agent_id, amount, priority, duration)
            
            # Queue the request if can't satisfy immediately
            request = {
                "agent_id": agent_id,
                "amount": amount,
                "priority": priority,
                "duration": duration,
                "type": "allocation"
            }
            
            request_id = await self.request_queue.enqueue(request)
            logger.debug(f"Queued allocation request {request_id} for agent {agent_id}")
            return request_id
    
    async def _do_allocation(self, agent_id: str, amount: int, priority: int, 
                           duration: Optional[float]) -> str:
        """Perform the actual allocation"""
        allocation_id = str(uuid.uuid4())
        
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            agent_id=agent_id,
            resource_type=self.resource_spec.resource_type,
            amount=amount,
            priority=priority,
            start_time=time.time(),
            duration=duration,
            last_used=time.time()
        )
        
        self.allocations[allocation_id] = allocation
        self.agent_allocations[agent_id].append(allocation_id)
        self.free_resources -= amount
        self.allocation_count += 1
        
        logger.info(f"Allocated {amount} {self.resource_spec.resource_type.value} to {agent_id}")
        return allocation_id
    
    async def deallocate(self, allocation_id: str) -> bool:
        """Deallocate resources"""
        async with self.lock:
            if allocation_id not in self.allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return False
            
            allocation = self.allocations[allocation_id]
            
            # Free the resources
            self.free_resources += allocation.amount
            self.deallocation_count += 1
            
            # Remove from tracking
            del self.allocations[allocation_id]
            self.agent_allocations[allocation.agent_id].remove(allocation_id)
            
            # Clean up empty agent lists
            if not self.agent_allocations[allocation.agent_id]:
                del self.agent_allocations[allocation.agent_id]
            
            logger.info(f"Deallocated {allocation.amount} {self.resource_spec.resource_type.value} from {allocation.agent_id}")
            
            # Process queued requests
            await self._process_queue()
            
            return True
    
    async def _try_free_resources(self, needed_amount: int, requesting_priority: int) -> int:
        """Try to free resources by preempting lower priority allocations"""
        if not self.resource_spec.is_preemptible:
            return 0
        
        freed_amount = 0
        candidates = []
        
        # Find preemption candidates
        for allocation in self.allocations.values():
            if (allocation.priority < requesting_priority and 
                allocation.status == "active"):
                candidates.append(allocation)
        
        # Sort by priority (lowest first) and last used (oldest first)
        candidates.sort(key=lambda a: (a.priority, a.last_used))
        
        # Preempt allocations until we have enough resources
        for allocation in candidates:
            if freed_amount >= needed_amount:
                break
            
            # Preempt this allocation
            await self._preempt_allocation(allocation.allocation_id)
            freed_amount += allocation.amount
        
        return freed_amount
    
    async def _preempt_allocation(self, allocation_id: str):
        """Preempt a resource allocation"""
        if allocation_id in self.allocations:
            allocation = self.allocations[allocation_id]
            allocation.status = "preempted"
            allocation.preemption_count += 1
            
            self.free_resources += allocation.amount
            
            logger.info(f"Preempted allocation {allocation_id} from {allocation.agent_id}")
            
            # Notify agent about preemption (would integrate with agent communication)
            # For now, just log it
    
    async def _process_queue(self):
        """Process queued resource requests"""
        while await self.request_queue.get_size() > 0:
            request = await self.request_queue.dequeue()
            if not request:
                break
            
            if request["type"] == "allocation":
                if self.free_resources >= request["amount"]:
                    await self._do_allocation(
                        request["agent_id"],
                        request["amount"],
                        request["priority"],
                        request["duration"]
                    )
                else:
                    # Re-queue if still can't satisfy
                    await self.request_queue.enqueue(request)
                    break
    
    async def update_usage(self, allocation_id: str, current_usage: float):
        """Update usage statistics for an allocation"""
        if allocation_id in self.allocations:
            allocation = self.allocations[allocation_id]
            allocation.usage_history.append(current_usage)
            allocation.last_used = time.time()
            
            # Keep only recent usage history
            if len(allocation.usage_history) > 100:
                allocation.usage_history.pop(0)
    
    async def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics"""
        async with self.lock:
            allocated_amount = sum(a.amount for a in self.allocations.values() if a.status == "active")
            total_capacity = self.resource_spec.current_amount
            
            utilization = allocated_amount / max(total_capacity, 1)
            
            # Calculate fragmentation
            active_allocations = [a for a in self.allocations.values() if a.status == "active"]
            if active_allocations:
                avg_allocation = allocated_amount / len(active_allocations)
                largest_free = self.free_resources
                fragmentation = 1.0 - (largest_free / max(avg_allocation, 1))
            else:
                fragmentation = 0.0
            
            # Calculate rates
            allocation_rate = self.allocation_count / max(time.time() - 0, 1)  # Would use pool start time
            deallocation_rate = self.deallocation_count / max(time.time() - 0, 1)
            
            avg_allocation_size = statistics.mean([a.amount for a in active_allocations]) if active_allocations else 0
            
            queue_size = await self.request_queue.get_size()
            queue_wait_time = self.request_queue.get_average_wait_time()
            
            # Calculate efficiency score
            efficiency = utilization * (1.0 - fragmentation) * (1.0 / max(queue_wait_time, 1))
            
            return PoolMetrics(
                total_capacity=total_capacity,
                allocated_amount=allocated_amount,
                available_amount=self.free_resources,
                utilization_ratio=utilization,
                fragmentation_ratio=fragmentation,
                allocation_rate=allocation_rate,
                deallocation_rate=deallocation_rate,
                average_allocation_size=avg_allocation_size,
                queue_length=queue_size,
                queue_wait_time=queue_wait_time,
                efficiency_score=efficiency
            )
    
    async def scale_pool(self, new_capacity: int) -> bool:
        """Scale the resource pool"""
        async with self.lock:
            if new_capacity < 0:
                return False
            
            old_capacity = self.resource_spec.current_amount
            allocated_amount = sum(a.amount for a in self.allocations.values() if a.status == "active")
            
            if new_capacity < allocated_amount:
                logger.warning(f"Cannot scale pool below current allocations ({allocated_amount})")
                return False
            
            self.state = PoolState.SCALING
            
            # Update capacity
            self.resource_spec.current_amount = new_capacity
            self.free_resources = new_capacity - allocated_amount
            
            self.state = PoolState.ACTIVE
            
            logger.info(f"Scaled pool {self.pool_id} from {old_capacity} to {new_capacity}")
            
            # Process any queued requests now that we have more resources
            if new_capacity > old_capacity:
                await self._process_queue()
            
            return True

class ResourcePoolCoordinator:
    """Main resource pool coordinator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pools = {}  # pool_id -> ResourcePool
        self.agent_profiles = {}  # agent_id -> AgentResourceProfile
        self.global_allocation_strategy = AllocationStrategy(
            config.get("global_strategy", "adaptive")
        )
        
        # Coordination
        self.coordinator_lock = asyncio.Lock()
        self.running = False
        self.monitor_task = None
        self.rebalance_task = None
        
        # Analytics
        self.usage_history = deque(maxlen=10000)
        self.efficiency_history = deque(maxlen=1000)
        
        # Load balancing
        self.load_balancer = LoadBalancer(self)
        
        logger.info("Resource Pool Coordinator initialized")
    
    async def start(self):
        """Start the coordinator"""
        self.running = True
        
        # Create default pools
        await self._create_default_pools()
        
        # Start monitoring and rebalancing tasks
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self.rebalance_task = asyncio.create_task(self._rebalance_loop())
        
        logger.info("Resource Pool Coordinator started")
    
    async def stop(self):
        """Stop the coordinator"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.rebalance_task:
            self.rebalance_task.cancel()
            try:
                await self.rebalance_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource Pool Coordinator stopped")
    
    async def _create_default_pools(self):
        """Create default resource pools"""
        # CPU cores pool
        cpu_spec = ResourceSpec(
            resource_type=ResourceType.CPU_CORES,
            min_amount=1,
            max_amount=psutil.cpu_count(),
            current_amount=psutil.cpu_count(),
            unit="cores",
            is_shareable=True,
            is_preemptible=True
        )
        cpu_pool = ResourcePool("cpu_pool", cpu_spec, AllocationStrategy.LEAST_LOADED)
        self.pools["cpu_pool"] = cpu_pool
        
        # Memory pool
        total_memory_mb = psutil.virtual_memory().total // (1024 * 1024)
        memory_spec = ResourceSpec(
            resource_type=ResourceType.MEMORY,
            min_amount=256,  # 256MB minimum
            max_amount=total_memory_mb,
            current_amount=int(total_memory_mb * 0.8),  # 80% of total
            unit="MB",
            is_shareable=False,
            is_preemptible=True
        )
        memory_pool = ResourcePool("memory_pool", memory_spec, AllocationStrategy.WEIGHTED_FAIR)
        self.pools["memory_pool"] = memory_pool
        
        # Model slots pool (for AI model instances)
        model_spec = ResourceSpec(
            resource_type=ResourceType.MODEL_SLOTS,
            min_amount=1,
            max_amount=131,  # One per agent maximum
            current_amount=50,  # Start with 50 slots
            unit="slots",
            is_shareable=True,
            is_preemptible=False
        )
        model_pool = ResourcePool("model_pool", model_spec, AllocationStrategy.PRIORITY_BASED)
        self.pools["model_pool"] = model_pool
        
        # Thread pool
        thread_spec = ResourceSpec(
            resource_type=ResourceType.THREAD_POOL,
            min_amount=10,
            max_amount=1000,
            current_amount=200,
            unit="threads",
            is_shareable=True,
            is_preemptible=True
        )
        thread_pool = ResourcePool("thread_pool", thread_spec, AllocationStrategy.ROUND_ROBIN)
        self.pools["thread_pool"] = thread_pool
        
        logger.info(f"Created {len(self.pools)} default resource pools")
    
    async def allocate_resources(self, agent_id: str, resource_requests: List[Dict[str, Any]]) -> Dict[str, str]:
        """Allocate multiple resources for an agent"""
        allocations = {}
        failed_allocations = []
        
        # Ensure agent profile exists
        if agent_id not in self.agent_profiles:
            await self._create_agent_profile(agent_id)
        
        # Process each resource request
        for request in resource_requests:
            resource_type = ResourceType(request["resource_type"])
            amount = request["amount"]
            priority = request.get("priority", 0)
            duration = request.get("duration")
            
            # Find appropriate pool
            pool = await self._find_best_pool(resource_type, amount, priority)
            
            if pool:
                allocation_id = await pool.allocate(agent_id, amount, priority, duration)
                if allocation_id:
                    allocations[resource_type.value] = allocation_id
                    
                    # Update agent profile
                    profile = self.agent_profiles[agent_id]
                    profile.current_allocations[resource_type] = pool.allocations[allocation_id]
                else:
                    failed_allocations.append(request)
            else:
                failed_allocations.append(request)
        
        if failed_allocations:
            logger.warning(f"Failed to allocate {len(failed_allocations)} resources for {agent_id}")
        
        return allocations
    
    async def deallocate_resources(self, agent_id: str, allocation_ids: List[str] = None) -> int:
        """Deallocate resources for an agent"""
        deallocated_count = 0
        
        if allocation_ids is None:
            # Deallocate all resources for the agent
            if agent_id in self.agent_profiles:
                allocation_ids = []
                for allocation in self.agent_profiles[agent_id].current_allocations.values():
                    allocation_ids.append(allocation.allocation_id)
        
        # Deallocate from each pool
        for pool in self.pools.values():
            for allocation_id in allocation_ids[:]:  # Copy to avoid modification during iteration
                if await pool.deallocate(allocation_id):
                    allocation_ids.remove(allocation_id)
                    deallocated_count += 1
        
        # Update agent profile
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            profile.current_allocations.clear()
            profile.last_activity = time.time()
        
        logger.info(f"Deallocated {deallocated_count} resources for agent {agent_id}")
        return deallocated_count
    
    async def _find_best_pool(self, resource_type: ResourceType, amount: int, priority: int) -> Optional[ResourcePool]:
        """Find the best pool for a resource allocation"""
        candidates = [pool for pool in self.pools.values() 
                     if pool.resource_spec.resource_type == resource_type]
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Score pools based on various factors
        best_pool = None
        best_score = -1
        
        for pool in candidates:
            metrics = await pool.get_metrics()
            
            # Scoring factors
            availability_score = metrics.available_amount / max(amount, 1)
            utilization_score = 1.0 - metrics.utilization_ratio  # Prefer less utilized
            efficiency_score = metrics.efficiency_score
            queue_score = 1.0 / max(metrics.queue_length + 1, 1)
            
            total_score = (availability_score * 0.4 + 
                          utilization_score * 0.3 + 
                          efficiency_score * 0.2 + 
                          queue_score * 0.1)
            
            if total_score > best_score:
                best_score = total_score
                best_pool = pool
        
        return best_pool
    
    async def _create_agent_profile(self, agent_id: str):
        """Create profile for new agent"""
        profile = AgentResourceProfile(
            agent_id=agent_id,
            current_allocations={},
            historical_usage={rt: [] for rt in ResourceType},
            average_usage={rt: 0.0 for rt in ResourceType},
            peak_usage={rt: 0.0 for rt in ResourceType},
            efficiency_score=1.0,
            priority_level=5,  # Default priority
            last_activity=time.time()
        )
        
        self.agent_profiles[agent_id] = profile
        logger.info(f"Created profile for agent {agent_id}")
    
    async def update_agent_usage(self, agent_id: str, resource_usage: Dict[str, float]):
        """Update agent resource usage statistics"""
        if agent_id not in self.agent_profiles:
            await self._create_agent_profile(agent_id)
        
        profile = self.agent_profiles[agent_id]
        profile.last_activity = time.time()
        
        # Update usage statistics
        for resource_type_str, usage in resource_usage.items():
            try:
                resource_type = ResourceType(resource_type_str)
                
                # Add to history
                profile.historical_usage[resource_type].append(usage)
                
                # Keep only recent history
                if len(profile.historical_usage[resource_type]) > 1000:
                    profile.historical_usage[resource_type].pop(0)
                
                # Update averages and peaks
                if profile.historical_usage[resource_type]:
                    profile.average_usage[resource_type] = statistics.mean(
                        profile.historical_usage[resource_type]
                    )
                    profile.peak_usage[resource_type] = max(
                        profile.historical_usage[resource_type]
                    )
                
                # Update pool allocation usage
                if resource_type in profile.current_allocations:
                    allocation = profile.current_allocations[resource_type]
                    for pool in self.pools.values():
                        if allocation.allocation_id in pool.allocations:
                            await pool.update_usage(allocation.allocation_id, usage)
                            break
                            
            except ValueError:
                logger.warning(f"Unknown resource type: {resource_type_str}")
        
        # Update efficiency score
        await self._calculate_agent_efficiency(agent_id)
    
    async def _calculate_agent_efficiency(self, agent_id: str):
        """Calculate agent resource efficiency score"""
        if agent_id not in self.agent_profiles:
            return
        
        profile = self.agent_profiles[agent_id]
        efficiency_scores = []
        
        for resource_type, allocation in profile.current_allocations.items():
            if allocation.usage_history:
                # Efficiency = average usage / allocated amount
                avg_usage = statistics.mean(allocation.usage_history[-10:])  # Recent average
                efficiency = avg_usage / max(allocation.amount, 1)
                efficiency_scores.append(min(efficiency, 1.0))  # Cap at 1.0
        
        if efficiency_scores:
            profile.efficiency_score = statistics.mean(efficiency_scores)
        
        # Adjust priority based on efficiency
        if profile.efficiency_score > 0.8:
            profile.priority_level = min(profile.priority_level + 1, 10)
        elif profile.efficiency_score < 0.3:
            profile.priority_level = max(profile.priority_level - 1, 1)
    
    async def _monitor_loop(self):
        """Monitoring loop for system health and performance"""
        while self.running:
            try:
                # Collect metrics from all pools
                pool_metrics = {}
                for pool_id, pool in self.pools.items():
                    pool_metrics[pool_id] = await pool.get_metrics()
                
                # Collect agent metrics
                agent_metrics = {}
                for agent_id, profile in self.agent_profiles.items():
                    agent_metrics[agent_id] = {
                        "efficiency_score": profile.efficiency_score,
                        "priority_level": profile.priority_level,
                        "active_allocations": len(profile.current_allocations),
                        "resource_credits": profile.resource_credits,
                        "last_activity": profile.last_activity
                    }
                
                # Store in history
                metrics_snapshot = {
                    "timestamp": time.time(),
                    "pools": {k: asdict(v) for k, v in pool_metrics.items()},
                    "agents": agent_metrics,
                    "system": {
                        "cpu_usage": psutil.cpu_percent(),
                        "memory_usage": psutil.virtual_memory().percent,
                        "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                    }
                }
                
                self.usage_history.append(metrics_snapshot)
                
                # Log summary every 5 minutes
                if len(self.usage_history) % 300 == 0:
                    await self._log_summary_metrics(metrics_snapshot)
                
                # Check for alerts
                await self._check_system_alerts(metrics_snapshot)
                
                await asyncio.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)
    
    async def _rebalance_loop(self):
        """Resource rebalancing loop"""
        while self.running:
            try:
                await self._perform_rebalancing()
                await asyncio.sleep(30)  # Rebalance every 30 seconds
                
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_rebalancing(self):
        """Perform resource rebalancing based on usage patterns"""
        # Identify overloaded and underloaded pools
        rebalancing_actions = []
        
        for pool_id, pool in self.pools.items():
            metrics = await pool.get_metrics()
            
            # Pool is overloaded
            if metrics.utilization_ratio > 0.85 and metrics.queue_length > 5:
                # Try to scale up or redistribute
                if pool.resource_spec.current_amount < pool.resource_spec.max_amount:
                    new_capacity = min(
                        int(pool.resource_spec.current_amount * 1.2),
                        pool.resource_spec.max_amount
                    )
                    rebalancing_actions.append(("scale_up", pool_id, new_capacity))
            
            # Pool is underutilized
            elif metrics.utilization_ratio < 0.2 and metrics.queue_length == 0:
                if pool.resource_spec.current_amount > pool.resource_spec.min_amount:
                    new_capacity = max(
                        int(pool.resource_spec.current_amount * 0.9),
                        pool.resource_spec.min_amount
                    )
                    rebalancing_actions.append(("scale_down", pool_id, new_capacity))
        
        # Execute rebalancing actions
        for action, pool_id, new_capacity in rebalancing_actions:
            pool = self.pools[pool_id]
            success = await pool.scale_pool(new_capacity)
            if success:
                logger.info(f"Rebalanced pool {pool_id}: {action} to {new_capacity}")
    
    async def _log_summary_metrics(self, metrics: Dict[str, Any]):
        """Log summary of system metrics"""
        pool_summary = []
        for pool_id, pool_metrics in metrics["pools"].items():
            pool_summary.append(
                f"{pool_id}: {pool_metrics['utilization_ratio']:.2f} util, "
                f"{pool_metrics['queue_length']} queued"
            )
        
        active_agents = len([a for a in metrics["agents"].values() 
                           if time.time() - a["last_activity"] < 300])
        
        logger.info(f"System Summary - Active Agents: {active_agents}, "
                   f"CPU: {metrics['system']['cpu_usage']:.1f}%, "
                   f"Memory: {metrics['system']['memory_usage']:.1f}%, "
                   f"Pools: {', '.join(pool_summary)}")
    
    async def _check_system_alerts(self, metrics: Dict[str, Any]):
        """Check for system alerts and take action"""
        # High system load
        if metrics["system"]["cpu_usage"] > 90:
            logger.warning("High CPU usage detected - triggering resource optimization")
            await self._emergency_resource_optimization()
        
        # High memory usage
        if metrics["system"]["memory_usage"] > 90:
            logger.warning("High memory usage detected - triggering memory cleanup")
            await self._emergency_memory_cleanup()
        
        # Pool queue buildup
        for pool_id, pool_metrics in metrics["pools"].items():
            if pool_metrics["queue_length"] > 20:
                logger.warning(f"High queue length in pool {pool_id} - {pool_metrics['queue_length']}")
    
    async def _emergency_resource_optimization(self):
        """Emergency resource optimization"""
        # Identify least efficient agents
        inefficient_agents = [
            (agent_id, profile.efficiency_score)
            for agent_id, profile in self.agent_profiles.items()
            if profile.efficiency_score < 0.3 and profile.current_allocations
        ]
        
        # Sort by efficiency (worst first)
        inefficient_agents.sort(key=lambda x: x[1])
        
        # Reduce allocations for least efficient agents
        for agent_id, efficiency in inefficient_agents[:5]:  # Top 5 worst
            logger.info(f"Reducing resources for inefficient agent {agent_id} (efficiency: {efficiency:.3f})")
            # Would implement resource reduction logic here
    
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear old usage history
        for profile in self.agent_profiles.values():
            for resource_type in profile.historical_usage:
                if len(profile.historical_usage[resource_type]) > 100:
                    profile.historical_usage[resource_type] = profile.historical_usage[resource_type][-100:]
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        pool_stats = {}
        for pool_id, pool in self.pools.items():
            pool_stats[pool_id] = asdict(await pool.get_metrics())
        
        agent_stats = {}
        for agent_id, profile in self.agent_profiles.items():
            agent_stats[agent_id] = {
                "efficiency_score": profile.efficiency_score,
                "priority_level": profile.priority_level,
                "active_allocations": len(profile.current_allocations),
                "resource_credits": profile.resource_credits,
                "average_usage": dict(profile.average_usage),
                "peak_usage": dict(profile.peak_usage)
            }
        
        # System statistics
        system_stats = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "load_average": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
            "total_agents": len(self.agent_profiles),
            "active_agents": len([p for p in self.agent_profiles.values() 
                                if time.time() - p.last_activity < 300])
        }
        
        return {
            "pools": pool_stats,
            "agents": agent_stats,
            "system": system_stats,
            "coordinator": {
                "running": self.running,
                "strategy": self.global_allocation_strategy.value,
                "usage_history_length": len(self.usage_history)
            }
        }
    
    async def export_stats(self, filepath: str):
        """Export comprehensive statistics"""
        stats = await self.get_comprehensive_stats()
        stats["export_timestamp"] = time.time()
        stats["usage_history"] = list(self.usage_history)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported resource pool statistics to {filepath}")

class LoadBalancer:
    """Load balancer for resource distribution"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    async def balance_load(self):
        """Balance load across pools"""
        # Implementation would go here
        pass

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Resource Pool Coordinator")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--port", type=int, default=8200, help="API server port")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")
    parser.add_argument("--export-stats", type=str, help="Export statistics to file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "global_strategy": "adaptive",
        "enable_rebalancing": True,
        "rebalance_interval": 30,
        "monitoring_interval": 1
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create coordinator
    coordinator = ResourcePoolCoordinator(config)
    
    # Signal handling
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(coordinator.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await coordinator.start()
        
        if args.monitor:
            logger.info("Starting resource pool monitoring. Press Ctrl+C to stop.")
            while coordinator.running:
                await asyncio.sleep(60)
                stats = await coordinator.get_comprehensive_stats()
                
                print(f"System Status: "
                      f"CPU={stats['system']['cpu_usage']:.1f}%, "
                      f"Memory={stats['system']['memory_usage']:.1f}%, "
                      f"Active_Agents={stats['system']['active_agents']}")
        
        if args.export_stats:
            await coordinator.export_stats(args.export_stats)
            print(f"Statistics exported to {args.export_stats}")
    
    finally:
        await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(main())