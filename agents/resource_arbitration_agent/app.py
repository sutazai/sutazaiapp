#!/usr/bin/env python3
"""
Resource Arbitration Agent - Real Implementation with RabbitMQ
Manages resource allocation, conflict resolution, and capacity planning
"""
import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
from enum import Enum
import uuid

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psutil

# Add parent directory to path for imports
sys.path.append('/app')
sys.path.append('/opt/sutazaiapp/agents')

from core.messaging import (
    RabbitMQClient, MessageProcessor,
    ResourceMessage, StatusMessage, ErrorMessage,
    MessageType, Priority
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "resource-arbitration-agent"
REDIS_ALLOCATION_KEY = "resource:allocations"
REDIS_RESERVATION_KEY = "resource:reservations"
ALLOCATION_TTL_SECONDS = 3600
MAX_CPU_ALLOCATION = 0.8  # 80% max
MAX_MEMORY_ALLOCATION = 0.85  # 85% max
MAX_GPU_ALLOCATION = 0.9  # 90% max

# Resource Types
class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"

# Data Models
class ResourceAllocation(BaseModel):
    """Resource allocation record"""
    allocation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    resource_type: ResourceType
    amount: float
    unit: str  # cores, GB, Mbps, etc.
    allocated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    exclusive: bool = False
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = {}

class ResourceReservation(BaseModel):
    """Resource reservation request"""
    reservation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    resource_type: ResourceType
    requested_amount: float
    unit: str
    duration_seconds: Optional[int] = None
    priority: Priority = Priority.NORMAL
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, approved, denied, expired

class ResourceCapacity(BaseModel):
    """System resource capacity"""
    resource_type: ResourceType
    total_capacity: float
    allocated: float
    available: float
    unit: str
    utilization_percent: float

class AllocationPolicy(BaseModel):
    """Resource allocation policy"""
    resource_type: ResourceType
    max_allocation_percent: float = 80.0
    max_per_agent_percent: float = 30.0
    priority_weight: float = 1.0
    allow_oversubscription: bool = False
    oversubscription_ratio: float = 1.2

class ConflictResolution(BaseModel):
    """Resource conflict resolution"""
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType
    conflicting_agents: List[str]
    resolution_strategy: str  # priority_based, fair_share, preemption, queuing
    resolved_at: Optional[datetime] = None
    resolution_details: Dict[str, Any] = {}


class ResourceArbitrationMessageProcessor(MessageProcessor):
    """Message processor for resource arbitration"""
    
    def __init__(self, arbitrator):
        super().__init__(AGENT_ID)
        self.arbitrator = arbitrator
        
    async def handle_resource_request(self, message: Dict[str, Any]):
        """Handle resource allocation requests"""
        try:
            resource_msg = ResourceMessage(**message)
            
            # Create reservation
            reservation = ResourceReservation(
                agent_id=resource_msg.source_agent,
                resource_type=ResourceType(resource_msg.resource_type),
                requested_amount=resource_msg.resource_amount,
                unit=resource_msg.resource_unit,
                duration_seconds=resource_msg.duration_seconds,
                priority=Priority(resource_msg.priority)
            )
            
            # Process allocation
            allocation = await self.arbitrator.process_allocation_request(reservation)
            
            if allocation:
                # Send success response
                response = ResourceMessage(
                    message_id=f"{AGENT_ID}_response_{datetime.utcnow().timestamp()}",
                    message_type=MessageType.RESOURCE_RESPONSE,
                    source_agent=AGENT_ID,
                    resource_type=resource_msg.resource_type,
                    resource_amount=allocation.amount,
                    resource_unit=allocation.unit,
                    correlation_id=resource_msg.message_id
                )
                
                await self.rabbitmq_client.publish_message(
                    response,
                    routing_key=f"agent.{resource_msg.source_agent}.resource"
                )
                
                logger.info(f"Allocated {allocation.amount} {allocation.unit} of {allocation.resource_type} to {allocation.agent_id}")
            else:
                # Send denial
                await self.rabbitmq_client.publish_error(
                    error_code="RESOURCE_UNAVAILABLE",
                    error_message=f"Cannot allocate {resource_msg.resource_amount} {resource_msg.resource_unit} of {resource_msg.resource_type}",
                    original_message_id=resource_msg.message_id
                )
            
        except Exception as e:
            logger.error(f"Error handling resource request: {e}")
            await self.rabbitmq_client.publish_error(
                error_code="RESOURCE_REQUEST_ERROR",
                error_message=str(e),
                original_message_id=message.get("message_id")
            )
    
    async def handle_resource_release(self, message: Dict[str, Any]):
        """Handle resource release messages"""
        try:
            agent_id = message.get("source_agent")
            allocation_id = message.get("payload", {}).get("allocation_id")
            
            if allocation_id:
                await self.arbitrator.release_allocation(allocation_id)
                logger.info(f"Released allocation {allocation_id} from agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling resource release: {e}")


class ResourceArbitrationAgent:
    """Main resource arbitration implementation"""
    
    def __init__(self):
        self.redis_client = None
        self.message_processor = None
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.reservations: Dict[str, ResourceReservation] = {}
        self.policies: Dict[ResourceType, AllocationPolicy] = {}
        self.system_resources: Dict[ResourceType, ResourceCapacity] = {}
        self.conflicts: List[ConflictResolution] = []
        self.running = False
        self.allocation_lock = asyncio.Lock()
        
        # Initialize default policies
        self._init_default_policies()
        
    def _init_default_policies(self):
        """Initialize default allocation policies"""
        self.policies = {
            ResourceType.CPU: AllocationPolicy(
                resource_type=ResourceType.CPU,
                max_allocation_percent=MAX_CPU_ALLOCATION * 100,
                max_per_agent_percent=30.0
            ),
            ResourceType.MEMORY: AllocationPolicy(
                resource_type=ResourceType.MEMORY,
                max_allocation_percent=MAX_MEMORY_ALLOCATION * 100,
                max_per_agent_percent=25.0
            ),
            ResourceType.GPU: AllocationPolicy(
                resource_type=ResourceType.GPU,
                max_allocation_percent=MAX_GPU_ALLOCATION * 100,
                max_per_agent_percent=50.0,
                allow_oversubscription=False
            ),
            ResourceType.DISK: AllocationPolicy(
                resource_type=ResourceType.DISK,
                max_allocation_percent=90.0,
                max_per_agent_percent=20.0
            ),
            ResourceType.NETWORK: AllocationPolicy(
                resource_type=ResourceType.NETWORK,
                max_allocation_percent=95.0,
                max_per_agent_percent=15.0,
                allow_oversubscription=True,
                oversubscription_ratio=1.5
            )
        }
    
    async def initialize(self):
        """Initialize the arbitrator"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Initialize message processor
            self.message_processor = ResourceArbitrationMessageProcessor(self)
            
            # Register custom handlers
            self.message_processor.rabbitmq_client.register_handler(
                MessageType.RESOURCE_REQUEST,
                self.message_processor.handle_resource_request
            )
            
            await self.message_processor.start()
            logger.info("Message processor started")
            
            # Load existing allocations from Redis
            await self.load_allocations()
            
            # Discover system resources
            await self.discover_system_resources()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self.resource_monitor())
            asyncio.create_task(self.allocation_cleanup())
            asyncio.create_task(self.conflict_resolver())
            asyncio.create_task(self.capacity_planner())
            
            logger.info("Resource Arbitration Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize arbitrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the arbitrator"""
        self.running = False
        
        # Save allocations to Redis
        await self.save_allocations()
        
        if self.message_processor:
            await self.message_processor.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Resource Arbitration Agent shutdown complete")
    
    async def discover_system_resources(self):
        """Discover available system resources"""
        try:
            # CPU resources
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            self.system_resources[ResourceType.CPU] = ResourceCapacity(
                resource_type=ResourceType.CPU,
                total_capacity=cpu_count,
                allocated=self._calculate_allocated(ResourceType.CPU),
                available=cpu_count * (1 - cpu_percent/100),
                unit="cores",
                utilization_percent=cpu_percent
            )
            
            # Memory resources
            memory = psutil.virtual_memory()
            
            self.system_resources[ResourceType.MEMORY] = ResourceCapacity(
                resource_type=ResourceType.MEMORY,
                total_capacity=memory.total / (1024**3),  # Convert to GB
                allocated=self._calculate_allocated(ResourceType.MEMORY),
                available=memory.available / (1024**3),
                unit="GB",
                utilization_percent=memory.percent
            )
            
            # Disk resources
            disk = psutil.disk_usage('/')
            
            self.system_resources[ResourceType.DISK] = ResourceCapacity(
                resource_type=ResourceType.DISK,
                total_capacity=disk.total / (1024**3),
                allocated=self._calculate_allocated(ResourceType.DISK),
                available=disk.free / (1024**3),
                unit="GB",
                utilization_percent=disk.percent
            )
            
            # Network resources (simplified)
            self.system_resources[ResourceType.NETWORK] = ResourceCapacity(
                resource_type=ResourceType.NETWORK,
                total_capacity=1000.0,  # Assume 1Gbps
                allocated=self._calculate_allocated(ResourceType.NETWORK),
                available=900.0,  # Simplified
                unit="Mbps",
                utilization_percent=10.0
            )
            
            # GPU resources (check if available)
            # This is simplified - in production, use nvidia-ml-py or similar
            gpu_available = os.path.exists("/dev/nvidia0")
            if gpu_available:
                self.system_resources[ResourceType.GPU] = ResourceCapacity(
                    resource_type=ResourceType.GPU,
                    total_capacity=1.0,  # Number of GPUs
                    allocated=self._calculate_allocated(ResourceType.GPU),
                    available=1.0,
                    unit="GPUs",
                    utilization_percent=0.0
                )
            
            logger.info("System resources discovered")
            
        except Exception as e:
            logger.error(f"Error discovering system resources: {e}")
    
    def _calculate_allocated(self, resource_type: ResourceType) -> float:
        """Calculate total allocated resources of a type"""
        total = 0.0
        for allocation in self.allocations.values():
            if allocation.resource_type == resource_type:
                total += allocation.amount
        return total
    
    async def process_allocation_request(
        self, 
        reservation: ResourceReservation
    ) -> Optional[ResourceAllocation]:
        """Process a resource allocation request"""
        async with self.allocation_lock:
            try:
                # Check if resource type is available
                if reservation.resource_type not in self.system_resources:
                    logger.warning(f"Resource type {reservation.resource_type} not available")
                    reservation.status = "denied"
                    return None
                
                capacity = self.system_resources[reservation.resource_type]
                policy = self.policies.get(reservation.resource_type)
                
                # Check capacity constraints
                if not await self._check_capacity(reservation, capacity, policy):
                    reservation.status = "denied"
                    return None
                
                # Check for conflicts
                conflicts = await self._detect_conflicts(reservation)
                if conflicts:
                    # Try to resolve conflicts
                    if not await self._resolve_conflicts(conflicts, reservation):
                        reservation.status = "denied"
                        return None
                
                # Create allocation
                allocation = ResourceAllocation(
                    agent_id=reservation.agent_id,
                    resource_type=reservation.resource_type,
                    amount=reservation.requested_amount,
                    unit=reservation.unit,
                    priority=reservation.priority,
                    exclusive=reservation.resource_type == ResourceType.GPU  # GPUs are exclusive
                )
                
                if reservation.duration_seconds:
                    allocation.expires_at = datetime.utcnow() + timedelta(seconds=reservation.duration_seconds)
                
                # Store allocation
                self.allocations[allocation.allocation_id] = allocation
                reservation.status = "approved"
                
                # Update capacity
                capacity.allocated += allocation.amount
                capacity.available -= allocation.amount
                capacity.utilization_percent = (capacity.allocated / capacity.total_capacity) * 100
                
                # Store in Redis
                await self.redis_client.hset(
                    REDIS_ALLOCATION_KEY,
                    allocation.allocation_id,
                    allocation.json()
                )
                
                return allocation
                
            except Exception as e:
                logger.error(f"Error processing allocation request: {e}")
                reservation.status = "denied"
                return None
    
    async def _check_capacity(
        self,
        reservation: ResourceReservation,
        capacity: ResourceCapacity,
        policy: Optional[AllocationPolicy]
    ) -> bool:
        """Check if capacity is available for allocation"""
        if not policy:
            policy = AllocationPolicy(resource_type=reservation.resource_type)
        
        # Check total capacity limit
        max_allowed = capacity.total_capacity * (policy.max_allocation_percent / 100)
        if capacity.allocated + reservation.requested_amount > max_allowed:
            if not policy.allow_oversubscription:
                logger.warning(f"Capacity limit exceeded for {reservation.resource_type}")
                return False
            
            # Check oversubscription limit
            max_oversubscribed = capacity.total_capacity * policy.oversubscription_ratio
            if capacity.allocated + reservation.requested_amount > max_oversubscribed:
                logger.warning(f"Oversubscription limit exceeded for {reservation.resource_type}")
                return False
        
        # Check per-agent limit
        agent_allocated = sum(
            a.amount for a in self.allocations.values()
            if a.agent_id == reservation.agent_id and a.resource_type == reservation.resource_type
        )
        
        max_per_agent = capacity.total_capacity * (policy.max_per_agent_percent / 100)
        if agent_allocated + reservation.requested_amount > max_per_agent:
            logger.warning(f"Per-agent limit exceeded for {reservation.agent_id}")
            return False
        
        return True
    
    async def _detect_conflicts(
        self, 
        reservation: ResourceReservation
    ) -> List[ConflictResolution]:
        """Detect resource conflicts"""
        conflicts = []
        
        # Check for exclusive resource conflicts (e.g., GPU)
        if reservation.resource_type == ResourceType.GPU:
            gpu_allocations = [
                a for a in self.allocations.values()
                if a.resource_type == ResourceType.GPU and a.exclusive
            ]
            
            if gpu_allocations:
                conflict = ConflictResolution(
                    resource_type=ResourceType.GPU,
                    conflicting_agents=[a.agent_id for a in gpu_allocations] + [reservation.agent_id],
                    resolution_strategy="priority_based"
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _resolve_conflicts(
        self,
        conflicts: List[ConflictResolution],
        reservation: ResourceReservation
    ) -> bool:
        """Resolve resource conflicts"""
        for conflict in conflicts:
            if conflict.resolution_strategy == "priority_based":
                # Compare priorities
                for allocation_id, allocation in list(self.allocations.items()):
                    if (allocation.agent_id in conflict.conflicting_agents and
                        allocation.priority < reservation.priority):
                        # Preempt lower priority allocation
                        await self.release_allocation(allocation_id)
                        conflict.resolved_at = datetime.utcnow()
                        conflict.resolution_details = {
                            "preempted": allocation.agent_id,
                            "reason": "lower_priority"
                        }
                        return True
            
            elif conflict.resolution_strategy == "fair_share":
                # Implement fair sharing logic
                pass
            
            elif conflict.resolution_strategy == "queuing":
                # Queue the request for later
                self.reservations[reservation.reservation_id] = reservation
                return False
        
        return False
    
    async def release_allocation(self, allocation_id: str):
        """Release a resource allocation"""
        async with self.allocation_lock:
            try:
                if allocation_id in self.allocations:
                    allocation = self.allocations[allocation_id]
                    
                    # Update capacity
                    if allocation.resource_type in self.system_resources:
                        capacity = self.system_resources[allocation.resource_type]
                        capacity.allocated -= allocation.amount
                        capacity.available += allocation.amount
                        capacity.utilization_percent = (capacity.allocated / capacity.total_capacity) * 100
                    
                    # Remove allocation
                    del self.allocations[allocation_id]
                    
                    # Remove from Redis
                    await self.redis_client.hdel(REDIS_ALLOCATION_KEY, allocation_id)
                    
                    logger.info(f"Released allocation {allocation_id}")
                    
                    # Check if any queued reservations can now be fulfilled
                    await self._process_queued_reservations()
                    
            except Exception as e:
                logger.error(f"Error releasing allocation: {e}")
    
    async def _process_queued_reservations(self):
        """Process queued reservations after resources are freed"""
        for reservation_id, reservation in list(self.reservations.items()):
            if reservation.status == "pending":
                allocation = await self.process_allocation_request(reservation)
                if allocation:
                    del self.reservations[reservation_id]
    
    async def resource_monitor(self):
        """Monitor resource utilization"""
        while self.running:
            try:
                await self.discover_system_resources()
                
                # Publish metrics
                metrics = {
                    resource_type.value: {
                        "total": capacity.total_capacity,
                        "allocated": capacity.allocated,
                        "available": capacity.available,
                        "utilization_percent": capacity.utilization_percent
                    }
                    for resource_type, capacity in self.system_resources.items()
                }
                
                await self.redis_client.hset(
                    "resource:metrics",
                    AGENT_ID,
                    json.dumps(metrics)
                )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(60)
    
    async def allocation_cleanup(self):
        """Clean up expired allocations"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                expired_allocations = []
                
                for allocation_id, allocation in self.allocations.items():
                    if allocation.expires_at and current_time > allocation.expires_at:
                        expired_allocations.append(allocation_id)
                
                for allocation_id in expired_allocations:
                    await self.release_allocation(allocation_id)
                    logger.info(f"Cleaned up expired allocation {allocation_id}")
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in allocation cleanup: {e}")
                await asyncio.sleep(120)
    
    async def conflict_resolver(self):
        """Periodic conflict resolution"""
        while self.running:
            try:
                # Process any unresolved conflicts
                for conflict in self.conflicts:
                    if not conflict.resolved_at:
                        # Attempt resolution
                        pass
                
                await asyncio.sleep(45)
                
            except Exception as e:
                logger.error(f"Error in conflict resolver: {e}")
                await asyncio.sleep(90)
    
    async def capacity_planner(self):
        """Plan for future capacity needs"""
        while self.running:
            try:
                # Analyze allocation trends
                # Predict future needs
                # Generate recommendations
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in capacity planner: {e}")
                await asyncio.sleep(600)
    
    async def load_allocations(self):
        """Load existing allocations from Redis"""
        try:
            allocations_data = await self.redis_client.hgetall(REDIS_ALLOCATION_KEY)
            
            for allocation_id, allocation_json in allocations_data.items():
                allocation = ResourceAllocation(**json.loads(allocation_json))
                self.allocations[allocation_id] = allocation
            
            logger.info(f"Loaded {len(self.allocations)} existing allocations")
            
        except Exception as e:
            logger.error(f"Error loading allocations: {e}")
    
    async def save_allocations(self):
        """Save allocations to Redis"""
        try:
            for allocation_id, allocation in self.allocations.items():
                await self.redis_client.hset(
                    REDIS_ALLOCATION_KEY,
                    allocation_id,
                    allocation.json()
                )
            
            logger.info(f"Saved {len(self.allocations)} allocations")
            
        except Exception as e:
            logger.error(f"Error saving allocations: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get arbitrator status"""
        return {
            "status": "healthy",
            "total_allocations": len(self.allocations),
            "pending_reservations": len([r for r in self.reservations.values() if r.status == "pending"]),
            "resources": {
                resource_type.value: {
                    "allocated": capacity.allocated,
                    "available": capacity.available,
                    "utilization": round(capacity.utilization_percent, 2)
                }
                for resource_type, capacity in self.system_resources.items()
            }
        }


# Global arbitrator instance
arbitrator = ResourceArbitrationAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await arbitrator.initialize()
    yield
    # Shutdown
    await arbitrator.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Resource Arbitration Agent",
    description="Real implementation with resource management and RabbitMQ",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = await arbitrator.get_status()
    return {
        "status": "healthy",
        "agent": AGENT_ID,
        "timestamp": datetime.utcnow().isoformat(),
        **status
    }

@app.get("/resources")
async def get_resources():
    """Get resource capacity information"""
    resources = {}
    for resource_type, capacity in arbitrator.system_resources.items():
        resources[resource_type.value] = capacity.dict()
    return {"resources": resources}

@app.get("/allocations")
async def get_allocations():
    """Get current allocations"""
    allocations = []
    for allocation in arbitrator.allocations.values():
        allocations.append({
            "allocation_id": allocation.allocation_id,
            "agent_id": allocation.agent_id,
            "resource_type": allocation.resource_type.value,
            "amount": allocation.amount,
            "unit": allocation.unit,
            "allocated_at": allocation.allocated_at.isoformat(),
            "expires_at": allocation.expires_at.isoformat() if allocation.expires_at else None
        })
    return {"allocations": allocations}

@app.post("/allocate")
async def request_allocation(reservation: ResourceReservation):
    """Request resource allocation"""
    allocation = await arbitrator.process_allocation_request(reservation)
    
    if allocation:
        return {
            "status": "approved",
            "allocation_id": allocation.allocation_id,
            "amount": allocation.amount,
            "unit": allocation.unit
        }
    else:
        return {
            "status": "denied",
            "reason": "Insufficient resources or policy violation"
        }

@app.delete("/allocations/{allocation_id}")
async def release_allocation(allocation_id: str):
    """Release a resource allocation"""
    await arbitrator.release_allocation(allocation_id)
    return {"status": "released", "allocation_id": allocation_id}

@app.get("/policies")
async def get_policies():
    """Get allocation policies"""
    policies = {}
    for resource_type, policy in arbitrator.policies.items():
        policies[resource_type.value] = policy.dict()
    return {"policies": policies}

@app.put("/policies/{resource_type}")
async def update_policy(resource_type: str, policy: AllocationPolicy):
    """Update allocation policy for a resource type"""
    try:
        rt = ResourceType(resource_type)
        arbitrator.policies[rt] = policy
        return {"status": "updated", "resource_type": resource_type}
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")

@app.get("/status")
async def get_arbitrator_status():
    """Get detailed arbitrator status"""
    return await arbitrator.get_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8588"))
    uvicorn.run(app, host="0.0.0.0", port=port)