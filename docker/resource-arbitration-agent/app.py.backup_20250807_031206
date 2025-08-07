#!/usr/bin/env python3
"""
Resource Arbitration Agent - Manages resource allocation across agents
"""
import os
import asyncio
import json
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceRequest(BaseModel):
    agent_id: str
    request_id: str
    cpu_cores: float = Field(default=1.0, description="Requested CPU cores")
    memory_mb: int = Field(default=1024, description="Requested memory in MB")
    gpu_count: int = Field(default=0, description="Requested GPU count")
    priority: int = Field(default=5, description="Request priority (1-10)")
    duration_minutes: int = Field(default=60, description="Expected usage duration")

class ResourceAllocation(BaseModel):
    allocation_id: str
    agent_id: str
    request_id: str
    allocated_cpu: float
    allocated_memory_mb: int
    allocated_gpu: int
    start_time: datetime
    end_time: datetime
    status: str  # allocated, active, released

class SystemResources(BaseModel):
    total_cpu_cores: float
    available_cpu_cores: float
    total_memory_mb: int
    available_memory_mb: int
    total_gpu_count: int
    available_gpu_count: int
    cpu_utilization: float
    memory_utilization: float

class ResourceArbitrationAgent:
    def __init__(self):
        self.redis_client = None
        self.resource_requests = {}
        self.active_allocations = {}
        self.system_resources = None
        self.allocation_history = []
        
        # Resource limits and policies
        self.max_cpu_allocation = 0.8  # Max 80% of total CPU
        self.max_memory_allocation = 0.8  # Max 80% of total memory
        self.reservation_buffer = 0.1  # 10% buffer for system processes
        
    async def initialize(self):
        """Initialize Redis connection and start monitoring"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Initialize system resource monitoring
            await self.update_system_resources()
            
            # Start background tasks
            asyncio.create_task(self.resource_monitor())
            asyncio.create_task(self.allocation_cleanup())
            asyncio.create_task(self.usage_tracker())
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def resource_monitor(self):
        """Background task to monitor system resources"""
        while True:
            try:
                await self.update_system_resources()
                await self.process_pending_requests()
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                await asyncio.sleep(30)
    
    async def allocation_cleanup(self):
        """Clean up expired allocations"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_allocations = []
                
                for allocation_id, allocation in self.active_allocations.items():
                    if allocation.end_time <= current_time:
                        expired_allocations.append(allocation_id)
                
                # Release expired allocations
                for allocation_id in expired_allocations:
                    await self.release_allocation(allocation_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in allocation cleanup: {e}")
                await asyncio.sleep(300)
    
    async def usage_tracker(self):
        """Track actual resource usage vs allocated"""
        while True:
            try:
                # Monitor actual usage of allocated resources
                for allocation_id, allocation in self.active_allocations.items():
                    if allocation.status == "active":
                        # Here we would ideally check actual usage from the agent
                        # For now, we'll log the allocation for monitoring
                        logger.debug(f"Tracking allocation {allocation_id} for agent {allocation.agent_id}")
                
                await asyncio.sleep(30)  # Track every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in usage tracker: {e}")
                await asyncio.sleep(60)
    
    async def update_system_resources(self):
        """Update current system resource information"""
        try:
            # Get CPU information
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_available = cpu_count * (1 - cpu_percent / 100.0)
            
            # Get memory information
            memory = psutil.virtual_memory()
            total_memory_mb = memory.total // (1024 * 1024)
            available_memory_mb = memory.available // (1024 * 1024)
            
            # For GPU, we'll use a simplified approach (would need nvidia-ml-py for real GPU info)
            total_gpu = int(os.getenv("TOTAL_GPU_COUNT", "0"))
            
            # Calculate allocated resources
            allocated_cpu = sum(alloc.allocated_cpu for alloc in self.active_allocations.values())
            allocated_memory = sum(alloc.allocated_memory_mb for alloc in self.active_allocations.values())
            allocated_gpu = sum(alloc.allocated_gpu for alloc in self.active_allocations.values())
            
            available_cpu = max(0, cpu_count * self.max_cpu_allocation - allocated_cpu)
            available_memory = max(0, total_memory_mb * self.max_memory_allocation - allocated_memory)
            available_gpu = max(0, total_gpu - allocated_gpu)
            
            self.system_resources = SystemResources(
                total_cpu_cores=cpu_count,
                available_cpu_cores=available_cpu,
                total_memory_mb=total_memory_mb,
                available_memory_mb=available_memory,
                total_gpu_count=total_gpu,
                available_gpu_count=available_gpu,
                cpu_utilization=cpu_percent,
                memory_utilization=(1 - memory.available / memory.total) * 100
            )
            
            # Store in Redis for other services
            if self.redis_client:
                await self.redis_client.set(
                    "system_resources",
                    json.dumps(self.system_resources.dict()),
                    ex=60  # Expire after 60 seconds
                )
                
        except Exception as e:
            logger.error(f"Error updating system resources: {e}")
    
    async def request_resources(self, request: ResourceRequest) -> Dict[str, Any]:
        """Handle a resource allocation request"""
        try:
            # Validate request
            if request.cpu_cores <= 0 or request.memory_mb <= 0:
                raise HTTPException(status_code=400, detail="Invalid resource requirements")
            
            # Check if resources are available
            if not await self.can_allocate_resources(request):
                # Queue the request if resources not immediately available
                self.resource_requests[request.request_id] = {
                    "request": request,
                    "queued_at": datetime.utcnow(),
                    "status": "queued"
                }
                
                return {
                    "request_id": request.request_id,
                    "status": "queued",
                    "message": "Request queued due to insufficient resources",
                    "estimated_wait_minutes": await self.estimate_wait_time(request)
                }
            
            # Allocate resources immediately
            allocation = await self.allocate_resources(request)
            
            return {
                "request_id": request.request_id,
                "allocation_id": allocation.allocation_id,
                "status": "allocated",
                "allocated_resources": {
                    "cpu_cores": allocation.allocated_cpu,
                    "memory_mb": allocation.allocated_memory_mb,
                    "gpu_count": allocation.allocated_gpu
                },
                "expires_at": allocation.end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing resource request {request.request_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def can_allocate_resources(self, request: ResourceRequest) -> bool:
        """Check if requested resources can be allocated"""
        if not self.system_resources:
            return False
        
        return (
            request.cpu_cores <= self.system_resources.available_cpu_cores and
            request.memory_mb <= self.system_resources.available_memory_mb and
            request.gpu_count <= self.system_resources.available_gpu_count
        )
    
    async def allocate_resources(self, request: ResourceRequest) -> ResourceAllocation:
        """Allocate resources for a request"""
        allocation_id = f"alloc_{request.request_id}_{int(datetime.utcnow().timestamp())}"
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=request.duration_minutes)
        
        allocation = ResourceAllocation(
            allocation_id=allocation_id,
            agent_id=request.agent_id,
            request_id=request.request_id,
            allocated_cpu=request.cpu_cores,
            allocated_memory_mb=request.memory_mb,
            allocated_gpu=request.gpu_count,
            start_time=start_time,
            end_time=end_time,
            status="allocated"
        )
        
        self.active_allocations[allocation_id] = allocation
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.hset(
                "resource_allocations",
                allocation_id,
                json.dumps(allocation.dict(), default=str)
            )
        
        # Remove from pending requests if it was queued
        if request.request_id in self.resource_requests:
            del self.resource_requests[request.request_id]
        
        logger.info(f"Allocated resources for agent {request.agent_id}: {allocation_id}")
        
        return allocation
    
    async def release_allocation(self, allocation_id: str) -> Dict[str, Any]:
        """Release a resource allocation"""
        if allocation_id not in self.active_allocations:
            raise HTTPException(status_code=404, detail="Allocation not found")
        
        allocation = self.active_allocations[allocation_id]
        allocation.status = "released"
        
        # Move to history
        self.allocation_history.append(allocation)
        del self.active_allocations[allocation_id]
        
        # Update Redis
        if self.redis_client:
            await self.redis_client.hdel("resource_allocations", allocation_id)
            await self.redis_client.hset(
                "allocation_history",
                allocation_id,
                json.dumps(allocation.dict(), default=str)
            )
        
        logger.info(f"Released allocation {allocation_id} for agent {allocation.agent_id}")
        
        return {
            "allocation_id": allocation_id,
            "status": "released",
            "message": "Resources released successfully"
        }
    
    async def process_pending_requests(self):
        """Process queued resource requests"""
        # Sort requests by priority and queue time
        sorted_requests = sorted(
            self.resource_requests.items(),
            key=lambda x: (-x[1]["request"].priority, x[1]["queued_at"])
        )
        
        for request_id, request_data in sorted_requests:
            request = request_data["request"]
            
            if await self.can_allocate_resources(request):
                try:
                    await self.allocate_resources(request)
                    logger.info(f"Processed queued request {request_id}")
                except Exception as e:
                    logger.error(f"Error processing queued request {request_id}: {e}")
    
    async def estimate_wait_time(self, request: ResourceRequest) -> int:
        """Estimate wait time for a resource request in minutes"""
        # Simple estimation based on current allocations
        # In a real system, this would be more sophisticated
        
        earliest_available = datetime.utcnow()
        
        # Find when enough resources will be available
        for allocation in self.active_allocations.values():
            if (request.cpu_cores <= allocation.allocated_cpu or
                request.memory_mb <= allocation.allocated_memory_mb or
                request.gpu_count <= allocation.allocated_gpu):
                if allocation.end_time > earliest_available:
                    earliest_available = allocation.end_time
        
        wait_minutes = max(0, (earliest_available - datetime.utcnow()).total_seconds() / 60)
        return int(wait_minutes)
    
    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        total_allocations = len(self.active_allocations)
        queued_requests = len(self.resource_requests)
        
        # Calculate utilization
        if self.system_resources:
            cpu_utilization = (
                (self.system_resources.total_cpu_cores - self.system_resources.available_cpu_cores) /
                self.system_resources.total_cpu_cores * 100
            )
            memory_utilization = (
                (self.system_resources.total_memory_mb - self.system_resources.available_memory_mb) /
                self.system_resources.total_memory_mb * 100
            )
        else:
            cpu_utilization = memory_utilization = 0
        
        return {
            "system_resources": self.system_resources.dict() if self.system_resources else {},
            "active_allocations": total_allocations,
            "queued_requests": queued_requests,
            "resource_utilization": {
                "cpu_percent": cpu_utilization,
                "memory_percent": memory_utilization
            },
            "allocation_history_count": len(self.allocation_history)
        }

# Global arbitration agent instance
arbitration_agent = ResourceArbitrationAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await arbitration_agent.initialize()
    yield
    # Shutdown
    if arbitration_agent.redis_client:
        await arbitration_agent.redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Resource Arbitration Agent",
    description="Manages resource allocation across agents",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "resource-arbitration-agent",
        "timestamp": datetime.utcnow().isoformat(),
        "active_allocations": len(arbitration_agent.active_allocations),
        "queued_requests": len(arbitration_agent.resource_requests)
    }

@app.post("/request_resources")
async def request_resources(request: ResourceRequest):
    """Request resource allocation"""
    return await arbitration_agent.request_resources(request)

@app.post("/release_allocation/{allocation_id}")
async def release_allocation(allocation_id: str):
    """Release a resource allocation"""
    return await arbitration_agent.release_allocation(allocation_id)

@app.get("/resource_usage")
async def get_resource_usage():
    """Get current resource usage statistics"""
    return await arbitration_agent.get_resource_usage()

@app.get("/allocations")
async def list_allocations():
    """List all active allocations"""
    allocations = []
    for allocation_id, allocation in arbitration_agent.active_allocations.items():
        allocations.append({
            "allocation_id": allocation_id,
            "agent_id": allocation.agent_id,
            "allocated_cpu": allocation.allocated_cpu,
            "allocated_memory_mb": allocation.allocated_memory_mb,
            "allocated_gpu": allocation.allocated_gpu,
            "start_time": allocation.start_time.isoformat(),
            "end_time": allocation.end_time.isoformat(),
            "status": allocation.status
        })
    return {"allocations": allocations}

@app.get("/requests")
async def list_pending_requests():
    """List all pending resource requests"""
    requests = []
    for request_id, request_data in arbitration_agent.resource_requests.items():
        request = request_data["request"]
        requests.append({
            "request_id": request_id,
            "agent_id": request.agent_id,
            "cpu_cores": request.cpu_cores,
            "memory_mb": request.memory_mb,
            "gpu_count": request.gpu_count,
            "priority": request.priority,
            "queued_at": request_data["queued_at"].isoformat(),
            "status": request_data["status"]
        })
    return {"requests": requests}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "resource-arbitration-agent",
        "status": "running",
        "description": "Resource Allocation and Management Service"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8588"))
    uvicorn.run(app, host="0.0.0.0", port=port)