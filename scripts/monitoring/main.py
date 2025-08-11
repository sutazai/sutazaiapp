#!/usr/bin/env python3
"""
SutazAI Resource Manager Service
Manages CPU, memory, and service allocation across the AI system
"""

import os
import asyncio
import json
import psutil
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import psycopg2
import consul
import pika
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from contextlib import asynccontextmanager

# Metrics
REQUEST_COUNT = Counter('resource_manager_requests_total', 'Total requests', ['method', 'endpoint'])
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percentage')
ACTIVE_SERVICES = Gauge('active_services_count', 'Number of active services')
RESPONSE_TIME = Histogram('resource_manager_response_time_seconds', 'Response time')

class ResourceAllocation(BaseModel):
    service_name: str
    cpu_cores: float
    memory_mb: int
    priority: int = 1

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_services: int
    timestamp: datetime

class ResourceManager:
    def __init__(self):
        self.redis_client = None
        self.consul_client = None
        self.rabbitmq_connection = None
        self.postgres_conn = None
        self.cpu_cores = int(os.getenv('CPU_CORES', '12'))
        self.total_memory_gb = int(os.getenv('TOTAL_MEMORY_GB', '29'))
        self.allocated_resources: Dict[str, ResourceAllocation] = {}
        
    async def initialize(self):
        """Initialize connections to external services"""
        try:
            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/2')
            self.redis_client = redis.from_url(redis_url)
            
            # Consul connection
            consul_url = os.getenv('CONSUL_URL', 'http://consul:8500')
            self.consul_client = consul.Consul(host=consul_url.split('://')[1].split(':')[0])
            
            # PostgreSQL connection
            postgres_url = os.getenv('POSTGRES_URL')
            if postgres_url:
                self.postgres_conn = psycopg2.connect(postgres_url)
                
            # RabbitMQ connection
            rabbitmq_url = os.getenv('RABBITMQ_URL')
            if rabbitmq_url:
                params = pika.URLParameters(rabbitmq_url)
                self.rabbitmq_connection = pika.BlockingConnection(params)
                
            print("Resource Manager initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Resource Manager: {e}")
            
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Update Prometheus metrics
        CPU_USAGE.set(cpu_percent)
        MEMORY_USAGE.set(memory.percent)
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            active_services=len(self.allocated_resources),
            timestamp=datetime.now()
        )
    
    async def allocate_resources(self, allocation: ResourceAllocation) -> bool:
        """Allocate resources for a service"""
        try:
            # Check if resources are available
            current_cpu = sum(r.cpu_cores for r in self.allocated_resources.values())
            current_memory = sum(r.memory_mb for r in self.allocated_resources.values())
            
            if (current_cpu + allocation.cpu_cores > self.cpu_cores or 
                current_memory + allocation.memory_mb > self.total_memory_gb * 1024):
                return False
                
            # Allocate resources
            self.allocated_resources[allocation.service_name] = allocation
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    'resource_allocations',
                    allocation.service_name,
                    json.dumps(allocation.dict())
                )
            
            # Register with Consul
            if self.consul_client:
                self.consul_client.agent.service.register(
                    name=allocation.service_name,
                    service_id=f"{allocation.service_name}-{allocation.priority}",
                    tags=[f"cpu:{allocation.cpu_cores}", f"memory:{allocation.memory_mb}"]
                )
                
            ACTIVE_SERVICES.set(len(self.allocated_resources))
            return True
            
        except Exception as e:
            print(f"Failed to allocate resources for {allocation.service_name}: {e}")
            return False
    
    async def deallocate_resources(self, service_name: str) -> bool:
        """Deallocate resources for a service"""
        try:
            if service_name in self.allocated_resources:
                del self.allocated_resources[service_name]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.hdel('resource_allocations', service_name)
                
                # Deregister from Consul
                if self.consul_client:
                    services = self.consul_client.agent.services()
                    for service_id, service_info in services.items():
                        if service_info['Service'] == service_name:
                            self.consul_client.agent.service.deregister(service_id)
                            
                ACTIVE_SERVICES.set(len(self.allocated_resources))
                return True
                
        except Exception as e:
            print(f"Failed to deallocate resources for {service_name}: {e}")
            
        return False

# Initialize Resource Manager
resource_manager = ResourceManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await resource_manager.initialize()
    yield
    # Shutdown
    if resource_manager.rabbitmq_connection:
        resource_manager.rabbitmq_connection.close()
    if resource_manager.postgres_conn:
        resource_manager.postgres_conn.close()

# FastAPI app
app = FastAPI(
    title="SutazAI Resource Manager",
    description="Manages system resources and service allocation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:10011",  # Frontend Streamlit UI
        "http://localhost:10010",  # Backend API
        "http://127.0.0.1:10011",  # Alternative localhost
        "http://127.0.0.1:10010",  # Alternative localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNT.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy", "service": "resource-manager"}

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/system/metrics")
async def get_system_metrics():
    """Get current system metrics"""
    REQUEST_COUNT.labels(method="GET", endpoint="/system/metrics").inc()
    return resource_manager.get_system_metrics()

@app.post("/allocate")
async def allocate_resources(allocation: ResourceAllocation):
    """Allocate resources for a service"""
    REQUEST_COUNT.labels(method="POST", endpoint="/allocate").inc()
    success = await resource_manager.allocate_resources(allocation)
    if success:
        return {"status": "allocated", "service": allocation.service_name}
    else:
        raise HTTPException(status_code=400, detail="Insufficient resources")

@app.delete("/deallocate/{service_name}")
async def deallocate_resources(service_name: str):
    """Deallocate resources for a service"""
    REQUEST_COUNT.labels(method="DELETE", endpoint="/deallocate").inc()
    success = await resource_manager.deallocate_resources(service_name)
    if success:
        return {"status": "deallocated", "service": service_name}
    else:
        raise HTTPException(status_code=404, detail="Service not found")

@app.get("/allocations")
async def get_allocations():
    """Get current resource allocations"""
    REQUEST_COUNT.labels(method="GET", endpoint="/allocations").inc()
    return resource_manager.allocated_resources

@app.post("/alerts/webhook")
async def webhook_alert(alert_data: dict):
    """Handle webhook alerts from Alertmanager"""
    REQUEST_COUNT.labels(method="POST", endpoint="/alerts/webhook").inc()
    print(f"Received alert: {alert_data}")
    return {"status": "received"}

@app.post("/alerts/critical")
async def critical_alert(alert_data: dict):
    """Handle critical alerts"""
    REQUEST_COUNT.labels(method="POST", endpoint="/alerts/critical").inc()
    print(f"CRITICAL ALERT: {alert_data}")
    # Implement emergency resource management here
    return {"status": "processed"}

@app.post("/alerts/ai-agent")
async def ai_agent_alert(alert_data: dict):
    """Handle AI agent alerts"""
    REQUEST_COUNT.labels(method="POST", endpoint="/alerts/ai-agent").inc()
    print(f"AI Agent Alert: {alert_data}")
    return {"status": "processed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)