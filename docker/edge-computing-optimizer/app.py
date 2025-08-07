import json
import time
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import redis
from loguru import logger
import psutil

# Configuration
AGENT_NAME = "edge-computing-optimizer"
AGENT_ROLE = "Edge Computing Optimizer Agent"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:10104")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PORT = int(os.getenv("PORT", "8080"))

# Initialize FastAPI app
app = FastAPI(
    title=AGENT_NAME,
    version="2.2.0",
    description=f"{AGENT_ROLE} - Optimizes edge computing resources and deployments"
)

# Initialize Redis connection
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Pydantic models
class TaskRequest(BaseModel):
    task_type: str
    parameters: Dict[str, Any]
    priority: Optional[str] = "normal"
    timeout: Optional[int] = 300

class TaskResponse(BaseModel):
    task_id: str
    status: str
    agent: str
    created_at: str
    estimated_completion: int

# Health tracking
startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {AGENT_NAME} - {AGENT_ROLE}")
    # Register agent in Redis if available
    if redis_client:
        try:
            redis_client.hset(
                f"agent:{AGENT_NAME}",
                mapping={
                    "status": "active",
                    "role": AGENT_ROLE,
                    "started_at": datetime.utcnow().isoformat(),
                    "capabilities": json.dumps([
                        "edge_optimization",
                        "resource_allocation",
                        "deployment_planning",
                        "performance_monitoring"
                    ])
                }
            )
            redis_client.expire(f"agent:{AGENT_NAME}", 3600)  # TTL 1 hour
        except Exception as e:
            logger.error(f"Failed to register agent in Redis: {e}")

@app.get("/")
async def root():
    return {
        "agent": AGENT_NAME,
        "role": AGENT_ROLE,
        "status": "active",
        "version": "2.2.0",
        "capabilities": [
            "edge_optimization",
            "resource_allocation",
            "deployment_planning",
            "performance_monitoring"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    uptime = time.time() - startup_time
    memory = psutil.Process().memory_info()
    
    health_status = {
        "status": "healthy",
        "agent": AGENT_NAME,
        "uptime_seconds": round(uptime, 2),
        "memory_usage_mb": round(memory.rss / 1024 / 1024, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check Redis connection
    redis_healthy = False
    if redis_client:
        try:
            redis_client.ping()
            redis_healthy = True
        except:
            pass
    
    health_status["redis_connected"] = redis_healthy
    
    # Check Ollama connection
    ollama_healthy = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                ollama_healthy = True
    except:
        pass
    
    health_status["ollama_connected"] = ollama_healthy
    
    return health_status

@app.get("/capabilities")
async def capabilities():
    return {
        "agent": AGENT_NAME,
        "role": AGENT_ROLE,
        "capabilities": {
            "edge_optimization": {
                "description": "Optimize edge computing resources",
                "methods": ["resource_allocation", "workload_distribution", "latency_optimization"]
            },
            "deployment_planning": {
                "description": "Plan and optimize edge deployments",
                "methods": ["site_selection", "capacity_planning", "failover_strategies"]
            },
            "performance_monitoring": {
                "description": "Monitor edge computing performance",
                "methods": ["metrics_collection", "anomaly_detection", "performance_tuning"]
            }
        },
        "supported_models": ["tinyllama", "tinyllama", "tinyllama"],
        "api_version": "2.2.0"
    }

@app.post("/task", response_model=TaskResponse)
async def execute_task(task: TaskRequest):
    task_id = f"edge_{int(time.time() * 1000)}"
    
    # Store task in Redis if available
    if redis_client:
        try:
            task_data = {
                "task_id": task_id,
                "agent": AGENT_NAME,
                "task_type": task.task_type,
                "parameters": json.dumps(task.parameters),
                "priority": task.priority,
                "status": "queued",
                "created_at": datetime.utcnow().isoformat()
            }
            redis_client.hset(f"task:{task_id}", mapping=task_data)
            redis_client.expire(f"task:{task_id}", 86400)  # TTL 24 hours
            
            # Add to task queue
            redis_client.lpush("task_queue:edge_computing", task_id)
        except Exception as e:
            logger.error(f"Failed to store task in Redis: {e}")
    
    # Simulate task processing
    estimated_completion = 30
    if task.task_type == "optimize_resources":
        estimated_completion = 45
    elif task.task_type == "deploy_edge_node":
        estimated_completion = 120
    elif task.task_type == "analyze_performance":
        estimated_completion = 60
    
    return TaskResponse(
        task_id=task_id,
        status="processing",
        agent=AGENT_NAME,
        created_at=datetime.utcnow().isoformat(),
        estimated_completion=estimated_completion
    )

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics_lines = [
        f'# HELP {AGENT_NAME}_uptime_seconds Agent uptime in seconds',
        f'# TYPE {AGENT_NAME}_uptime_seconds gauge',
        f'{AGENT_NAME}_uptime_seconds {time.time() - startup_time}',
        f'# HELP {AGENT_NAME}_memory_usage_bytes Memory usage in bytes',
        f'# TYPE {AGENT_NAME}_memory_usage_bytes gauge',
        f'{AGENT_NAME}_memory_usage_bytes {psutil.Process().memory_info().rss}',
    ]
    
    return "\n".join(metrics_lines)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {AGENT_NAME} on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")