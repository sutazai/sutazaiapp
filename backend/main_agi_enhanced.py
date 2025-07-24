#!/usr/bin/env python3
"""
SutazAI AGI/ASI Enhanced Backend
Integrates with all 30+ AI agents and provides unified AGI capabilities
"""

import json
import asyncio
import httpx
from datetime import datetime
from typing import Dict, Optional, List, Any, Union
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import logging
import uvicorn
from contextlib import asynccontextmanager
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Service URLs
SERVICE_URLS = {
    "ollama": "http://ollama:11434",
    "litellm": "http://litellm:4000",
    "service_hub": "http://service-hub:8080",
    "chromadb": "http://chromadb:8000",
    "qdrant": "http://qdrant:6333",
    # AI Agents
    "autogpt": "http://autogpt:8080",
    "crewai": "http://crewai:8080",
    "aider": "http://aider:8080",
    "gpt_engineer": "http://gpt-engineer:8080",
    "localagi": "http://localagi:8080",
    "autogen": "http://autogen:8080",
    "agentzero": "http://agentzero:8080",
    "bigagi": "http://bigagi:3000",
    "dify": "http://dify:5000",
    "opendevin": "http://opendevin:3000",
    "finrobot": "http://finrobot:8080",
    "code_improver": "http://code-improver:8080",
    # Workflow
    "langflow": "http://langflow:7860",
    "flowise": "http://flowise:3000",
    "n8n": "http://n8n:5678"
}

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "sutazai"),
    "user": os.getenv("POSTGRES_USER", "sutazai"),
    "password": os.getenv("POSTGRES_PASSWORD", "sutazai123")
}

# Redis configuration
REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "redis"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "decode_responses": True
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting SutazAI AGI/ASI Backend...")
    
    # Initialize connections
    app.state.redis = redis.Redis(**REDIS_CONFIG)
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    
    # Test database connection
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.close()
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AGI/ASI Backend...")
    await app.state.http_client.aclose()
    app.state.redis.close()

# Create FastAPI app
app = FastAPI(
    title="SutazAI AGI/ASI Backend",
    description="Unified backend for AGI/ASI system with 30+ AI agents",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Request/Response Models =====

class AgentTask(BaseModel):
    agent: str
    task: str
    parameters: Dict[str, Any] = {}
    timeout: int = 300

class OrchestrationRequest(BaseModel):
    task_type: str
    task_data: Dict[str, Any]
    agents: List[str]
    parallel: bool = True
    max_iterations: int = 10

class ModelGenerateRequest(BaseModel):
    prompt: str
    model: str = "deepseek-r1:8b"
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False

class WorkflowRequest(BaseModel):
    workflow_id: str
    steps: List[Dict[str, Any]]
    parameters: Dict[str, Any] = {}

# ===== Helper Functions =====

async def call_service(service: str, endpoint: str, method: str = "GET", 
                      data: Dict = None, timeout: int = 30):
    """Call an internal service"""
    if service not in SERVICE_URLS:
        raise ValueError(f"Unknown service: {service}")
    
    url = f"{SERVICE_URLS[service]}{endpoint}"
    
    try:
        client = app.state.http_client
        
        if method == "GET":
            response = await client.get(url, timeout=timeout)
        elif method == "POST":
            response = await client.post(url, json=data, timeout=timeout)
        elif method == "PUT":
            response = await client.put(url, json=data, timeout=timeout)
        elif method == "DELETE":
            response = await client.delete(url, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code >= 400:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Service {service} returned error: {response.text}"
            )
        
        return response.json()
    
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Service {service} timed out"
        )
    except Exception as e:
        logger.error(f"Error calling {service}: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service {service} unavailable: {str(e)}"
        )

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SutazAI AGI/ASI Backend",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "30+ AI Agents Integration",
            "Multi-Agent Orchestration",
            "Unified Model Management",
            "Workflow Automation",
            "Vector Database Integration",
            "Real-time Monitoring"
        ]
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check critical services
    critical_services = ["ollama", "litellm", "service_hub", "chromadb"]
    
    for service in critical_services:
        try:
            if service == "ollama":
                result = await call_service(service, "/api/tags", timeout=5)
                health_status["services"][service] = "healthy"
            elif service == "litellm":
                result = await call_service(service, "/health", timeout=5)
                health_status["services"][service] = "healthy"
            elif service == "service_hub":
                result = await call_service(service, "/health", timeout=5)
                health_status["services"][service] = "healthy"
            elif service == "chromadb":
                result = await call_service(service, "/api/v1/heartbeat", timeout=5)
                health_status["services"][service] = "healthy"
        except:
            health_status["services"][service] = "unhealthy"
            health_status["status"] = "degraded"
    
    # Check database
    try:
        conn = get_db_connection()
        conn.close()
        health_status["services"]["database"] = "healthy"
    except:
        health_status["services"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        app.state.redis.ping()
        health_status["services"]["redis"] = "healthy"
    except:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status

# ===== AI Agent Management =====

@app.get("/api/v1/agents")
async def list_agents():
    """List all available AI agents"""
    try:
        # Get agent list from service hub
        result = await call_service("service_hub", "/services")
        
        # Categorize agents
        agents = {
            "core_agents": [],
            "advanced_agents": [],
            "specialized_agents": [],
            "workflow_agents": []
        }
        
        agent_mapping = {
            "core_agents": ["autogpt", "crewai", "aider", "gpt-engineer", "llamaindex"],
            "advanced_agents": ["localagi", "autogen", "agentzero", "bigagi", "dify"],
            "specialized_agents": ["opendevin", "finrobot", "realtimestt", "code-improver"],
            "workflow_agents": ["langflow", "flowise", "n8n"]
        }
        
        if result and "services" in result:
            for category, agent_list in agent_mapping.items():
                for agent in agent_list:
                    if agent in result["services"]:
                        agents[category].append({
                            "name": agent,
                            "status": "available",
                            "endpoint": SERVICE_URLS.get(agent.replace("-", "_"))
                        })
        
        return {
            "agents": agents,
            "total": sum(len(v) for v in agents.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agents/execute")
async def execute_agent_task(request: AgentTask):
    """Execute a task on a specific agent"""
    try:
        # Validate agent
        agent = request.agent.replace("-", "_")
        if agent not in SERVICE_URLS:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent} not found")
        
        # Route to appropriate agent endpoint
        agent_endpoints = {
            "autogpt": "/api/agent/tasks",
            "crewai": "/api/crews/execute",
            "aider": "/api/code/generate",
            "gpt_engineer": "/api/projects/create",
            "autogen": "/api/assistants/run",
            "localagi": "/api/agents/execute",
            "opendevin": "/generate",
            "finrobot": "/api/analyze"
        }
        
        endpoint = agent_endpoints.get(agent, "/api/execute")
        
        # Execute task
        result = await call_service(
            agent,
            endpoint,
            method="POST",
            data={
                "task": request.task,
                "parameters": request.parameters
            },
            timeout=request.timeout
        )
        
        # Log to Redis
        task_record = {
            "agent": request.agent,
            "task": request.task,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed"
        }
        app.state.redis.lpush("agent_tasks", json.dumps(task_record))
        app.state.redis.ltrim("agent_tasks", 0, 999)
        
        return {
            "agent": request.agent,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error executing agent task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Multi-Agent Orchestration =====

@app.post("/api/v1/orchestrate")
async def orchestrate_agents(request: OrchestrationRequest):
    """Orchestrate multiple agents for complex tasks"""
    try:
        # Use service hub for orchestration
        result = await call_service(
            "service_hub",
            "/orchestrate",
            method="POST",
            data={
                "task_type": request.task_type,
                "task_data": request.task_data,
                "agents": request.agents
            }
        )
        
        # Store orchestration record
        orch_record = {
            "task_type": request.task_type,
            "agents": request.agents,
            "timestamp": datetime.utcnow().isoformat(),
            "result_summary": result.get("summary", {})
        }
        
        app.state.redis.lpush("orchestrations", json.dumps(orch_record))
        app.state.redis.ltrim("orchestrations", 0, 99)
        
        return result
    
    except Exception as e:
        logger.error(f"Orchestration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Model Management =====

@app.get("/api/v1/models")
async def list_models():
    """List available models"""
    try:
        # Get models from Ollama
        ollama_models = await call_service("ollama", "/api/tags")
        
        # Get LiteLLM mappings
        models = {
            "ollama_models": [
                {
                    "name": model["name"],
                    "size": model.get("size", "unknown"),
                    "modified": model.get("modified_at", "")
                }
                for model in ollama_models.get("models", [])
            ],
            "litellm_mappings": {
                "gpt-4": "deepseek-r1:8b",
                "gpt-3.5-turbo": "qwen2.5:3b",
                "code-davinci-002": "codellama:7b",
                "text-embedding-ada-002": "nomic-embed-text"
            }
        }
        
        return models
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate")
async def generate_completion(request: ModelGenerateRequest):
    """Generate completion using specified model"""
    try:
        if request.stream:
            # Streaming response
            async def stream_response():
                async with httpx.AsyncClient() as client:
                    async with client.stream(
                        "POST",
                        f"{SERVICE_URLS['ollama']}/api/generate",
                        json={
                            "model": request.model,
                            "prompt": request.prompt,
                            "temperature": request.temperature,
                            "stream": True
                        }
                    ) as response:
                        async for line in response.aiter_lines():
                            if line:
                                yield f"data: {line}\n\n"
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            # Non-streaming response
            result = await call_service(
                "ollama",
                "/api/generate",
                method="POST",
                data={
                    "model": request.model,
                    "prompt": request.prompt,
                    "temperature": request.temperature,
                    "stream": False
                }
            )
            
            return {
                "response": result.get("response", ""),
                "model": request.model,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Workflow Automation =====

@app.post("/api/v1/workflows/execute")
async def execute_workflow(request: WorkflowRequest):
    """Execute a workflow across multiple services"""
    try:
        results = []
        
        for step in request.steps:
            step_type = step.get("type")
            
            if step_type == "agent_task":
                # Execute agent task
                result = await execute_agent_task(AgentTask(
                    agent=step["agent"],
                    task=step["task"],
                    parameters=step.get("parameters", {})
                ))
                results.append(result)
                
            elif step_type == "model_generate":
                # Generate with model
                result = await generate_completion(ModelGenerateRequest(
                    prompt=step["prompt"],
                    model=step.get("model", "deepseek-r1:8b")
                ))
                results.append(result)
                
            elif step_type == "data_process":
                # Process data (placeholder)
                results.append({"step": step_type, "status": "completed"})
        
        return {
            "workflow_id": request.workflow_id,
            "steps_completed": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Workflow execution error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== Monitoring & Metrics =====

@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        # Get recent tasks from Redis
        recent_tasks = []
        task_data = app.state.redis.lrange("agent_tasks", 0, 9)
        for task in task_data:
            recent_tasks.append(json.loads(task))
        
        # Get recent orchestrations
        recent_orchestrations = []
        orch_data = app.state.redis.lrange("orchestrations", 0, 4)
        for orch in orch_data:
            recent_orchestrations.append(json.loads(orch))
        
        # Get service health from hub
        health_data = await call_service("service_hub", "/health")
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "tasks": {
                "recent": recent_tasks,
                "total_24h": len(recent_tasks) * 10  # Estimate
            },
            "orchestrations": {
                "recent": recent_orchestrations,
                "total_24h": len(recent_orchestrations) * 5  # Estimate
            },
            "services": health_data.get("summary", {}),
            "models": {
                "available": 11,
                "primary": "deepseek-r1:8b"
            }
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# ===== WebSocket for Real-time Updates =====

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    
    try:
        # Send initial status
        health = await health_check()
        await websocket.send_json({
            "type": "health",
            "data": health
        })
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Wait for client message or timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "get_metrics":
                    metrics = await get_metrics()
                    await websocket.send_json({
                        "type": "metrics",
                        "data": metrics
                    })
                
            except asyncio.TimeoutError:
                # Send periodic health update
                health = await health_check()
                await websocket.send_json({
                    "type": "health",
                    "data": health
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

# ===== Database Endpoints =====

@app.get("/api/v1/db/health")
async def database_health():
    """Check database health"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        return {"status": "healthy", "database": "postgresql"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")

@app.get("/api/v1/cache/health")
async def cache_health():
    """Check cache health"""
    try:
        app.state.redis.ping()
        return {"status": "healthy", "cache": "redis"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Cache unhealthy: {str(e)}")

# ===== Main Entry Point =====

if __name__ == "__main__":
    uvicorn.run(
        "main_agi_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )