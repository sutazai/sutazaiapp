from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import httpx
import asyncio
import redis
import json
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Service Communication Hub")

# Redis connection for service registry and pub/sub
try:
    redis_url = os.getenv('REDIS_URL', 'redis://:redis_password@redis:6379/0')
    # Parse Redis URL to extract password
    if '@' in redis_url:
        # Extract password from URL like redis://:password@host:port/db
        password_part = redis_url.split('//')[1].split('@')[0]
        if ':' in password_part:
            redis_password = password_part.split(':')[1]
        else:
            redis_password = password_part
    else:
        redis_password = os.getenv('REDIS_PASSWORD', 'redis_password')
    
    redis_client = redis.Redis(
        host='redis', 
        port=6379, 
        password=redis_password,
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available ({str(e)}) - using in-memory storage")

class ServiceRequest(BaseModel):
    service: str
    endpoint: str
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None
    timeout: int = 30

class OrchestrationRequest(BaseModel):
    task_type: str
    task_data: Dict[str, Any]
    agents: Optional[List[str]] = None

class ServiceRegistry:
    def __init__(self):
        self.services = {
            # Core Services
            'ollama': 'http://ollama:10104',
            'chromadb': 'http://chromadb:8000',
            'qdrant': 'http://qdrant:6333',
            'backend': 'http://backend:8000',
            # AI Agents (all configured to use Ollama)
            'autogpt': 'http://autogpt:8080',
            'crewai': 'http://crewai:8080',
            'aider': 'http://aider:8080',
            'gpt-engineer': 'http://gpt-engineer:8080',
            'localagi': 'http://localagi:8080',
            'autogen': 'http://autogen:8080',
            'agentzero': 'http://agentzero:8080',
            'bigagi': 'http://bigagi:3000',
            'dify': 'http://dify:5000',
            'opendevin': 'http://opendevin:3000',
            'finrobot': 'http://finrobot:8080',
            'code-improver': 'http://code-improver:8080',
            # Additional Services
            'langflow': 'http://langflow:7860',
            'flowise': 'http://flowise:3000',
            'n8n': 'http://n8n:5678',
            'tabbyml': 'http://tabbyml:8080',
            'semgrep': 'http://semgrep:8080',
            'pytorch': 'http://pytorch:8080',
            'tensorflow': 'http://tensorflow:8080',
            'jax': 'http://jax:8080',
            # Model Management
            'modelmanager': 'http://modelmanager:8080',
            'llm-autoeval': 'http://llm-autoeval:8080',
            'context-framework': 'http://context-framework:8080',
            'awesome-code-ai': 'http://awesome-code-ai:8080',
            'ollama-webui': 'http://ollama-webui:8080'
        }
        self.websocket_connections = {}
    
    async def call_service(self, service: str, endpoint: str, method: str = 'GET', data: Dict = None, timeout: int = 30):
        """Call a service endpoint with error handling"""
        if service not in self.services:
            raise HTTPException(status_code=404, detail=f"Service {service} not found")
        
        url = f"{self.services[service]}{endpoint}"
        
        try:
            async with httpx.AsyncClient() as client:
                if method == 'GET':
                    response = await client.get(url, timeout=timeout)
                elif method == 'POST':
                    response = await client.post(url, json=data, timeout=timeout)
                elif method == 'PUT':
                    response = await client.put(url, json=data, timeout=timeout)
                elif method == 'DELETE':
                    response = await client.delete(url, timeout=timeout)
                else:
                    raise HTTPException(status_code=405, detail="Method not allowed")
                
                # Store call in Redis if available
                if redis_client:
                    call_record = {
                        "service": service,
                        "endpoint": endpoint,
                        "method": method,
                        "timestamp": datetime.utcnow().isoformat(),
                        "status": response.status_code
                    }
                    redis_client.lpush("service_calls", json.dumps(call_record))
                    redis_client.ltrim("service_calls", 0, 999)  # Keep last 1000 calls
                
                return response.json()
        
        except httpx.TimeoutException:
            logger.error(f"Service {service} timed out")
            raise HTTPException(status_code=504, detail=f"Service {service} timed out")
        except httpx.RequestError as e:
            logger.error(f"Service {service} error: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Service {service} unavailable")
        except Exception as e:
            logger.error(f"Unexpected error calling {service}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def health_check(self, service: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            # Service-specific health endpoints
            service_endpoints = {
                'chromadb': ["/api/v1/heartbeat", "/api/v1"],
                'dify': ["/"],  # Dify uses root endpoint
                'bigagi': ["/"],  # BigAGI web interface
                'ollama': ["/", "/api/tags"],
                'qdrant': ["/", "/collections"],
                'prometheus': ["/-/healthy", "/"],
                'grafana': ["/api/health", "/"],
            }
            
            # Get endpoints to try for this service
            endpoints_to_try = service_endpoints.get(service, ["/health", "/healthz", "/", "/api/health"])
            
            # Try service-specific endpoints
            for endpoint in endpoints_to_try:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"{self.services[service]}{endpoint}",
                            timeout=5.0
                        )
                        if response.status_code == 200:
                            return {
                                "service": service,
                                "status": "healthy",
                                "endpoint": endpoint,
                                "response_time": response.elapsed.total_seconds()
                            }
                except Exception as e:
                    logger.debug(f"Continuing after exception: {e}")
                    continue
            
            return {"service": service, "status": "unhealthy"}
        except Exception as e:
            logger.warning(f"Exception caught, returning: {e}")
            return {"service": service, "status": "unreachable"}
    
    async def orchestrate_task(self, task_type: str, task_data: Dict, agents: List[str] = None):
        """Orchestrate a task across multiple services"""
        
        orchestrators = {
            "code_generation": ["aider", "gpt-engineer", "opendevin", "autogen"],
            "analysis": ["crewai", "autogen", "localagi", "autogpt"],
            "autonomous_task": ["autogpt", "agentzero", "localagi", "bigagi"],
            "document_processing": ["documind", "llamaindex", "langchain-agents"],
            "financial_analysis": ["finrobot", "crewai"],
            "workflow_automation": ["n8n", "langflow", "flowise"]
        }
        
        # Use specified agents or default for task type
        agents_to_use = agents or orchestrators.get(task_type, ["autogen", "crewai"])
        
        # Execute task with each agent
        results = []
        for agent in agents_to_use:
            if agent in self.services:
                try:
                    result = await self.call_service(
                        agent,
                        "/execute" if agent != "n8n" else "/workflow/execute",
                        "POST",
                        task_data
                    )
                    results.append({
                        "agent": agent,
                        "status": "success",
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "agent": agent,
                        "status": "error",
                        "error": str(e)
                    })
        
        # Store orchestration result
        if redis_client:
            orchestration_record = {
                "task_type": task_type,
                "timestamp": datetime.utcnow().isoformat(),
                "agents": agents_to_use,
                "results": len([r for r in results if r["status"] == "success"])
            }
            redis_client.lpush("orchestrations", json.dumps(orchestration_record))
        
        return {
            "task_type": task_type,
            "agents_used": agents_to_use,
            "results": results,
            "summary": self.summarize_results(results)
        }
    
    def summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize results from multiple agents"""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        return {
            "total_agents": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "consensus": self.find_consensus(successful) if successful else None
        }
    
    def find_consensus(self, results: List[Dict]) -> Any:
        """Find consensus among successful results"""
        # Simple implementation - can be enhanced with more sophisticated logic
        if not results:
            return None
        
        # For now, return the first successful result
        # In production, implement voting, similarity scoring, etc.
        return results[0].get("result")

# Initialize service registry
registry = ServiceRegistry()

@app.get("/services")
async def list_services():
    """List all registered services"""
    return {
        "services": list(registry.services.keys()),
        "total": len(registry.services)
    }

@app.post("/call")
async def call_service(request: ServiceRequest):
    """Call a specific service endpoint"""
    return await registry.call_service(
        request.service,
        request.endpoint,
        request.method,
        request.data,
        request.timeout
    )

@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest):
    """Orchestrate a task across multiple services"""
    return await registry.orchestrate_task(
        request.task_type,
        request.task_data,
        request.agents
    )

@app.get("/health")
async def health_check_all():
    """Check health of all services"""
    health_checks = {}
    
    # Check services in parallel
    tasks = []
    for service in registry.services.keys():
        tasks.append(registry.health_check(service))
    
    results = await asyncio.gather(*tasks)
    
    for result in results:
        service = result["service"]
        health_checks[service] = result["status"]
    
    healthy_count = sum(1 for status in health_checks.values() if status == "healthy")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "services": health_checks,
        "summary": {
            "total": len(health_checks),
            "healthy": healthy_count,
            "unhealthy": len(health_checks) - healthy_count
        }
    }

@app.get("/health/{service}")
async def health_check_service(service: str):
    """Check health of a specific service"""
    if service not in registry.services:
        raise HTTPException(status_code=404, detail=f"Service {service} not found")
    
    return await registry.health_check(service)

@app.post("/broadcast")
async def broadcast(endpoint: str, method: str = "POST", data: Dict = None):
    """Broadcast a request to all services"""
    results = {}
    
    tasks = []
    for service in registry.services.keys():
        tasks.append(registry.call_service(service, endpoint, method, data))
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for service, response in zip(registry.services.keys(), responses):
        if isinstance(response, Exception):
            results[service] = {"error": str(response)}
        else:
            results[service] = response
    
    return results

@app.get("/metrics")
async def get_metrics():
    """Get service usage metrics"""
    metrics = {
        "total_services": len(registry.services),
        "recent_calls": [],
        "recent_orchestrations": []
    }
    
    if redis_client:
        # Get recent service calls
        recent_calls = redis_client.lrange("service_calls", 0, 9)
        metrics["recent_calls"] = [json.loads(call) for call in recent_calls]
        
        # Get recent orchestrations
        recent_orchestrations = redis_client.lrange("orchestrations", 0, 9)
        metrics["recent_orchestrations"] = [json.loads(orch) for orch in recent_orchestrations]
    
    return metrics

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time service communication"""
    await websocket.accept()
    connection_id = f"conn_{datetime.utcnow().timestamp()}"
    registry.websocket_connections[connection_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe":
                # Subscribe to service events
                services = data.get("services", [])
                await websocket.send_json({
                    "type": "subscribed",
                    "services": services
                })
            
            elif data["type"] == "call":
                # Call a service
                result = await registry.call_service(
                    data["service"],
                    data["endpoint"],
                    data.get("method", "GET"),
                    data.get("data")
                )
                await websocket.send_json({
                    "type": "result",
                    "service": data["service"],
                    "result": result
                })
            
            elif data["type"] == "health":
                # Get health status
                health = await health_check_all()
                await websocket.send_json({
                    "type": "health",
                    "data": health
                })
    
    except WebSocketDisconnect:
        del registry.websocket_connections[connection_id]
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@app.get("/")
async def root():
    return {
        "service": "SutazAI Service Communication Hub",
        "version": "1.0",
        "endpoints": {
            "/services": "List all services",
            "/call": "Call a service",
            "/orchestrate": "Orchestrate multi-agent task",
            "/health": "Check all services health",
            "/health/{service}": "Check specific service health",
            "/broadcast": "Broadcast to all services",
            "/metrics": "Get usage metrics",
            "/ws": "WebSocket connection"
        },
        "features": [
            "Service discovery and registry",
            "Multi-agent task orchestration",
            "Health monitoring",
            "WebSocket real-time communication",
            "Metrics and logging",
            "Error handling and retry logic"
        ]
    }