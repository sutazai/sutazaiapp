#!/usr/bin/env python3
"""
MCP Bridge Server - Phase 7
Unified communication bridge for all AI agents and services
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import httpx
import websockets
from typing import Union
import aio_pika
import redis.asyncio as aioredis
from consul import Consul

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SutazAI MCP Bridge",
    description="Message Control Protocol Bridge for AI Agent Integration",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Registry
SERVICE_REGISTRY = {
    # Core Services
    "postgres": {"url": "postgresql://jarvis:sutazai_secure_2024@localhost:10000/jarvis_ai", "type": "database"},
    "redis": {"url": "redis://localhost:10001", "type": "cache"},
    "rabbitmq": {"url": "amqp://sutazai:sutazai_secure_2024@localhost:10004", "type": "queue"},
    "neo4j": {"url": "bolt://localhost:10003", "type": "graph"},
    "consul": {"url": "http://localhost:10006", "type": "discovery"},
    "kong": {"url": "http://localhost:10009", "type": "gateway"},
    
    # Vector Databases
    "chromadb": {"url": "http://localhost:10100", "type": "vector"},
    "qdrant": {"url": "http://localhost:10101", "type": "vector"},
    "faiss": {"url": "http://localhost:10103", "type": "vector"},
    
    # Backend Services
    "backend": {"url": "http://localhost:10200", "type": "api"},
    "frontend": {"url": "http://localhost:11000", "type": "ui"},
    
    # AI Agents (when deployed)
    "letta": {"url": "http://localhost:11100", "type": "agent", "status": "pending"},
    "autogpt": {"url": "http://localhost:11101", "type": "agent", "status": "pending"},
    "crewai": {"url": "http://localhost:11102", "type": "agent", "status": "pending"},
    "aider": {"url": "http://localhost:11103", "type": "agent", "status": "pending"},
    "private-gpt": {"url": "http://localhost:11104", "type": "agent", "status": "pending"},
}

# Agent Registry
AGENT_REGISTRY = {
    "letta": {
        "name": "Letta (MemGPT)",
        "capabilities": ["memory", "conversation", "task-automation"],
        "port": 11100,
        "status": "offline"
    },
    "autogpt": {
        "name": "AutoGPT",
        "capabilities": ["autonomous", "web-search", "task-execution"],
        "port": 11101,
        "status": "offline"
    },
    "crewai": {
        "name": "CrewAI",
        "capabilities": ["multi-agent", "orchestration", "collaboration"],
        "port": 11102,
        "status": "offline"
    },
    "aider": {
        "name": "Aider",
        "capabilities": ["code-editing", "pair-programming", "refactoring"],
        "port": 11103,
        "status": "offline"
    },
    "private-gpt": {
        "name": "Private-GPT",
        "capabilities": ["document-qa", "local-llm", "privacy"],
        "port": 11104,
        "status": "offline"
    }
}

# Message routing table
MESSAGE_ROUTES = {
    "task.automation": ["letta", "autogpt"],
    "code.generation": ["aider", "gpt-engineer"],
    "document.analysis": ["private-gpt", "llama-index"],
    "multi.agent": ["crewai", "autogen"],
    "conversation": ["letta", "jarvis"],
}

# Active connections
active_connections: Dict[str, WebSocket] = {}

# Global connections
rabbitmq_connection = None
rabbitmq_channel = None
redis_client = None
consul_client = None

# Request/Response Models
class MCPMessage(BaseModel):
    id: str
    source: str
    target: str
    type: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None

class ServiceRequest(BaseModel):
    service: str
    method: str
    params: Optional[Dict] = None
    timeout: Optional[int] = 30

class AgentTask(BaseModel):
    task_id: str
    task_type: str
    description: str
    agent: Optional[str] = None
    params: Optional[Dict] = None
    priority: Optional[str] = "medium"

class BridgeStatus(BaseModel):
    status: str
    services: Dict[str, str]
    agents: Dict[str, str]
    active_connections: int
    timestamp: str

# Initialization Functions
async def init_rabbitmq():
    """Initialize RabbitMQ connection and setup exchanges/queues"""
    global rabbitmq_connection, rabbitmq_channel
    try:
        # Connect to RabbitMQ
        rabbitmq_connection = await aio_pika.connect_robust(
            "amqp://sutazai:sutazai_secure_2024@localhost:10004/"
        )
        rabbitmq_channel = await rabbitmq_connection.channel()
        
        # Declare exchange for agent communication
        exchange = await rabbitmq_channel.declare_exchange(
            "mcp.exchange",
            aio_pika.ExchangeType.TOPIC,
            durable=True
        )
        
        # Declare queues for each agent
        for agent_id in AGENT_REGISTRY.keys():
            queue = await rabbitmq_channel.declare_queue(
                f"agent.{agent_id}",
                durable=True
            )
            # Bind queue to exchange with routing key
            await queue.bind(exchange, routing_key=f"agent.{agent_id}.*")
            await queue.bind(exchange, routing_key="agent.all.*")
        
        # Declare queue for bridge itself
        bridge_queue = await rabbitmq_channel.declare_queue(
            "mcp.bridge",
            durable=True
        )
        await bridge_queue.bind(exchange, routing_key="bridge.*")
        
        logger.info("RabbitMQ initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RabbitMQ: {e}")
        return False

async def init_redis():
    """Initialize Redis connection for caching and session management"""
    global redis_client
    try:
        redis_client = await aioredis.from_url(
            "redis://localhost:10001",
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Redis initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        return False

async def init_consul():
    """Initialize Consul for service discovery"""
    global consul_client
    try:
        consul_client = Consul(host='localhost', port=10006)
        # Register MCP Bridge service
        consul_client.agent.service.register(
            name='mcp-bridge',
            service_id='mcp-bridge-1',
            address='localhost',
            port=11100,
            tags=['bridge', 'mcp', 'ai'],
            check={"http": "http://localhost:11100/health", "interval": "30s"}
        )
        logger.info("Consul initialized and service registered")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Consul: {e}")
        return False

async def publish_to_rabbitmq(routing_key: str, message: Dict):
    """Publish message to RabbitMQ"""
    if not rabbitmq_channel:
        logger.error("RabbitMQ channel not initialized")
        return False
    
    try:
        exchange = await rabbitmq_channel.get_exchange("mcp.exchange")
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                content_type="application/json"
            ),
            routing_key=routing_key
        )
        logger.info(f"Published message to {routing_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to publish to RabbitMQ: {e}")
        return False

async def cache_to_redis(key: str, value: Any, ttl: int = 3600):
    """Cache data to Redis"""
    if not redis_client:
        return False
    
    try:
        await redis_client.setex(
            key,
            ttl,
            json.dumps(value) if not isinstance(value, str) else value
        )
        return True
    except Exception as e:
        logger.error(f"Failed to cache to Redis: {e}")
        return False

async def get_from_redis(key: str):
    """Get data from Redis cache"""
    if not redis_client:
        return None
    
    try:
        value = await redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Failed to get from Redis: {e}")
        return None

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Initialize all connections on startup"""
    logger.info("Initializing MCP Bridge connections...")
    
    # Initialize RabbitMQ
    await init_rabbitmq()
    
    # Initialize Redis
    await init_redis()
    
    # Initialize Consul
    await init_consul()
    
    logger.info("MCP Bridge initialization complete")

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    if rabbitmq_connection:
        await rabbitmq_connection.close()
    if redis_client:
        await redis_client.close()
    if consul_client:
        consul_client.agent.service.deregister('mcp-bridge-1')
    logger.info("MCP Bridge shutdown complete")

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "mcp-bridge",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Service Registry Endpoints
@app.get("/services")
async def get_services():
    """Get all registered services"""
    return SERVICE_REGISTRY

@app.get("/services/{service_name}")
async def get_service(service_name: str):
    """Get specific service information"""
    if service_name not in SERVICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    return SERVICE_REGISTRY[service_name]

@app.post("/services/{service_name}/health")
async def check_service_health(service_name: str):
    """Check health of a specific service"""
    if service_name not in SERVICE_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    
    service = SERVICE_REGISTRY[service_name]
    
    # Check service health based on type
    try:
        if service["type"] in ["api", "ui", "agent", "vector"]:
            # HTTP health check
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{service['url']}/health", timeout=5)
                return {
                    "service": service_name,
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_code": response.status_code
                }
        else:
            # Basic connectivity check for databases
            return {
                "service": service_name,
                "status": "assumed_healthy",
                "note": "Direct connectivity check not implemented"
            }
    except Exception as e:
        return {
            "service": service_name,
            "status": "unhealthy",
            "error": str(e)
        }

# Agent Registry Endpoints
@app.get("/agents")
async def get_agents():
    """Get all registered agents"""
    return AGENT_REGISTRY

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent information"""
    if agent_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return AGENT_REGISTRY[agent_id]

@app.post("/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, status: str):
    """Update agent status"""
    if agent_id not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    AGENT_REGISTRY[agent_id]["status"] = status
    return {"agent": agent_id, "new_status": status}

# Message Routing Endpoints
@app.post("/route")
async def route_message(message: MCPMessage):
    """Route message to appropriate service or agent"""
    message.timestamp = message.timestamp or datetime.now().isoformat()
    
    # Determine routing based on target
    if message.target in SERVICE_REGISTRY:
        # Route to service
        service = SERVICE_REGISTRY[message.target]
        return await forward_to_service(message, service)
    elif message.target in AGENT_REGISTRY:
        # Route to agent
        agent = AGENT_REGISTRY[message.target]
        return await forward_to_agent(message, agent)
    else:
        # Try pattern-based routing
        for pattern, targets in MESSAGE_ROUTES.items():
            if pattern in message.type:
                # Route to first available target
                for target in targets:
                    if target in AGENT_REGISTRY and AGENT_REGISTRY[target]["status"] == "online":
                        return await forward_to_agent(message, AGENT_REGISTRY[target])
        
        raise HTTPException(status_code=404, detail=f"No route found for target: {message.target}")

async def forward_to_service(message: MCPMessage, service: Dict):
    """Forward message to a service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service['url']}/mcp/receive",
                json=message.dict(),
                timeout=30
            )
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to forward to service: {e}")
        return {"error": str(e), "status": "failed"}

async def forward_to_agent(message: MCPMessage, agent: Dict):
    """Forward message to an agent via RabbitMQ and HTTP fallback"""
    if agent["status"] != "online":
        # Try to publish to RabbitMQ anyway (agent might be listening)
        routing_key = f"agent.{message.target}.message"
        if await publish_to_rabbitmq(routing_key, message.dict()):
            return {"status": "queued", "message": "Message queued for offline agent"}
        return {"error": f"Agent {agent['name']} is not online", "status": "offline"}
    
    # First try RabbitMQ
    routing_key = f"agent.{message.target}.message"
    if await publish_to_rabbitmq(routing_key, message.dict()):
        # Cache the message for tracking
        cache_key = f"message:{message.id}"
        await cache_to_redis(cache_key, message.dict(), ttl=300)
        return {"status": "published", "message_id": message.id}
    
    # Fallback to HTTP if RabbitMQ fails
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://localhost:{agent['port']}/mcp/receive",
                json=message.dict(),
                timeout=30
            )
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to forward to agent: {e}")
        return {"error": str(e), "status": "failed"}

# Task Orchestration Endpoints
@app.post("/tasks/submit")
async def submit_task(task: AgentTask):
    """Submit a task for execution"""
    # Determine best agent for task
    if task.agent:
        # Use specified agent
        if task.agent not in AGENT_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Agent {task.agent} not found")
        agent = AGENT_REGISTRY[task.agent]
    else:
        # Auto-select based on task type
        agent = select_agent_for_task(task.task_type)
        if not agent:
            raise HTTPException(status_code=400, detail=f"No suitable agent for task type: {task.task_type}")
    
    # Create MCP message for task
    message = MCPMessage(
        id=task.task_id,
        source="mcp-bridge",
        target=agent["name"].lower().replace(" ", "-"),
        type=f"task.{task.task_type}",
        payload={
            "task_id": task.task_id,
            "description": task.description,
            "params": task.params or {},
            "priority": task.priority
        }
    )
    
    # Forward to agent
    result = await forward_to_agent(message, agent)
    return {
        "task_id": task.task_id,
        "agent": agent["name"],
        "status": "submitted",
        "result": result
    }

def select_agent_for_task(task_type: str) -> Optional[Dict]:
    """Select best agent for a task type"""
    # Map task types to agent capabilities
    capability_map = {
        "automation": ["task-automation", "autonomous"],
        "code": ["code-editing", "pair-programming"],
        "document": ["document-qa", "local-llm"],
        "conversation": ["conversation", "memory"],
        "orchestration": ["multi-agent", "orchestration"]
    }
    
    for key, capabilities in capability_map.items():
        if key in task_type.lower():
            # Find online agent with matching capability
            for agent_id, agent in AGENT_REGISTRY.items():
                if agent["status"] == "online":
                    if any(cap in agent["capabilities"] for cap in capabilities):
                        return agent
    
    return None

# WebSocket for Real-time Communication
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process message
            if data.get("type") == "broadcast":
                # Broadcast to all connected clients
                for conn_id, conn in active_connections.items():
                    if conn_id != client_id:
                        await conn.send_json({
                            "from": client_id,
                            "data": data.get("payload"),
                            "timestamp": datetime.now().isoformat()
                        })
            elif data.get("type") == "direct":
                # Send to specific client
                target = data.get("target")
                if target in active_connections:
                    await active_connections[target].send_json({
                        "from": client_id,
                        "data": data.get("payload"),
                        "timestamp": datetime.now().isoformat()
                    })
            
    except WebSocketDisconnect:
        del active_connections[client_id]
        logger.info(f"Client {client_id} disconnected")

# Bridge Status Endpoint
@app.get("/status", response_model=BridgeStatus)
async def get_bridge_status():
    """Get current bridge status"""
    # Check service statuses
    service_statuses = {}
    for service_name in SERVICE_REGISTRY:
        try:
            result = await check_service_health(service_name)
            service_statuses[service_name] = result.get("status", "unknown")
        except:
            service_statuses[service_name] = "unknown"
    
    # Get agent statuses
    agent_statuses = {
        agent_id: agent["status"] 
        for agent_id, agent in AGENT_REGISTRY.items()
    }
    
    return BridgeStatus(
        status="operational",
        services=service_statuses,
        agents=agent_statuses,
        active_connections=len(active_connections),
        timestamp=datetime.now().isoformat()
    )

# Metrics Endpoint
@app.get("/metrics")
async def get_metrics():
    """Get bridge metrics"""
    return {
        "total_services": len(SERVICE_REGISTRY),
        "total_agents": len(AGENT_REGISTRY),
        "online_agents": sum(1 for a in AGENT_REGISTRY.values() if a["status"] == "online"),
        "active_connections": len(active_connections),
        "message_routes": len(MESSAGE_ROUTES),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting MCP Bridge Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=11100,
        log_level="info"
    )