#!/usr/bin/env python3
"""
SutazAI Agent Message Bus
Real-time communication hub for all AI agents
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uuid

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "10485760"))  # 10MB
MESSAGE_TTL = int(os.getenv("MESSAGE_TTL", "3600"))  # 1 hour
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # 30 seconds

# Pydantic models
class Message(BaseModel):
    id: str = None
    sender: str
    recipient: str = None  # None for broadcast
    channel: str
    content: dict
    timestamp: datetime = None
    ttl: int = MESSAGE_TTL
    priority: str = "normal"  # low, normal, high, urgent

class AgentStatus(BaseModel):
    agent_id: str
    status: str  # online, offline, busy, error
    capabilities: List[str]
    load: float = 0.0
    last_heartbeat: datetime = None

class BroadcastMessage(BaseModel):
    channel: str
    content: dict
    sender: str
    priority: str = "normal"

# FastAPI app
app = FastAPI(title="SutazAI Agent Message Bus", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis_client: Optional[redis.Redis] = None
active_connections: Dict[str, WebSocket] = {}
agent_subscriptions: Dict[str, Set[str]] = {}
message_stats = {
    "total_messages": 0,
    "active_agents": 0,
    "channels": set(),
    "start_time": datetime.now()
}

@app.on_event("startup")
async def startup():
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    
    # Test Redis connection
    try:
        await redis_client.ping()
        logger.info("✅ Connected to Redis successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise
    
    # Start background tasks
    asyncio.create_task(cleanup_expired_messages())
    asyncio.create_task(monitor_agent_heartbeats())
    
    logger.info("🚀 Agent Message Bus started successfully")

@app.on_event("shutdown")
async def shutdown():
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("🛑 Agent Message Bus stopped")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await redis_client.ping()
        return {
            "status": "healthy",
            "redis": "connected",
            "active_connections": len(active_connections),
            "total_messages": message_stats["total_messages"],
            "uptime": str(datetime.now() - message_stats["start_time"])
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/message/send")
async def send_message(message: Message):
    """Send a message to a specific agent or broadcast to a channel"""
    try:
        # Generate message ID if not provided
        if not message.id:
            message.id = str(uuid.uuid4())
        
        # Set timestamp if not provided
        if not message.timestamp:
            message.timestamp = datetime.now()
        
        # Validate message size
        message_json = message.json()
        if len(message_json.encode()) > MAX_MESSAGE_SIZE:
            raise HTTPException(status_code=413, detail="Message too large")
        
        # Store message in Redis
        message_key = f"message:{message.id}"
        await redis_client.setex(
            message_key,
            message.ttl,
            message_json
        )
        
        # Route message
        if message.recipient:
            # Direct message
            channel = f"agent:{message.recipient}"
            await redis_client.publish(channel, message_json)
        else:
            # Broadcast to channel
            channel = f"channel:{message.channel}"
            await redis_client.publish(channel, message_json)
        
        # Update stats
        message_stats["total_messages"] += 1
        message_stats["channels"].add(message.channel)
        
        # Send via WebSocket if agent is connected
        if message.recipient and message.recipient in active_connections:
            try:
                await active_connections[message.recipient].send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message to {message.recipient}: {e}")
        
        logger.info(f"📨 Message {message.id} sent from {message.sender} to {message.recipient or 'broadcast'}")
        
        return {"status": "sent", "message_id": message.id}
        
    except Exception as e:
        logger.error(f"❌ Failed to send message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/message/broadcast")
async def broadcast_message(broadcast: BroadcastMessage):
    """Broadcast a message to all agents subscribed to a channel"""
    try:
        message = Message(
            sender=broadcast.sender,
            channel=broadcast.channel,
            content=broadcast.content,
            priority=broadcast.priority
        )
        
        # Send as broadcast (no specific recipient)
        result = await send_message(message)
        
        logger.info(f"📢 Broadcast message sent to channel {broadcast.channel}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to broadcast message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages/{agent_id}")
async def get_messages(agent_id: str, limit: int = 100):
    """Get messages for a specific agent"""
    try:
        # Get messages from agent's personal channel
        channel = f"agent:{agent_id}"
        messages = []
        
        # Get recent messages (this is a simplified implementation)
        # In production, you might want to store messages in a more structured way
        pattern = f"message:*"
        keys = await redis_client.keys(pattern)
        
        for key in keys[-limit:]:
            message_data = await redis_client.get(key)
            if message_data:
                message = json.loads(message_data)
                if message.get("recipient") == agent_id or not message.get("recipient"):
                    messages.append(message)
        
        return {"messages": messages, "count": len(messages)}
        
    except Exception as e:
        logger.error(f"❌ Failed to get messages for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/register")
async def register_agent(status: AgentStatus):
    """Register an agent with the message bus"""
    try:
        # Store agent status
        agent_key = f"agent_status:{status.agent_id}"
        status.last_heartbeat = datetime.now()
        
        await redis_client.setex(
            agent_key,
            HEARTBEAT_INTERVAL * 3,  # 3x heartbeat interval timeout
            status.json()
        )
        
        # Update stats
        message_stats["active_agents"] = len(await redis_client.keys("agent_status:*"))
        
        logger.info(f"🤖 Agent {status.agent_id} registered successfully")
        
        return {"status": "registered", "agent_id": status.agent_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to register agent {status.agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/heartbeat/{agent_id}")
async def agent_heartbeat(agent_id: str, load: float = 0.0):
    """Agent heartbeat to maintain connection"""
    try:
        # Update heartbeat timestamp
        agent_key = f"agent_status:{agent_id}"
        agent_data = await redis_client.get(agent_key)
        
        if agent_data:
            agent_status = AgentStatus.parse_raw(agent_data)
            agent_status.last_heartbeat = datetime.now()
            agent_status.load = load
            
            await redis_client.setex(
                agent_key,
                HEARTBEAT_INTERVAL * 3,
                agent_status.json()
            )
            
            return {"status": "acknowledged", "agent_id": agent_id}
        else:
            raise HTTPException(status_code=404, detail="Agent not registered")
            
    except Exception as e:
        logger.error(f"❌ Heartbeat failed for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all registered agents"""
    try:
        agent_keys = await redis_client.keys("agent_status:*")
        agents = []
        
        for key in agent_keys:
            agent_data = await redis_client.get(key)
            if agent_data:
                agent_status = json.loads(agent_data)
                agents.append(agent_status)
        
        return {
            "agents": agents,
            "total_count": len(agents),
            "online_count": len([a for a in agents if a.get("status") == "online"])
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get agents status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{agent_id}")
async def websocket_endpoint(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[agent_id] = websocket
    
    try:
        logger.info(f"🔌 WebSocket connected for agent {agent_id}")
        
        # Subscribe to agent's channel
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"agent:{agent_id}")
        
        # Also subscribe to any channels the agent is interested in
        agent_key = f"agent_status:{agent_id}"
        agent_data = await redis_client.get(agent_key)
        if agent_data:
            agent_status = json.loads(agent_data)
            for channel in agent_status.get("subscribed_channels", []):
                await pubsub.subscribe(f"channel:{channel}")
        
        # Listen for messages
        async def listen_for_messages():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        await websocket.send_text(message["data"])
                    except WebSocketDisconnect:
                        break
        
        # Start listening task
        listen_task = asyncio.create_task(listen_for_messages())
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            # Handle incoming message from agent
            try:
                message_data = json.loads(data)
                if message_data.get("type") == "heartbeat":
                    await agent_heartbeat(agent_id, message_data.get("load", 0.0))
                elif message_data.get("type") == "message":
                    message = Message(**message_data["data"])
                    await send_message(message)
            except Exception as e:
                logger.error(f"❌ Error processing WebSocket message from {agent_id}: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket error for agent {agent_id}: {e}")
    finally:
        # Cleanup
        if agent_id in active_connections:
            del active_connections[agent_id]
        
        try:
            await pubsub.unsubscribe()
            await pubsub.close()
        except:
            pass

@app.get("/stats")
async def get_stats():
    """Get message bus statistics"""
    return {
        "total_messages": message_stats["total_messages"],
        "active_connections": len(active_connections),
        "active_agents": message_stats["active_agents"],
        "channels": list(message_stats["channels"]),
        "uptime": str(datetime.now() - message_stats["start_time"]),
        "redis_connected": True
    }

# Background tasks
async def cleanup_expired_messages():
    """Cleanup expired messages periodically"""
    while True:
        try:
            # This is handled by Redis TTL, but we could add additional cleanup here
            await asyncio.sleep(300)  # Run every 5 minutes
        except Exception as e:
            logger.error(f"❌ Error in cleanup task: {e}")

async def monitor_agent_heartbeats():
    """Monitor agent heartbeats and mark inactive agents as offline"""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
            # Check all registered agents
            agent_keys = await redis_client.keys("agent_status:*")
            current_time = datetime.now()
            
            for key in agent_keys:
                agent_data = await redis_client.get(key)
                if agent_data:
                    agent_status = json.loads(agent_data)
                    last_heartbeat = datetime.fromisoformat(
                        agent_status["last_heartbeat"].replace("Z", "+00:00")
                    )
                    
                    # Check if agent is inactive
                    if (current_time - last_heartbeat).seconds > HEARTBEAT_INTERVAL * 2:
                        agent_status["status"] = "offline"
                        await redis_client.setex(key, HEARTBEAT_INTERVAL, json.dumps(agent_status))
                        logger.warning(f"⚠️ Agent {agent_status['agent_id']} marked as offline")
                        
        except Exception as e:
            logger.error(f"❌ Error in heartbeat monitor: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "message_bus:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )