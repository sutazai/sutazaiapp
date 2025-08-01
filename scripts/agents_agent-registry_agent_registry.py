#!/usr/bin/env python3
"""
SutazAI Agent Registry
Central registry for agent discovery and capability matching
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import uuid

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sutazai:sutazai_password@postgres:5432/sutazai")
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "30"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "180"))

# Database setup
Base = declarative_base()

class AgentRecord(Base):
    __tablename__ = "agent_registry"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    specialization = Column(String)
    capabilities = Column(Text)  # JSON string
    status = Column(String, default="offline")
    endpoint = Column(String)
    health_check_url = Column(String)
    load = Column(Integer, default=0)
    max_load = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat = Column(DateTime)
    active = Column(Boolean, default=True)
    metadata = Column(Text)  # JSON string for additional data

# Pydantic models
class AgentRegistration(BaseModel):
    id: str
    name: str
    type: str
    specialization: str = None
    capabilities: List[str] = []
    endpoint: str
    health_check_url: str = None
    max_load: int = 100
    metadata: dict = {}

class AgentUpdate(BaseModel):
    status: str = None
    load: int = None
    capabilities: List[str] = None
    metadata: dict = None

class AgentQuery(BaseModel):
    type: str = None
    specialization: str = None
    capabilities: List[str] = []
    status: str = "online"
    max_load_threshold: int = 80

class TaskRequest(BaseModel):
    task_type: str
    capabilities_required: List[str]
    priority: str = "normal"
    estimated_load: int = 10
    constraints: dict = {}

# FastAPI app
app = FastAPI(title="SutazAI Agent Registry", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis_client: Optional[redis.Redis] = None
db_engine = None
SessionLocal = None

@app.on_event("startup")
async def startup():
    global redis_client, db_engine, SessionLocal
    
    # Connect to Redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await redis_client.ping()
        logger.info("✅ Connected to Redis successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise
    
    # Connect to PostgreSQL
    try:
        db_engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        
        # Create tables
        Base.metadata.create_all(bind=db_engine)
        logger.info("✅ Connected to PostgreSQL successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
        raise
    
    # Load existing agent configurations
    await load_agent_configurations()
    
    # Start background tasks
    asyncio.create_task(monitor_agent_health())
    asyncio.create_task(cleanup_inactive_agents())
    
    logger.info("🚀 Agent Registry started successfully")

@app.on_event("shutdown")
async def shutdown():
    if redis_client:
        await redis_client.close()
    if db_engine:
        db_engine.dispose()
    logger.info("🛑 Agent Registry stopped")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        await redis_client.ping()
        
        # Check database
        db = SessionLocal()
        agent_count = db.query(AgentRecord).count()
        db.close()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "database": "connected",
            "registered_agents": agent_count
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/agents/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent"""
    try:
        db = SessionLocal()
        
        # Check if agent already exists
        existing = db.query(AgentRecord).filter(AgentRecord.id == registration.id).first()
        
        if existing:
            # Update existing registration
            existing.name = registration.name
            existing.type = registration.type
            existing.specialization = registration.specialization
            existing.capabilities = json.dumps(registration.capabilities)
            existing.endpoint = registration.endpoint
            existing.health_check_url = registration.health_check_url
            existing.max_load = registration.max_load
            existing.metadata = json.dumps(registration.metadata)
            existing.updated_at = datetime.utcnow()
            existing.last_heartbeat = datetime.utcnow()
            existing.status = "online"
            existing.active = True
        else:
            # Create new registration
            agent_record = AgentRecord(
                id=registration.id,
                name=registration.name,
                type=registration.type,
                specialization=registration.specialization,
                capabilities=json.dumps(registration.capabilities),
                endpoint=registration.endpoint,
                health_check_url=registration.health_check_url,
                max_load=registration.max_load,
                metadata=json.dumps(registration.metadata),
                last_heartbeat=datetime.utcnow(),
                status="online"
            )
            db.add(agent_record)
        
        db.commit()
        db.close()
        
        # Cache in Redis for fast lookup
        await redis_client.setex(
            f"agent:{registration.id}",
            AGENT_TIMEOUT,
            json.dumps(registration.dict())
        )
        
        # Update capability index
        await update_capability_index(registration.id, registration.capabilities)
        
        logger.info(f"🤖 Agent {registration.id} registered successfully")
        
        return {"status": "registered", "agent_id": registration.id}
        
    except Exception as e:
        logger.error(f"❌ Failed to register agent {registration.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/agents/{agent_id}")
async def update_agent(agent_id: str, update: AgentUpdate):
    """Update agent information"""
    try:
        db = SessionLocal()
        agent = db.query(AgentRecord).filter(AgentRecord.id == agent_id).first()
        
        if not agent:
            db.close()
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update fields
        if update.status is not None:
            agent.status = update.status
        if update.load is not None:
            agent.load = update.load
        if update.capabilities is not None:
            agent.capabilities = json.dumps(update.capabilities)
            await update_capability_index(agent_id, update.capabilities)
        if update.metadata is not None:
            agent.metadata = json.dumps(update.metadata)
        
        agent.updated_at = datetime.utcnow()
        agent.last_heartbeat = datetime.utcnow()
        
        db.commit()
        db.close()
        
        # Update Redis cache
        cached_data = await redis_client.get(f"agent:{agent_id}")
        if cached_data:
            agent_data = json.loads(cached_data)
            if update.status:
                agent_data["status"] = update.status
            if update.load is not None:
                agent_data["load"] = update.load
            if update.capabilities:
                agent_data["capabilities"] = update.capabilities
            
            await redis_client.setex(
                f"agent:{agent_id}",
                AGENT_TIMEOUT,
                json.dumps(agent_data)
            )
        
        logger.info(f"🔄 Agent {agent_id} updated successfully")
        
        return {"status": "updated", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to update agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/heartbeat")
async def agent_heartbeat(agent_id: str, load: int = 0):
    """Agent heartbeat"""
    try:
        db = SessionLocal()
        agent = db.query(AgentRecord).filter(AgentRecord.id == agent_id).first()
        
        if not agent:
            db.close()
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent.last_heartbeat = datetime.utcnow()
        agent.load = load
        agent.status = "online"
        
        db.commit()
        db.close()
        
        # Update Redis
        await redis_client.setex(f"heartbeat:{agent_id}", HEARTBEAT_INTERVAL * 2, str(datetime.utcnow().timestamp()))
        
        return {"status": "acknowledged", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"❌ Heartbeat failed for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents(
    type: str = Query(None),
    specialization: str = Query(None),
    status: str = Query(None),
    limit: int = Query(100),
    offset: int = Query(0)
):
    """List registered agents with optional filtering"""
    try:
        db = SessionLocal()
        query = db.query(AgentRecord).filter(AgentRecord.active == True)
        
        if type:
            query = query.filter(AgentRecord.type == type)
        if specialization:
            query = query.filter(AgentRecord.specialization == specialization)
        if status:
            query = query.filter(AgentRecord.status == status)
        
        agents = query.offset(offset).limit(limit).all()
        
        result = []
        for agent in agents:
            agent_data = {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "specialization": agent.specialization,
                "capabilities": json.loads(agent.capabilities) if agent.capabilities else [],
                "status": agent.status,
                "endpoint": agent.endpoint,
                "load": agent.load,
                "max_load": agent.max_load,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                "metadata": json.loads(agent.metadata) if agent.metadata else {}
            }
            result.append(agent_data)
        
        db.close()
        
        return {
            "agents": result,
            "total": len(result),
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent information"""
    try:
        # Try Redis cache first
        cached_data = await redis_client.get(f"agent:{agent_id}")
        if cached_data:
            return json.loads(cached_data)
        
        # Fall back to database
        db = SessionLocal()
        agent = db.query(AgentRecord).filter(AgentRecord.id == agent_id).first()
        
        if not agent:
            db.close()
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent_data = {
            "id": agent.id,
            "name": agent.name,
            "type": agent.type,
            "specialization": agent.specialization,
            "capabilities": json.loads(agent.capabilities) if agent.capabilities else [],
            "status": agent.status,
            "endpoint": agent.endpoint,
            "load": agent.load,
            "max_load": agent.max_load,
            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            "metadata": json.loads(agent.metadata) if agent.metadata else {}
        }
        
        db.close()
        
        return agent_data
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/find")
async def find_agents(query: AgentQuery):
    """Find agents matching specific criteria"""
    try:
        db = SessionLocal()
        db_query = db.query(AgentRecord).filter(AgentRecord.active == True)
        
        if query.type:
            db_query = db_query.filter(AgentRecord.type == query.type)
        if query.specialization:
            db_query = db_query.filter(AgentRecord.specialization == query.specialization)
        if query.status:
            db_query = db_query.filter(AgentRecord.status == query.status)
        
        # Filter by load threshold
        db_query = db_query.filter(AgentRecord.load <= query.max_load_threshold)
        
        agents = db_query.all()
        
        # Filter by capabilities if specified
        if query.capabilities:
            filtered_agents = []
            for agent in agents:
                agent_caps = json.loads(agent.capabilities) if agent.capabilities else []
                if all(cap in agent_caps for cap in query.capabilities):
                    filtered_agents.append(agent)
            agents = filtered_agents
        
        result = []
        for agent in agents:
            agent_data = {
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "specialization": agent.specialization,
                "capabilities": json.loads(agent.capabilities) if agent.capabilities else [],
                "status": agent.status,
                "endpoint": agent.endpoint,
                "load": agent.load,
                "max_load": agent.max_load,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            }
            result.append(agent_data)
        
        db.close()
        
        # Sort by load (least loaded first)
        result.sort(key=lambda x: x["load"])
        
        return {
            "agents": result,
            "count": len(result)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to find agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tasks/assign")
async def assign_task(task: TaskRequest):
    """Find the best agent for a task"""
    try:
        # Find agents with required capabilities
        query = AgentQuery(
            capabilities=task.capabilities_required,
            status="online",
            max_load_threshold=100 - task.estimated_load
        )
        
        result = await find_agents(query)
        suitable_agents = result["agents"]
        
        if not suitable_agents:
            raise HTTPException(status_code=404, detail="No suitable agents found")
        
        # Score agents based on load, capabilities match, and priority
        scored_agents = []
        for agent in suitable_agents:
            score = 0
            
            # Lower load is better
            score += (100 - agent["load"]) * 0.4
            
            # More capabilities is better (specialization bonus)
            capability_match = len(set(agent["capabilities"]) & set(task.capabilities_required))
            score += capability_match * 0.3
            
            # Specialization match
            if agent["specialization"] and task.task_type in agent["specialization"]:
                score += 20
            
            # Type match
            if task.task_type in agent["type"]:
                score += 10
            
            scored_agents.append({
                "agent": agent,
                "score": score
            })
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 3 recommendations
        recommendations = [item["agent"] for item in scored_agents[:3]]
        
        return {
            "task_id": str(uuid.uuid4()),
            "recommended_agents": recommendations,
            "total_candidates": len(suitable_agents)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to assign task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str):
    """Deregister an agent"""
    try:
        db = SessionLocal()
        agent = db.query(AgentRecord).filter(AgentRecord.id == agent_id).first()
        
        if not agent:
            db.close()
            raise HTTPException(status_code=404, detail="Agent not found")
        
        agent.active = False
        agent.status = "offline"
        agent.updated_at = datetime.utcnow()
        
        db.commit()
        db.close()
        
        # Remove from Redis
        await redis_client.delete(f"agent:{agent_id}")
        await redis_client.delete(f"heartbeat:{agent_id}")
        
        logger.info(f"🗑️ Agent {agent_id} deregistered")
        
        return {"status": "deregistered", "agent_id": agent_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to deregister agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_registry_stats():
    """Get registry statistics"""
    try:
        db = SessionLocal()
        
        total_agents = db.query(AgentRecord).filter(AgentRecord.active == True).count()
        online_agents = db.query(AgentRecord).filter(
            AgentRecord.active == True,
            AgentRecord.status == "online"
        ).count()
        
        # Get agent types
        types = db.query(AgentRecord.type).filter(AgentRecord.active == True).distinct().all()
        type_counts = {}
        for type_tuple in types:
            type_name = type_tuple[0]
            count = db.query(AgentRecord).filter(
                AgentRecord.active == True,
                AgentRecord.type == type_name
            ).count()
            type_counts[type_name] = count
        
        db.close()
        
        return {
            "total_agents": total_agents,
            "online_agents": online_agents,
            "offline_agents": total_agents - online_agents,
            "agent_types": type_counts,
            "registry_uptime": "active"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def update_capability_index(agent_id: str, capabilities: List[str]):
    """Update capability index for fast lookups"""
    try:
        for capability in capabilities:
            await redis_client.sadd(f"capability:{capability}", agent_id)
    except Exception as e:
        logger.error(f"❌ Failed to update capability index: {e}")

async def load_agent_configurations():
    """Load agent configurations from files"""
    try:
        config_dir = "/app/agent_definitions"
        if os.path.exists(config_dir):
            for filename in os.listdir(config_dir):
                if filename.endswith(".json"):
                    async with aiofiles.open(os.path.join(config_dir, filename), 'r') as f:
                        config_data = json.loads(await f.read())
                        # Process configuration data
                        logger.info(f"📋 Loaded configuration for {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to load agent configurations: {e}")

# Background tasks
async def monitor_agent_health():
    """Monitor agent health and update status"""
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            
            db = SessionLocal()
            agents = db.query(AgentRecord).filter(
                AgentRecord.active == True,
                AgentRecord.status == "online"
            ).all()
            
            current_time = datetime.utcnow()
            
            for agent in agents:
                if agent.last_heartbeat:
                    time_diff = (current_time - agent.last_heartbeat).seconds
                    if time_diff > AGENT_TIMEOUT:
                        agent.status = "offline"
                        logger.warning(f"⚠️ Agent {agent.id} marked as offline (no heartbeat)")
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"❌ Error in health monitor: {e}")

async def cleanup_inactive_agents():
    """Cleanup inactive agents periodically"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            db = SessionLocal()
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Mark agents as inactive if they haven't sent heartbeat in 24 hours
            inactive_agents = db.query(AgentRecord).filter(
                AgentRecord.active == True,
                AgentRecord.last_heartbeat < cutoff_time
            ).all()
            
            for agent in inactive_agents:
                agent.active = False
                agent.status = "offline"
                logger.info(f"🧹 Agent {agent.id} marked as inactive (24h+ without heartbeat)")
            
            db.commit()
            db.close()
            
        except Exception as e:
            logger.error(f"❌ Error in cleanup task: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "agent_registry:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )