#!/usr/bin/env python3
"""
AI Agent Orchestrator - Manages agent interactions and coordination
"""
import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentInfo(BaseModel):
    agent_id: str
    name: str
    capabilities: List[str]
    endpoint: str
    status: str = "online"
    load: float = 0.0
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    performance_score: float = 1.0

class AgentInteraction(BaseModel):
    interaction_id: str
    source_agent: str
    target_agent: str
    interaction_type: str  # request, response, notification, error
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"

class ConflictResolution(BaseModel):
    conflict_id: str
    agents_involved: List[str]
    conflict_type: str
    description: str
    resolution_strategy: str
    status: str = "pending"

class AIAgentOrchestrator:
    def __init__(self):
        self.redis_client = None
        self.ollama_client = None
        self.agent_registry = {}
        self.active_interactions = {}
        self.conflict_resolutions = {}
        self.performance_metrics = {}
        
    async def initialize(self):
        """Initialize connections and start background processes"""
        try:
            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Initialize Ollama
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:10104")
            self.ollama_client = ollama.AsyncClient(host=ollama_base_url)
            logger.info("Connected to Ollama successfully")
            
            # Start background tasks
            asyncio.create_task(self.agent_discovery())
            asyncio.create_task(self.interaction_monitor())
            asyncio.create_task(self.conflict_detector())
            asyncio.create_task(self.performance_optimizer())
            asyncio.create_task(self.health_monitor())
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def agent_discovery(self):
        """Continuously discover and register agents"""
        while True:
            try:
                # Discover agents from Redis registry
                if self.redis_client:
                    agent_keys = await self.redis_client.keys("agent:*")
                    for key in agent_keys:
                        agent_data = await self.redis_client.get(key)
                        if agent_data:
                            agent_info = json.loads(agent_data)
                            await self.register_agent(agent_info)
                
                await asyncio.sleep(60)  # Discovery every minute
                
            except Exception as e:
                logger.error(f"Error in agent discovery: {e}")
                await asyncio.sleep(120)
    
    async def interaction_monitor(self):
        """Monitor agent interactions"""
        while True:
            try:
                # Process active interactions
                current_time = datetime.utcnow()
                expired_interactions = []
                
                for interaction_id, interaction in self.active_interactions.items():
                    # Check for timeouts
                    if (current_time - interaction.timestamp).seconds > 300:  # 5 minute timeout
                        expired_interactions.append(interaction_id)
                        logger.warning(f"Interaction {interaction_id} timed out")
                
                # Clean up expired interactions
                for interaction_id in expired_interactions:
                    del self.active_interactions[interaction_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in interaction monitor: {e}")
                await asyncio.sleep(60)
    
    async def conflict_detector(self):
        """Detect and resolve conflicts between agents"""
        while True:
            try:
                # Analyze interaction patterns for conflicts
                await self.detect_resource_conflicts()
                await self.detect_task_conflicts()
                await self.resolve_pending_conflicts()
                
                await asyncio.sleep(45)  # Check every 45 seconds
                
            except Exception as e:
                logger.error(f"Error in conflict detector: {e}")
                await asyncio.sleep(90)
    
    async def performance_optimizer(self):
        """Optimize agent performance based on metrics"""
        while True:
            try:
                # Analyze performance metrics
                await self.collect_performance_metrics()
                await self.optimize_agent_assignments()
                await self.suggest_improvements()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance optimizer: {e}")
                await asyncio.sleep(600)
    
    async def health_monitor(self):
        """Monitor health of all registered agents"""
        while True:
            try:
                for agent_id, agent_info in self.agent_registry.items():
                    await self.check_agent_health(agent_id, agent_info)
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(120)
    
    async def register_agent(self, agent_data: Dict[str, Any]):
        """Register or update an agent in the registry"""
        try:
            agent_id = agent_data.get("agent_id")
            if not agent_id:
                return
            
            agent_info = AgentInfo(
                agent_id=agent_id,
                name=agent_data.get("name", agent_id),
                capabilities=agent_data.get("capabilities", []),
                endpoint=agent_data.get("endpoint", f"http://{agent_id}:8080"),
                status=agent_data.get("status", "online"),
                load=agent_data.get("load", 0.0),
                performance_score=agent_data.get("performance_score", 1.0)
            )
            
            self.agent_registry[agent_id] = agent_info
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "orchestrator:agents",
                    agent_id,
                    json.dumps(agent_info.dict(), default=str)
                )
            
            logger.info(f"Registered/updated agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
    
    async def orchestrate_interaction(self, interaction: AgentInteraction) -> Dict[str, Any]:
        """Orchestrate an interaction between agents"""
        try:
            self.active_interactions[interaction.interaction_id] = interaction
            
            # Validate agents exist
            if interaction.source_agent not in self.agent_registry:
                raise HTTPException(status_code=404, detail=f"Source agent {interaction.source_agent} not found")
            
            if interaction.target_agent not in self.agent_registry:
                raise HTTPException(status_code=404, detail=f"Target agent {interaction.target_agent} not found")
            
            # Check for conflicts
            conflict = await self.check_interaction_conflicts(interaction)
            if conflict:
                return {
                    "interaction_id": interaction.interaction_id,
                    "status": "blocked",
                    "reason": "conflict_detected",
                    "conflict_id": conflict.conflict_id
                }
            
            # Route the interaction
            target_agent = self.agent_registry[interaction.target_agent]
            
            # Send to target agent
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{target_agent.endpoint}/interact",
                    json={
                        "interaction_id": interaction.interaction_id,
                        "source_agent": interaction.source_agent,
                        "interaction_type": interaction.interaction_type,
                        "data": interaction.data
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    interaction.status = "completed"
                    
                    # Update performance metrics
                    await self.update_performance_metrics(interaction.target_agent, True)
                    
                    return {
                        "interaction_id": interaction.interaction_id,
                        "status": "completed",
                        "result": result
                    }
                else:
                    interaction.status = "failed"
                    await self.update_performance_metrics(interaction.target_agent, False)
                    raise Exception(f"Target agent returned status {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error orchestrating interaction {interaction.interaction_id}: {e}")
            interaction.status = "failed"
            return {
                "interaction_id": interaction.interaction_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def check_interaction_conflicts(self, interaction: AgentInteraction) -> Optional[ConflictResolution]:
        """Check if an interaction would cause conflicts"""
        # Simple conflict detection logic
        # In a real system, this would be more sophisticated
        
        target_agent = self.agent_registry.get(interaction.target_agent)
        if not target_agent:
            return None
        
        # Check if target agent is overloaded
        if target_agent.load > 0.9:
            conflict_id = f"conflict_{int(datetime.utcnow().timestamp())}"
            conflict = ConflictResolution(
                conflict_id=conflict_id,
                agents_involved=[interaction.target_agent],
                conflict_type="overload",
                description=f"Agent {interaction.target_agent} is overloaded (load: {target_agent.load})",
                resolution_strategy="queue_or_redirect"
            )
            self.conflict_resolutions[conflict_id] = conflict
            return conflict
        
        return None
    
    async def detect_resource_conflicts(self):
        """Detect resource conflicts between agents"""
        # Analyze resource usage patterns
        high_load_agents = [
            agent_id for agent_id, agent_info in self.agent_registry.items()
            if agent_info.load > 0.8
        ]
        
        if len(high_load_agents) > 3:  # Arbitrary threshold
            conflict_id = f"resource_conflict_{int(datetime.utcnow().timestamp())}"
            conflict = ConflictResolution(
                conflict_id=conflict_id,
                agents_involved=high_load_agents,
                conflict_type="resource_contention",
                description="Multiple agents experiencing high load",
                resolution_strategy="load_balancing"
            )
            self.conflict_resolutions[conflict_id] = conflict
    
    async def detect_task_conflicts(self):
        """Detect task-level conflicts between agents"""
        # Analyze for duplicate or conflicting tasks
        pass
    
    async def resolve_pending_conflicts(self):
        """Resolve pending conflicts"""
        for conflict_id, conflict in list(self.conflict_resolutions.items()):
            if conflict.status == "pending":
                await self.resolve_conflict(conflict)
    
    async def resolve_conflict(self, conflict: ConflictResolution):
        """Resolve a specific conflict"""
        try:
            if conflict.resolution_strategy == "load_balancing":
                # Implement load balancing
                await self.balance_agent_loads(conflict.agents_involved)
            elif conflict.resolution_strategy == "queue_or_redirect":
                # Implement queuing or redirection
                await self.implement_queuing_strategy(conflict.agents_involved)
            
            conflict.status = "resolved"
            logger.info(f"Resolved conflict {conflict.conflict_id}")
            
        except Exception as e:
            logger.error(f"Error resolving conflict {conflict.conflict_id}: {e}")
            conflict.status = "failed"
    
    async def balance_agent_loads(self, agent_ids: List[str]):
        """Balance loads across specified agents"""
        # Implement load balancing logic
        logger.info(f"Implementing load balancing for agents: {agent_ids}")
    
    async def implement_queuing_strategy(self, agent_ids: List[str]):
        """Implement queuing strategy for overloaded agents"""
        logger.info(f"Implementing queuing strategy for agents: {agent_ids}")
    
    async def collect_performance_metrics(self):
        """Collect performance metrics from all agents"""
        for agent_id, agent_info in self.agent_registry.items():
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{agent_info.endpoint}/metrics")
                    if response.status_code == 200:
                        metrics = response.json()
                        self.performance_metrics[agent_id] = metrics
            except Exception as e:
                logger.warning(f"Could not collect metrics from {agent_id}: {e}")
    
    async def optimize_agent_assignments(self):
        """Optimize task assignments based on performance"""
        # Analyze performance metrics and suggest optimizations
        pass
    
    async def suggest_improvements(self):
        """Suggest improvements using AI analysis"""
        if not self.ollama_client:
            return
        
        try:
            # Use Ollama to analyze system state and suggest improvements
            system_state = {
                "agents": len(self.agent_registry),
                "active_interactions": len(self.active_interactions),
                "conflicts": len(self.conflict_resolutions),
                "performance_metrics": self.performance_metrics
            }
            
            # This would use the AI to suggest optimizations
            # For now, just log the system state
            logger.info(f"System state for AI analysis: {json.dumps(system_state, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error in AI-based suggestions: {e}")
    
    async def update_performance_metrics(self, agent_id: str, success: bool):
        """Update performance metrics for an agent"""
        if agent_id in self.agent_registry:
            agent_info = self.agent_registry[agent_id]
            if success:
                agent_info.performance_score = min(10.0, agent_info.performance_score * 1.01)
            else:
                agent_info.performance_score = max(0.1, agent_info.performance_score * 0.99)
    
    async def check_agent_health(self, agent_id: str, agent_info: AgentInfo):
        """Check health of a specific agent"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{agent_info.endpoint}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    agent_info.status = "online"
                    agent_info.load = health_data.get("load", 0.0)
                    agent_info.last_seen = datetime.utcnow()
                else:
                    agent_info.status = "offline"
                    
        except Exception as e:
            logger.warning(f"Agent {agent_id} health check failed: {e}")
            agent_info.status = "offline"
    
    async def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status"""
        return {
            "registered_agents": len(self.agent_registry),
            "active_interactions": len(self.active_interactions),
            "pending_conflicts": len([c for c in self.conflict_resolutions.values() if c.status == "pending"]),
            "resolved_conflicts": len([c for c in self.conflict_resolutions.values() if c.status == "resolved"]),
            "system_health": "healthy" if len([a for a in self.agent_registry.values() if a.status == "online"]) > 0 else "degraded"
        }

# Global orchestrator instance
orchestrator = AIAgentOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await orchestrator.initialize()
    yield
    # Shutdown
    if orchestrator.redis_client:
        await orchestrator.redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="AI Agent Orchestrator",
    description="Manages agent interactions and coordination",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "ai-agent-orchestrator",
        "timestamp": datetime.utcnow().isoformat(),
        "registered_agents": len(orchestrator.agent_registry),
        "active_interactions": len(orchestrator.active_interactions)
    }

@app.post("/register_agent")
async def register_agent(agent_data: Dict[str, Any]):
    """Register a new agent"""
    await orchestrator.register_agent(agent_data)
    return {"message": f"Agent {agent_data.get('agent_id', 'unknown')} registered successfully"}

@app.post("/orchestrate_interaction")
async def orchestrate_interaction(interaction: AgentInteraction):
    """Orchestrate an interaction between agents"""
    return await orchestrator.orchestrate_interaction(interaction)

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = []
    for agent_id, agent_info in orchestrator.agent_registry.items():
        agents.append({
            "agent_id": agent_id,
            "name": agent_info.name,
            "capabilities": agent_info.capabilities,
            "status": agent_info.status,
            "load": agent_info.load,
            "performance_score": agent_info.performance_score,
            "last_seen": agent_info.last_seen.isoformat()
        })
    return {"agents": agents}

@app.get("/interactions")
async def list_interactions():
    """List active interactions"""
    interactions = []
    for interaction_id, interaction in orchestrator.active_interactions.items():
        interactions.append({
            "interaction_id": interaction_id,
            "source_agent": interaction.source_agent,
            "target_agent": interaction.target_agent,
            "interaction_type": interaction.interaction_type,
            "status": interaction.status,
            "timestamp": interaction.timestamp.isoformat()
        })
    return {"interactions": interactions}

@app.get("/conflicts")
async def list_conflicts():
    """List all conflicts"""
    conflicts = []
    for conflict_id, conflict in orchestrator.conflict_resolutions.items():
        conflicts.append({
            "conflict_id": conflict_id,
            "agents_involved": conflict.agents_involved,
            "conflict_type": conflict.conflict_type,
            "description": conflict.description,
            "resolution_strategy": conflict.resolution_strategy,
            "status": conflict.status
        })
    return {"conflicts": conflicts}

@app.get("/status")
async def get_orchestration_status():
    """Get orchestration status"""
    return await orchestrator.get_orchestration_status()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "ai-agent-orchestrator",
        "status": "running",
        "description": "AI Agent Interaction and Coordination Service"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8589"))
    uvicorn.run(app, host="0.0.0.0", port=port)