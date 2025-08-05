#!/usr/bin/env python3
"""
AgentZero Coordinator - Master orchestration agent for SUTAZAIAPP
"""
import os
import asyncio
import json
import logging
import docker
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

class SystemStatus(BaseModel):
    total_agents: int
    active_agents: int
    failed_agents: int
    system_load: float
    memory_usage: float
    health_score: float

class AgentCommand(BaseModel):
    command_id: str
    target_agent: str
    command_type: str  # start, stop, restart, scale, configure
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, description="Command priority (1-10)")

class EmergencyAction(BaseModel):
    action_id: str
    action_type: str  # shutdown, restart, isolate, scale_down
    affected_agents: List[str]
    reason: str
    auto_triggered: bool = True

class AgentZeroCoordinator:
    def __init__(self):
        self.redis_client = None
        self.docker_client = None
        self.ollama_client = None
        self.system_status = None
        self.agent_registry = {}
        self.command_queue = asyncio.Queue()
        self.emergency_actions = {}
        self.coordination_metrics = {}
        
        # Coordination settings
        self.max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "50"))
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
        self.emergency_threshold = float(os.getenv("EMERGENCY_SHUTDOWN_THRESHOLD", "0.5"))
        
    async def initialize(self):
        """Initialize all connections and start coordination processes"""
        try:
            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("AgentZero: Connected to Redis successfully")
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            logger.info("AgentZero: Connected to Docker successfully")
            
            # Initialize Ollama
            try:
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
                self.ollama_client = ollama.AsyncClient(host=ollama_base_url)
                logger.info("AgentZero: Connected to Ollama successfully")
            except Exception as e:
                logger.warning(f"AgentZero: Could not connect to Ollama: {e}")
            
            # Start coordination processes
            asyncio.create_task(self.system_monitor())
            asyncio.create_task(self.agent_lifecycle_manager())
            asyncio.create_task(self.command_processor())
            asyncio.create_task(self.emergency_monitor())
            asyncio.create_task(self.performance_coordinator())
            asyncio.create_task(self.auto_scaling_manager())
            
            logger.info("AgentZero: All coordination processes started")
            
        except Exception as e:
            logger.error(f"AgentZero: Failed to initialize: {e}")
            raise
    
    async def system_monitor(self):
        """Monitor overall system health and status"""
        while True:
            try:
                await self.update_system_status()
                await self.check_system_health()
                await self.update_coordination_metrics()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"AgentZero: Error in system monitor: {e}")
                await asyncio.sleep(60)
    
    async def agent_lifecycle_manager(self):
        """Manage agent lifecycles - start, stop, restart"""
        while True:
            try:
                # Check for failed agents and restart them
                failed_agents = await self.identify_failed_agents()
                for agent_id in failed_agents:
                    await self.restart_agent(agent_id)
                
                # Check for agents that need scaling
                await self.check_scaling_needs()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"AgentZero: Error in lifecycle manager: {e}")
                await asyncio.sleep(120)
    
    async def command_processor(self):
        """Process commands from the command queue"""
        while True:
            try:
                # Process commands with timeout
                try:
                    command = await asyncio.wait_for(self.command_queue.get(), timeout=5.0)
                    await self.execute_command(command)
                except asyncio.TimeoutError:
                    continue  # No commands to process
                    
            except Exception as e:
                logger.error(f"AgentZero: Error in command processor: {e}")
                await asyncio.sleep(10)
    
    async def emergency_monitor(self):
        """Monitor for emergency situations"""
        while True:
            try:
                if self.system_status:
                    # Check for emergency conditions
                    if self.system_status.health_score < self.emergency_threshold:
                        logger.warning(f"AgentZero: Emergency threshold reached: {self.system_status.health_score}")
                        await self.trigger_emergency_action("low_health", "System health below threshold")
                    
                    # Check for resource exhaustion
                    if self.system_status.memory_usage > 0.95:
                        logger.warning("AgentZero: Memory exhaustion detected")
                        await self.trigger_emergency_action("memory_exhaustion", "System memory usage critical")
                
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"AgentZero: Error in emergency monitor: {e}")
                await asyncio.sleep(30)
    
    async def performance_coordinator(self):
        """Coordinate performance across all agents"""
        while True:
            try:
                # Collect performance data from all agents
                await self.collect_agent_performance_data()
                
                # Optimize agent assignments based on performance
                await self.optimize_agent_performance()
                
                # Update coordination strategies
                await self.update_coordination_strategies()
                
                await asyncio.sleep(300)  # Coordinate every 5 minutes
                
            except Exception as e:
                logger.error(f"AgentZero: Error in performance coordinator: {e}")
                await asyncio.sleep(600)
    
    async def auto_scaling_manager(self):
        """Manage automatic scaling of agents"""
        while True:
            try:
                if os.getenv("AUTO_SCALING_ENABLED", "true").lower() == "true":
                    await self.evaluate_scaling_decisions()
                    await self.execute_scaling_actions()
                
                await asyncio.sleep(120)  # Evaluate every 2 minutes
                
            except Exception as e:
                logger.error(f"AgentZero: Error in auto-scaling manager: {e}")
                await asyncio.sleep(240)
    
    async def update_system_status(self):
        """Update current system status"""
        try:
            # Count agents by status
            total_agents = 0
            active_agents = 0
            failed_agents = 0
            
            # Get Docker containers
            containers = self.docker_client.containers.list(all=True)
            sutazai_containers = [c for c in containers if 'sutazai' in c.name]
            
            total_agents = len(sutazai_containers)
            active_agents = len([c for c in sutazai_containers if c.status == 'running'])
            failed_agents = total_agents - active_agents
            
            # Calculate system metrics
            system_load = await self.calculate_system_load()
            memory_usage = await self.calculate_memory_usage()
            health_score = self.calculate_health_score(active_agents, total_agents, system_load)
            
            self.system_status = SystemStatus(
                total_agents=total_agents,
                active_agents=active_agents,
                failed_agents=failed_agents,
                system_load=system_load,
                memory_usage=memory_usage,
                health_score=health_score
            )
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.set(
                    "agentzero:system_status",
                    json.dumps(self.system_status.dict()),
                    ex=120
                )
                
        except Exception as e:
            logger.error(f"AgentZero: Error updating system status: {e}")
    
    async def calculate_system_load(self) -> float:
        """Calculate overall system load"""
        try:
            # Simple load calculation based on active agents
            if self.system_status:
                return min(1.0, self.system_status.active_agents / self.max_concurrent_agents)
            return 0.0
        except Exception:
            return 0.0
    
    async def calculate_memory_usage(self) -> float:
        """Calculate system memory usage"""
        try:
            # This would ideally use system metrics
            # For now, estimate based on active containers
            if self.system_status:
                estimated_memory_per_agent = 512  # MB
                total_estimated = self.system_status.active_agents * estimated_memory_per_agent
                system_memory = 8192  # Assume 8GB system
                return min(1.0, total_estimated / system_memory)
            return 0.0
        except Exception:
            return 0.0
    
    def calculate_health_score(self, active: int, total: int, load: float) -> float:
        """Calculate overall system health score"""
        if total == 0:
            return 1.0
        
        # Health score based on agent availability and system load
        availability_score = active / total
        load_score = 1.0 - min(1.0, load)
        
        return (availability_score * 0.7) + (load_score * 0.3)
    
    async def identify_failed_agents(self) -> List[str]:
        """Identify agents that have failed"""
        failed_agents = []
        try:
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                if 'sutazai' in container.name and container.status != 'running':
                    failed_agents.append(container.name)
        except Exception as e:
            logger.error(f"AgentZero: Error identifying failed agents: {e}")
        
        return failed_agents
    
    async def restart_agent(self, agent_id: str):
        """Restart a failed agent"""
        try:
            container = self.docker_client.containers.get(agent_id)
            container.restart()
            logger.info(f"AgentZero: Restarted agent {agent_id}")
            
            # Log the restart action
            if self.redis_client:
                await self.redis_client.lpush(
                    "agentzero:restart_log",
                    json.dumps({
                        "agent_id": agent_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "action": "restart"
                    })
                )
                
        except Exception as e:
            logger.error(f"AgentZero: Error restarting agent {agent_id}: {e}")
    
    async def execute_command(self, command: AgentCommand):
        """Execute a coordination command"""
        try:
            logger.info(f"AgentZero: Executing command {command.command_type} for {command.target_agent}")
            
            if command.command_type == "start":
                await self.start_agent(command.target_agent, command.parameters)
            elif command.command_type == "stop":
                await self.stop_agent(command.target_agent)
            elif command.command_type == "restart":
                await self.restart_agent(command.target_agent)
            elif command.command_type == "scale":
                await self.scale_agent(command.target_agent, command.parameters)
            elif command.command_type == "configure":
                await self.configure_agent(command.target_agent, command.parameters)
            
        except Exception as e:
            logger.error(f"AgentZero: Error executing command {command.command_id}: {e}")
    
    async def start_agent(self, agent_id: str, parameters: Dict[str, Any]):
        """Start an agent"""
        # Implementation would depend on specific agent deployment method
        logger.info(f"AgentZero: Starting agent {agent_id} with parameters {parameters}")
    
    async def stop_agent(self, agent_id: str):
        """Stop an agent"""
        try:
            container = self.docker_client.containers.get(agent_id)
            container.stop()
            logger.info(f"AgentZero: Stopped agent {agent_id}")
        except Exception as e:
            logger.error(f"AgentZero: Error stopping agent {agent_id}: {e}")
    
    async def scale_agent(self, agent_id: str, parameters: Dict[str, Any]):
        """Scale an agent"""
        logger.info(f"AgentZero: Scaling agent {agent_id} with parameters {parameters}")
    
    async def configure_agent(self, agent_id: str, parameters: Dict[str, Any]):
        """Configure an agent"""
        logger.info(f"AgentZero: Configuring agent {agent_id} with parameters {parameters}")
    
    async def trigger_emergency_action(self, action_type: str, reason: str):
        """Trigger an emergency action"""
        action_id = f"emergency_{int(datetime.utcnow().timestamp())}"
        
        emergency_action = EmergencyAction(
            action_id=action_id,
            action_type=action_type,
            affected_agents=list(self.agent_registry.keys()),
            reason=reason
        )
        
        self.emergency_actions[action_id] = emergency_action
        
        logger.critical(f"AgentZero: Emergency action triggered - {action_type}: {reason}")
        
        # Store in Redis for alerting
        if self.redis_client:
            await self.redis_client.lpush(
                "agentzero:emergency_log",
                json.dumps(emergency_action.dict(), default=str)
            )
    
    async def collect_agent_performance_data(self):
        """Collect performance data from all agents"""
        pass
    
    async def optimize_agent_performance(self):
        """Optimize agent performance based on collected data"""
        pass
    
    async def update_coordination_strategies(self):
        """Update coordination strategies based on system performance"""
        pass
    
    async def check_scaling_needs(self):
        """Check if any agents need scaling"""
        pass
    
    async def evaluate_scaling_decisions(self):
        """Evaluate scaling decisions"""
        pass
    
    async def execute_scaling_actions(self):
        """Execute scaling actions"""
        pass
    
    async def check_system_health(self):
        """Check overall system health"""
        if self.system_status and self.system_status.health_score < 0.3:
            logger.warning(f"AgentZero: System health degraded - Score: {self.system_status.health_score}")
    
    async def update_coordination_metrics(self):
        """Update coordination metrics"""
        if self.system_status:
            self.coordination_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": self.system_status.dict(),
                "active_commands": self.command_queue.qsize(),
                "emergency_actions": len(self.emergency_actions),
                "uptime_minutes": int((datetime.utcnow() - datetime(2025, 8, 5)).total_seconds() / 60)
            }
    
    async def submit_command(self, command: AgentCommand) -> Dict[str, Any]:
        """Submit a command for execution"""
        await self.command_queue.put(command)
        
        return {
            "command_id": command.command_id,
            "status": "queued",
            "message": f"Command {command.command_type} queued for {command.target_agent}"
        }
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "system_status": self.system_status.dict() if self.system_status else {},
            "coordination_metrics": self.coordination_metrics,
            "command_queue_size": self.command_queue.qsize(),
            "emergency_actions": len(self.emergency_actions),
            "agent_registry_size": len(self.agent_registry)
        }

# Global coordinator instance
coordinator = AgentZeroCoordinator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await coordinator.initialize()
    yield
    # Shutdown
    if coordinator.redis_client:
        await coordinator.redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="AgentZero Coordinator",
    description="Master orchestration agent for SUTAZAIAPP",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "agentzero-coordinator",
        "timestamp": datetime.utcnow().isoformat(),
        "system_health_score": coordinator.system_status.health_score if coordinator.system_status else 0.0,
        "active_agents": coordinator.system_status.active_agents if coordinator.system_status else 0
    }

@app.post("/submit_command")
async def submit_command(command: AgentCommand):
    """Submit a coordination command"""
    return await coordinator.submit_command(command)

@app.get("/status")
async def get_coordination_status():
    """Get coordination status"""
    return await coordinator.get_coordination_status()

@app.get("/system_status")
async def get_system_status():
    """Get system status"""
    return coordinator.system_status.dict() if coordinator.system_status else {}

@app.get("/emergency_actions")
async def list_emergency_actions():
    """List emergency actions"""
    actions = []
    for action_id, action in coordinator.emergency_actions.items():
        actions.append({
            "action_id": action_id,
            "action_type": action.action_type,
            "affected_agents": action.affected_agents,
            "reason": action.reason,
            "auto_triggered": action.auto_triggered
        })
    return {"emergency_actions": actions}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "agentzero-coordinator",
        "status": "running",
        "description": "Master Agent Coordination and Management Service",
        "role": "Master Orchestrator"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8586"))
    uvicorn.run(app, host="0.0.0.0", port=port)