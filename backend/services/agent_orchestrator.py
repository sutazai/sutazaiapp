#!/usr/bin/env python3
"""
SutazAI Agent Orchestrator
Manages and coordinates AI agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
import json

from ..core.config import settings

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates AI agents across different services"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.active_agents = {}
        self.agent_configs = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the agent orchestrator"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Agent Orchestrator...")
            
            # Load agent configurations
            await self._load_agent_configs()
            
            # Test connections to agent services
            await self._test_agent_connections()
            
            self._initialized = True
            logger.info("Agent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Agent Orchestrator: {e}")
            raise
    
    async def _load_agent_configs(self):
        """Load agent configurations"""
        self.agent_configs = {
            "autogpt": {
                "url": settings.AUTOGPT_URL,
                "capabilities": ["reasoning", "web_search", "code_execution", "file_management"],
                "max_instances": 3,
                "health_endpoint": "/health"
            },
            "localagi": {
                "url": settings.LOCALAGI_URL,
                "capabilities": ["local_inference", "model_management", "text_generation"],
                "max_instances": 2,
                "health_endpoint": "/health"
            },
            "tabbyml": {
                "url": settings.TABBYML_URL,
                "capabilities": ["code_completion", "code_analysis", "programming_assistance"],
                "max_instances": 5,
                "health_endpoint": "/health"
            },
            "agentzero": {
                "url": settings.AGENTZERO_URL,
                "capabilities": ["task_automation", "workflow_execution", "decision_making"],
                "max_instances": 2,
                "health_endpoint": "/health"
            },
            "bigagi": {
                "url": settings.BIGAGI_URL,
                "capabilities": ["multi_modal", "advanced_reasoning", "creative_tasks"],
                "max_instances": 1,
                "health_endpoint": "/health"
            }
        }
    
    async def _test_agent_connections(self):
        """Test connections to all agent services"""
        for agent_name, config in self.agent_configs.items():
            try:
                response = await self.http_client.get(
                    f"{config['url']}{config['health_endpoint']}"
                )
                if response.status_code == 200:
                    logger.info(f"Agent {agent_name} is healthy")
                else:
                    logger.warning(f"Agent {agent_name} health check failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"Cannot reach agent {agent_name}: {e}")
    
    async def shutdown(self):
        """Shutdown the agent orchestrator"""
        logger.info("Shutting down Agent Orchestrator...")
        
        # Stop all active agents
        for agent_id in list(self.active_agents.keys()):
            await self.stop_agent(agent_id)
        
        self._initialized = False
        logger.info("Agent Orchestrator shutdown complete")
    
    async def create_agent(self, agent_type: str, task: Dict[str, Any], agent_id: str = None) -> Dict[str, Any]:
        """Create a new agent instance"""
        try:
            if agent_type not in self.agent_configs:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            config = self.agent_configs[agent_type]
            
            # Check if we can create more instances
            active_count = len([a for a in self.active_agents.values() if a["type"] == agent_type])
            if active_count >= config["max_instances"]:
                raise ValueError(f"Maximum instances reached for agent type: {agent_type}")
            
            # Generate agent ID if not provided
            if not agent_id:
                agent_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create agent instance
            agent_data = {
                "agent_id": agent_id,
                "type": agent_type,
                "url": config["url"],
                "capabilities": config["capabilities"],
                "task": task,
                "status": "initializing",
                "created_at": datetime.utcnow().isoformat(),
                "last_update": datetime.utcnow().isoformat()
            }
            
            # Start the agent
            await self._start_agent(agent_data)
            
            # Add to active agents
            self.active_agents[agent_id] = agent_data
            
            logger.info(f"Created agent {agent_id} of type {agent_type}")
            
            return {
                "agent_id": agent_id,
                "type": agent_type,
                "status": "created",
                "capabilities": config["capabilities"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}")
            raise
    
    async def _start_agent(self, agent_data: Dict[str, Any]):
        """Start an agent instance"""
        try:
            # Send start request to agent service
            response = await self.http_client.post(
                f"{agent_data['url']}/start",
                json={
                    "agent_id": agent_data["agent_id"],
                    "task": agent_data["task"]
                }
            )
            
            if response.status_code == 200:
                agent_data["status"] = "running"
                agent_data["last_update"] = datetime.utcnow().isoformat()
            else:
                agent_data["status"] = "failed"
                agent_data["error"] = f"Failed to start: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Failed to start agent {agent_data['agent_id']}: {e}")
            agent_data["status"] = "failed"
            agent_data["error"] = str(e)
    
    async def stop_agent(self, agent_id: str) -> bool:
        """Stop an agent instance"""
        try:
            if agent_id not in self.active_agents:
                return False
            
            agent_data = self.active_agents[agent_id]
            
            # Send stop request to agent service
            response = await self.http_client.post(
                f"{agent_data['url']}/stop",
                json={"agent_id": agent_id}
            )
            
            # Remove from active agents
            del self.active_agents[agent_id]
            
            logger.info(f"Stopped agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}")
            return False
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an agent"""
        if agent_id not in self.active_agents:
            return None
        
        agent_data = self.active_agents[agent_id]
        
        try:
            # Get current status from agent service
            response = await self.http_client.get(
                f"{agent_data['url']}/status/{agent_id}"
            )
            
            if response.status_code == 200:
                status_data = response.json()
                agent_data.update(status_data)
                agent_data["last_update"] = datetime.utcnow().isoformat()
            
            return agent_data
            
        except Exception as e:
            logger.error(f"Failed to get status for agent {agent_id}: {e}")
            return agent_data
    
    async def execute_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with an agent"""
        try:
            if agent_id not in self.active_agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent_data = self.active_agents[agent_id]
            
            # Send task to agent
            response = await self.http_client.post(
                f"{agent_data['url']}/execute",
                json={
                    "agent_id": agent_id,
                    "task": task
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                agent_data["last_update"] = datetime.utcnow().isoformat()
                return result
            else:
                raise Exception(f"Task execution failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to execute task with agent {agent_id}: {e}")
            raise
    
    async def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents"""
        return list(self.active_agents.values())
    
    async def get_active_agents_count(self) -> int:
        """Get count of active agents"""
        return len(self.active_agents)
    
    async def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get agents by capability"""
        matching_agents = []
        
        for agent_data in self.active_agents.values():
            if capability in agent_data.get("capabilities", []):
                matching_agents.append(agent_data)
        
        return matching_agents
    
    async def orchestrate_multi_agent_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a task across multiple agents"""
        try:
            # Analyze task requirements
            required_capabilities = task.get("capabilities", [])
            
            # Find suitable agents
            suitable_agents = []
            for capability in required_capabilities:
                agents = await self.get_agent_by_capability(capability)
                suitable_agents.extend(agents)
            
            # Remove duplicates
            suitable_agents = list({agent["agent_id"]: agent for agent in suitable_agents}.values())
            
            if not suitable_agents:
                raise ValueError("No suitable agents found for task")
            
            # Execute task with suitable agents
            results = []
            for agent in suitable_agents:
                try:
                    result = await self.execute_task(agent["agent_id"], task)
                    results.append({
                        "agent_id": agent["agent_id"],
                        "result": result,
                        "status": "success"
                    })
                except Exception as e:
                    results.append({
                        "agent_id": agent["agent_id"],
                        "error": str(e),
                        "status": "failed"
                    })
            
            return {
                "task_id": task.get("task_id"),
                "results": results,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Multi-agent task orchestration failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for agent orchestrator"""
        try:
            health_data = {
                "status": "healthy",
                "active_agents": len(self.active_agents),
                "agent_types": list(self.agent_configs.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check individual agent health
            agent_health = {}
            for agent_name, config in self.agent_configs.items():
                try:
                    response = await self.http_client.get(
                        f"{config['url']}{config['health_endpoint']}"
                    )
                    agent_health[agent_name] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                    }
                except Exception as e:
                    agent_health[agent_name] = {
                        "status": "unreachable",
                        "error": str(e)
                    }
            
            health_data["agent_health"] = agent_health
            
            return health_data
            
        except Exception as e:
            logger.error(f"Agent orchestrator health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }