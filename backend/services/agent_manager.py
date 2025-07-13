"""
Comprehensive Agent Manager for SutazAI
Manages all AI agents and their interactions
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import importlib
import sys
import time

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/agents.json"):
        self.config_path = Path(config_path)
        self.agents = {}
        self.agent_configs = {}
        self.agent_stats = {}
        self.load_agent_configurations()
        self.initialize_agents()
    
    def load_agent_configurations(self):
        """Load agent configurations from JSON"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.agent_configs = json.load(f)
                logger.info(f"✅ Loaded {len(self.agent_configs)} agent configurations")
            else:
                logger.warning(f"Agent config file not found: {self.config_path}")
                self.agent_configs = {}
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            self.agent_configs = {}
    
    def initialize_agents(self):
        """Initialize all configured agents"""
        for agent_name, config in self.agent_configs.items():
            try:
                if config.get("enabled", True):
                    agent_instance = self._create_agent(agent_name, config)
                    if agent_instance:
                        self.agents[agent_name] = agent_instance
                        self.agent_stats[agent_name] = {
                            "created_at": time.time(),
                            "tasks_executed": 0,
                            "last_used": None,
                            "status": "active"
                        }
                        logger.info(f"✅ Initialized agent: {agent_name}")
                else:
                    logger.info(f"⏸️ Agent {agent_name} is disabled")
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_name}: {e}")
    
    def _create_agent(self, agent_name: str, config: Dict[str, Any]):
        """Create an agent instance based on configuration"""
        agent_type = config.get("type", "").lower()
        
        # Try to import and create actual agent implementations
        if agent_type == "ollama":
            return self._create_ollama_agent(config)
        elif agent_type == "ml_nlp":
            return self._create_ml_nlp_agent(config)
        elif agent_type == "document_processor":
            return self._create_document_processor_agent(config)
        elif agent_type == "code_generator":
            return self._create_code_generator_agent(config)
        else:
            # Create placeholder agent for other types
            return self._create_placeholder_agent(agent_type, config)
    
    def _create_ollama_agent(self, config: Dict[str, Any]):
        """Create Ollama agent"""
        try:
            sys.path.append('/opt/sutazaiapp')
            from ai_agents.ollama_agent import OllamaAgent
            return OllamaAgent(config.get("config", {}))
        except Exception as e:
            logger.error(f"Failed to create Ollama agent: {e}")
            return self._create_placeholder_agent("ollama", config)
    
    def _create_ml_nlp_agent(self, config: Dict[str, Any]):
        """Create ML/NLP agent"""
        try:
            sys.path.append('/opt/sutazaiapp')
            from ai_agents.ml_nlp_service import MLNLPService
            return MLNLPService(config.get("config", {}))
        except Exception as e:
            logger.error(f"Failed to create ML/NLP agent: {e}")
            return self._create_placeholder_agent("ml_nlp", config)
    
    def _create_document_processor_agent(self, config: Dict[str, Any]):
        """Create document processor agent"""
        return self._create_placeholder_agent("document_processor", config)
    
    def _create_code_generator_agent(self, config: Dict[str, Any]):
        """Create code generator agent"""
        return self._create_placeholder_agent("code_generator", config)
    
    def _create_placeholder_agent(self, agent_type: str, config: Dict[str, Any]):
        """Create a placeholder agent for types not yet implemented"""
        return {
            "type": agent_type,
            "config": config,
            "status": "placeholder",
            "capabilities": config.get("capabilities", []),
            "description": config.get("description", f"{agent_type} agent")
        }
    
    async def execute_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the specified agent"""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return {"error": f"Agent {agent_name} not found or not initialized"}
            
            # Update stats
            self.agent_stats[agent_name]["tasks_executed"] += 1
            self.agent_stats[agent_name]["last_used"] = time.time()
            
            # Route to appropriate execution method
            if hasattr(agent, 'execute_task'):
                result = await agent.execute_task(task)
            elif hasattr(agent, 'process_task'):
                result = await agent.process_task(task)
            elif isinstance(agent, dict) and agent.get("status") == "placeholder":
                result = {
                    "status": "placeholder_response",
                    "message": f"Agent {agent_name} ({agent['type']}) is not yet fully implemented",
                    "capabilities": agent.get("capabilities", []),
                    "task_received": task
                }
            else:
                result = {"error": f"Agent {agent_name} does not support task execution"}
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute task with agent {agent_name}: {e}")
            return {"error": str(e)}
    
    async def get_agent_status(self, agent_name: str = None) -> Dict[str, Any]:
        """Get status of one or all agents"""
        if agent_name:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                stats = self.agent_stats.get(agent_name, {})
                
                # Try to get health status from agent
                health_status = "unknown"
                if hasattr(agent, 'get_health_status'):
                    try:
                        health_result = await agent.get_health_status()
                        health_status = health_result.get("status", "unknown")
                    except:
                        health_status = "error"
                elif isinstance(agent, dict):
                    health_status = agent.get("status", "placeholder")
                
                return {
                    "name": agent_name,
                    "type": self.agent_configs[agent_name].get("type"),
                    "health": health_status,
                    "stats": stats,
                    "capabilities": self.agent_configs[agent_name].get("capabilities", []),
                    "enabled": self.agent_configs[agent_name].get("enabled", True)
                }
            else:
                return {"error": f"Agent {agent_name} not found"}
        else:
            # Return status for all agents
            status_map = {}
            for name in self.agents.keys():
                status_map[name] = await self.get_agent_status(name)
            return status_map
    
    async def list_agents(self) -> Dict[str, Any]:
        """List all available agents"""
        agent_list = {}
        for agent_name, config in self.agent_configs.items():
            agent_list[agent_name] = {
                "type": config.get("type"),
                "enabled": config.get("enabled", True),
                "description": config.get("description", ""),
                "capabilities": config.get("capabilities", []),
                "initialized": agent_name in self.agents,
                "stats": self.agent_stats.get(agent_name, {})
            }
        return agent_list
    
    async def get_capabilities(self) -> Dict[str, List[str]]:
        """Get all capabilities provided by agents"""
        capabilities = {}
        for agent_name, config in self.agent_configs.items():
            if config.get("enabled", True):
                capabilities[agent_name] = config.get("capabilities", [])
        return capabilities
    
    async def find_agents_by_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability"""
        matching_agents = []
        for agent_name, config in self.agent_configs.items():
            if (config.get("enabled", True) and 
                capability in config.get("capabilities", [])):
                matching_agents.append(agent_name)
        return matching_agents
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        total_agents = len(self.agent_configs)
        enabled_agents = sum(1 for config in self.agent_configs.values() if config.get("enabled", True))
        initialized_agents = len(self.agents)
        
        # Check individual agent health
        agent_health = {}
        for agent_name in self.agents.keys():
            status = await self.get_agent_status(agent_name)
            agent_health[agent_name] = status.get("health", "unknown")
        
        healthy_agents = sum(1 for health in agent_health.values() if health in ["healthy", "active"])
        
        return {
            "status": "healthy" if healthy_agents > 0 else "degraded",
            "total_agents": total_agents,
            "enabled_agents": enabled_agents,
            "initialized_agents": initialized_agents,
            "healthy_agents": healthy_agents,
            "agent_health": agent_health,
            "uptime": time.time() - min([stats.get("created_at", time.time()) 
                                       for stats in self.agent_stats.values()], default=time.time())
        }
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent"""
        return self.agent_configs.get(agent_name)
    
    async def reload_configuration(self) -> bool:
        """Reload agent configurations from file"""
        try:
            old_configs = self.agent_configs.copy()
            self.load_agent_configurations()
            
            # Re-initialize agents if configs changed
            if self.agent_configs != old_configs:
                logger.info("Agent configurations changed, reinitializing...")
                self.agents.clear()
                self.agent_stats.clear()
                self.initialize_agents()
            
            logger.info("✅ Agent configurations reloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to reload configurations: {e}")
            return False

# Global agent manager instance
agent_manager = AgentManager()

# Convenience functions
async def execute_task(agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute task with specific agent"""
    return await agent_manager.execute_task(agent_name, task)

async def get_agent_status(agent_name: str = None) -> Dict[str, Any]:
    """Get agent status"""
    return await agent_manager.get_agent_status(agent_name)

async def list_agents() -> Dict[str, Any]:
    """List all agents"""
    return await agent_manager.list_agents()

async def health_check() -> Dict[str, Any]:
    """Perform health check"""
    return await agent_manager.health_check()

async def get_capabilities() -> Dict[str, List[str]]:
    """Get all agent capabilities"""
    return await agent_manager.get_capabilities()

async def find_agents_by_capability(capability: str) -> List[str]:
    """Find agents by capability"""
    return await agent_manager.find_agents_by_capability(capability)