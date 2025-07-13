#!/usr/bin/env python3
"""
Enhanced Agent Factory for SutazAI
Incorporates improvements from v3 and advanced framework integrations
"""

import asyncio
import importlib
import inspect
import os
from typing import Dict, List, Type, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from loguru import logger

from .base_agent import BaseAgent
from .ollama_agent import OllamaAgent, create_ollama_agent
from .ml_nlp_service import MLNLPService, create_ml_nlp_service


class AgentType(str, Enum):
    """Available agent types"""
    OLLAMA = "ollama"
    ML_NLP = "ml_nlp"
    AUTO_GPT = "auto_gpt"
    LOCAL_AGI = "local_agi"
    AUTO_GEN = "auto_gen"
    BIG_AGI = "big_agi"
    AGENT_ZERO = "agent_zero"
    BROWSER_USE = "browser_use"
    SKYVERN = "skyvern"
    OPEN_WEBUI = "open_webui"
    TABBY_ML = "tabby_ml"
    SEMGREP = "semgrep"
    DOCUMENT_PROCESSOR = "document_processor"
    CODE_GENERATOR = "code_generator"
    SUPREME_AI = "supreme_ai"


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    success_rate: float = 100.0
    last_used: Optional[str] = None
    health_score: float = 1.0
    
    def update_success_rate(self):
        """Update success rate based on completed vs failed tasks"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.success_rate = (self.tasks_completed / total_tasks) * 100
        else:
            self.success_rate = 100.0


@dataclass
class AgentConfiguration:
    """Agent configuration and metadata"""
    agent_type: AgentType
    name: str
    description: str
    enabled: bool = True
    auto_initialize: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    def validate(self) -> bool:
        """Validate agent configuration"""
        if not self.name or not self.description:
            return False
        return True


@dataclass
class AgentInstance:
    """Agent instance with metadata and status"""
    agent: BaseAgent
    config: AgentConfiguration
    status: AgentStatus = AgentStatus.INACTIVE
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    created_at: str = field(default_factory=lambda: str(asyncio.get_event_loop().time()))
    last_health_check: Optional[str] = None
    error_message: Optional[str] = None


class EnhancedAgentFactory:
    """Enhanced agent factory with v3 improvements and advanced capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.agents: Dict[str, AgentInstance] = {}
        self.agent_registry: Dict[AgentType, Callable] = {}
        self.config_path = config_path or "config/agents.json"
        self.global_config = {}
        
        # Performance tracking
        self.total_tasks_executed = 0
        self.factory_start_time = asyncio.get_event_loop().time()
        
        # Register built-in agents
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """Register built-in agent types"""
        self.agent_registry.update({
            AgentType.OLLAMA: create_ollama_agent,
            AgentType.ML_NLP: create_ml_nlp_service,
        })
        
        logger.info(f"Registered {len(self.agent_registry)} built-in agent types")
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the agent factory"""
        try:
            self.global_config = config or {}
            
            # Load agent configurations
            await self._load_agent_configurations()
            
            # Discover and register additional agents
            await self._discover_agents()
            
            # Initialize enabled agents
            await self._initialize_enabled_agents()
            
            logger.info(f"Agent factory initialized with {len(self.agents)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent factory: {e}")
            return False
    
    async def _load_agent_configurations(self):
        """Load agent configurations from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs = json.load(f)
                
                for agent_config in configs.get('agents', []):
                    config = AgentConfiguration(**agent_config)
                    if config.validate():
                        logger.info(f"Loaded configuration for agent: {config.name}")
                    else:
                        logger.warning(f"Invalid configuration for agent: {config.name}")
            else:
                # Create default configurations
                await self._create_default_configurations()
                
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            await self._create_default_configurations()
    
    async def _create_default_configurations(self):
        """Create default agent configurations"""
        default_configs = [
            AgentConfiguration(
                agent_type=AgentType.OLLAMA,
                name="Ollama Local LLM",
                description="Local Large Language Model using Ollama",
                capabilities=["text_generation", "chat", "embeddings", "code_generation"],
                config={
                    "host": "localhost",
                    "port": 11434,
                    "default_model": "llama3.1:8b",
                    "timeout": 60
                }
            ),
            AgentConfiguration(
                agent_type=AgentType.ML_NLP,
                name="Advanced ML/NLP Service",
                description="Multi-framework ML and NLP processing service",
                capabilities=["sentiment_analysis", "ner", "classification", "summarization", "embeddings"],
                dependencies=["transformers", "spacy", "nltk"],
                config={}
            ),
        ]
        
        # Save default configurations
        await self._save_configurations(default_configs)
    
    async def _save_configurations(self, configs: List[AgentConfiguration]):
        """Save agent configurations to file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                "version": "1.0.0",
                "agents": [
                    {
                        "agent_type": config.agent_type.value,
                        "name": config.name,
                        "description": config.description,
                        "enabled": config.enabled,
                        "auto_initialize": config.auto_initialize,
                        "config": config.config,
                        "dependencies": config.dependencies,
                        "capabilities": config.capabilities,
                        "resource_requirements": config.resource_requirements,
                        "version": config.version
                    }
                    for config in configs
                ]
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Saved {len(configs)} agent configurations")
            
        except Exception as e:
            logger.error(f"Failed to save agent configurations: {e}")
    
    async def _discover_agents(self):
        """Discover additional agent implementations"""
        try:
            # Look for agent implementations in the ai_agents directory
            agents_dir = Path(__file__).parent
            
            for python_file in agents_dir.glob("*_agent.py"):
                if python_file.name.startswith("base_") or python_file.name.startswith("enhanced_"):
                    continue
                
                try:
                    module_name = python_file.stem
                    spec = importlib.util.spec_from_file_location(module_name, python_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for agent classes or factory functions
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseAgent) and 
                            obj != BaseAgent):
                            logger.info(f"Discovered agent class: {name} in {module_name}")
                        
                        elif (inspect.isfunction(obj) and 
                              name.startswith("create_") and 
                              name.endswith("_agent")):
                            logger.info(f"Discovered agent factory: {name} in {module_name}")
                
                except Exception as e:
                    logger.warning(f"Failed to discover agents in {python_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
    
    async def _initialize_enabled_agents(self):
        """Initialize all enabled agents"""
        for agent_type, factory_func in self.agent_registry.items():
            try:
                # Get configuration for this agent type
                config = self.global_config.get(agent_type.value, {})
                
                # Check if agent is enabled
                if not config.get("enabled", True):
                    logger.info(f"Skipping disabled agent: {agent_type.value}")
                    continue
                
                # Create agent configuration
                agent_config = AgentConfiguration(
                    agent_type=agent_type,
                    name=f"{agent_type.value.title()} Agent",
                    description=f"Auto-generated {agent_type.value} agent",
                    config=config
                )
                
                # Create and initialize agent
                await self.create_agent(agent_config)
                
            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_type.value}: {e}")
    
    async def create_agent(self, config: AgentConfiguration) -> Optional[str]:
        """Create and register a new agent instance"""
        try:
            # Check if agent type is registered
            if config.agent_type not in self.agent_registry:
                logger.error(f"Unknown agent type: {config.agent_type}")
                return None
            
            # Create agent instance
            factory_func = self.agent_registry[config.agent_type]
            agent = factory_func(config.config)
            
            # Create agent instance wrapper
            agent_id = f"{config.agent_type.value}_{len(self.agents)}"
            instance = AgentInstance(
                agent=agent,
                config=config,
                status=AgentStatus.INITIALIZING
            )
            
            # Initialize agent if auto_initialize is enabled
            if config.auto_initialize:
                try:
                    success = await agent.initialize()
                    instance.status = AgentStatus.ACTIVE if success else AgentStatus.ERROR
                    if not success:
                        instance.error_message = "Initialization failed"
                except Exception as e:
                    instance.status = AgentStatus.ERROR
                    instance.error_message = str(e)
                    logger.error(f"Agent initialization failed: {e}")
            
            # Register agent
            self.agents[agent_id] = instance
            
            logger.info(f"Created agent: {agent_id} (Status: {instance.status.value})")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    async def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent instance by ID"""
        instance = self.agents.get(agent_id)
        return instance.agent if instance else None
    
    async def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """List all registered agents with their status"""
        return {
            agent_id: {
                "name": instance.config.name,
                "type": instance.config.agent_type.value,
                "status": instance.status.value,
                "metrics": {
                    "tasks_completed": instance.metrics.tasks_completed,
                    "tasks_failed": instance.metrics.tasks_failed,
                    "success_rate": instance.metrics.success_rate,
                    "health_score": instance.metrics.health_score
                },
                "capabilities": instance.config.capabilities,
                "created_at": instance.created_at,
                "error_message": instance.error_message
            }
            for agent_id, instance in self.agents.items()
        }
    
    async def execute_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using specific agent"""
        instance = self.agents.get(agent_id)
        if not instance:
            return {
                "status": "error",
                "error": f"Agent not found: {agent_id}"
            }
        
        if instance.status != AgentStatus.ACTIVE:
            return {
                "status": "error",
                "error": f"Agent not active: {instance.status.value}"
            }
        
        try:
            import time
            start_time = time.time()
            
            # Execute task
            result = await instance.agent.execute_task(task)
            
            # Update metrics
            execution_time = time.time() - start_time
            instance.metrics.average_response_time = (
                (instance.metrics.average_response_time * instance.metrics.tasks_completed + execution_time) /
                (instance.metrics.tasks_completed + 1)
            )
            
            if result.get("status") == "completed":
                instance.metrics.tasks_completed += 1
            else:
                instance.metrics.tasks_failed += 1
            
            instance.metrics.update_success_rate()
            instance.metrics.last_used = str(time.time())
            
            self.total_tasks_executed += 1
            
            return result
            
        except Exception as e:
            instance.metrics.tasks_failed += 1
            instance.metrics.update_success_rate()
            logger.error(f"Task execution failed for agent {agent_id}: {e}")
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def health_check(self, agent_id: str = None) -> Dict[str, Any]:
        """Perform health check on agents"""
        if agent_id:
            # Check specific agent
            instance = self.agents.get(agent_id)
            if not instance:
                return {"status": "error", "error": "Agent not found"}
            
            try:
                if hasattr(instance.agent, 'get_health_status'):
                    health = await instance.agent.get_health_status()
                else:
                    health = {"status": "unknown"}
                
                instance.last_health_check = str(asyncio.get_event_loop().time())
                return health
                
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        else:
            # Check all agents
            results = {}
            for aid, instance in self.agents.items():
                try:
                    if hasattr(instance.agent, 'get_health_status'):
                        health = await instance.agent.get_health_status()
                    else:
                        health = {"status": instance.status.value}
                    
                    results[aid] = health
                    instance.last_health_check = str(asyncio.get_event_loop().time())
                    
                except Exception as e:
                    results[aid] = {"status": "error", "error": str(e)}
            
            return results
    
    async def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory performance statistics"""
        active_agents = sum(1 for instance in self.agents.values() 
                          if instance.status == AgentStatus.ACTIVE)
        
        total_tasks = sum(instance.metrics.tasks_completed + instance.metrics.tasks_failed 
                         for instance in self.agents.values())
        
        average_success_rate = (
            sum(instance.metrics.success_rate for instance in self.agents.values()) / 
            len(self.agents) if self.agents else 0
        )
        
        uptime = asyncio.get_event_loop().time() - self.factory_start_time
        
        return {
            "total_agents": len(self.agents),
            "active_agents": active_agents,
            "registered_types": len(self.agent_registry),
            "total_tasks_executed": total_tasks,
            "average_success_rate": average_success_rate,
            "uptime_seconds": uptime,
            "factory_version": "2.0.0"
        }
    
    async def shutdown(self):
        """Shutdown all agents and cleanup resources"""
        logger.info("Shutting down agent factory...")
        
        for agent_id, instance in self.agents.items():
            try:
                if hasattr(instance.agent, 'cleanup'):
                    await instance.agent.cleanup()
                instance.status = AgentStatus.INACTIVE
                logger.info(f"Shutdown agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to shutdown agent {agent_id}: {e}")
        
        logger.info("Agent factory shutdown complete")


# Global factory instance
factory = EnhancedAgentFactory()


# Convenience functions
async def initialize_factory(config: Dict[str, Any] = None) -> bool:
    """Initialize the global agent factory"""
    return await factory.initialize(config)


async def get_agent(agent_id: str) -> Optional[BaseAgent]:
    """Get agent by ID"""
    return await factory.get_agent(agent_id)


async def list_agents() -> Dict[str, Dict[str, Any]]:
    """List all agents"""
    return await factory.list_agents()


async def execute_task(agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute task with specific agent"""
    return await factory.execute_task(agent_id, task)


async def health_check(agent_id: str = None) -> Dict[str, Any]:
    """Perform health check"""
    return await factory.health_check(agent_id)