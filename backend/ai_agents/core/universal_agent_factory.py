"""
Universal Agent Factory - Dynamic Agent Creation System
======================================================

This factory creates any type of AI agent dynamically based on configuration.
It supports multiple agent types and can instantiate them with local Ollama models
or any other AI provider. The factory is completely independent from external APIs.

Features:
- Dynamic agent type registration
- Template-based agent creation
- Local model integration
- Capability-based agent selection
- Plugin architecture for custom agents
"""

import asyncio
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable

from agents.core.base_agent import BaseAgent, AgentConfig, AgentCapability, AgentStatus


class AgentTemplate:
    """Template for creating agents of a specific type"""
    
    def __init__(self, agent_type: str, class_path: str, default_config: Dict[str, Any]):
        self.agent_type = agent_type
        self.class_path = class_path
        self.default_config = default_config
        self.agent_class: Optional[Type[BaseAgent]] = None
        self._loaded = False
    
    async def load_class(self):
        """Dynamically load the agent class"""
        if self._loaded:
            return
        
        try:
            module_path, class_name = self.class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            self.agent_class = getattr(module, class_name)
            self._loaded = True
        except Exception as e:
            raise ImportError(f"Failed to load agent class {self.class_path}: {e}")
    
    def create_config(self, agent_id: str, overrides: Dict[str, Any] = None) -> AgentConfig:
        """Create agent configuration from template"""
        config_data = self.default_config.copy()
        if overrides:
            config_data.update(overrides)
        
        # Ensure required fields
        config_data.setdefault("agent_id", agent_id)
        config_data.setdefault("agent_type", self.agent_type)
        
        # Convert capability strings to enums
        if "capabilities" in config_data:
            capabilities = []
            for cap in config_data["capabilities"]:
                if isinstance(cap, str):
                    try:
                        capabilities.append(AgentCapability(cap))
                    except ValueError:
                        logging.warning(f"Unknown capability: {cap}")
                else:
                    capabilities.append(cap)
            config_data["capabilities"] = capabilities
        
        return AgentConfig(**config_data)
    
    async def create_agent(self, agent_id: str, overrides: Dict[str, Any] = None) -> BaseAgent:
        """Create agent instance"""
        await self.load_class()
        config = self.create_config(agent_id, overrides)
        return self.agent_class(config)


class UniversalAgentFactory:
    """
    Universal Agent Factory
    
    Creates and manages AI agents of any type. Supports dynamic registration
    of agent types and provides a unified interface for agent creation.
    """
    
    def __init__(self, config_dir: str = None):
        self.logger = logging.getLogger("universal_agent_factory")
        self.config_dir = config_dir or "/opt/sutazaiapp/backend/ai_agents/configs"
        self.templates: Dict[str, AgentTemplate] = {}
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_stats = {
            "created": 0,
            "destroyed": 0,
            "active": 0,
            "types": set()
        }
        
        # Default agent types
        self._register_default_templates()
        
        # Load custom templates from config directory
        self._load_custom_templates()
    
    def _register_default_templates(self):
        """Register built-in agent templates"""
        
        # Code Generation Agent
        self.register_template(AgentTemplate(
            agent_type="code_generator",
            class_path="backend.ai_agents.specialized.code_generator.CodeGeneratorAgent",
            default_config={
                "name": "Code Generator",
                "description": "Generates code using local AI models",
                "capabilities": ["code_generation", "reasoning"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.2,
                    "max_tokens": 4000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 3
            }
        ))
        
        # Security Analyzer Agent
        self.register_template(AgentTemplate(
            agent_type="security_analyzer",
            class_path="backend.ai_agents.specialized.security_analyzer.SecurityAnalyzerAgent",
            default_config={
                "name": "Security Analyzer",
                "description": "Analyzes code for security vulnerabilities",
                "capabilities": ["security_analysis", "code_analysis"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.1,
                    "max_tokens": 3000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 2
            }
        ))
        
        # Test Agent
        self.register_template(AgentTemplate(
            agent_type="test_agent",
            class_path="backend.ai_agents.specialized.test_agent.TestAgent",
            default_config={
                "name": "Test Agent",
                "description": "Creates and runs tests for code",
                "capabilities": ["testing", "code_analysis"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 5
            }
        ))
        
        # Orchestrator Agent
        self.register_template(AgentTemplate(
            agent_type="orchestrator",
            class_path="backend.ai_agents.specialized.orchestrator.OrchestratorAgent",
            default_config={
                "name": "Orchestrator",
                "description": "Coordinates multiple agents and workflows",
                "capabilities": ["orchestration", "communication", "reasoning"],
                "model_config": {
                    "model": "llama2",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.4,
                    "max_tokens": 3000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 10
            }
        ))
        
        # Data Processing Agent
        self.register_template(AgentTemplate(
            agent_type="data_processor",
            class_path="backend.ai_agents.specialized.data_processor.DataProcessorAgent",
            default_config={
                "name": "Data Processor",
                "description": "Processes and analyzes data",
                "capabilities": ["data_processing", "file_operations"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.5,
                    "max_tokens": 2500
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 4
            }
        ))
        
        # Deployment Agent
        self.register_template(AgentTemplate(
            agent_type="deployment_agent",
            class_path="backend.ai_agents.specialized.deployment_agent.DeploymentAgent",
            default_config={
                "name": "Deployment Agent",
                "description": "Handles application deployment and operations",
                "capabilities": ["deployment", "monitoring", "autonomous_execution"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.2,
                    "max_tokens": 2000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 2
            }
        ))
        
        # API Integration Agent
        self.register_template(AgentTemplate(
            agent_type="api_agent",
            class_path="backend.ai_agents.specialized.api_agent.APIAgent",
            default_config={
                "name": "API Agent",
                "description": "Handles API integrations and external communications",
                "capabilities": ["api_integration", "communication"],
                "model_config": {
                    "model": "tinyllama",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 8
            }
        ))
        
        # Learning Agent
        self.register_template(AgentTemplate(
            agent_type="learning_agent",
            class_path="backend.ai_agents.specialized.learning_agent.LearningAgent",
            default_config={
                "name": "Learning Agent",
                "description": "Continuously learns and improves from experience",
                "capabilities": ["learning", "reasoning", "autonomous_execution"],
                "model_config": {
                    "model": "llama2",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.6,
                    "max_tokens": 3000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 3
            }
        ))
        
        # Generic Agent (fallback)
        self.register_template(AgentTemplate(
            agent_type="generic",
            class_path="backend.ai_agents.specialized.generic_agent.GenericAgent",
            default_config={
                "name": "Generic Agent",
                "description": "General purpose agent for any task",
                "capabilities": ["reasoning", "communication"],
                "model_config": {
                    "model": "llama2",
                    "ollama_url": "http://localhost:10104",
                    "temperature": 0.5,
                    "max_tokens": 2000
                },
                "redis_config": {
                    "url": "redis://localhost:6379"
                },
                "max_concurrent_tasks": 5
            }
        ))
    
    def _load_custom_templates(self):
        """Load custom agent templates from configuration directory"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            return
        
        for config_file in Path(self.config_dir).glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                if "agent_type" in config_data and "class_path" in config_data:
                    template = AgentTemplate(
                        agent_type=config_data["agent_type"],
                        class_path=config_data["class_path"],
                        default_config=config_data.get("default_config", {})
                    )
                    self.register_template(template)
                    self.logger.info(f"Loaded custom template: {config_data['agent_type']}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load template from {config_file}: {e}")
    
    def register_template(self, template: AgentTemplate):
        """Register an agent template"""
        self.templates[template.agent_type] = template
        self.agent_stats["types"].add(template.agent_type)
        self.logger.info(f"Registered agent template: {template.agent_type}")
    
    def get_available_types(self) -> List[str]:
        """Get list of available agent types"""
        return list(self.templates.keys())
    
    def get_template(self, agent_type: str) -> Optional[AgentTemplate]:
        """Get agent template by type"""
        return self.templates.get(agent_type)
    
    async def create_agent(self, agent_id: str, agent_type: str, 
                          config_overrides: Dict[str, Any] = None) -> BaseAgent:
        """Create a new agent instance"""
        if agent_id in self.active_agents:
            raise ValueError(f"Agent with ID {agent_id} already exists")
        
        template = self.templates.get(agent_type)
        if not template:
            # Fall back to generic agent
            template = self.templates.get("generic")
            if not template:
                raise ValueError(f"No template found for agent type: {agent_type}")
        
        try:
            # Create agent
            agent = await template.create_agent(agent_id, config_overrides)
            
            # Initialize agent
            success = await agent.initialize()
            if not success:
                raise RuntimeError(f"Failed to initialize agent {agent_id}")
            
            # Track agent
            self.active_agents[agent_id] = agent
            self.agent_stats["created"] += 1
            self.agent_stats["active"] += 1
            
            self.logger.info(f"Created agent {agent_id} of type {agent_type}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {e}")
            raise
    
    async def create_agent_by_capabilities(self, agent_id: str, 
                                         required_capabilities: List[Union[str, AgentCapability]],
                                         preferred_type: str = None,
                                         config_overrides: Dict[str, Any] = None) -> BaseAgent:
        """Create agent based on required capabilities"""
        
        # Convert string capabilities to enums
        capabilities = []
        for cap in required_capabilities:
            if isinstance(cap, str):
                try:
                    capabilities.append(AgentCapability(cap))
                except ValueError:
                    self.logger.warning(f"Unknown capability: {cap}")
            else:
                capabilities.append(cap)
        
        # Find suitable agent type
        suitable_types = []
        for agent_type, template in self.templates.items():
            template_caps = set(template.default_config.get("capabilities", []))
            if all(cap.value in template_caps or cap in template_caps for cap in capabilities):
                suitable_types.append(agent_type)
        
        if not suitable_types:
            # Fall back to generic agent
            agent_type = "generic"
        elif preferred_type and preferred_type in suitable_types:
            agent_type = preferred_type
        else:
            # Choose first suitable type
            agent_type = suitable_types[0]
        
        return await self.create_agent(agent_id, agent_type, config_overrides)
    
    async def create_agent_fleet(self, fleet_config: Dict[str, Any]) -> Dict[str, BaseAgent]:
        """Create multiple agents as a fleet"""
        fleet = {}
        
        for agent_spec in fleet_config.get("agents", []):
            agent_id = agent_spec["agent_id"]
            agent_type = agent_spec["agent_type"]
            overrides = agent_spec.get("config", {})
            
            try:
                agent = await self.create_agent(agent_id, agent_type, overrides)
                fleet[agent_id] = agent
            except Exception as e:
                self.logger.error(f"Failed to create fleet agent {agent_id}: {e}")
        
        self.logger.info(f"Created agent fleet with {len(fleet)} agents")
        return fleet
    
    async def destroy_agent(self, agent_id: str):
        """Destroy an agent and clean up resources"""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        agent = self.active_agents[agent_id]
        
        try:
            await agent.shutdown()
            del self.active_agents[agent_id]
            self.agent_stats["destroyed"] += 1
            self.agent_stats["active"] -= 1
            
            self.logger.info(f"Destroyed agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error destroying agent {agent_id}: {e}")
            raise
    
    async def destroy_all_agents(self):
        """Destroy all active agents"""
        agent_ids = list(self.active_agents.keys())
        
        for agent_id in agent_ids:
            try:
                await self.destroy_agent(agent_id)
            except Exception as e:
                self.logger.error(f"Error destroying agent {agent_id}: {e}")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an active agent by ID"""
        return self.active_agents.get(agent_id)
    
    def get_active_agents(self) -> Dict[str, BaseAgent]:
        """Get all active agents"""
        return self.active_agents.copy()
    
    async def get_agents_by_capability(self, capability: Union[str, AgentCapability]) -> List[BaseAgent]:
        """Get all agents that have a specific capability"""
        if isinstance(capability, str):
            try:
                capability = AgentCapability(capability)
            except ValueError:
                return []
        
        matching_agents = []
        for agent in self.active_agents.values():
            if agent.has_capability(capability):
                matching_agents.append(agent)
        
        return matching_agents
    
    async def get_agents_by_status(self, status: AgentStatus) -> List[BaseAgent]:
        """Get all agents with a specific status"""
        return [agent for agent in self.active_agents.values() if agent.status == status]
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        return {
            "total_created": self.agent_stats["created"],
            "total_destroyed": self.agent_stats["destroyed"],
            "currently_active": self.agent_stats["active"],
            "available_types": len(self.agent_stats["types"]),
            "registered_types": list(self.agent_stats["types"]),
            "active_agent_ids": list(self.active_agents.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_report = {
            "factory_status": "healthy",
            "total_agents": len(self.active_agents),
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "agent_details": {}
        }
        
        for agent_id, agent in self.active_agents.items():
            agent_info = await agent.get_agent_info()
            is_healthy = agent.status not in [AgentStatus.ERROR, AgentStatus.OFFLINE]
            
            health_report["agent_details"][agent_id] = {
                "status": agent.status.value,
                "healthy": is_healthy,
                "uptime": agent_info["uptime"],
                "error_count": agent_info["error_count"],
                "active_tasks": agent_info["active_tasks"]
            }
            
            if is_healthy:
                health_report["healthy_agents"] += 1
            else:
                health_report["unhealthy_agents"] += 1
        
        return health_report
    
    async def save_configuration(self, filepath: str):
        """Save current factory configuration to file"""
        config = {
            "templates": {},
            "stats": self.get_factory_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for agent_type, template in self.templates.items():
            config["templates"][agent_type] = {
                "agent_type": template.agent_type,
                "class_path": template.class_path,
                "default_config": template.default_config
            }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Factory configuration saved to {filepath}")
    
    async def load_configuration(self, filepath: str):
        """Load factory configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            # Load templates
            for agent_type, template_data in config.get("templates", {}).items():
                template = AgentTemplate(
                    agent_type=template_data["agent_type"],
                    class_path=template_data["class_path"],
                    default_config=template_data["default_config"]
                )
                self.register_template(template)
            
            self.logger.info(f"Factory configuration loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {filepath}: {e}")
            raise


# Global factory instance
_factory_instance: Optional[UniversalAgentFactory] = None


def get_factory() -> UniversalAgentFactory:
    """Get the global factory instance"""
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = UniversalAgentFactory()
    return _factory_instance


def set_factory(factory: UniversalAgentFactory):
    """Set the global factory instance"""
    global _factory_instance
    _factory_instance = factory


# Convenience functions
async def create_agent(agent_id: str, agent_type: str, 
                      config_overrides: Dict[str, Any] = None) -> BaseAgent:
    """Create agent using global factory"""
    factory = get_factory()
    return await factory.create_agent(agent_id, agent_type, config_overrides)


async def create_agent_by_capabilities(agent_id: str, 
                                     capabilities: List[Union[str, AgentCapability]],
                                     preferred_type: str = None,
                                     config_overrides: Dict[str, Any] = None) -> BaseAgent:
    """Create agent by capabilities using global factory"""
    factory = get_factory()
    return await factory.create_agent_by_capabilities(
        agent_id, capabilities, preferred_type, config_overrides
    )


async def get_agent(agent_id: str) -> Optional[BaseAgent]:
    """Get agent using global factory"""
    factory = get_factory()
    return factory.get_agent(agent_id)


async def destroy_agent(agent_id: str):
    """Destroy agent using global factory"""
    factory = get_factory()
    await factory.destroy_agent(agent_id)