"""
SutazAI Universal Agent System - Core Components
===============================================

This module provides the core infrastructure for the SutazAI universal agent system.
It includes all the fundamental components needed to create, manage, and coordinate
AI agents that operate independently using local Ollama models and Redis messaging.

Components:
- BaseAgent: Foundation class for all agents
- UniversalAgentFactory: Dynamic agent creation and management
- AgentMessageBus: Advanced inter-agent communication
- OrchestrationController: Multi-agent workflow coordination
- AgentRegistry: Centralized agent discovery and management

Usage:
    from backend.ai_agents.core import (
        BaseAgent, UniversalAgentFactory, AgentMessageBus,
        OrchestrationController, AgentRegistry
    )
    
    # Initialize the system
    system = UniversalAgentSystem()
    await system.initialize()
    
    # Create agents
    agent = await system.create_agent("my-agent", "code_generator")
    
    # Execute workflows
    workflow_id = await system.execute_workflow({
        "name": "Build Web App",
        "tasks": [...]
    })
"""

from .base_agent import (
    BaseAgent, 
    AgentMessage, 
    AgentStatus, 
    AgentCapability, 
    AgentConfig
)

from .universal_agent_factory import (
    UniversalAgentFactory,
    AgentTemplate,
    get_factory,
    set_factory,
    create_agent,
    create_agent_by_capabilities,
    get_agent,
    destroy_agent
)

from .agent_message_bus import (
    AgentMessageBus,
    MessagePriority,
    MessageType,
    RoutingStrategy,
    MessageRoute,
    MessageStats,
    get_message_bus,
    set_message_bus,
    send_message,
    broadcast_message
)

from .orchestration_controller import (
    OrchestrationController,
    Workflow,
    Task,
    WorkflowStatus,
    TaskStatus,
    TaskPriority,
    get_orchestration_controller,
    set_orchestration_controller
)

from .agent_registry import (
    AgentRegistry,
    AgentRegistration,
    AgentMetrics,
    AgentHealth,
    AgentSelector,
    get_agent_registry,
    set_agent_registry
)

__all__ = [
    # Base components
    "BaseAgent",
    "AgentMessage", 
    "AgentStatus",
    "AgentCapability",
    "AgentConfig",
    
    # Factory
    "UniversalAgentFactory",
    "AgentTemplate",
    "get_factory",
    "set_factory",
    "create_agent",
    "create_agent_by_capabilities",
    "get_agent",
    "destroy_agent",
    
    # Message Bus
    "AgentMessageBus",
    "MessagePriority",
    "MessageType",
    "RoutingStrategy",
    "MessageRoute",
    "MessageStats",
    "get_message_bus",
    "set_message_bus",
    "send_message",
    "broadcast_message",
    
    # Orchestration
    "OrchestrationController",
    "Workflow",
    "Task",
    "WorkflowStatus",
    "TaskStatus",
    "TaskPriority",
    "get_orchestration_controller",
    "set_orchestration_controller",
    
    # Registry
    "AgentRegistry",
    "AgentRegistration",
    "AgentMetrics",
    "AgentHealth",
    "AgentSelector",
    "get_agent_registry",
    "set_agent_registry",
    
    # System
    "UniversalAgentSystem"
]


class UniversalAgentSystem:
    """
    Complete Universal Agent System
    
    This class provides a unified interface to initialize and manage
    the entire universal agent system infrastructure.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 ollama_url: str = "http://localhost:11434",
                 namespace: str = "sutazai"):
        
        self.redis_url = redis_url
        self.ollama_url = ollama_url
        self.namespace = namespace
        
        self.factory: UniversalAgentFactory = None
        self.message_bus: AgentMessageBus = None
        self.orchestrator: OrchestrationController = None
        self.registry: AgentRegistry = None
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the complete agent system"""
        try:
            # Initialize registry
            self.registry = AgentRegistry(self.redis_url, self.namespace)
            success = await self.registry.initialize()
            if not success:
                raise RuntimeError("Failed to initialize agent registry")
            set_agent_registry(self.registry)
            
            # Initialize message bus
            self.message_bus = AgentMessageBus(self.redis_url, self.namespace)
            success = await self.message_bus.initialize()
            if not success:
                raise RuntimeError("Failed to initialize message bus")
            set_message_bus(self.message_bus)
            
            # Initialize factory
            self.factory = UniversalAgentFactory()
            set_factory(self.factory)
            
            # Initialize orchestration controller
            self.orchestrator = OrchestrationController(
                self.factory, 
                self.message_bus
            )
            success = await self.orchestrator.initialize()
            if not success:
                raise RuntimeError("Failed to initialize orchestration controller")
            set_orchestration_controller(self.orchestrator)
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize Universal Agent System: {e}")
            return False
    
    async def create_agent(self, agent_id: str, agent_type: str, 
                          config_overrides: dict = None) -> BaseAgent:
        """Create a new agent"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        return await self.factory.create_agent(agent_id, agent_type, config_overrides)
    
    async def create_workflow(self, workflow_spec: dict) -> str:
        """Create and execute a workflow"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        return await self.orchestrator.create_workflow(workflow_spec)
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        return await self.orchestrator.start_workflow(workflow_id)
    
    def get_system_status(self) -> dict:
        """Get overall system status"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "running",
            "registry_stats": self.registry.get_registry_stats(),
            "message_bus_stats": self.message_bus.get_stats().__dict__,
            "factory_stats": self.factory.get_factory_stats(),
            "orchestrator_stats": self.orchestrator.get_execution_stats()
        }
    
    async def shutdown(self):
        """Shutdown the entire system"""
        if not self.initialized:
            return
        
        try:
            # Shutdown orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Shutdown factory (destroys all agents)
            if self.factory:
                await self.factory.destroy_all_agents()
            
            # Shutdown message bus
            if self.message_bus:
                await self.message_bus.shutdown()
            
            # Shutdown registry
            if self.registry:
                await self.registry.shutdown()
            
            self.initialized = False
            
        except Exception as e:
            print(f"Error during system shutdown: {e}")


# Global system instance
_system_instance: UniversalAgentSystem = None


def get_system() -> UniversalAgentSystem:
    """Get the global system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = UniversalAgentSystem()
    return _system_instance


def set_system(system: UniversalAgentSystem):
    """Set the global system instance"""
    global _system_instance
    _system_instance = system