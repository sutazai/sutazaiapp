#!/usr/bin/env python3
"""
SutazAI Agent Orchestration Framework
Advanced multi-agent system with coordination, communication, and collaboration
"""

from .orchestrator import AgentOrchestrator, OrchestratorConfig
from .agent_manager import EnhancedAgentManager, AgentManagerConfig
from .agent_coordinator import AgentCoordinator, CoordinationConfig
from .communication_system import CommunicationSystem, CommunicationConfig
from .task_scheduler import TaskScheduler, SchedulerConfig
from .collaboration_engine import CollaborationEngine, CollaborationConfig
from .resource_manager import ResourceManager, ResourceConfig
from .workflow_engine import WorkflowEngine, WorkflowConfig
from .agent_registry import AgentRegistry, RegistryConfig
from .performance_monitor import PerformanceMonitor, MonitorConfig

__version__ = "1.0.0"
__all__ = [
    "AgentOrchestrator",
    "OrchestratorConfig",
    "EnhancedAgentManager",
    "AgentManagerConfig",
    "AgentCoordinator",
    "CoordinationConfig",
    "CommunicationSystem",
    "CommunicationConfig",
    "TaskScheduler",
    "SchedulerConfig",
    "CollaborationEngine",
    "CollaborationConfig",
    "ResourceManager",
    "ResourceConfig",
    "WorkflowEngine",
    "WorkflowConfig",
    "AgentRegistry",
    "RegistryConfig",
    "PerformanceMonitor",
    "MonitorConfig"
]

def create_agent_orchestrator(config: dict = None) -> AgentOrchestrator:
    """Factory function to create agent orchestrator"""
    return AgentOrchestrator(config=config)

def create_agent_manager(config: dict = None) -> EnhancedAgentManager:
    """Factory function to create enhanced agent manager"""
    return EnhancedAgentManager(config=config)

def create_communication_system(config: dict = None) -> CommunicationSystem:
    """Factory function to create communication system"""
    return CommunicationSystem(config=config)