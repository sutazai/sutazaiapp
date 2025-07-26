"""
Agent Registry for SutazAI
Manages registration and discovery of available agents
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration"""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    name: str
    description: str
    capabilities: List[str]
    status: AgentStatus = AgentStatus.OFFLINE
    endpoint: Optional[str] = None
    max_concurrent_tasks: int = 1
    current_tasks: int = 0
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentRegistry:
    """
    Registry for managing AI agents in the SutazAI system
    """
    
    def __init__(self):
        self.logger = logging.getLogger("sutazai.AgentRegistry")
        self._agents: Dict[str, AgentInfo] = {}
        self._capabilities_index: Dict[str, Set[str]] = {}
        self.logger.info("AgentRegistry initialized")
    
    def register_agent(
        self,
        name: str,
        description: str,
        capabilities: List[str],
        endpoint: Optional[str] = None,
        max_concurrent_tasks: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a new agent
        
        Args:
            name: Unique agent name
            description: Agent description
            capabilities: List of agent capabilities
            endpoint: Agent endpoint URL
            max_concurrent_tasks: Maximum concurrent tasks
            metadata: Additional metadata
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if name in self._agents:
                self.logger.warning(f"Agent {name} already registered, updating...")
            
            agent_info = AgentInfo(
                name=name,
                description=description,
                capabilities=capabilities,
                endpoint=endpoint,
                max_concurrent_tasks=max_concurrent_tasks,
                status=AgentStatus.AVAILABLE,
                last_seen=datetime.now(),
                metadata=metadata or {}
            )
            
            self._agents[name] = agent_info
            
            # Update capabilities index
            for capability in capabilities:
                if capability not in self._capabilities_index:
                    self._capabilities_index[capability] = set()
                self._capabilities_index[capability].add(name)
            
            self.logger.info(f"Agent {name} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {name}: {e}")
            return False
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent
        
        Args:
            name: Agent name to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if name not in self._agents:
                self.logger.warning(f"Agent {name} not found for unregistration")
                return False
            
            agent_info = self._agents[name]
            
            # Remove from capabilities index
            for capability in agent_info.capabilities:
                if capability in self._capabilities_index:
                    self._capabilities_index[capability].discard(name)
                    if not self._capabilities_index[capability]:
                        del self._capabilities_index[capability]
            
            del self._agents[name]
            self.logger.info(f"Agent {name} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {name}: {e}")
            return False
    
    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """
        Get agent information by name
        
        Args:
            name: Agent name
            
        Returns:
            AgentInfo if found, None otherwise
        """
        return self._agents.get(name)
    
    def list_agents(self, status: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """
        List all registered agents
        
        Args:
            status: Filter by agent status
            
        Returns:
            List of agent information
        """
        agents = list(self._agents.values())
        if status:
            agents = [agent for agent in agents if agent.status == status]
        return agents
    
    def find_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """
        Find agents that have a specific capability
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agents with the capability
        """
        if capability not in self._capabilities_index:
            return []
        
        agent_names = self._capabilities_index[capability]
        return [self._agents[name] for name in agent_names if name in self._agents]
    
    def get_available_agents(self) -> List[AgentInfo]:
        """
        Get all available agents (not busy)
        
        Returns:
            List of available agents
        """
        return [
            agent for agent in self._agents.values()
            if agent.status == AgentStatus.AVAILABLE and agent.current_tasks < agent.max_concurrent_tasks
        ]
    
    def update_agent_status(self, name: str, status: AgentStatus) -> bool:
        """
        Update agent status
        
        Args:
            name: Agent name
            status: New status
            
        Returns:
            True if update successful, False otherwise
        """
        if name not in self._agents:
            self.logger.warning(f"Agent {name} not found for status update")
            return False
        
        self._agents[name].status = status
        self._agents[name].last_seen = datetime.now()
        self.logger.debug(f"Agent {name} status updated to {status}")
        return True
    
    def increment_task_count(self, name: str) -> bool:
        """
        Increment agent's current task count
        
        Args:
            name: Agent name
            
        Returns:
            True if increment successful, False otherwise
        """
        if name not in self._agents:
            return False
        
        agent = self._agents[name]
        if agent.current_tasks >= agent.max_concurrent_tasks:
            return False
        
        agent.current_tasks += 1
        if agent.current_tasks >= agent.max_concurrent_tasks:
            agent.status = AgentStatus.BUSY
        
        return True
    
    def decrement_task_count(self, name: str) -> bool:
        """
        Decrement agent's current task count
        
        Args:
            name: Agent name
            
        Returns:
            True if decrement successful, False otherwise
        """
        if name not in self._agents:
            return False
        
        agent = self._agents[name]
        if agent.current_tasks > 0:
            agent.current_tasks -= 1
            if agent.current_tasks < agent.max_concurrent_tasks and agent.status == AgentStatus.BUSY:
                agent.status = AgentStatus.AVAILABLE
        
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Dictionary with registry statistics
        """
        total_agents = len(self._agents)
        status_counts = {}
        
        for status in AgentStatus:
            status_counts[status.value] = len([
                agent for agent in self._agents.values()
                if agent.status == status
            ])
        
        return {
            "total_agents": total_agents,
            "status_counts": status_counts,
            "total_capabilities": len(self._capabilities_index),
            "capabilities": list(self._capabilities_index.keys())
        } 