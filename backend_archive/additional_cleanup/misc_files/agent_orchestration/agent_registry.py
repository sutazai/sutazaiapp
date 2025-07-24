#!/usr/bin/env python3
"""
Agent Registry - Adapted from Model Registry
Central repository for managing and discovering AI agents
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Agent type enumeration"""
    GENERAL = "general"
    SPECIALIZED = "specialized"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SUPERVISOR = "supervisor"

class AgentStatus(Enum):
    """Agent status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class RegistryConfig:
    """Configuration for agent registry"""
    max_agents: int = 100
    cleanup_interval: float = 300.0  # 5 minutes
    heartbeat_timeout: float = 30.0  # 30 seconds

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    name: str
    agent_type: AgentType
    description: str
    capabilities: List[str]
    status: AgentStatus
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    last_heartbeat: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}

class AgentRegistry:
    """
    Central agent registry for managing and discovering AI agents
    Adapted from ModelRegistry for agent management
    """
    
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.agents: Dict[str, AgentRegistration] = {}
        self.agent_heartbeats: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info("Agent registry initialized")
    
    async def initialize(self) -> bool:
        """Initialize the agent registry"""
        try:
            logger.info("Initializing agent registry...")
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Agent registry initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Agent registry initialization failed: {e}")
            return False
    
    async def start(self):
        """Start the agent registry"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Agent registry started")
    
    async def stop(self):
        """Stop the agent registry"""
        self._shutdown = True
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent registry stopped")
    
    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent"""
        try:
            with self._lock:
                if len(self.agents) >= self.config.max_agents:
                    logger.warning(f"Maximum agents reached: {self.config.max_agents}")
                    return False
                
                self.agents[registration.agent_id] = registration
                self.agent_heartbeats[registration.agent_id] = datetime.now(timezone.utc)
                
                logger.info(f"Agent registered: {registration.agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering agent {registration.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            with self._lock:
                if agent_id in self.agents:
                    del self.agents[agent_id]
                    if agent_id in self.agent_heartbeats:
                        del self.agent_heartbeats[agent_id]
                    logger.info(f"Agent unregistered: {agent_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False
    
    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID"""
        return self.agents.get(agent_id)
    
    async def list_agents(self, 
                         agent_type: Optional[AgentType] = None,
                         status: Optional[AgentStatus] = None) -> List[AgentRegistration]:
        """List agents with optional filtering"""
        agents = list(self.agents.values())
        
        if agent_type:
            agents = [agent for agent in agents if agent.agent_type == agent_type]
        
        if status:
            agents = [agent for agent in agents if agent.status == status]
        
        return agents
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status"""
        try:
            with self._lock:
                if agent_id in self.agents:
                    self.agents[agent_id].status = status
                    return True
                return False
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return False
    
    async def heartbeat(self, agent_id: str) -> bool:
        """Record agent heartbeat"""
        try:
            with self._lock:
                if agent_id in self.agents:
                    self.agent_heartbeats[agent_id] = datetime.now(timezone.utc)
                    self.agents[agent_id].last_heartbeat = datetime.now(timezone.utc)
                    return True
                return False
        except Exception as e:
            logger.error(f"Error recording heartbeat for {agent_id}: {e}")
            return False
    
    async def find_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Find agents with specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities and agent.status == AgentStatus.ACTIVE
        ]
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get agent registry statistics"""
        try:
            total_agents = len(self.agents)
            active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
            
            agents_by_type = {}
            agents_by_status = {}
            
            for agent in self.agents.values():
                # Count by type
                type_key = agent.agent_type.value
                agents_by_type[type_key] = agents_by_type.get(type_key, 0) + 1
                
                # Count by status
                status_key = agent.status.value
                agents_by_status[status_key] = agents_by_status.get(status_key, 0) + 1
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'agents_by_type': agents_by_type,
                'agents_by_status': agents_by_status,
                'max_agents': self.config.max_agents
            }
            
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check agent registry health"""
        try:
            # Check if any agents are responsive
            now = datetime.now(timezone.utc)
            active_agents = 0
            
            for agent_id, last_heartbeat in self.agent_heartbeats.items():
                time_since_heartbeat = (now - last_heartbeat).total_seconds()
                if time_since_heartbeat < self.config.heartbeat_timeout:
                    active_agents += 1
            
            # Registry is healthy if we have at least one responsive agent
            # or if we're just starting up (no agents yet)
            return active_agents > 0 or len(self.agents) == 0
            
        except Exception as e:
            logger.error(f"Agent registry health check failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get registry status"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
            "registry_healthy": self.health_check(),
            "max_capacity": self.config.max_agents
        }
    
    async def _cleanup_loop(self):
        """Cleanup loop to remove stale agents"""
        while not self._shutdown:
            try:
                await self._cleanup_stale_agents()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_stale_agents(self):
        """Remove agents that haven't sent heartbeat"""
        try:
            now = datetime.now(timezone.utc)
            stale_agents = []
            
            with self._lock:
                for agent_id, last_heartbeat in self.agent_heartbeats.items():
                    time_since_heartbeat = (now - last_heartbeat).total_seconds()
                    if time_since_heartbeat > self.config.heartbeat_timeout:
                        stale_agents.append(agent_id)
                
                # Remove stale agents
                for agent_id in stale_agents:
                    if agent_id in self.agents:
                        logger.warning(f"Removing stale agent: {agent_id}")
                        del self.agents[agent_id]
                    if agent_id in self.agent_heartbeats:
                        del self.agent_heartbeats[agent_id]
                        
        except Exception as e:
            logger.error(f"Error cleaning up stale agents: {e}")