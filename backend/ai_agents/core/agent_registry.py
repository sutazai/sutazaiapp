"""
Agent Registry - Centralized Agent Management and Discovery
==========================================================

This registry provides centralized management and discovery of all agents
in the SutazAI system. It tracks agent metadata, capabilities, health status,
and provides sophisticated agent selection and load balancing capabilities.

Features:
- Centralized agent registration and discovery
- Health monitoring and status tracking
- Capability-based agent selection
- Load balancing and resource management
- Agent lifecycle management
- Performance metrics and analytics
- Dynamic agent scaling
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from app.orchestration.event_utils import register_event_handler as _reg_handler
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

import aioredis
from pydantic import BaseModel

from agents.core.base_agent import BaseAgent, AgentStatus, AgentCapability


class RegistryStatus(Enum):
    """Registry operational status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class AgentHealth(Enum):
    """Agent health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNRESPONSIVE = "unresponsive"
    UNKNOWN = "unknown"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_task_duration: float = 0.0
    peak_memory_usage: float = 0.0
    current_memory_usage: float = 0.0
    cpu_utilization: float = 0.0
    uptime_seconds: float = 0.0
    last_activity: Optional[datetime] = None
    error_rate: float = 0.0
    throughput_tasks_per_minute: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate task success rate"""
        total = self.total_tasks_completed + self.total_tasks_failed
        if total == 0:
            return 0.0
        return (self.total_tasks_completed / total) * 100.0


@dataclass
class AgentRegistration:
    """Complete agent registration information"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    capabilities: Set[AgentCapability]
    status: AgentStatus
    health: AgentHealth
    host_info: Dict[str, Any]
    model_config: Dict[str, Any]
    max_concurrent_tasks: int
    current_task_count: int
    metrics: AgentMetrics
    registered_at: datetime
    last_heartbeat: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "description": self.description,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "health": self.health.value,
            "host_info": self.host_info,
            "model_config": self.model_config,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_task_count": self.current_task_count,
            "metrics": {
                "total_tasks_completed": self.metrics.total_tasks_completed,
                "total_tasks_failed": self.metrics.total_tasks_failed,
                "average_task_duration": self.metrics.average_task_duration,
                "peak_memory_usage": self.metrics.peak_memory_usage,
                "current_memory_usage": self.metrics.current_memory_usage,
                "cpu_utilization": self.metrics.cpu_utilization,
                "uptime_seconds": self.metrics.uptime_seconds,
                "last_activity": self.metrics.last_activity.isoformat() if self.metrics.last_activity else None,
                "error_rate": self.metrics.error_rate,
                "throughput_tasks_per_minute": self.metrics.throughput_tasks_per_minute,
                "success_rate": self.metrics.success_rate()
            },
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata,
            "tags": list(self.tags)
        }


class AgentSelector:
    """Advanced agent selection strategies"""
    
    @staticmethod
    def by_capability(agents: List[AgentRegistration], 
                     required_capabilities: List[AgentCapability]) -> List[AgentRegistration]:
        """Select agents by required capabilities"""
        return [agent for agent in agents 
                if all(cap in agent.capabilities for cap in required_capabilities)]
    
    @staticmethod
    def by_load(agents: List[AgentRegistration], 
                max_load_percentage: float = 80.0) -> List[AgentRegistration]:
        """Select agents below load threshold"""
        return [agent for agent in agents 
                if (agent.current_task_count / agent.max_concurrent_tasks * 100) <= max_load_percentage]
    
    @staticmethod
    def by_health(agents: List[AgentRegistration], 
                  min_health: AgentHealth = AgentHealth.WARNING) -> List[AgentRegistration]:
        """Select agents above minimum health threshold"""
        health_order = {
            AgentHealth.HEALTHY: 4,
            AgentHealth.WARNING: 3,
            AgentHealth.CRITICAL: 2,
            AgentHealth.UNRESPONSIVE: 1,
            AgentHealth.UNKNOWN: 0
        }
        min_level = health_order.get(min_health, 0)
        return [agent for agent in agents 
                if health_order.get(agent.health, 0) >= min_level]
    
    @staticmethod
    def by_performance(agents: List[AgentRegistration], 
                      min_success_rate: float = 80.0) -> List[AgentRegistration]:
        """Select agents above minimum performance threshold"""
        return [agent for agent in agents 
                if agent.metrics.success_rate() >= min_success_rate]
    
    @staticmethod
    def best_fit(agents: List[AgentRegistration], 
                 required_capabilities: List[AgentCapability],
                 priority_weights: Dict[str, float] = None) -> Optional[AgentRegistration]:
        """Select best agent using weighted scoring"""
        if not agents:
            return None
        
        weights = priority_weights or {
            "load": 0.3,
            "performance": 0.25,
            "health": 0.2,
            "capability_match": 0.15,
            "uptime": 0.1
        }
        
        scored_agents = []
        
        for agent in agents:
            score = 0.0
            
            # Load score (lower load is better)
            load_percentage = agent.current_task_count / agent.max_concurrent_tasks * 100
            load_score = max(0, 100 - load_percentage) / 100
            score += load_score * weights.get("load", 0)
            
            # Performance score
            performance_score = agent.metrics.success_rate() / 100
            score += performance_score * weights.get("performance", 0)
            
            # Health score
            health_scores = {
                AgentHealth.HEALTHY: 1.0,
                AgentHealth.WARNING: 0.7,
                AgentHealth.CRITICAL: 0.4,
                AgentHealth.UNRESPONSIVE: 0.1,
                AgentHealth.UNKNOWN: 0.0
            }
            health_score = health_scores.get(agent.health, 0)
            score += health_score * weights.get("health", 0)
            
            # Capability match score
            if required_capabilities:
                matched_caps = sum(1 for cap in required_capabilities 
                                 if cap in agent.capabilities)
                capability_score = matched_caps / len(required_capabilities)
            else:
                capability_score = 1.0
            score += capability_score * weights.get("capability_match", 0)
            
            # Uptime score
            uptime_hours = agent.metrics.uptime_seconds / 3600
            uptime_score = min(1.0, uptime_hours / 24)  # Max score after 24 hours
            score += uptime_score * weights.get("uptime", 0)
            
            scored_agents.append((agent, score))
        
        # Return agent with highest score
        return max(scored_agents, key=lambda x: x[1])[0]


class AgentRegistry:
    """
    Centralized Agent Registry and Discovery Service
    
    Manages all agents in the SutazAI system, providing registration,
    discovery, health monitoring, and intelligent agent selection.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 namespace: str = "sutazai"):
        
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        
        # Registry state
        self.status = RegistryStatus.INITIALIZING
        self.agents: Dict[str, AgentRegistration] = {}
        self.agent_types: Dict[str, Set[str]] = defaultdict(set)  # type -> agent_ids
        self.capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.heartbeat_timeout = 120  # seconds
        self.unhealthy_agents: Set[str] = set()
        
        # Performance tracking
        self.registry_stats = {
            "total_registrations": 0,
            "active_agents": 0,
            "unhealthy_agents": 0,
            "agent_types": 0,
            "total_capabilities": 0,
            "health_checks_performed": 0,
            "failed_health_checks": 0
        }
        
        # Event callbacks
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("agent_registry")
    
    async def initialize(self) -> bool:
        """Initialize the agent registry"""
        try:
            self.logger.info("Initializing Agent Registry")
            
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            
            # Load existing registrations from Redis
            await self._load_registrations()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.status = RegistryStatus.ACTIVE
            self.logger.info("Agent Registry initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent registry: {e}")
            self.status = RegistryStatus.ERROR
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        tasks = [
            self._health_monitor(),
            self._stats_collector(),
            self._cleanup_expired_agents(),
            self._performance_analyzer()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _load_registrations(self):
        """Load existing agent registrations from Redis"""
        try:
            # Load agent registrations
            agent_keys = await self.redis.keys(f"{self.namespace}:agent:*")
            
            for key in agent_keys:
                agent_data = await self.redis.get(key)
                if agent_data:
                    try:
                        agent_dict = json.loads(agent_data)
                        registration = self._dict_to_registration(agent_dict)
                        
                        # Check if agent is still alive
                        time_since_heartbeat = (datetime.utcnow() - registration.last_heartbeat).total_seconds()
                        
                        if time_since_heartbeat > self.heartbeat_timeout:
                            registration.health = AgentHealth.UNRESPONSIVE
                            registration.status = AgentStatus.OFFLINE
                        
                        self._add_to_indexes(registration)
                        self.agents[registration.agent_id] = registration
                        
                    except Exception as e:
                        self.logger.error(f"Error loading agent registration from {key}: {e}")
            
            self.logger.info(f"Loaded {len(self.agents)} agent registrations")
            
        except Exception as e:
            self.logger.error(f"Error loading registrations: {e}")
    
    def _dict_to_registration(self, agent_dict: Dict[str, Any]) -> AgentRegistration:
        """Convert dictionary to AgentRegistration"""
        metrics_data = agent_dict.get("metrics", {})
        metrics = AgentMetrics(
            total_tasks_completed=metrics_data.get("total_tasks_completed", 0),
            total_tasks_failed=metrics_data.get("total_tasks_failed", 0),
            average_task_duration=metrics_data.get("average_task_duration", 0.0),
            peak_memory_usage=metrics_data.get("peak_memory_usage", 0.0),
            current_memory_usage=metrics_data.get("current_memory_usage", 0.0),
            cpu_utilization=metrics_data.get("cpu_utilization", 0.0),
            uptime_seconds=metrics_data.get("uptime_seconds", 0.0),
            last_activity=datetime.fromisoformat(metrics_data["last_activity"]) 
                         if metrics_data.get("last_activity") else None,
            error_rate=metrics_data.get("error_rate", 0.0),
            throughput_tasks_per_minute=metrics_data.get("throughput_tasks_per_minute", 0.0)
        )
        
        return AgentRegistration(
            agent_id=agent_dict["agent_id"],
            agent_type=agent_dict["agent_type"],
            name=agent_dict["name"],
            description=agent_dict["description"],
            capabilities=set(AgentCapability(cap) for cap in agent_dict["capabilities"]),
            status=AgentStatus(agent_dict["status"]),
            health=AgentHealth(agent_dict["health"]),
            host_info=agent_dict["host_info"],
            model_config=agent_dict["model_config"],
            max_concurrent_tasks=agent_dict["max_concurrent_tasks"],
            current_task_count=agent_dict["current_task_count"],
            metrics=metrics,
            registered_at=datetime.fromisoformat(agent_dict["registered_at"]),
            last_heartbeat=datetime.fromisoformat(agent_dict["last_heartbeat"]),
            metadata=agent_dict.get("metadata", {}),
            tags=set(agent_dict.get("tags", []))
        )
    
    async def register_agent(self, agent_info: Dict[str, Any]) -> bool:
        """Register a new agent"""
        try:
            agent_id = agent_info["agent_id"]
            
            # Create registration
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_info["agent_type"],
                name=agent_info["name"],
                description=agent_info.get("description", ""),
                capabilities=set(AgentCapability(cap) for cap in agent_info.get("capabilities", [])),
                status=AgentStatus(agent_info.get("status", "idle")),
                health=AgentHealth.HEALTHY,
                host_info=agent_info.get("host_info", {}),
                model_config=agent_info.get("model_config", {}),
                max_concurrent_tasks=agent_info.get("max_concurrent_tasks", 5),
                current_task_count=0,
                metrics=AgentMetrics(),
                registered_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                metadata=agent_info.get("metadata", {}),
                tags=set(agent_info.get("tags", []))
            )
            
            # Store in Redis
            await self._store_registration(registration)
            
            # Update local indexes
            self._add_to_indexes(registration)
            self.agents[agent_id] = registration
            
            # Update stats
            self.registry_stats["total_registrations"] += 1
            self._update_stats()
            
            # Trigger event
            await self._trigger_event("agent_registered", {"agent_id": agent_id})
            
            self.logger.info(f"Registered agent {agent_id} of type {registration.agent_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
            return False
    
    async def _store_registration(self, registration: AgentRegistration):
        """Store agent registration in Redis"""
        key = f"{self.namespace}:agent:{registration.agent_id}"
        await self.redis.set(key, json.dumps(registration.to_dict()))
    
    def _add_to_indexes(self, registration: AgentRegistration):
        """Add agent to search indexes"""
        agent_id = registration.agent_id
        
        # Type index
        self.agent_types[registration.agent_type].add(agent_id)
        
        # Capability index
        for capability in registration.capabilities:
            self.capability_index[capability].add(agent_id)
        
        # Tag index
        for tag in registration.tags:
            self.tag_index[tag].add(agent_id)
    
    def _remove_from_indexes(self, registration: AgentRegistration):
        """Remove agent from search indexes"""
        agent_id = registration.agent_id
        
        # Type index
        self.agent_types[registration.agent_type].discard(agent_id)
        
        # Capability index
        for capability in registration.capabilities:
            self.capability_index[capability].discard(agent_id)
        
        # Tag index
        for tag in registration.tags:
            self.tag_index[tag].discard(agent_id)
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
            
            registration = self.agents[agent_id]
            
            # Remove from Redis
            key = f"{self.namespace}:agent:{agent_id}"
            await self.redis.delete(key)
            
            # Remove from indexes
            self._remove_from_indexes(registration)
            
            # Remove from local storage
            del self.agents[agent_id]
            self.unhealthy_agents.discard(agent_id)
            
            # Update stats
            self._update_stats()
            
            # Trigger event
            await self._trigger_event("agent_unregistered", {"agent_id": agent_id})
            
            self.logger.info(f"Unregistered agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def update_agent_heartbeat(self, agent_id: str, status_info: Dict[str, Any]):
        """Update agent heartbeat and status"""
        if agent_id not in self.agents:
            self.logger.warning(f"Heartbeat from unknown agent {agent_id}")
            return
        
        registration = self.agents[agent_id]
        registration.last_heartbeat = datetime.utcnow()
        
        # Update status information
        if "status" in status_info:
            try:
                registration.status = AgentStatus(status_info["status"])
            except ValueError:
                pass
        
        if "current_task_count" in status_info:
            registration.current_task_count = status_info["current_task_count"]
        
        # Update metrics
        metrics = registration.metrics
        if "task_count" in status_info:
            metrics.total_tasks_completed = status_info["task_count"]
        
        if "error_count" in status_info:
            total_tasks = metrics.total_tasks_completed + metrics.total_tasks_failed
            if total_tasks > 0:
                metrics.error_rate = (status_info["error_count"] / total_tasks) * 100
        
        if "uptime" in status_info:
            metrics.uptime_seconds = status_info["uptime"]
        
        if "memory_usage" in status_info:
            metrics.current_memory_usage = status_info["memory_usage"]
            metrics.peak_memory_usage = max(metrics.peak_memory_usage, 
                                          status_info["memory_usage"])
        
        if "cpu_utilization" in status_info:
            metrics.cpu_utilization = status_info["cpu_utilization"]
        
        metrics.last_activity = datetime.utcnow()
        
        # Update health based on status
        registration.health = self._calculate_agent_health(registration)
        
        # Store updated registration
        await self._store_registration(registration)
        
        # Remove from unhealthy set if healthy now
        if registration.health == AgentHealth.HEALTHY:
            self.unhealthy_agents.discard(agent_id)
    
    def _calculate_agent_health(self, registration: AgentRegistration) -> AgentHealth:
        """Calculate agent health based on metrics"""
        metrics = registration.metrics
        
        # Check if agent is responsive
        time_since_heartbeat = (datetime.utcnow() - registration.last_heartbeat).total_seconds()
        if time_since_heartbeat > self.heartbeat_timeout:
            return AgentHealth.UNRESPONSIVE
        
        # Check error rate
        if metrics.error_rate > 50:
            return AgentHealth.CRITICAL
        elif metrics.error_rate > 20:
            return AgentHealth.WARNING
        
        # Check resource usage
        if metrics.cpu_utilization > 90 or metrics.current_memory_usage > 90:
            return AgentHealth.CRITICAL
        elif metrics.cpu_utilization > 75 or metrics.current_memory_usage > 75:
            return AgentHealth.WARNING
        
        # Check task load
        load_percentage = (registration.current_task_count / 
                          registration.max_concurrent_tasks) * 100
        if load_percentage > 95:
            return AgentHealth.WARNING
        
        return AgentHealth.HEALTHY
    
    # Agent discovery and selection methods
    def find_agents_by_type(self, agent_type: str) -> List[AgentRegistration]:
        """Find agents by type"""
        agent_ids = self.agent_types.get(agent_type, set())
        return [self.agents[agent_id] for agent_id in agent_ids 
                if agent_id in self.agents]
    
    def find_agents_by_capability(self, capabilities: List[AgentCapability]) -> List[AgentRegistration]:
        """Find agents with specific capabilities"""
        if not capabilities:
            return list(self.agents.values())
        
        # Start with agents that have the first capability
        agent_ids = self.capability_index.get(capabilities[0], set())
        
        # Intersect with agents that have remaining capabilities
        for capability in capabilities[1:]:
            agent_ids = agent_ids.intersection(self.capability_index.get(capability, set()))
        
        return [self.agents[agent_id] for agent_id in agent_ids 
                if agent_id in self.agents]
    
    def find_agents_by_tags(self, tags: List[str]) -> List[AgentRegistration]:
        """Find agents with specific tags"""
        if not tags:
            return list(self.agents.values())
        
        # Start with agents that have the first tag
        agent_ids = self.tag_index.get(tags[0], set())
        
        # Intersect with agents that have remaining tags
        for tag in tags[1:]:
            agent_ids = agent_ids.intersection(self.tag_index.get(tag, set()))
        
        return [self.agents[agent_id] for agent_id in agent_ids 
                if agent_id in self.agents]
    
    def find_available_agents(self, max_load_percentage: float = 80.0) -> List[AgentRegistration]:
        """Find agents with available capacity"""
        available_agents = []
        
        for registration in self.agents.values():
            if registration.status not in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                continue
            
            if registration.health in [AgentHealth.UNRESPONSIVE, AgentHealth.CRITICAL]:
                continue
            
            load_percentage = (registration.current_task_count / 
                             registration.max_concurrent_tasks) * 100
            
            if load_percentage <= max_load_percentage:
                available_agents.append(registration)
        
        return available_agents
    
    def select_best_agent(self, required_capabilities: List[AgentCapability] = None,
                         agent_type: str = None,
                         tags: List[str] = None,
                         selection_criteria: Dict[str, float] = None) -> Optional[AgentRegistration]:
        """Select the best agent based on criteria"""
        
        # Start with all agents
        candidates = list(self.agents.values())
        
        # Filter by type
        if agent_type:
            candidates = [agent for agent in candidates if agent.agent_type == agent_type]
        
        # Filter by capabilities
        if required_capabilities:
            candidates = AgentSelector.by_capability(candidates, required_capabilities)
        
        # Filter by tags
        if tags:
            candidates = [agent for agent in candidates 
                         if all(tag in agent.tags for tag in tags)]
        
        # Filter by availability
        candidates = AgentSelector.by_load(candidates)
        candidates = AgentSelector.by_health(candidates)
        
        if not candidates:
            return None
        
        # Select best agent
        return AgentSelector.best_fit(candidates, required_capabilities or [], selection_criteria)
    
    def get_agent_load_distribution(self) -> Dict[str, float]:
        """Get load distribution across all agents"""
        load_distribution = {}
        
        for agent_id, registration in self.agents.items():
            load_percentage = (registration.current_task_count / 
                             registration.max_concurrent_tasks) * 100
            load_distribution[agent_id] = load_percentage
        
        return load_distribution
    
    def get_capability_coverage(self) -> Dict[str, int]:
        """Get coverage of each capability across agents"""
        coverage = {}
        
        for capability in AgentCapability:
            agent_count = len(self.capability_index.get(capability, set()))
            coverage[capability.value] = agent_count
        
        return coverage
    
    # Background monitoring tasks
    async def _health_monitor(self):
        """Monitor agent health"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                unhealthy_agents = set()
                
                for agent_id, registration in self.agents.items():
                    # Check heartbeat timeout
                    time_since_heartbeat = (current_time - registration.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        registration.health = AgentHealth.UNRESPONSIVE
                        registration.status = AgentStatus.OFFLINE
                        unhealthy_agents.add(agent_id)
                        
                        await self._trigger_event("agent_unhealthy", {
                            "agent_id": agent_id,
                            "reason": "heartbeat_timeout",
                            "last_heartbeat": registration.last_heartbeat.isoformat()
                        })
                    
                    # Update health status
                    new_health = self._calculate_agent_health(registration)
                    if new_health != registration.health:
                        old_health = registration.health
                        registration.health = new_health
                        
                        await self._trigger_event("agent_health_changed", {
                            "agent_id": agent_id,
                            "old_health": old_health.value,
                            "new_health": new_health.value
                        })
                
                # Update unhealthy agents set
                self.unhealthy_agents = unhealthy_agents
                
                # Update stats
                self.registry_stats["health_checks_performed"] += 1
                self.registry_stats["failed_health_checks"] += len(unhealthy_agents)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _stats_collector(self):
        """Collect registry statistics"""
        while not self._shutdown_event.is_set():
            try:
                self._update_stats()
                await asyncio.sleep(60)  # Update stats every minute
                
            except Exception as e:
                self.logger.error(f"Stats collector error: {e}")
                await asyncio.sleep(60)
    
    def _update_stats(self):
        """Update registry statistics"""
        self.registry_stats["active_agents"] = len([
            agent for agent in self.agents.values()
            if agent.status not in [AgentStatus.OFFLINE, AgentStatus.ERROR]
        ])
        
        self.registry_stats["unhealthy_agents"] = len(self.unhealthy_agents)
        self.registry_stats["agent_types"] = len(self.agent_types)
        self.registry_stats["total_capabilities"] = len(self.capability_index)
    
    async def _cleanup_expired_agents(self):
        """Clean up expired agent registrations"""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                cleanup_threshold = timedelta(hours=24)  # Remove agents offline for 24 hours
                
                agents_to_remove = []
                
                for agent_id, registration in self.agents.items():
                    if (registration.status == AgentStatus.OFFLINE and
                        current_time - registration.last_heartbeat > cleanup_threshold):
                        agents_to_remove.append(agent_id)
                
                for agent_id in agents_to_remove:
                    await self.unregister_agent(agent_id)
                    self.logger.info(f"Cleaned up expired agent {agent_id}")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_analyzer(self):
        """Analyze agent performance trends"""
        while not self._shutdown_event.is_set():
            try:
                # Log performance summary
                total_agents = len(self.agents)
                healthy_agents = len([a for a in self.agents.values() 
                                    if a.health == AgentHealth.HEALTHY])
                
                if total_agents > 0:
                    health_percentage = (healthy_agents / total_agents) * 100
                    avg_load = sum(
                        (a.current_task_count / a.max_concurrent_tasks) * 100
                        for a in self.agents.values()
                    ) / total_agents if total_agents > 0 else 0
                    
                    self.logger.info(
                        f"Registry Health: {health_percentage:.1f}% healthy agents, "
                        f"avg load: {avg_load:.1f}%"
                    )
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance analyzer error: {e}")
                await asyncio.sleep(300)
    
    # Event system
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler (canonical util)"""
        _reg_handler(self.event_handlers, event_type, handler)
    
    async def _trigger_event(self, event_type: str, event_data: Dict[str, Any]):
        """Trigger an event"""
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data)
                else:
                    handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event_type}: {e}")
    
    # Public query methods
    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, AgentRegistration]:
        """Get all registered agents"""
        return self.agents.copy()
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return self.registry_stats.copy()
    
    def get_agent_types(self) -> List[str]:
        """Get list of registered agent types"""
        return list(self.agent_types.keys())
    
    def get_available_capabilities(self) -> List[str]:
        """Get list of available capabilities"""
        return [cap.value for cap in self.capability_index.keys()]
    
    async def shutdown(self):
        """Shutdown the agent registry"""
        self.logger.info("Shutting down Agent Registry")
        
        self.status = RegistryStatus.SHUTDOWN
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.logger.info("Agent Registry shutdown complete")


# Global registry instance
_agent_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> Optional[AgentRegistry]:
    """Get the global agent registry"""
    return _agent_registry


def set_agent_registry(registry: AgentRegistry):
    """Set the global agent registry"""
    global _agent_registry
    _agent_registry = registry
