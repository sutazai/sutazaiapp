#!/usr/bin/env python3
"""
Agent Coordinator - Coordinates agent activities and interactions
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CoordinationMode(Enum):
    """Coordination modes"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    DEMOCRATIC = "democratic"
    LEADER_FOLLOWER = "leader_follower"
    SWARM = "swarm"

class CoordinationStrategy(Enum):
    """Coordination strategies"""
    ROUND_ROBIN = "round_robin"
    LOAD_BASED = "load_based"
    CAPABILITY_BASED = "capability_based"
    PRIORITY_BASED = "priority_based"
    RANDOM = "random"

@dataclass
class CoordinationConfig:
    """Configuration for agent coordination"""
    mode: CoordinationMode = CoordinationMode.HIERARCHICAL
    strategy: CoordinationStrategy = CoordinationStrategy.LOAD_BASED
    max_coordination_groups: int = 10
    group_size_limit: int = 20
    enable_dynamic_grouping: bool = True
    enable_conflict_resolution: bool = True
    enable_consensus: bool = True
    consensus_threshold: float = 0.67
    coordination_timeout: float = 30.0
    heartbeat_interval: float = 5.0
    enable_leader_election: bool = True
    leader_election_timeout: float = 10.0

@dataclass
class CoordinationGroup:
    """Agent coordination group"""
    group_id: str
    leader_id: Optional[str] = None
    members: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    group_type: str = "general"
    coordination_mode: CoordinationMode = CoordinationMode.HIERARCHICAL
    active_tasks: Set[str] = field(default_factory=set)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class CoordinationTask:
    """Task for agent coordination"""
    task_id: str
    task_type: str
    priority: int
    assigned_agents: Set[str]
    required_capabilities: List[str]
    dependencies: List[str]
    deadline: Optional[datetime] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentCoordinator:
    """Agent coordinator for managing agent interactions"""
    
    def __init__(self, config: CoordinationConfig = None, agent_manager=None, 
                 task_scheduler=None, collaboration_engine=None, workflow_engine=None):
        self.config = config or CoordinationConfig()
        self.agent_manager = agent_manager
        self.task_scheduler = task_scheduler
        self.collaboration_engine = collaboration_engine
        self.workflow_engine = workflow_engine
        
        # Coordination state
        self.coordination_groups: Dict[str, CoordinationGroup] = {}
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, str] = {}  # agent_id -> group_id
        
        # Task coordination
        self.coordination_tasks: Dict[str, CoordinationTask] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # Leadership tracking
        self.leaders: Dict[str, str] = {}  # group_id -> leader_id
        self.leader_elections: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.coordination_metrics = {
            "total_groups": 0,
            "active_coordinations": 0,
            "successful_coordinations": 0,
            "failed_coordinations": 0,
            "average_coordination_time": 0.0,
            "consensus_success_rate": 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._heartbeat_task = None
        self._maintenance_task = None
        
        logger.info("Agent coordinator initialized")
    
    async def initialize(self) -> bool:
        """Initialize agent coordinator"""
        try:
            # Start background tasks
            self._start_heartbeat()
            self._start_maintenance()
            
            logger.info("Agent coordinator initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Agent coordinator initialization failed: {e}")
            return False
    
    def _start_heartbeat(self):
        """Start coordination heartbeat"""
        def heartbeat_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._coordination_heartbeat())
                    self._shutdown_event.wait(self.config.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    self._shutdown_event.wait(10)
        
        self._heartbeat_task = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_task.start()
    
    def _start_maintenance(self):
        """Start coordination maintenance"""
        def maintenance_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._coordination_maintenance())
                    self._shutdown_event.wait(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                    self._shutdown_event.wait(60)
        
        self._maintenance_task = threading.Thread(target=maintenance_loop, daemon=True)
        self._maintenance_task.start()
    
    async def create_coordination_group(self, group_config: Dict[str, Any]) -> str:
        """Create a new coordination group"""
        try:
            with self._lock:
                if len(self.coordination_groups) >= self.config.max_coordination_groups:
                    raise RuntimeError("Maximum coordination groups reached")
                
                group_id = f"group_{uuid.uuid4().hex[:8]}"
                
                # Create coordination group
                group = CoordinationGroup(
                    group_id=group_id,
                    group_type=group_config.get("type", "general"),
                    coordination_mode=CoordinationMode(group_config.get("mode", "hierarchical"))
                )
                
                # Add initial members if specified
                initial_members = group_config.get("members", [])
                for member_id in initial_members:
                    await self.add_agent_to_group(group_id, member_id)
                
                self.coordination_groups[group_id] = group
                self.coordination_metrics["total_groups"] += 1
                
                # Elect leader if needed
                if self.config.enable_leader_election and group.coordination_mode == CoordinationMode.HIERARCHICAL:
                    await self._elect_leader(group_id)
                
                logger.info(f"Coordination group created: {group_id}")
                return group_id
                
        except Exception as e:
            logger.error(f"Failed to create coordination group: {e}")
            raise
    
    async def add_agent_to_group(self, group_id: str, agent_id: str) -> bool:
        """Add agent to coordination group"""
        try:
            with self._lock:
                if group_id not in self.coordination_groups:
                    raise ValueError(f"Coordination group not found: {group_id}")
                
                group = self.coordination_groups[group_id]
                
                # Check group size limit
                if len(group.members) >= self.config.group_size_limit:
                    logger.warning(f"Group size limit reached for group {group_id}")
                    return False
                
                # Add agent to group
                group.members.add(agent_id)
                self.agent_assignments[agent_id] = group_id
                group.last_activity = datetime.now(timezone.utc)
                
                # Update leader if this is the first member
                if len(group.members) == 1 and group.leader_id is None:
                    group.leader_id = agent_id
                    self.leaders[group_id] = agent_id
                
                logger.info(f"Agent {agent_id} added to group {group_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add agent to group: {e}")
            return False
    
    async def remove_agent_from_group(self, group_id: str, agent_id: str) -> bool:
        """Remove agent from coordination group"""
        try:
            with self._lock:
                if group_id not in self.coordination_groups:
                    return False
                
                group = self.coordination_groups[group_id]
                
                if agent_id not in group.members:
                    return False
                
                # Remove agent from group
                group.members.remove(agent_id)
                if agent_id in self.agent_assignments:
                    del self.agent_assignments[agent_id]
                
                group.last_activity = datetime.now(timezone.utc)
                
                # Handle leader removal
                if group.leader_id == agent_id:
                    group.leader_id = None
                    if group_id in self.leaders:
                        del self.leaders[group_id]
                    
                    # Elect new leader if needed
                    if group.members and self.config.enable_leader_election:
                        await self._elect_leader(group_id)
                
                # Remove group if empty
                if not group.members:
                    del self.coordination_groups[group_id]
                    self.coordination_metrics["total_groups"] -= 1
                
                logger.info(f"Agent {agent_id} removed from group {group_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to remove agent from group: {e}")
            return False
    
    async def coordinate_task(self, task_config: Dict[str, Any]) -> str:
        """Coordinate a task across agents"""
        try:
            task_id = f"coord_task_{uuid.uuid4().hex[:8]}"
            
            # Create coordination task
            task = CoordinationTask(
                task_id=task_id,
                task_type=task_config.get("type", "general"),
                priority=task_config.get("priority", 1),
                assigned_agents=set(task_config.get("assigned_agents", [])),
                required_capabilities=task_config.get("required_capabilities", []),
                dependencies=task_config.get("dependencies", []),
                deadline=task_config.get("deadline"),
                metadata=task_config.get("metadata", {})
            )
            
            # Find suitable coordination group or create one
            group_id = await self._find_coordination_group(task)
            if not group_id:
                # Create new group for this task
                group_config = {
                    "type": "task_coordination",
                    "mode": "hierarchical",
                    "members": list(task.assigned_agents)
                }
                group_id = await self.create_coordination_group(group_config)
            
            # Add task to group
            group = self.coordination_groups[group_id]
            group.active_tasks.add(task_id)
            
            # Store task
            self.coordination_tasks[task_id] = task
            self.coordination_metrics["active_coordinations"] += 1
            
            # Start coordination
            coordination_id = await self._start_coordination(group_id, task_id)
            
            logger.info(f"Task coordination started: {task_id}")
            return coordination_id
            
        except Exception as e:
            logger.error(f"Task coordination failed: {e}")
            raise
    
    async def _find_coordination_group(self, task: CoordinationTask) -> Optional[str]:
        """Find suitable coordination group for task"""
        with self._lock:
            # Look for existing group with required agents
            for group_id, group in self.coordination_groups.items():
                if task.assigned_agents.issubset(group.members):
                    return group_id
            
            # Look for group with available capacity
            for group_id, group in self.coordination_groups.items():
                if len(group.active_tasks) < 3:  # Arbitrary limit
                    return group_id
            
            return None
    
    async def _start_coordination(self, group_id: str, task_id: str) -> str:
        """Start coordination for a task"""
        try:
            coordination_id = f"coord_{uuid.uuid4().hex[:8]}"
            
            # Create coordination record
            coordination = {
                "coordination_id": coordination_id,
                "group_id": group_id,
                "task_id": task_id,
                "status": "active",
                "start_time": datetime.now(timezone.utc),
                "participants": list(self.coordination_groups[group_id].members),
                "leader": self.coordination_groups[group_id].leader_id,
                "progress": 0.0,
                "messages": []
            }
            
            self.active_coordinations[coordination_id] = coordination
            
            # Notify participants
            await self._notify_coordination_start(coordination)
            
            # Start coordination process based on mode
            group = self.coordination_groups[group_id]
            if group.coordination_mode == CoordinationMode.HIERARCHICAL:
                await self._hierarchical_coordination(coordination_id)
            elif group.coordination_mode == CoordinationMode.PEER_TO_PEER:
                await self._peer_to_peer_coordination(coordination_id)
            elif group.coordination_mode == CoordinationMode.DEMOCRATIC:
                await self._democratic_coordination(coordination_id)
            
            return coordination_id
            
        except Exception as e:
            logger.error(f"Failed to start coordination: {e}")
            raise
    
    async def _hierarchical_coordination(self, coordination_id: str):
        """Handle hierarchical coordination"""
        try:
            coordination = self.active_coordinations[coordination_id]
            leader_id = coordination["leader"]
            
            if not leader_id:
                # Elect leader
                group_id = coordination["group_id"]
                await self._elect_leader(group_id)
                leader_id = self.leaders.get(group_id)
            
            if leader_id:
                # Leader coordinates the task
                await self._leader_coordinate_task(coordination_id, leader_id)
            else:
                # Fall back to peer-to-peer
                await self._peer_to_peer_coordination(coordination_id)
                
        except Exception as e:
            logger.error(f"Hierarchical coordination failed: {e}")
            await self._handle_coordination_failure(coordination_id, str(e))
    
    async def _peer_to_peer_coordination(self, coordination_id: str):
        """Handle peer-to-peer coordination"""
        try:
            coordination = self.active_coordinations[coordination_id]
            participants = coordination["participants"]
            
            # Distribute task among participants
            if len(participants) > 1:
                # Simple round-robin distribution
                for i, participant in enumerate(participants):
                    subtask = {
                        "participant": participant,
                        "portion": i / len(participants),
                        "coordination_id": coordination_id
                    }
                    await self._assign_subtask(subtask)
            
            # Monitor progress
            await self._monitor_coordination_progress(coordination_id)
            
        except Exception as e:
            logger.error(f"Peer-to-peer coordination failed: {e}")
            await self._handle_coordination_failure(coordination_id, str(e))
    
    async def _democratic_coordination(self, coordination_id: str):
        """Handle democratic coordination"""
        try:
            coordination = self.active_coordinations[coordination_id]
            participants = coordination["participants"]
            
            # Voting-based coordination
            if len(participants) > 1:
                # Simulate voting process
                votes = await self._collect_votes(coordination_id)
                decision = await self._make_consensus_decision(votes)
                
                if decision:
                    await self._execute_consensus_decision(coordination_id, decision)
                else:
                    # Fall back to leader-based coordination
                    await self._hierarchical_coordination(coordination_id)
            
        except Exception as e:
            logger.error(f"Democratic coordination failed: {e}")
            await self._handle_coordination_failure(coordination_id, str(e))
    
    async def _elect_leader(self, group_id: str):
        """Elect leader for coordination group"""
        try:
            if group_id not in self.coordination_groups:
                return
            
            group = self.coordination_groups[group_id]
            if not group.members:
                return
            
            # Simple election: choose first available agent
            # In practice, this would be more sophisticated
            leader_id = next(iter(group.members))
            
            group.leader_id = leader_id
            self.leaders[group_id] = leader_id
            
            logger.info(f"Leader elected for group {group_id}: {leader_id}")
            
        except Exception as e:
            logger.error(f"Leader election failed: {e}")
    
    async def _leader_coordinate_task(self, coordination_id: str, leader_id: str):
        """Leader coordinates task execution"""
        try:
            coordination = self.active_coordinations[coordination_id]
            
            # Leader makes decisions and delegates
            delegation_plan = await self._create_delegation_plan(coordination_id, leader_id)
            
            # Execute delegation
            for delegation in delegation_plan:
                await self._execute_delegation(delegation)
            
            # Monitor execution
            await self._monitor_coordination_progress(coordination_id)
            
        except Exception as e:
            logger.error(f"Leader coordination failed: {e}")
            raise
    
    async def _create_delegation_plan(self, coordination_id: str, leader_id: str) -> List[Dict[str, Any]]:
        """Create delegation plan for task"""
        coordination = self.active_coordinations[coordination_id]
        task_id = coordination["task_id"]
        task = self.coordination_tasks[task_id]
        
        # Simple delegation: divide task among participants
        participants = [p for p in coordination["participants"] if p != leader_id]
        
        delegations = []
        for i, participant in enumerate(participants):
            delegation = {
                "participant": participant,
                "task_portion": f"subtask_{i}",
                "coordination_id": coordination_id,
                "deadline": task.deadline
            }
            delegations.append(delegation)
        
        return delegations
    
    async def _execute_delegation(self, delegation: Dict[str, Any]):
        """Execute delegation to agent"""
        try:
            participant = delegation["participant"]
            
            # This would integrate with the agent manager
            # For now, just log the delegation
            logger.info(f"Delegating task to {participant}: {delegation['task_portion']}")
            
        except Exception as e:
            logger.error(f"Delegation execution failed: {e}")
    
    async def _assign_subtask(self, subtask: Dict[str, Any]):
        """Assign subtask to participant"""
        try:
            participant = subtask["participant"]
            
            # This would integrate with the agent manager
            logger.info(f"Assigning subtask to {participant}: {subtask['portion']}")
            
        except Exception as e:
            logger.error(f"Subtask assignment failed: {e}")
    
    async def _collect_votes(self, coordination_id: str) -> List[Dict[str, Any]]:
        """Collect votes from participants"""
        coordination = self.active_coordinations[coordination_id]
        participants = coordination["participants"]
        
        # Simulate voting process
        votes = []
        for participant in participants:
            vote = {
                "participant": participant,
                "decision": "approve",  # Simplified
                "confidence": 0.8,
                "timestamp": datetime.now(timezone.utc)
            }
            votes.append(vote)
        
        return votes
    
    async def _make_consensus_decision(self, votes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Make consensus decision from votes"""
        if not votes:
            return None
        
        # Simple majority consensus
        approve_count = sum(1 for vote in votes if vote["decision"] == "approve")
        total_votes = len(votes)
        
        if approve_count / total_votes >= self.config.consensus_threshold:
            return {
                "decision": "approve",
                "confidence": approve_count / total_votes,
                "votes": votes
            }
        
        return None
    
    async def _execute_consensus_decision(self, coordination_id: str, decision: Dict[str, Any]):
        """Execute consensus decision"""
        try:
            coordination = self.active_coordinations[coordination_id]
            
            # Execute the decided action
            logger.info(f"Executing consensus decision for {coordination_id}: {decision['decision']}")
            
            # Update coordination status
            coordination["status"] = "executing"
            coordination["decision"] = decision
            
        except Exception as e:
            logger.error(f"Consensus execution failed: {e}")
    
    async def _monitor_coordination_progress(self, coordination_id: str):
        """Monitor coordination progress"""
        try:
            coordination = self.active_coordinations[coordination_id]
            
            # Simulate progress monitoring
            # In practice, this would track actual task progress
            coordination["progress"] = 0.5  # 50% complete
            
            # Check if coordination is complete
            if coordination["progress"] >= 1.0:
                await self._complete_coordination(coordination_id)
            
        except Exception as e:
            logger.error(f"Progress monitoring failed: {e}")
    
    async def _complete_coordination(self, coordination_id: str):
        """Complete coordination"""
        try:
            coordination = self.active_coordinations[coordination_id]
            
            # Update status
            coordination["status"] = "completed"
            coordination["end_time"] = datetime.now(timezone.utc)
            
            # Update metrics
            self.coordination_metrics["successful_coordinations"] += 1
            self.coordination_metrics["active_coordinations"] -= 1
            
            # Clean up
            group_id = coordination["group_id"]
            task_id = coordination["task_id"]
            
            if group_id in self.coordination_groups:
                self.coordination_groups[group_id].active_tasks.discard(task_id)
            
            logger.info(f"Coordination completed: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Coordination completion failed: {e}")
    
    async def _handle_coordination_failure(self, coordination_id: str, error: str):
        """Handle coordination failure"""
        try:
            coordination = self.active_coordinations[coordination_id]
            
            # Update status
            coordination["status"] = "failed"
            coordination["error"] = error
            coordination["end_time"] = datetime.now(timezone.utc)
            
            # Update metrics
            self.coordination_metrics["failed_coordinations"] += 1
            self.coordination_metrics["active_coordinations"] -= 1
            
            logger.error(f"Coordination failed: {coordination_id} - {error}")
            
        except Exception as e:
            logger.error(f"Coordination failure handling failed: {e}")
    
    async def _notify_coordination_start(self, coordination: Dict[str, Any]):
        """Notify participants about coordination start"""
        try:
            # This would integrate with the communication system
            logger.info(f"Notifying participants about coordination start: {coordination['coordination_id']}")
            
        except Exception as e:
            logger.error(f"Coordination notification failed: {e}")
    
    async def _coordination_heartbeat(self):
        """Coordination heartbeat"""
        try:
            with self._lock:
                # Check active coordinations
                current_time = datetime.now(timezone.utc)
                
                for coordination_id, coordination in list(self.active_coordinations.items()):
                    if coordination["status"] == "active":
                        # Check for timeouts
                        elapsed = (current_time - coordination["start_time"]).total_seconds()
                        if elapsed > self.config.coordination_timeout:
                            await self._handle_coordination_failure(
                                coordination_id, 
                                "Coordination timeout"
                            )
                
                # Update group activity
                for group in self.coordination_groups.values():
                    if group.members:
                        group.last_activity = current_time
                        
        except Exception as e:
            logger.error(f"Coordination heartbeat failed: {e}")
    
    async def _coordination_maintenance(self):
        """Coordination maintenance"""
        try:
            with self._lock:
                current_time = datetime.now(timezone.utc)
                
                # Clean up old coordinations
                old_coordinations = [
                    coord_id for coord_id, coord in self.active_coordinations.items()
                    if coord["status"] in ["completed", "failed"] and
                    (current_time - coord.get("end_time", current_time)).total_seconds() > 3600
                ]
                
                for coord_id in old_coordinations:
                    del self.active_coordinations[coord_id]
                
                # Clean up empty groups
                empty_groups = [
                    group_id for group_id, group in self.coordination_groups.items()
                    if not group.members and not group.active_tasks
                ]
                
                for group_id in empty_groups:
                    del self.coordination_groups[group_id]
                    if group_id in self.leaders:
                        del self.leaders[group_id]
                    self.coordination_metrics["total_groups"] -= 1
                
        except Exception as e:
            logger.error(f"Coordination maintenance failed: {e}")
    
    async def start(self) -> bool:
        """Start agent coordinator"""
        try:
            logger.info("Starting agent coordinator...")
            return True
        except Exception as e:
            logger.error(f"Failed to start agent coordinator: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop agent coordinator"""
        try:
            logger.info("Stopping agent coordinator...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Complete active coordinations
            for coordination_id in list(self.active_coordinations.keys()):
                await self._handle_coordination_failure(coordination_id, "System shutdown")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent coordinator: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        with self._lock:
            return {
                "total_groups": len(self.coordination_groups),
                "active_coordinations": len([
                    c for c in self.active_coordinations.values() 
                    if c["status"] == "active"
                ]),
                "total_agents": len(self.agent_assignments),
                "metrics": self.coordination_metrics,
                "config": {
                    "mode": self.config.mode.value,
                    "strategy": self.config.strategy.value,
                    "max_groups": self.config.max_coordination_groups
                }
            }
    
    def get_coordination_info(self, coordination_id: str) -> Optional[Dict[str, Any]]:
        """Get coordination information"""
        with self._lock:
            return self.active_coordinations.get(coordination_id)
    
    def get_group_info(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group information"""
        with self._lock:
            group = self.coordination_groups.get(group_id)
            if group:
                return {
                    "group_id": group.group_id,
                    "leader_id": group.leader_id,
                    "members": list(group.members),
                    "group_type": group.group_type,
                    "coordination_mode": group.coordination_mode.value,
                    "active_tasks": list(group.active_tasks),
                    "created_at": group.created_at.isoformat(),
                    "last_activity": group.last_activity.isoformat()
                }
            return None
    
    def health_check(self) -> bool:
        """Check coordinator health"""
        try:
            return (
                len(self.coordination_groups) <= self.config.max_coordination_groups and
                len(self.active_coordinations) <= self.config.max_coordination_groups * 5
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Factory function
def create_agent_coordinator(config: Optional[Dict[str, Any]] = None) -> AgentCoordinator:
    """Create agent coordinator instance"""
    if config:
        coord_config = CoordinationConfig(**config)
    else:
        coord_config = CoordinationConfig()
    
    return AgentCoordinator(config=coord_config)