#!/usr/bin/env python3
"""
Task Assignment Coordinator with Dynamic Queue Management
Implements intelligent task routing based on agent capabilities and load.
"""
import asyncio
import json
import logging
import os
import sys
import uuid
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict
from enum import Enum

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

from agents.core.messaging_agent_base import MessagingAgent
from agents.core.rabbitmq_client import RabbitMQClient
from schemas.task_messages import (
    TaskRequestMessage, TaskAssignmentMessage, 
    TaskStatusUpdateMessage, TaskCompletionMessage
)
from schemas.agent_messages import AgentHeartbeatMessage, AgentStatusMessage
from schemas.system_messages import ErrorMessage, SystemAlertMessage
from schemas.base import TaskStatus, AgentStatus, Priority, MessageType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Add trace_id to logger
class TraceIDAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra.get('trace_id', 'NO_TRACE'), msg), kwargs


class AssignmentStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY_BASED = "priority_based"
    CAPABILITY_MATCH = "capability_match"


class AgentInfo:
    """Tracks agent status and metrics"""
    def __init__(self, agent_id: str, capabilities: List[str], priority: int = 1):
        self.agent_id = agent_id
        self.capabilities = set(capabilities)
        self.priority = priority
        self.status = AgentStatus.OFFLINE
        self.current_load = 0.0
        self.active_tasks = 0
        self.max_tasks = 10
        self.last_heartbeat = None
        self.last_assignment = None
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.average_completion_time = 0.0
        self.queue_name = f"agent.{agent_id}"
        
    def is_available(self) -> bool:
        """Check if agent is available for tasks"""
        if self.status != AgentStatus.READY:
            return False
        if self.active_tasks >= self.max_tasks:
            return False
        if self.current_load > 0.9:  # 90% load threshold
            return False
        return True
    
    def is_stale(self, threshold_seconds: int = 120) -> bool:
        """Check if agent heartbeat is stale"""
        if not self.last_heartbeat:
            return True
        age = (datetime.utcnow() - self.last_heartbeat).total_seconds()
        return age > threshold_seconds


class TaskAssignmentCoordinator(MessagingAgent):
    """
    Intelligent task assignment coordinator with dynamic routing.
    Monitors agent health and assigns tasks based on capabilities and load.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="task_assignment_coordinator",
            agent_type="coordinator",
            capabilities=[
                "task_assignment",
                "load_balancing", 
                "priority_management",
                "queue_management"
            ],
            version="3.0.0",
            port=8551
        )
        
        # Load configuration
        self.config = self._load_config()
        
        # Agent tracking
        self.agents: Dict[str, AgentInfo] = {}
        self._initialize_agents()
        
        # Task tracking
        self.pending_tasks: Dict[str, Dict] = {}
        self.active_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.task_trace_ids: Dict[str, str] = {}  # task_id -> trace_id
        
        # Assignment strategy
        self.assignment_strategy = AssignmentStrategy.CAPABILITY_MATCH
        self.round_robin_index = 0
        
        # Metrics
        self.metrics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_assignment_time": 0.0
        }
        
        # Background tasks
        self.monitor_task = None
        self.cleanup_task = None
    
    def _load_config(self) -> Dict:
        """Load agent configuration from YAML"""
        config_path = "/opt/sutazaiapp/config/agents.yaml"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {"agents": {}, "task_routing": {}, "global_settings": {}}
    
    def _initialize_agents(self):
        """Initialize agent registry from configuration"""
        for agent_id, agent_config in self.config.get("agents", {}).items():
            self.agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                capabilities=agent_config.get("capabilities", []),
                priority=agent_config.get("priority", 1)
            )
            self.agents[agent_id].max_tasks = agent_config.get("max_concurrent_tasks", 10)
            self.agents[agent_id].queue_name = agent_config.get("queue", f"agent.{agent_id}")
            logger.info(f"Registered agent: {agent_id} with capabilities: {agent_config.get('capabilities')}")
    
    async def _register_default_handlers(self):
        """Register message handlers"""
        await self.register_handler("task.assign", self.handle_task_assign)
        await self.register_handler("agent.status", self.handle_agent_status)
        await self.register_handler("agent.heartbeat", self.handle_agent_heartbeat)
        await self.register_handler("task.completion", self.handle_task_completion)
    
    async def handle_task_assign(self, message_data: Dict, raw_message):
        """
        Handle task assignment request.
        Routes tasks to appropriate agents based on capabilities and load.
        """
        trace_id = message_data.get("correlation_id", str(uuid.uuid4()))
        log = TraceIDAdapter(logger, {'trace_id': trace_id})
        
        try:
            task_id = message_data.get("task_id")
            task_type = message_data.get("task_type")
            payload = message_data.get("payload", {})
            priority = message_data.get("priority", Priority.NORMAL)
            
            log.info(f"Received task assignment request: {task_id}, type: {task_type}")
            
            # Store trace ID
            self.task_trace_ids[task_id] = trace_id
            
            # Find eligible agents
            eligible_agents = await self._find_eligible_agents(task_type, log)
            
            if not eligible_agents:
                log.warning(f"No eligible agents found for task type: {task_type}")
                await self._send_assignment_failed(task_id, trace_id, "No eligible agents available")
                return
            
            # Select best agent
            selected_agent = await self._select_agent(eligible_agents, priority, log)
            
            if not selected_agent:
                log.warning("No available agents after selection")
                await self._send_assignment_failed(task_id, trace_id, "All agents at capacity")
                return
            
            # Dispatch task to agent
            await self._dispatch_task(task_id, selected_agent, message_data, trace_id, log)
            
            # Update metrics
            self.metrics["total_assignments"] += 1
            self.metrics["successful_assignments"] += 1
            
        except Exception as e:
            log.error(f"Error handling task assignment: {e}")
            await self._send_assignment_failed(
                message_data.get("task_id", "unknown"),
                trace_id,
                f"Assignment error: {str(e)}"
            )
    
    async def _find_eligible_agents(self, task_type: str, log) -> List[AgentInfo]:
        """Find agents capable of handling the task type"""
        eligible = []
        
        # Get required capabilities from config
        task_config = self.config.get("task_routing", {}).get(task_type, {})
        required_capabilities = set(task_config.get("required_capabilities", []))
        preferred_agents = task_config.get("preferred_agents", [])
        
        log.info(f"Required capabilities for {task_type}: {required_capabilities}")
        
        # Find agents with required capabilities
        for agent_id, agent in self.agents.items():
            # Check if agent is not stale
            if agent.is_stale():
                log.debug(f"Agent {agent_id} is stale")
                continue
            
            # Check capabilities match
            if required_capabilities and not required_capabilities.issubset(agent.capabilities):
                continue
            
            # Check if agent is preferred
            if preferred_agents and agent_id not in preferred_agents:
                continue
            
            eligible.append(agent)
            log.debug(f"Agent {agent_id} is eligible")
        
        return eligible
    
    async def _select_agent(self, eligible_agents: List[AgentInfo], priority: int, log) -> Optional[AgentInfo]:
        """Select best agent based on assignment strategy"""
        available_agents = [a for a in eligible_agents if a.is_available()]
        
        if not available_agents:
            return None
        
        log.info(f"Selecting from {len(available_agents)} available agents using {self.assignment_strategy}")
        
        if self.assignment_strategy == AssignmentStrategy.ROUND_ROBIN:
            # Round-robin selection
            selected = available_agents[self.round_robin_index % len(available_agents)]
            self.round_robin_index += 1
            
        elif self.assignment_strategy == AssignmentStrategy.LEAST_LOADED:
            # Select least loaded agent
            selected = min(available_agents, key=lambda a: a.current_load)
            
        elif self.assignment_strategy == AssignmentStrategy.PRIORITY_BASED:
            # Select by agent priority and task priority
            if priority >= Priority.HIGH:
                # High priority tasks go to high priority agents
                selected = min(available_agents, key=lambda a: (a.priority, a.current_load))
            else:
                selected = min(available_agents, key=lambda a: a.current_load)
                
        else:  # CAPABILITY_MATCH (default)
            # Best capability match with load consideration
            selected = min(available_agents, key=lambda a: (a.current_load, -len(a.capabilities)))
        
        log.info(f"Selected agent: {selected.agent_id} (load: {selected.current_load:.2f})")
        return selected
    
    async def _dispatch_task(self, task_id: str, agent: AgentInfo, task_data: Dict, trace_id: str, log):
        """Dispatch task to selected agent"""
        try:
            # Create assignment message
            assignment = TaskAssignmentMessage(
                source_agent=self.agent_id,
                target_agent=agent.agent_id,
                correlation_id=trace_id,
                task_id=task_id,
                assigned_agent=agent.agent_id,
                assignment_time=datetime.utcnow(),
                priority_boost=0,
                resource_allocation={},
                execution_constraints={}
            )
            
            # Publish to agent's queue
            await self.rabbitmq.publish(
                assignment,
                exchange="sutazai.tasks",
                routing_key=f"task.dispatch.{agent.agent_id}"
            )
            
            # Update tracking
            self.active_assignments[task_id] = agent.agent_id
            agent.active_tasks += 1
            agent.last_assignment = datetime.utcnow()
            
            log.info(f"Task {task_id} dispatched to {agent.agent_id}")
            
            # Send status update
            status_update = TaskStatusUpdateMessage(
                source_agent=self.agent_id,
                correlation_id=trace_id,
                task_id=task_id,
                status=TaskStatus.ASSIGNED,
                progress=0.0,
                message=f"Assigned to {agent.agent_id}"
            )
            await self.rabbitmq.publish(status_update)
            
        except Exception as e:
            log.error(f"Failed to dispatch task: {e}")
            raise
    
    async def handle_agent_status(self, message_data: Dict, raw_message):
        """Handle agent status updates"""
        try:
            agent_id = message_data.get("agent_id")
            if agent_id not in self.agents:
                # Register new agent
                self.agents[agent_id] = AgentInfo(
                    agent_id=agent_id,
                    capabilities=message_data.get("capabilities", [])
                )
            
            agent = self.agents[agent_id]
            agent.status = AgentStatus(message_data.get("status", "offline"))
            agent.current_load = message_data.get("current_load", 0.0)
            agent.active_tasks = message_data.get("active_tasks", 0)
            agent.last_heartbeat = datetime.utcnow()
            
            logger.debug(f"Updated status for {agent_id}: {agent.status}, load: {agent.current_load}")
            
        except Exception as e:
            logger.error(f"Error handling agent status: {e}")
    
    async def handle_agent_heartbeat(self, message_data: Dict, raw_message):
        """Handle agent heartbeat messages"""
        try:
            agent_id = message_data.get("agent_id")
            if agent_id not in self.agents:
                # Auto-register agent
                self.agents[agent_id] = AgentInfo(agent_id, [])
            
            agent = self.agents[agent_id]
            agent.status = AgentStatus(message_data.get("status", "ready"))
            agent.current_load = message_data.get("current_load", 0.0)
            agent.active_tasks = message_data.get("active_tasks", 0)
            agent.last_heartbeat = datetime.utcnow()
            
            # Update capacity if provided
            if "available_capacity" in message_data:
                agent.max_tasks = agent.active_tasks + message_data["available_capacity"]
            
            logger.debug(f"Heartbeat from {agent_id}: load={agent.current_load:.2f}, tasks={agent.active_tasks}")
            
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")
    
    async def handle_task_completion(self, message_data: Dict, raw_message):
        """Handle task completion notifications"""
        try:
            task_id = message_data.get("task_id")
            status = TaskStatus(message_data.get("status", "completed"))
            
            if task_id in self.active_assignments:
                agent_id = self.active_assignments[task_id]
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    agent.active_tasks = max(0, agent.active_tasks - 1)
                    
                    if status == TaskStatus.COMPLETED:
                        agent.total_tasks_completed += 1
                    else:
                        agent.total_tasks_failed += 1
                    
                    logger.info(f"Task {task_id} completed by {agent_id} with status: {status}")
                
                # Clean up tracking
                del self.active_assignments[task_id]
                if task_id in self.task_trace_ids:
                    del self.task_trace_ids[task_id]
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
    
    async def _send_assignment_failed(self, task_id: str, trace_id: str, reason: str):
        """Send assignment failure notification"""
        try:
            # Send error message
            error = ErrorMessage(
                source_agent=self.agent_id,
                correlation_id=trace_id,
                error_id=f"assignment_failed_{task_id}",
                error_code="ASSIGNMENT_FAILED",
                error_message=reason,
                error_type="assignment_error",
                severity=Priority.HIGH,
                affected_task_id=task_id,
                retry_possible=True,
                context={"task_id": task_id, "reason": reason}
            )
            
            await self.rabbitmq.publish(
                error,
                exchange="sutazai.system",
                routing_key="assignment.failed"
            )
            
            # Update metrics
            self.metrics["failed_assignments"] += 1
            
            logger.warning(f"Assignment failed for task {task_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Failed to send assignment failure: {e}")
    
    async def monitor_agents(self):
        """Background task to monitor agent health"""
        while not self.shutdown_event.is_set():
            try:
                stale_threshold = self.config.get("global_settings", {}).get(
                    "stale_agent_threshold_seconds", 120
                )
                
                for agent_id, agent in self.agents.items():
                    if agent.is_stale(stale_threshold):
                        if agent.status != AgentStatus.OFFLINE:
                            logger.warning(f"Agent {agent_id} marked as offline (stale heartbeat)")
                            agent.status = AgentStatus.OFFLINE
                            
                            # Send alert
                            alert = SystemAlertMessage(
                                source_agent=self.agent_id,
                                alert_id=f"agent_offline_{agent_id}",
                                severity="warning",
                                category="agent_health",
                                title=f"Agent {agent_id} Offline",
                                description=f"No heartbeat received for {stale_threshold} seconds",
                                affected_components=[agent_id]
                            )
                            await self.rabbitmq.publish(alert)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(30)
    
    async def cleanup_stale_tasks(self):
        """Background task to clean up stale assignments"""
        while not self.shutdown_event.is_set():
            try:
                # Clean up old trace IDs (older than 1 hour)
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                stale_tasks = []
                
                for task_id, agent_id in self.active_assignments.items():
                    if agent_id in self.agents:
                        agent = self.agents[agent_id]
                        if agent.last_assignment and agent.last_assignment < cutoff_time:
                            stale_tasks.append(task_id)
                
                for task_id in stale_tasks:
                    logger.warning(f"Cleaning up stale task: {task_id}")
                    if task_id in self.active_assignments:
                        del self.active_assignments[task_id]
                    if task_id in self.task_trace_ids:
                        del self.task_trace_ids[task_id]
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)
    
    async def get_status(self) -> Dict:
        """Get coordinator status and metrics"""
        return {
            "coordinator_id": self.agent_id,
            "status": self.status.value,
            "assignment_strategy": self.assignment_strategy.value,
            "agents": {
                agent_id: {
                    "status": agent.status.value,
                    "load": agent.current_load,
                    "active_tasks": agent.active_tasks,
                    "available": agent.is_available()
                }
                for agent_id, agent in self.agents.items()
            },
            "active_assignments": len(self.active_assignments),
            "metrics": self.metrics
        }
    
    async def run(self):
        """Main coordinator run loop"""
        try:
            # Initialize messaging
            if not await self.initialize():
                return
            
            # Start consuming from queues
            await self.rabbitmq.consume("task.assign", self.handle_task_assign)
            await self.rabbitmq.consume("agent.status", self.handle_agent_status)
            
            # Start background tasks
            self.monitor_task = asyncio.create_task(self.monitor_agents())
            self.cleanup_task = asyncio.create_task(self.cleanup_stale_tasks())
            
            logger.info(f"Task Assignment Coordinator started with strategy: {self.assignment_strategy}")
            
            # Keep running until shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Coordinator run error: {e}")
        finally:
            # Cleanup
            if self.monitor_task:
                self.monitor_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            await self.shutdown()


async def main():
    """Main entry point"""
    coordinator = TaskAssignmentCoordinator()
    await coordinator.run()


if __name__ == "__main__":
    asyncio.run(main())