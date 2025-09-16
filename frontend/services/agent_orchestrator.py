"""
Agent Orchestrator Module
Manages multi-agent coordination and task delegation
"""

import asyncio
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import uuid
from enum import Enum
from dataclasses import dataclass
import concurrent.futures
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"

class TaskStatus(Enum):
    """Task execution states"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Agent:
    """Agent representation"""
    id: str
    name: str
    type: str
    capabilities: List[str]
    status: AgentStatus
    current_task: Optional[str] = None
    performance_score: float = 1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    metadata: Dict = None

@dataclass
class Task:
    """Task representation"""
    id: str
    type: str
    description: str
    requirements: List[str]
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict = None

class AgentOrchestrator:
    """Multi-agent orchestration system"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.result_callbacks: Dict[str, Callable] = {}
        
        # Threading for background orchestration
        self.orchestration_thread = None
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_completion_time": 0,
            "agent_utilization": {}
        }
        
        # Agent capabilities mapping
        self.capability_matrix = {
            "code_generation": ["letta", "autogpt", "crewai", "chatdev", "metgpt"],
            "web_search": ["autogpt", "gpt-researcher", "agentgpt"],
            "data_analysis": ["baby-agi", "finrobot", "workgpt"],
            "conversation": ["jarvis", "letta", "localagi"],
            "planning": ["autogpt", "baby-agi", "superagi"],
            "debugging": ["aider", "sweep", "gpt-pilot"],
            "documentation": ["gpt-engineer", "demogpt", "gpt-migrate"],
            "memory_management": ["letta", "agent-zero", "localgpt"]
        }
        
        # Initialize default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default AI agents"""
        default_agents = [
            Agent(
                id="jarvis-001",
                name="JARVIS",
                type="jarvis",
                capabilities=["conversation", "planning", "coordination"],
                status=AgentStatus.IDLE,
                metadata={"primary": True}
            ),
            Agent(
                id="letta-001",
                name="Letta (MemGPT)",
                type="letta",
                capabilities=["memory_management", "conversation", "code_generation"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="autogpt-001",
                name="AutoGPT",
                type="autogpt",
                capabilities=["planning", "web_search", "code_generation"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="crewai-001",
                name="CrewAI",
                type="crewai",
                capabilities=["code_generation", "teamwork", "coordination"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="babyagi-001",
                name="BabyAGI",
                type="baby-agi",
                capabilities=["planning", "data_analysis", "task_decomposition"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="researcher-001",
                name="GPT-Researcher",
                type="gpt-researcher",
                capabilities=["web_search", "research", "documentation"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="aider-001",
                name="Aider",
                type="aider",
                capabilities=["debugging", "code_modification", "testing"],
                status=AgentStatus.OFFLINE
            ),
            Agent(
                id="gptpilot-001",
                name="GPT-Pilot",
                type="gpt-pilot",
                capabilities=["debugging", "code_generation", "testing"],
                status=AgentStatus.OFFLINE
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.id] = agent
    
    def start(self):
        """Start orchestration system"""
        if not self.is_running:
            self.is_running = True
            self.orchestration_thread = threading.Thread(
                target=self._orchestration_loop,
                daemon=True
            )
            self.orchestration_thread.start()
            logger.info("Agent orchestrator started")
            return True
        return False
    
    def stop(self):
        """Stop orchestration system"""
        self.is_running = False
        if self.orchestration_thread:
            self.orchestration_thread.join(timeout=2)
            self.orchestration_thread = None
        logger.info("Agent orchestrator stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.is_running:
            try:
                # Process task queue
                if not self.task_queue.empty():
                    priority, task_id = self.task_queue.get(timeout=1)
                    
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        if task.status == TaskStatus.PENDING:
                            self._assign_task(task)
                
                # Check task timeouts
                self._check_task_timeouts()
                
                # Update agent statuses
                self._update_agent_statuses()
                
                # Calculate metrics
                self._update_metrics()
                
                time.sleep(1)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Orchestration error: {e}")
    
    # Agent Management
    
    def register_agent(self, agent: Agent) -> bool:
        """Register a new agent"""
        if agent.id not in self.agents:
            self.agents[agent.id] = agent
            logger.info(f"Agent {agent.name} registered")
            return True
        return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if agent.status == AgentStatus.BUSY:
                # Cancel current task
                if agent.current_task:
                    self.cancel_task(agent.current_task)
            
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")
            return True
        return False
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            
            # Clear current task if going offline
            if status == AgentStatus.OFFLINE:
                agent = self.agents[agent_id]
                if agent.current_task:
                    self.reassign_task(agent.current_task)
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[Agent]:
        """Get agents with specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]
    
    def get_available_agents(self) -> List[Agent]:
        """Get all available agents"""
        return [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
        ]
    
    def select_best_agent(self, task_description: str) -> Optional[str]:
        """Select best agent for a task based on description"""
        # Simple keyword-based selection
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["code", "write", "program", "develop"]):
            capability = "code_generation"
        elif any(word in task_lower for word in ["search", "find", "research"]):
            capability = "web_search"
        elif any(word in task_lower for word in ["analyze", "data", "statistics"]):
            capability = "data_analysis"
        elif any(word in task_lower for word in ["chat", "talk", "conversation"]):
            capability = "conversation"
        else:
            capability = "planning"
        
        suitable_agents = self.get_agents_by_capability(capability)
        available = [a for a in suitable_agents if a.status == AgentStatus.IDLE]
        
        if available:
            # Return the name of the best agent
            return available[0].name
        elif suitable_agents:
            # Return a busy but suitable agent's name
            return suitable_agents[0].name
        
        # Default to JARVIS
        return "JARVIS"
    
    # Task Management
    
    def create_task(self, task_type: str, description: str, 
                   requirements: List[str] = None,
                   priority: TaskPriority = TaskPriority.MEDIUM,
                   callback: Optional[Callable] = None,
                   metadata: Optional[Dict] = None) -> str:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            description=description,
            requirements=requirements or [],
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.metrics["tasks_created"] += 1
        
        # Add to priority queue
        self.task_queue.put((-priority.value, task_id))
        
        # Register callback if provided
        if callback:
            self.result_callbacks[task_id] = callback
        
        logger.info(f"Task {task_id} created: {description}")
        return task_id
    
    def _assign_task(self, task: Task):
        """Assign task to best available agent"""
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(task)
        
        if not suitable_agents:
            logger.warning(f"No suitable agents for task {task.id}")
            return
        
        # Select best agent based on performance and availability
        best_agent = self._select_best_agent(suitable_agents, task)
        
        if best_agent:
            # Assign task
            task.status = TaskStatus.ASSIGNED
            task.assigned_to = best_agent.id
            task.started_at = datetime.now()
            
            best_agent.status = AgentStatus.BUSY
            best_agent.current_task = task.id
            
            logger.info(f"Task {task.id} assigned to {best_agent.name}")
            
            # Execute task asynchronously
            threading.Thread(
                target=self._execute_task,
                args=(task, best_agent),
                daemon=True
            ).start()
    
    def _find_suitable_agents(self, task: Task) -> List[Agent]:
        """Find agents suitable for a task"""
        suitable = []
        
        # Check capability requirements
        for agent in self.agents.values():
            if agent.status in [AgentStatus.IDLE, AgentStatus.BUSY]:
                # Check if agent has required capabilities
                if task.type in self.capability_matrix:
                    required_caps = self.capability_matrix[task.type]
                    if agent.type in required_caps:
                        suitable.append(agent)
                elif any(req in agent.capabilities for req in task.requirements):
                    suitable.append(agent)
        
        return suitable
    
    def _select_best_agent(self, agents: List[Agent], task: Task) -> Optional[Agent]:
        """Select best agent from candidates"""
        # Prioritize idle agents
        idle_agents = [a for a in agents if a.status == AgentStatus.IDLE]
        
        if idle_agents:
            # Sort by performance score
            idle_agents.sort(key=lambda a: a.performance_score, reverse=True)
            return idle_agents[0]
        
        # If no idle agents, queue for later
        return None
    
    def _execute_task(self, task: Task, agent: Agent):
        """Execute task with agent"""
        try:
            task.status = TaskStatus.RUNNING
            logger.info(f"Executing task {task.id} with {agent.name}")
            
            # Simulate task execution (replace with actual agent execution)
            time.sleep(5)  # Simulated work
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = {
                "success": True,
                "output": f"Task completed by {agent.name}",
                "execution_time": (task.completed_at - task.started_at).total_seconds()
            }
            
            # Update agent stats
            agent.tasks_completed += 1
            agent.current_task = None
            agent.status = AgentStatus.IDLE
            
            # Update performance score
            self._update_agent_performance(agent, True)
            
            self.metrics["tasks_completed"] += 1
            logger.info(f"Task {task.id} completed successfully")
            
            # Call callback if registered
            if task.id in self.result_callbacks:
                callback = self.result_callbacks[task.id]
                callback(task.result)
                del self.result_callbacks[task.id]
            
        except Exception as e:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            # Update agent stats
            agent.tasks_failed += 1
            agent.current_task = None
            agent.status = AgentStatus.IDLE
            
            # Update performance score
            self._update_agent_performance(agent, False)
            
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.id} failed: {e}")
            
            # Call callback with error
            if task.id in self.result_callbacks:
                callback = self.result_callbacks[task.id]
                callback({"error": str(e)})
                del self.result_callbacks[task.id]
    
    def _update_agent_performance(self, agent: Agent, success: bool):
        """Update agent performance score"""
        if success:
            agent.performance_score = min(2.0, agent.performance_score * 1.1)
        else:
            agent.performance_score = max(0.1, agent.performance_score * 0.9)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            
            if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                # Free up agent
                if task.assigned_to:
                    agent = self.agents.get(task.assigned_to)
                    if agent:
                        agent.current_task = None
                        agent.status = AgentStatus.IDLE
                
                logger.info(f"Task {task_id} cancelled")
                return True
        return False
    
    def reassign_task(self, task_id: str) -> bool:
        """Reassign a task to another agent"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            
            # Reset task status
            task.status = TaskStatus.PENDING
            task.assigned_to = None
            task.started_at = None
            
            # Add back to queue
            self.task_queue.put((-task.priority.value, task_id))
            
            logger.info(f"Task {task_id} reassigned")
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get tasks by status"""
        return [
            task for task in self.tasks.values()
            if task.status == status
        ]
    
    # Orchestration Strategies
    
    def execute_parallel_tasks(self, tasks: List[Dict]) -> List[str]:
        """Execute multiple tasks in parallel"""
        task_ids = []
        
        for task_config in tasks:
            task_id = self.create_task(
                task_type=task_config.get("type", "general"),
                description=task_config.get("description", ""),
                requirements=task_config.get("requirements", []),
                priority=task_config.get("priority", TaskPriority.MEDIUM),
                metadata=task_config.get("metadata", {})
            )
            task_ids.append(task_id)
        
        return task_ids
    
    def execute_sequential_tasks(self, tasks: List[Dict]) -> str:
        """Execute tasks sequentially"""
        workflow_id = str(uuid.uuid4())
        
        def create_next_task(index: int, previous_result: Any = None):
            if index >= len(tasks):
                return
            
            task_config = tasks[index]
            task_config["metadata"] = task_config.get("metadata", {})
            task_config["metadata"]["workflow_id"] = workflow_id
            task_config["metadata"]["step"] = index + 1
            task_config["metadata"]["previous_result"] = previous_result
            
            def callback(result):
                create_next_task(index + 1, result)
            
            self.create_task(
                task_type=task_config.get("type", "general"),
                description=task_config.get("description", ""),
                requirements=task_config.get("requirements", []),
                priority=task_config.get("priority", TaskPriority.MEDIUM),
                callback=callback,
                metadata=task_config["metadata"]
            )
        
        # Start first task
        create_next_task(0)
        return workflow_id
    
    def execute_collaborative_task(self, description: str, 
                                  agent_types: List[str]) -> str:
        """Execute task with multiple agents collaborating"""
        task_id = str(uuid.uuid4())
        
        # Create subtasks for each agent type
        subtask_ids = []
        for agent_type in agent_types:
            subtask_id = self.create_task(
                task_type=agent_type,
                description=f"[Collaborative] {description}",
                priority=TaskPriority.HIGH,
                metadata={
                    "parent_task": task_id,
                    "collaborative": True
                }
            )
            subtask_ids.append(subtask_id)
        
        # Store collaborative task metadata
        self.tasks[task_id] = Task(
            id=task_id,
            type="collaborative",
            description=description,
            requirements=agent_types,
            priority=TaskPriority.HIGH,
            status=TaskStatus.RUNNING,
            created_at=datetime.now(),
            metadata={"subtasks": subtask_ids}
        )
        
        return task_id
    
    # Monitoring & Metrics
    
    def _check_task_timeouts(self):
        """Check for task timeouts"""
        timeout_threshold = 300  # 5 minutes
        current_time = datetime.now()
        
        for task in self.tasks.values():
            if task.status == TaskStatus.RUNNING and task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                if elapsed > timeout_threshold:
                    logger.warning(f"Task {task.id} timeout")
                    task.status = TaskStatus.FAILED
                    task.error = "Timeout"
                    task.completed_at = current_time
                    
                    # Free up agent
                    if task.assigned_to:
                        agent = self.agents.get(task.assigned_to)
                        if agent:
                            agent.current_task = None
                            agent.status = AgentStatus.IDLE
                            agent.tasks_failed += 1
    
    def _update_agent_statuses(self):
        """Update agent statuses based on health checks"""
        # This would integrate with actual agent health checks
        pass
    
    def _update_metrics(self):
        """Update orchestration metrics"""
        # Calculate average completion time
        completed_tasks = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED and t.started_at and t.completed_at
        ]
        
        if completed_tasks:
            total_time = sum(
                (t.completed_at - t.started_at).total_seconds()
                for t in completed_tasks
            )
            self.metrics["average_completion_time"] = total_time / len(completed_tasks)
        
        # Calculate agent utilization
        for agent in self.agents.values():
            utilization = "busy" if agent.status == AgentStatus.BUSY else "idle"
            self.metrics["agent_utilization"][agent.id] = utilization
    
    def get_metrics(self) -> Dict:
        """Get orchestration metrics"""
        return {
            **self.metrics,
            "active_agents": len([a for a in self.agents.values() 
                                 if a.status != AgentStatus.OFFLINE]),
            "pending_tasks": len([t for t in self.tasks.values() 
                                if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in self.tasks.values() 
                                if t.status == TaskStatus.RUNNING])
        }
    
    def get_summary(self) -> Dict:
        """Get orchestration summary"""
        return {
            "agents": {
                "total": len(self.agents),
                "online": len([a for a in self.agents.values() 
                             if a.status != AgentStatus.OFFLINE]),
                "busy": len([a for a in self.agents.values() 
                           if a.status == AgentStatus.BUSY])
            },
            "tasks": {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() 
                              if t.status == TaskStatus.PENDING]),
                "running": len([t for t in self.tasks.values() 
                              if t.status == TaskStatus.RUNNING]),
                "completed": len([t for t in self.tasks.values() 
                                if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in self.tasks.values() 
                             if t.status == TaskStatus.FAILED])
            },
            "metrics": self.get_metrics()
        }
    
    def cleanup(self):
        """Cleanup orchestrator"""
        self.stop()
        self.agents.clear()
        self.tasks.clear()
        self.result_callbacks.clear()