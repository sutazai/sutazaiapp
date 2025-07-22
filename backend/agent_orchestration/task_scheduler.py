#!/usr/bin/env python3
"""
Task Scheduler - Advanced task scheduling and management for agent orchestration
"""

import asyncio
import logging
import threading
import uuid
import heapq
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class SchedulingStrategy(Enum):
    """Scheduling strategies"""
    FIFO = "fifo"
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    DEADLINE_FIRST = "deadline_first"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"

@dataclass
class Task:
    """Task definition"""
    task_id: str
    task_type: str
    priority: TaskPriority
    agent_requirements: Dict[str, Any]
    task_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    deadline: Optional[datetime] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskQueue:
    """Task queue with priority support"""
    queue_id: str
    priority_queues: Dict[TaskPriority, List[Task]] = field(default_factory=lambda: {
        TaskPriority.CRITICAL: [],
        TaskPriority.HIGH: [],
        TaskPriority.NORMAL: [],
        TaskPriority.LOW: [],
        TaskPriority.BACKGROUND: []
    })
    max_size: int = 10000
    current_size: int = 0
    strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SchedulerConfig:
    """Configuration for task scheduler"""
    max_concurrent_tasks: int = 100
    max_queue_size: int = 10000
    default_strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY
    enable_task_dependencies: bool = True
    enable_task_retries: bool = True
    enable_deadline_monitoring: bool = True
    enable_load_balancing: bool = True
    enable_task_persistence: bool = True
    scheduling_interval: float = 1.0
    heartbeat_interval: float = 5.0
    cleanup_interval: float = 300.0  # 5 minutes
    task_timeout: float = 300.0  # 5 minutes
    max_task_history: int = 10000
    enable_adaptive_scheduling: bool = True

class TaskScheduler:
    """Advanced task scheduler for agent orchestration"""
    
    def __init__(self, config: SchedulerConfig = None, agent_manager=None, resource_manager=None):
        self.config = config or SchedulerConfig()
        self.agent_manager = agent_manager
        self.resource_manager = resource_manager
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queues: Dict[str, TaskQueue] = {}
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Scheduling
        self.scheduling_strategy = self.config.default_strategy
        self.task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Agent assignment
        self.agent_task_assignments: Dict[str, str] = {}  # agent_id -> task_id
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self.agent_loads: Dict[str, int] = defaultdict(int)
        
        # Performance metrics
        self.scheduler_metrics = {
            "total_tasks": 0,
            "scheduled_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_queue_time": 0.0,
            "average_execution_time": 0.0,
            "throughput": 0.0,
            "agent_utilization": 0.0
        }
        
        # Priority queue for deadline monitoring
        self.deadline_queue: List[Tuple[datetime, str]] = []
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._scheduler_task = None
        self._monitor_task = None
        self._cleanup_task = None
        
        # Default queue
        self.default_queue = TaskQueue(queue_id="default")
        self.task_queues["default"] = self.default_queue
        
        logger.info("Task scheduler initialized")
    
    async def initialize(self) -> bool:
        """Initialize task scheduler"""
        try:
            # Start background tasks
            self._start_scheduler()
            self._start_monitor()
            self._start_cleanup()
            
            logger.info("Task scheduler initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Task scheduler initialization failed: {e}")
            return False
    
    def _start_scheduler(self):
        """Start scheduling background task"""
        def scheduler_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._schedule_tasks())
                    self._shutdown_event.wait(self.config.scheduling_interval)
                except Exception as e:
                    logger.error(f"Scheduling error: {e}")
                    self._shutdown_event.wait(5)
        
        self._scheduler_task = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_task.start()
    
    def _start_monitor(self):
        """Start monitoring background task"""
        def monitor_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._monitor_tasks())
                    self._shutdown_event.wait(self.config.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    self._shutdown_event.wait(10)
        
        self._monitor_task = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_task.start()
    
    def _start_cleanup(self):
        """Start cleanup background task"""
        def cleanup_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._cleanup_tasks())
                    self._shutdown_event.wait(self.config.cleanup_interval)
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    self._shutdown_event.wait(60)
        
        self._cleanup_task = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_task.start()
    
    async def schedule_task(self, task_id: str, task_definition: Dict[str, Any]) -> bool:
        """Schedule a task"""
        try:
            with self._lock:
                # Create task
                task = Task(
                    task_id=task_id,
                    task_type=task_definition.get("type", "default"),
                    priority=TaskPriority(task_definition.get("priority", TaskPriority.NORMAL.value)),
                    agent_requirements=task_definition.get("agent_requirements", {}),
                    task_data=task_definition.get("data", {}),
                    dependencies=task_definition.get("dependencies", []),
                    estimated_duration=task_definition.get("estimated_duration"),
                    deadline=task_definition.get("deadline"),
                    max_retries=task_definition.get("max_retries", 3),
                    retry_delay=task_definition.get("retry_delay", 1.0),
                    metadata=task_definition.get("metadata", {})
                )
                
                # Validate task
                if not self._validate_task(task):
                    logger.warning(f"Task validation failed: {task_id}")
                    return False
                
                # Store task
                self.tasks[task_id] = task
                self.scheduler_metrics["total_tasks"] += 1
                
                # Add to appropriate queue
                queue_id = task_definition.get("queue", "default")
                if queue_id not in self.task_queues:
                    self.task_queues[queue_id] = TaskQueue(queue_id=queue_id)
                
                queue = self.task_queues[queue_id]
                
                # Check queue capacity
                if queue.current_size >= queue.max_size:
                    logger.warning(f"Queue full: {queue_id}")
                    return False
                
                # Add to priority queue
                queue.priority_queues[task.priority].append(task)
                queue.current_size += 1
                
                # Handle dependencies
                if self.config.enable_task_dependencies and task.dependencies:
                    self._add_task_dependencies(task_id, task.dependencies)
                
                # Add to deadline monitoring
                if self.config.enable_deadline_monitoring and task.deadline:
                    heapq.heappush(self.deadline_queue, (task.deadline, task_id))
                
                task.status = TaskStatus.SCHEDULED
                task.scheduled_at = datetime.now(timezone.utc)
                
                logger.info(f"Task scheduled: {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Task scheduling failed: {e}")
            return False
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return False
                
                task = self.tasks[task_id]
                
                # Cancel running task
                if task.status == TaskStatus.RUNNING:
                    if task.assigned_agent:
                        # Signal agent to cancel task
                        await self._cancel_agent_task(task.assigned_agent, task_id)
                        
                        # Remove from agent assignments
                        if task.assigned_agent in self.agent_task_assignments:
                            del self.agent_task_assignments[task.assigned_agent]
                        
                        # Update agent load
                        self.agent_loads[task.assigned_agent] -= 1
                
                # Update task status
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now(timezone.utc)
                
                # Remove from queues
                self._remove_task_from_queues(task)
                
                # Remove from running tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                logger.info(f"Task cancelled: {task_id}")
                return True
                
        except Exception as e:
            logger.error(f"Task cancellation failed: {e}")
            return False
    
    async def _schedule_tasks(self):
        """Main scheduling loop"""
        try:
            with self._lock:
                # Get available agents
                available_agents = await self._get_available_agents()
                
                if not available_agents:
                    return
                
                # Schedule tasks from queues
                for queue_id, queue in self.task_queues.items():
                    scheduled_count = await self._schedule_from_queue(queue, available_agents)
                    if scheduled_count > 0:
                        self.scheduler_metrics["scheduled_tasks"] += scheduled_count
                
                # Update metrics
                self._update_scheduler_metrics()
                
        except Exception as e:
            logger.error(f"Task scheduling loop failed: {e}")
    
    async def _schedule_from_queue(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks from a specific queue"""
        scheduled_count = 0
        
        try:
            # Schedule based on strategy
            if queue.strategy == SchedulingStrategy.PRIORITY:
                scheduled_count = await self._schedule_priority(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.FIFO:
                scheduled_count = await self._schedule_fifo(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.ROUND_ROBIN:
                scheduled_count = await self._schedule_round_robin(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.SHORTEST_JOB_FIRST:
                scheduled_count = await self._schedule_shortest_job_first(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.DEADLINE_FIRST:
                scheduled_count = await self._schedule_deadline_first(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.LOAD_BALANCED:
                scheduled_count = await self._schedule_load_balanced(queue, available_agents)
            elif queue.strategy == SchedulingStrategy.ADAPTIVE:
                scheduled_count = await self._schedule_adaptive(queue, available_agents)
            
            return scheduled_count
            
        except Exception as e:
            logger.error(f"Queue scheduling failed: {e}")
            return 0
    
    async def _schedule_priority(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks by priority"""
        scheduled_count = 0
        
        # Process tasks in priority order
        for priority in TaskPriority:
            if priority not in queue.priority_queues:
                continue
                
            priority_queue = queue.priority_queues[priority]
            
            # Process tasks in this priority level
            remaining_tasks = []
            for task in priority_queue:
                if not available_agents:
                    remaining_tasks.append(task)
                    continue
                
                # Check dependencies
                if not self._are_dependencies_satisfied(task.task_id):
                    remaining_tasks.append(task)
                    continue
                
                # Find suitable agent
                agent_id = await self._find_suitable_agent(task, available_agents)
                if agent_id:
                    # Assign task to agent
                    if await self._assign_task_to_agent(task, agent_id):
                        available_agents.remove(agent_id)
                        scheduled_count += 1
                        queue.current_size -= 1
                    else:
                        remaining_tasks.append(task)
                else:
                    remaining_tasks.append(task)
            
            # Update priority queue
            queue.priority_queues[priority] = remaining_tasks
        
        return scheduled_count
    
    async def _schedule_fifo(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks in FIFO order"""
        scheduled_count = 0
        
        # Flatten all priority queues into FIFO order
        all_tasks = []
        for priority in TaskPriority:
            if priority in queue.priority_queues:
                all_tasks.extend(queue.priority_queues[priority])
        
        # Sort by creation time
        all_tasks.sort(key=lambda t: t.created_at)
        
        # Schedule tasks
        remaining_tasks = []
        for task in all_tasks:
            if not available_agents:
                remaining_tasks.append(task)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task.task_id):
                remaining_tasks.append(task)
                continue
            
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task, available_agents)
            if agent_id:
                if await self._assign_task_to_agent(task, agent_id):
                    available_agents.remove(agent_id)
                    scheduled_count += 1
                    queue.current_size -= 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Redistribute remaining tasks back to priority queues
        self._redistribute_tasks(queue, remaining_tasks)
        
        return scheduled_count
    
    async def _schedule_round_robin(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks in round-robin order across agents"""
        scheduled_count = 0
        agent_index = 0
        
        # Get all tasks
        all_tasks = []
        for priority in TaskPriority:
            if priority in queue.priority_queues:
                all_tasks.extend(queue.priority_queues[priority])
        
        # Sort by priority then creation time
        all_tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        
        # Schedule tasks in round-robin
        remaining_tasks = []
        for task in all_tasks:
            if not available_agents:
                remaining_tasks.append(task)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task.task_id):
                remaining_tasks.append(task)
                continue
            
            # Round-robin agent selection
            agent_id = available_agents[agent_index % len(available_agents)]
            
            # Check if agent is suitable
            if await self._is_agent_suitable(task, agent_id):
                if await self._assign_task_to_agent(task, agent_id):
                    available_agents.remove(agent_id)
                    scheduled_count += 1
                    queue.current_size -= 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
            
            agent_index += 1
        
        # Redistribute remaining tasks
        self._redistribute_tasks(queue, remaining_tasks)
        
        return scheduled_count
    
    async def _schedule_shortest_job_first(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks by estimated duration (shortest first)"""
        scheduled_count = 0
        
        # Get all tasks with estimated duration
        all_tasks = []
        for priority in TaskPriority:
            if priority in queue.priority_queues:
                all_tasks.extend(queue.priority_queues[priority])
        
        # Sort by estimated duration
        tasks_with_duration = [t for t in all_tasks if t.estimated_duration is not None]
        tasks_without_duration = [t for t in all_tasks if t.estimated_duration is None]
        
        tasks_with_duration.sort(key=lambda t: t.estimated_duration)
        
        # Schedule tasks with duration first
        remaining_tasks = []
        for task in tasks_with_duration + tasks_without_duration:
            if not available_agents:
                remaining_tasks.append(task)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task.task_id):
                remaining_tasks.append(task)
                continue
            
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task, available_agents)
            if agent_id:
                if await self._assign_task_to_agent(task, agent_id):
                    available_agents.remove(agent_id)
                    scheduled_count += 1
                    queue.current_size -= 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Redistribute remaining tasks
        self._redistribute_tasks(queue, remaining_tasks)
        
        return scheduled_count
    
    async def _schedule_deadline_first(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks by deadline (earliest first)"""
        scheduled_count = 0
        
        # Get all tasks with deadlines
        all_tasks = []
        for priority in TaskPriority:
            if priority in queue.priority_queues:
                all_tasks.extend(queue.priority_queues[priority])
        
        # Sort by deadline
        tasks_with_deadline = [t for t in all_tasks if t.deadline is not None]
        tasks_without_deadline = [t for t in all_tasks if t.deadline is None]
        
        tasks_with_deadline.sort(key=lambda t: t.deadline)
        
        # Schedule tasks with deadline first
        remaining_tasks = []
        for task in tasks_with_deadline + tasks_without_deadline:
            if not available_agents:
                remaining_tasks.append(task)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task.task_id):
                remaining_tasks.append(task)
                continue
            
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task, available_agents)
            if agent_id:
                if await self._assign_task_to_agent(task, agent_id):
                    available_agents.remove(agent_id)
                    scheduled_count += 1
                    queue.current_size -= 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Redistribute remaining tasks
        self._redistribute_tasks(queue, remaining_tasks)
        
        return scheduled_count
    
    async def _schedule_load_balanced(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Schedule tasks with load balancing"""
        scheduled_count = 0
        
        # Get all tasks
        all_tasks = []
        for priority in TaskPriority:
            if priority in queue.priority_queues:
                all_tasks.extend(queue.priority_queues[priority])
        
        # Sort by priority
        all_tasks.sort(key=lambda t: t.priority.value)
        
        # Schedule tasks with load balancing
        remaining_tasks = []
        for task in all_tasks:
            if not available_agents:
                remaining_tasks.append(task)
                continue
            
            # Check dependencies
            if not self._are_dependencies_satisfied(task.task_id):
                remaining_tasks.append(task)
                continue
            
            # Find least loaded suitable agent
            agent_id = await self._find_least_loaded_agent(task, available_agents)
            if agent_id:
                if await self._assign_task_to_agent(task, agent_id):
                    available_agents.remove(agent_id)
                    scheduled_count += 1
                    queue.current_size -= 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        # Redistribute remaining tasks
        self._redistribute_tasks(queue, remaining_tasks)
        
        return scheduled_count
    
    async def _schedule_adaptive(self, queue: TaskQueue, available_agents: List[str]) -> int:
        """Adaptive scheduling based on system state"""
        # This would implement adaptive scheduling based on:
        # - Current system load
        # - Historical performance
        # - Agent capabilities
        # - Task characteristics
        
        # For now, fall back to priority scheduling
        return await self._schedule_priority(queue, available_agents)
    
    async def _get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        try:
            if not self.agent_manager:
                return []
            
            # Get active agents from agent manager
            active_agents = self.agent_manager.get_active_agents()
            
            # Filter out agents that are at capacity
            available_agents = []
            for agent_id in active_agents:
                agent_load = self.agent_loads.get(agent_id, 0)
                if agent_load < self.config.max_concurrent_tasks:
                    available_agents.append(agent_id)
            
            return available_agents
            
        except Exception as e:
            logger.error(f"Failed to get available agents: {e}")
            return []
    
    async def _find_suitable_agent(self, task: Task, available_agents: List[str]) -> Optional[str]:
        """Find suitable agent for task"""
        try:
            # Check agent requirements
            for agent_id in available_agents:
                if await self._is_agent_suitable(task, agent_id):
                    return agent_id
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find suitable agent: {e}")
            return None
    
    async def _find_least_loaded_agent(self, task: Task, available_agents: List[str]) -> Optional[str]:
        """Find least loaded suitable agent"""
        try:
            suitable_agents = []
            
            # Find all suitable agents
            for agent_id in available_agents:
                if await self._is_agent_suitable(task, agent_id):
                    suitable_agents.append(agent_id)
            
            if not suitable_agents:
                return None
            
            # Return least loaded agent
            return min(suitable_agents, key=lambda a: self.agent_loads.get(a, 0))
            
        except Exception as e:
            logger.error(f"Failed to find least loaded agent: {e}")
            return None
    
    async def _is_agent_suitable(self, task: Task, agent_id: str) -> bool:
        """Check if agent is suitable for task"""
        try:
            # Check agent capabilities
            agent_capabilities = self.agent_capabilities.get(agent_id, {})
            
            # Check requirements
            for requirement, value in task.agent_requirements.items():
                if requirement not in agent_capabilities:
                    return False
                
                # Simple requirement matching
                if agent_capabilities[requirement] != value:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent suitability check failed: {e}")
            return False
    
    async def _assign_task_to_agent(self, task: Task, agent_id: str) -> bool:
        """Assign task to agent"""
        try:
            # Update task
            task.assigned_agent = agent_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            
            # Update agent assignments
            self.agent_task_assignments[agent_id] = task.task_id
            self.agent_loads[agent_id] += 1
            
            # Move to running tasks
            self.running_tasks[task.task_id] = task
            
            # Execute task on agent
            if self.agent_manager:
                # This would integrate with the agent manager
                await self._execute_task_on_agent(task, agent_id)
            
            logger.info(f"Task assigned to agent: {task.task_id} -> {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            return False
    
    async def _execute_task_on_agent(self, task: Task, agent_id: str):
        """Execute task on agent"""
        try:
            # This would integrate with the agent manager
            # For now, simulate task execution
            logger.info(f"Executing task {task.task_id} on agent {agent_id}")
            
            # Simulate task completion after delay
            asyncio.create_task(self._simulate_task_completion(task))
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _simulate_task_completion(self, task: Task):
        """Simulate task completion"""
        try:
            # Simulate execution time
            execution_time = task.estimated_duration or 1.0
            await asyncio.sleep(execution_time)
            
            # Complete task
            await self._complete_task(task.task_id, {"status": "completed"})
            
        except Exception as e:
            logger.error(f"Task simulation failed: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _complete_task(self, task_id: str, result: Dict[str, Any]):
        """Complete a task"""
        try:
            with self._lock:
                if task_id not in self.tasks:
                    return
                
                task = self.tasks[task_id]
                
                # Update task
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                task.result = result
                
                # Update agent assignments
                if task.assigned_agent:
                    if task.assigned_agent in self.agent_task_assignments:
                        del self.agent_task_assignments[task.assigned_agent]
                    self.agent_loads[task.assigned_agent] -= 1
                
                # Move to completed tasks
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                self.completed_tasks[task_id] = task
                
                # Update metrics
                self.scheduler_metrics["completed_tasks"] += 1
                
                # Check dependent tasks
                await self._check_dependent_tasks(task_id)
                
                logger.info(f"Task completed: {task_id}")
                
        except Exception as e:
            logger.error(f"Task completion failed: {e}")
    
    async def _handle_task_failure(self, task: Task, error: str):
        """Handle task failure"""
        try:
            with self._lock:
                # Update task
                task.error = error
                task.retry_count += 1
                
                # Check if we should retry
                if task.retry_count < task.max_retries:
                    # Retry task
                    task.status = TaskStatus.RETRYING
                    task.assigned_agent = None
                    
                    # Add back to queue after delay
                    await asyncio.sleep(task.retry_delay)
                    
                    # Find appropriate queue
                    queue_id = task.metadata.get("queue", "default")
                    if queue_id in self.task_queues:
                        queue = self.task_queues[queue_id]
                        queue.priority_queues[task.priority].append(task)
                        queue.current_size += 1
                        task.status = TaskStatus.SCHEDULED
                    
                    logger.info(f"Task retrying: {task.task_id} (attempt {task.retry_count})")
                else:
                    # Max retries reached
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now(timezone.utc)
                    
                    # Move to failed tasks
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.failed_tasks[task.task_id] = task
                    
                    # Update metrics
                    self.scheduler_metrics["failed_tasks"] += 1
                    
                    logger.error(f"Task failed: {task.task_id} - {error}")
                
                # Update agent assignments
                if task.assigned_agent:
                    if task.assigned_agent in self.agent_task_assignments:
                        del self.agent_task_assignments[task.assigned_agent]
                    self.agent_loads[task.assigned_agent] -= 1
                
        except Exception as e:
            logger.error(f"Task failure handling failed: {e}")
    
    def _validate_task(self, task: Task) -> bool:
        """Validate task"""
        try:
            # Check required fields
            if not task.task_id or not task.task_type:
                return False
            
            # Check deadline
            if task.deadline and task.deadline < datetime.now(timezone.utc):
                return False
            
            # Check circular dependencies
            if task.dependencies and task.task_id in task.dependencies:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Task validation failed: {e}")
            return False
    
    def _add_task_dependencies(self, task_id: str, dependencies: List[str]):
        """Add task dependencies"""
        try:
            for dep_id in dependencies:
                self.task_dependencies[task_id].add(dep_id)
                self.dependency_graph[dep_id].add(task_id)
                
        except Exception as e:
            logger.error(f"Adding task dependencies failed: {e}")
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        try:
            if task_id not in self.task_dependencies:
                return True
            
            for dep_id in self.task_dependencies[task_id]:
                if dep_id not in self.completed_tasks:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return False
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check and potentially schedule dependent tasks"""
        try:
            if completed_task_id not in self.dependency_graph:
                return
            
            # Check all dependent tasks
            for dependent_task_id in self.dependency_graph[completed_task_id]:
                if self._are_dependencies_satisfied(dependent_task_id):
                    # Dependencies satisfied, task can be scheduled
                    logger.info(f"Dependencies satisfied for task: {dependent_task_id}")
                    
        except Exception as e:
            logger.error(f"Dependent task check failed: {e}")
    
    def _remove_task_from_queues(self, task: Task):
        """Remove task from all queues"""
        try:
            for queue in self.task_queues.values():
                if task in queue.priority_queues[task.priority]:
                    queue.priority_queues[task.priority].remove(task)
                    queue.current_size -= 1
                    
        except Exception as e:
            logger.error(f"Task removal from queues failed: {e}")
    
    def _redistribute_tasks(self, queue: TaskQueue, tasks: List[Task]):
        """Redistribute tasks back to priority queues"""
        try:
            # Clear existing queues
            for priority in TaskPriority:
                queue.priority_queues[priority].clear()
            
            # Redistribute tasks
            for task in tasks:
                queue.priority_queues[task.priority].append(task)
                
        except Exception as e:
            logger.error(f"Task redistribution failed: {e}")
    
    async def _cancel_agent_task(self, agent_id: str, task_id: str):
        """Cancel task on agent"""
        try:
            # This would integrate with the agent manager
            logger.info(f"Cancelling task {task_id} on agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Agent task cancellation failed: {e}")
    
    async def _monitor_tasks(self):
        """Monitor running tasks"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Check for timed out tasks
            timed_out_tasks = []
            for task_id, task in self.running_tasks.items():
                if task.started_at:
                    elapsed = (current_time - task.started_at).total_seconds()
                    if elapsed > self.config.task_timeout:
                        timed_out_tasks.append(task_id)
            
            # Handle timed out tasks
            for task_id in timed_out_tasks:
                task = self.running_tasks[task_id]
                await self._handle_task_failure(task, "Task timeout")
            
            # Check deadlines
            while self.deadline_queue and self.deadline_queue[0][0] < current_time:
                deadline, task_id = heapq.heappop(self.deadline_queue)
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        logger.warning(f"Task deadline missed: {task_id}")
                        # Optionally increase priority or take other action
                        
        except Exception as e:
            logger.error(f"Task monitoring failed: {e}")
    
    async def _cleanup_tasks(self):
        """Clean up old completed and failed tasks"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Clean up completed tasks
            old_completed = []
            for task_id, task in self.completed_tasks.items():
                if task.completed_at:
                    elapsed = (current_time - task.completed_at).total_seconds()
                    if elapsed > 3600:  # 1 hour
                        old_completed.append(task_id)
            
            for task_id in old_completed[:max(0, len(self.completed_tasks) - self.config.max_task_history)]:
                del self.completed_tasks[task_id]
                if task_id in self.tasks:
                    del self.tasks[task_id]
            
            # Clean up failed tasks
            old_failed = []
            for task_id, task in self.failed_tasks.items():
                if task.completed_at:
                    elapsed = (current_time - task.completed_at).total_seconds()
                    if elapsed > 3600:  # 1 hour
                        old_failed.append(task_id)
            
            for task_id in old_failed[:max(0, len(self.failed_tasks) - self.config.max_task_history)]:
                del self.failed_tasks[task_id]
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    
        except Exception as e:
            logger.error(f"Task cleanup failed: {e}")
    
    def _update_scheduler_metrics(self):
        """Update scheduler metrics"""
        try:
            total_running = len(self.running_tasks)
            total_completed = len(self.completed_tasks)
            total_failed = len(self.failed_tasks)
            
            self.scheduler_metrics.update({
                "total_tasks": len(self.tasks),
                "running_tasks": total_running,
                "completed_tasks": total_completed,
                "failed_tasks": total_failed,
                "agent_utilization": len(self.agent_task_assignments) / max(1, len(self.agent_capabilities))
            })
            
            # Calculate averages
            if total_completed > 0:
                total_execution_time = sum(
                    (task.completed_at - task.started_at).total_seconds()
                    for task in self.completed_tasks.values()
                    if task.started_at and task.completed_at
                )
                self.scheduler_metrics["average_execution_time"] = total_execution_time / total_completed
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    async def start(self) -> bool:
        """Start task scheduler"""
        try:
            logger.info("Starting task scheduler...")
            return True
        except Exception as e:
            logger.error(f"Failed to start task scheduler: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop task scheduler"""
        try:
            logger.info("Stopping task scheduler...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel all running tasks
            for task_id in list(self.running_tasks.keys()):
                await self.cancel_task(task_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop task scheduler: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check scheduler health"""
        try:
            return (
                len(self.running_tasks) <= self.config.max_concurrent_tasks and
                sum(queue.current_size for queue in self.task_queues.values()) <= self.config.max_queue_size
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        with self._lock:
            return {
                "total_tasks": len(self.tasks),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "queued_tasks": sum(queue.current_size for queue in self.task_queues.values()),
                "agent_assignments": len(self.agent_task_assignments),
                "metrics": self.scheduler_metrics,
                "config": {
                    "max_concurrent_tasks": self.config.max_concurrent_tasks,
                    "default_strategy": self.config.default_strategy.value,
                    "scheduling_interval": self.config.scheduling_interval
                }
            }
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task information"""
        with self._lock:
            task = self.tasks.get(task_id)
            if task:
                return {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "assigned_agent": task.assigned_agent,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "retry_count": task.retry_count,
                    "error": task.error,
                    "result": task.result
                }
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        with self._lock:
            return self.scheduler_metrics.copy()

# Factory function
def create_task_scheduler(config: Optional[Dict[str, Any]] = None) -> TaskScheduler:
    """Create task scheduler instance"""
    if config:
        scheduler_config = SchedulerConfig(**config)
    else:
        scheduler_config = SchedulerConfig()
    
    return TaskScheduler(config=scheduler_config)