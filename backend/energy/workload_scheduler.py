"""
Energy-Aware Workload Scheduler - Intelligent task scheduling for power optimization
"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from app.schemas.message_types import TaskPriority
import heapq
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class SchedulingPolicy(Enum):
    """Scheduling policies for energy optimization"""
    ENERGY_FIRST = "energy_first"         # Minimize energy consumption
    PERFORMANCE_FIRST = "performance_first" # Maximize performance
    BALANCED = "balanced"                  # Balance energy and performance
    CARBON_AWARE = "carbon_aware"         # Schedule based on grid carbon intensity
    THERMAL_AWARE = "thermal_aware"       # Consider thermal constraints

# Canonical TaskPriority imported from app.schemas.message_types

@dataclass
class WorkloadMetrics:
    """Metrics for workload scheduling decisions"""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    temperature: float = 0.0
    power_consumption: float = 0.0
    carbon_intensity: float = 0.4  # kg CO2/kWh
    active_agents: int = 0
    pending_tasks: int = 0
    avg_response_time: float = 0.0

@dataclass
class Task:
    """Task definition for scheduling"""
    task_id: str
    agent_id: str
    task_type: str
    priority: TaskPriority
    estimated_duration: float  # seconds
    estimated_cpu_usage: float  # percentage
    estimated_memory_mb: float
    estimated_power_w: float
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    carbon_budget: Optional[float] = None  # grams CO2

@dataclass
class SchedulingSlot:
    """Time slot for scheduling tasks"""
    start_time: datetime
    end_time: datetime
    agent_id: str
    estimated_power: float
    estimated_cpu: float
    estimated_memory: float
    carbon_intensity: float
    available: bool = True

class EnergyOptimizedQueue:
    """Priority queue optimized for energy-aware scheduling"""
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.BALANCED):
        self.policy = policy
        self._heap: List[tuple] = []
        self._task_map: Dict[str, Task] = {}
        self._lock = threading.Lock()
    
    def add_task(self, task: Task, current_metrics: WorkloadMetrics) -> None:
        """Add a task to the queue with energy-aware priority"""
        priority_score = self._calculate_priority_score(task, current_metrics)
        
        with self._lock:
            # Use negative score for min-heap (higher score = higher priority)
            heapq.heappush(self._heap, (-priority_score, task.created_at, task.task_id))
            self._task_map[task.task_id] = task
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task based on energy-aware priority"""
        with self._lock:
            while self._heap:
                _, _, task_id = heapq.heappop(self._heap)
                if task_id in self._task_map:
                    task = self._task_map.pop(task_id)
                    return task
        return None
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """Remove a task from the queue"""
        with self._lock:
            return self._task_map.pop(task_id, None)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks"""
        with self._lock:
            return list(self._task_map.values())
    
    def size(self) -> int:
        """Get the number of pending tasks"""
        with self._lock:
            return len(self._task_map)
    
    def _calculate_priority_score(self, task: Task, metrics: WorkloadMetrics) -> float:
        """Calculate energy-aware priority score for a task"""
        base_score = TaskPriority.from_value(task.priority).rank * 100
        
        if self.policy == SchedulingPolicy.ENERGY_FIRST:
            # Prioritize tasks that consume less energy
            energy_penalty = task.estimated_power_w * task.estimated_duration / 3600  # Wh
            score = base_score - energy_penalty * 10
            
        elif self.policy == SchedulingPolicy.PERFORMANCE_FIRST:
            # Prioritize based on urgency and resource requirements
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds()
                urgency_bonus = max(0, 100 - time_to_deadline / 3600)  # Bonus for urgent tasks
                score = base_score + urgency_bonus
            else:
                score = base_score
                
        elif self.policy == SchedulingPolicy.CARBON_AWARE:
            # Schedule when carbon intensity is lower
            carbon_cost = task.estimated_power_w * task.estimated_duration / 3600 * metrics.carbon_intensity
            carbon_penalty = carbon_cost * 100  # Scale penalty
            score = base_score - carbon_penalty
            
        elif self.policy == SchedulingPolicy.THERMAL_AWARE:
            # Consider thermal impact
            if metrics.temperature > 70:  # High temperature
                thermal_penalty = task.estimated_cpu_usage * 2
                score = base_score - thermal_penalty
            else:
                score = base_score
                
        else:  # BALANCED
            # Balance multiple factors
            energy_factor = -(task.estimated_power_w * 2)
            urgency_factor = 0
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds()
                urgency_factor = max(0, 50 - time_to_deadline / 3600)
            
            thermal_factor = -max(0, metrics.temperature - 60) * 0.5
            carbon_factor = -(task.estimated_power_w * metrics.carbon_intensity)
            
            score = base_score + energy_factor + urgency_factor + thermal_factor + carbon_factor
        
        return score

class EnergyAwareScheduler:
    """Main energy-aware workload scheduler"""
    
    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.BALANCED, agent_manager=None):
        """
        Initialize the energy-aware scheduler
        
        Args:
            policy: Scheduling policy to use
            agent_manager: Reference to the main agent manager
        """
        self.policy = policy
        self.agent_manager = agent_manager
        
        self._task_queue = EnergyOptimizedQueue(policy)
        self._running_tasks: Dict[str, Task] = {}
        self._completed_tasks: List[Task] = []
        self._scheduling_history: List[Dict[str, Any]] = []
        
        self._scheduling = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Scheduling parameters
        self._scheduling_interval = 10  # seconds
        self._max_concurrent_tasks_per_agent = 2
        self._energy_budget_wh = 1000.0  # Daily energy budget in Wh
        self._carbon_budget_g = 400.0    # Daily carbon budget in grams
        
        # Current metrics
        self._current_metrics = WorkloadMetrics()
        
        # Energy and carbon tracking
        self._daily_energy_consumed = 0.0
        self._daily_carbon_emitted = 0.0
        self._last_reset = datetime.now().date()
    
    def start_scheduling(self) -> None:
        """Start the energy-aware scheduler"""
        if self._scheduling:
            return
        
        self._scheduling = True
        self._scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self._metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        
        self._scheduler_thread.start()
        self._metrics_thread.start()
        
        logger.info(f"Energy-aware scheduler started with {self.policy.value} policy")
    
    def stop_scheduling(self) -> None:
        """Stop the energy-aware scheduler"""
        self._scheduling = False
        
        if self._scheduler_thread:
            self._scheduler_thread.join()
        if self._metrics_thread:
            self._metrics_thread.join()
        
        logger.info("Energy-aware scheduler stopped")
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task for energy-aware scheduling"""
        # Validate task
        if not self._validate_task(task):
            logger.error(f"Task validation failed: {task.task_id}")
            return False
        
        # Check resource constraints
        if not self._check_resource_constraints(task):
            logger.warning(f"Task {task.task_id} exceeds resource constraints")
            return False
        
        # Add to queue
        self._task_queue.add_task(task, self._current_metrics)
        logger.info(f"Task {task.task_id} submitted for scheduling")
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        # Try to remove from queue first
        task = self._task_queue.remove_task(task_id)
        if task:
            logger.info(f"Cancelled pending task: {task_id}")
            return True
        
        # Check running tasks
        with self._lock:
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                # In a real implementation, you would stop the task execution
                logger.info(f"Cancelled running task: {task_id}")
                return True
        
        return False
    
    def _scheduling_loop(self) -> None:
        """Main scheduling loop"""
        while self._scheduling:
            try:
                self._reset_daily_budgets_if_needed()
                self._schedule_next_tasks()
                self._check_running_tasks()
                self._update_scheduling_metrics()
                
                time.sleep(self._scheduling_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(self._scheduling_interval)
    
    def _metrics_loop(self) -> None:
        """Metrics collection loop"""
        while self._scheduling:
            try:
                self._update_current_metrics()
                time.sleep(5)  # Update metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                time.sleep(5)
    
    def _schedule_next_tasks(self) -> None:
        """Schedule the next batch of tasks"""
        if not self.agent_manager:
            return
        
        try:
            # Get available agents
            active_agents = self.agent_manager.get_active_agents()
            
            # Calculate available capacity per agent
            agent_capacity = {}
            for agent_id in active_agents:
                with self._lock:
                    running_count = sum(1 for task in self._running_tasks.values() 
                                      if task.agent_id == agent_id)
                agent_capacity[agent_id] = max(0, self._max_concurrent_tasks_per_agent - running_count)
            
            # Schedule tasks while respecting constraints
            scheduled_tasks = []
            while self._task_queue.size() > 0:
                task = self._task_queue.get_next_task()
                if not task:
                    break
                
                # Check if agent has capacity
                if agent_capacity.get(task.agent_id, 0) <= 0:
                    # Try to find another suitable agent
                    alternative_agent = self._find_alternative_agent(task, agent_capacity)
                    if alternative_agent:
                        task.agent_id = alternative_agent
                    else:
                        # Put task back in queue for later
                        self._task_queue.add_task(task, self._current_metrics)
                        break
                
                # Check energy and carbon budgets
                if not self._check_budgets(task):
                    # Defer task if budgets are exceeded
                    self._task_queue.add_task(task, self._current_metrics)
                    break
                
                # Schedule the task
                if self._schedule_task(task):
                    scheduled_tasks.append(task)
                    agent_capacity[task.agent_id] -= 1
                else:
                    logger.warning(f"Failed to schedule task: {task.task_id}")
            
            if scheduled_tasks:
                logger.info(f"Scheduled {len(scheduled_tasks)} tasks")
                
        except Exception as e:
            logger.error(f"Error scheduling tasks: {e}")
    
    def _schedule_task(self, task: Task) -> bool:
        """Schedule a single task"""
        try:
            if not self.agent_manager:
                return False
            
            # Execute the task
            task.scheduled_at = datetime.now()
            task.started_at = datetime.now()
            
            with self._lock:
                self._running_tasks[task.task_id] = task
            
            # Update budgets
            estimated_energy = task.estimated_power_w * task.estimated_duration / 3600
            estimated_carbon = estimated_energy * self._current_metrics.carbon_intensity
            
            self._daily_energy_consumed += estimated_energy
            self._daily_carbon_emitted += estimated_carbon
            
            # Record scheduling event
            self._record_scheduling_event(task, "scheduled")
            
            # In a real implementation, you would start the actual task execution here
            # For now, we'll simulate task completion in _check_running_tasks
            
            logger.debug(f"Scheduled task {task.task_id} on agent {task.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error scheduling task {task.task_id}: {e}")
            return False
    
    def _check_running_tasks(self) -> None:
        """Check status of running tasks and handle completion"""
        current_time = datetime.now()
        completed_tasks = []
        
        with self._lock:
            for task_id, task in list(self._running_tasks.items()):
                # Simulate task completion based on estimated duration
                if task.started_at and (current_time - task.started_at).total_seconds() >= task.estimated_duration:
                    task.completed_at = current_time
                    completed_tasks.append(task)
                    del self._running_tasks[task_id]
        
        # Process completed tasks
        for task in completed_tasks:
            self._completed_tasks.append(task)
            self._record_scheduling_event(task, "completed")
            logger.debug(f"Task {task.task_id} completed")
            
            # Keep only recent completed tasks
            if len(self._completed_tasks) > 1000:
                self._completed_tasks = self._completed_tasks[-500:]
    
    def _find_alternative_agent(self, task: Task, agent_capacity: Dict[str, int]) -> Optional[str]:
        """Find an alternative agent for a task"""
        # Simple implementation - find any agent with capacity
        for agent_id, capacity in agent_capacity.items():
            if capacity > 0:
                return agent_id
        return None
    
    def _validate_task(self, task: Task) -> bool:
        """Validate a task before scheduling"""
        if not task.task_id or not task.agent_id:
            return False
        
        if task.estimated_duration <= 0:
            return False
        
        if task.estimated_power_w < 0 or task.estimated_cpu_usage < 0:
            return False
        
        return True
    
    def _check_resource_constraints(self, task: Task) -> bool:
        """Check if task meets resource constraints"""
        # Check maximum power consumption
        if task.estimated_power_w > 50.0:  # 50W max per task
            return False
        
        # Check maximum duration
        if task.estimated_duration > 3600:  # 1 hour max
            return False
        
        # Check memory requirements
        if task.estimated_memory_mb > 2048:  # 2GB max
            return False
        
        return True
    
    def _check_budgets(self, task: Task) -> bool:
        """Check if task fits within energy and carbon budgets"""
        estimated_energy = task.estimated_power_w * task.estimated_duration / 3600
        estimated_carbon = estimated_energy * self._current_metrics.carbon_intensity
        
        if self._daily_energy_consumed + estimated_energy > self._energy_budget_wh:
            return False
        
        if self._daily_carbon_emitted + estimated_carbon > self._carbon_budget_g:
            return False
        
        return True
    
    def _update_current_metrics(self) -> None:
        """Update current system metrics"""
        try:
            # Get system metrics
            self._current_metrics.cpu_utilization = psutil.cpu_percent(interval=1.0)
            self._current_metrics.memory_utilization = psutil.virtual_memory().percent
            
            # Get temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get CPU temperature
                    cpu_temps = temps.get('cpu_thermal', temps.get('coretemp', []))
                    if cpu_temps:
                        self._current_metrics.temperature = cpu_temps[0].current
            except:
                pass
            
            # Count active agents
            if self.agent_manager:
                self._current_metrics.active_agents = len(self.agent_manager.get_active_agents())
            
            # Count pending tasks
            self._current_metrics.pending_tasks = self._task_queue.size()
            
            # Calculate average response time
            recent_completed = [t for t in self._completed_tasks[-50:] 
                             if t.completed_at and t.started_at]
            if recent_completed:
                response_times = [(t.completed_at - t.created_at).total_seconds() 
                                for t in recent_completed]
                self._current_metrics.avg_response_time = sum(response_times) / len(response_times)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _reset_daily_budgets_if_needed(self) -> None:
        """Reset daily energy and carbon budgets if needed"""
        current_date = datetime.now().date()
        if current_date > self._last_reset:
            self._daily_energy_consumed = 0.0
            self._daily_carbon_emitted = 0.0
            self._last_reset = current_date
            logger.info("Daily energy and carbon budgets reset")
    
    def _record_scheduling_event(self, task: Task, event_type: str) -> None:
        """Record a scheduling event for analysis"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "task_id": task.task_id,
            "agent_id": task.agent_id,
            "task_type": task.task_type,
            "priority": task.priority.value,
            "estimated_power_w": task.estimated_power_w,
            "estimated_duration": task.estimated_duration,
            "current_metrics": {
                "cpu_utilization": self._current_metrics.cpu_utilization,
                "memory_utilization": self._current_metrics.memory_utilization,
                "temperature": self._current_metrics.temperature,
                "active_agents": self._current_metrics.active_agents
            }
        }
        
        self._scheduling_history.append(event)
        
        # Keep history manageable
        if len(self._scheduling_history) > 10000:
            self._scheduling_history = self._scheduling_history[-5000:]
    
    def _update_scheduling_metrics(self) -> None:
        """Update internal scheduling metrics"""
        # This would update performance metrics for the scheduler
        pass
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics"""
        with self._lock:
            running_count = len(self._running_tasks)
            
        pending_count = self._task_queue.size()
        completed_count = len(self._completed_tasks)
        
        # Calculate energy efficiency
        total_energy = sum(
            task.estimated_power_w * task.estimated_duration / 3600
            for task in self._completed_tasks
        )
        
        # Calculate average wait time
        recent_completed = [t for t in self._completed_tasks[-100:] 
                          if t.scheduled_at and t.created_at]
        avg_wait_time = 0.0
        if recent_completed:
            wait_times = [(t.scheduled_at - t.created_at).total_seconds() 
                         for t in recent_completed]
            avg_wait_time = sum(wait_times) / len(wait_times)
        
        return {
            "policy": self.policy.value,
            "tasks_pending": pending_count,
            "tasks_running": running_count,
            "tasks_completed": completed_count,
            "daily_energy_consumed_wh": self._daily_energy_consumed,
            "daily_carbon_emitted_g": self._daily_carbon_emitted,
            "energy_budget_utilization": self._daily_energy_consumed / self._energy_budget_wh,
            "carbon_budget_utilization": self._daily_carbon_emitted / self._carbon_budget_g,
            "avg_wait_time_seconds": avg_wait_time,
            "total_energy_scheduled": total_energy,
            "scheduling_active": self._scheduling,
            "current_metrics": {
                "cpu_utilization": self._current_metrics.cpu_utilization,
                "memory_utilization": self._current_metrics.memory_utilization,
                "temperature": self._current_metrics.temperature,
                "active_agents": self._current_metrics.active_agents,
                "avg_response_time": self._current_metrics.avg_response_time
            }
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        # Check running tasks
        with self._lock:
            if task_id in self._running_tasks:
                task = self._running_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "running",
                    "agent_id": task.agent_id,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "estimated_completion": (task.started_at + timedelta(seconds=task.estimated_duration)).isoformat() if task.started_at else None
                }
        
        # Check completed tasks
        for task in reversed(self._completed_tasks):
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "agent_id": task.agent_id,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "duration": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None
                }
        
        # Check pending tasks
        pending_tasks = self._task_queue.get_pending_tasks()
        for task in pending_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "agent_id": task.agent_id,
                    "created_at": task.created_at.isoformat(),
                    "priority": task.priority.value
                }
        
        return {"task_id": task_id, "status": "not_found"}
    
    def set_energy_budget(self, budget_wh: float) -> None:
        """Set daily energy budget in Watt-hours"""
        self._energy_budget_wh = budget_wh
        logger.info(f"Energy budget set to {budget_wh} Wh")
    
    def set_carbon_budget(self, budget_g: float) -> None:
        """Set daily carbon budget in grams CO2"""
        self._carbon_budget_g = budget_g
        logger.info(f"Carbon budget set to {budget_g} g CO2")
    
    def export_scheduling_data(self, filename: str) -> None:
        """Export scheduling data to JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "stats": self.get_scheduling_stats(),
            "scheduling_history": self._scheduling_history[-1000:],  # Last 1000 events
            "completed_tasks": [
                {
                    "task_id": task.task_id,
                    "agent_id": task.agent_id,
                    "task_type": task.task_type,
                    "priority": task.priority.value,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "estimated_power_w": task.estimated_power_w,
                    "estimated_duration": task.estimated_duration
                }
                for task in self._completed_tasks[-500:]  # Last 500 completed tasks
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Scheduling data exported to {filename}")

# Global scheduler instance
_global_scheduler: Optional[EnergyAwareScheduler] = None

def get_global_scheduler(
    policy: SchedulingPolicy = SchedulingPolicy.BALANCED, 
    agent_manager=None
) -> EnergyAwareScheduler:
    """Get or create global energy-aware scheduler instance"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = EnergyAwareScheduler(policy, agent_manager)
    return _global_scheduler
