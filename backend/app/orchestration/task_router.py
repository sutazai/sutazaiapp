"""
SutazAI Task Router and Load Balancer
Advanced task routing with load balancing, performance optimization,
and intelligent agent selection algorithms.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from app.schemas.message_types import TaskPriority
import heapq
from collections import defaultdict, deque
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_WEIGHTED = "capability_weighted"
    HYBRID_INTELLIGENT = "hybrid_intelligent"

@dataclass
class TaskRequest:
    id: str
    type: str
    description: str
    input_data: Any
    priority: TaskPriority
    requester_id: str
    capabilities_required: List[str]
    resource_requirements: Dict[str, Any]
    deadline: Optional[datetime] = None
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: float = 0.0
    created_at: datetime = None

@dataclass 
class AgentMetrics:
    agent_id: str
    current_load: float
    avg_response_time: float
    success_rate: float
    total_tasks: int
    active_tasks: int
    health_score: float
    resource_usage: Dict[str, float]
    capabilities_performance: Dict[str, float]
    last_updated: datetime

@dataclass
class RoutingDecision:
    selected_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[str]
    expected_completion_time: float
    load_balance_score: float

class PriorityTaskQueue:
    """Priority queue for tasks with dynamic reordering"""
    
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0
    
    def add_task(self, task: TaskRequest):
        """Add task to priority queue"""
        if task.id in self.entry_finder:
            self.remove_task(task.id)
        
        # Priority score calculation (higher score = higher priority)
        priority_score = self._calculate_priority_score(task)
        
        entry = [priority_score, self.counter, task]
        self.entry_finder[task.id] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1
    
    def remove_task(self, task_id: str):
        """Mark task as removed"""
        entry = self.entry_finder.pop(task_id, None)
        if entry:
            entry[-1] = self.REMOVED
    
    def pop_task(self) -> Optional[TaskRequest]:
        """Pop highest priority task"""
        while self.heap:
            priority, count, task = heapq.heappop(self.heap)
            if task is not self.REMOVED:
                del self.entry_finder[task.id]
                return task
        return None
    
    def peek_task(self) -> Optional[TaskRequest]:
        """Peek at highest priority task without removing"""
        while self.heap:
            priority, count, task = self.heap[0]
            if task is not self.REMOVED:
                return task
            heapq.heappop(self.heap)
        return None
    
    def _calculate_priority_score(self, task: TaskRequest) -> float:
        """Calculate dynamic priority score"""
        # Use canonical rank for numeric comparison; negative for min-heap
        base_priority = (-1000) * TaskPriority.from_value(task.priority).rank
        
        # Time urgency factor
        if task.deadline:
            time_until_deadline = (task.deadline - datetime.now()).total_seconds()
            urgency_factor = max(0, 1000 - time_until_deadline / 60)  # More urgent = higher priority
            base_priority -= urgency_factor
        
        # Age factor (older tasks get slightly higher priority)
        if task.created_at:
            age_minutes = (datetime.now() - task.created_at).total_seconds() / 60
            age_factor = min(age_minutes * 0.1, 100)
            base_priority -= age_factor
        
        return base_priority
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.entry_finder)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.entry_finder) == 0

class IntelligentTaskRouter:
    """
    Advanced task routing system with multiple load balancing algorithms
    and intelligent agent selection based on performance metrics.
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Task queues by priority
        self.task_queue = PriorityTaskQueue()
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.algorithm = LoadBalancingAlgorithm.HYBRID_INTELLIGENT
        
        # Performance tracking
        self.routing_history: deque = deque(maxlen=1000)
        self.performance_cache: Dict[str, Dict] = {}
        
        # Configuration
        self.max_queue_size = 10000
        self.routing_timeout = 30.0
        self.health_threshold = 0.5
        self.load_threshold = 0.9
        
        # Metrics
        self.metrics = {
            "tasks_routed": 0,
            "routing_failures": 0,
            "avg_routing_time": 0.0,
            "queue_size": 0,
            "successful_routings": 0
        }
    
    async def initialize(self):
        """Initialize the task router"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start background tasks
            asyncio.create_task(self._metrics_updater())
            asyncio.create_task(self._performance_analyzer())
            
            logger.info("Task router initialized successfully")
            
        except Exception as e:
            logger.error(f"Task router initialization failed: {e}")
            raise
    
    async def submit_task(self, task: TaskRequest) -> bool:
        """Submit a task for routing"""
        try:
            if self.task_queue.size() >= self.max_queue_size:
                logger.warning("Task queue is full, rejecting task")
                return False
            
            task.created_at = datetime.now()
            self.task_queue.add_task(task)
            self.metrics["queue_size"] = self.task_queue.size()
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                "pending_tasks",
                task.id,
                json.dumps(task.__dict__, default=str)
            )
            
            logger.debug(f"Task submitted for routing: {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.id}: {e}")
            return False
    
    async def route_next_task(self, available_agents: List[str]) -> Optional[Tuple[TaskRequest, str]]:
        """Route the next highest priority task to an optimal agent"""
        if self.task_queue.is_empty():
            return None
        
        start_time = time.time()
        
        try:
            task = self.task_queue.pop_task()
            if not task:
                return None
            
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(task, available_agents)
            if not suitable_agents:
                # Re-queue task for later retry
                task.retry_count += 1
                if task.retry_count <= task.max_retries:
                    self.task_queue.add_task(task)
                    logger.warning(f"No suitable agents for task {task.id}, re-queuing")
                else:
                    logger.error(f"Task {task.id} exceeded max retries, dropping")
                    self.metrics["routing_failures"] += 1
                return None
            
            # Select optimal agent using load balancing algorithm
            routing_decision = await self._select_optimal_agent(task, suitable_agents)
            
            # Update metrics
            routing_time = time.time() - start_time
            self.metrics["tasks_routed"] += 1
            self.metrics["successful_routings"] += 1
            self.metrics["avg_routing_time"] = (
                (self.metrics["avg_routing_time"] * (self.metrics["tasks_routed"] - 1) + routing_time) /
                self.metrics["tasks_routed"]
            )
            self.metrics["queue_size"] = self.task_queue.size()
            
            # Record routing decision
            self.routing_history.append({
                "task_id": task.id,
                "selected_agent": routing_decision.selected_agent,
                "confidence": routing_decision.confidence,
                "routing_time": routing_time,
                "timestamp": datetime.now()
            })
            
            # Remove from pending tasks
            await self.redis_client.hdel("pending_tasks", task.id)
            
            logger.info(f"Task {task.id} routed to agent {routing_decision.selected_agent}")
            return task, routing_decision.selected_agent
            
        except Exception as e:
            logger.error(f"Task routing failed: {e}")
            self.metrics["routing_failures"] += 1
            return None
    
    async def _find_suitable_agents(self, task: TaskRequest, available_agents: List[str]) -> List[str]:
        """Find agents suitable for the task based on capabilities and health"""
        suitable_agents = []
        
        for agent_id in available_agents:
            agent_metrics = self.agent_metrics.get(agent_id)
            if not agent_metrics:
                continue
            
            # Check health threshold
            if agent_metrics.health_score < self.health_threshold:
                continue
            
            # Check load threshold
            if agent_metrics.current_load > self.load_threshold:
                continue
            
            # Check capability requirements
            if task.capabilities_required:
                agent_capabilities = set(agent_metrics.capabilities_performance.keys())
                required_capabilities = set(task.capabilities_required)
                if not required_capabilities.issubset(agent_capabilities):
                    continue
            
            # Check resource requirements
            if task.resource_requirements:
                if not self._check_resource_availability(agent_metrics, task.resource_requirements):
                    continue
            
            suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _check_resource_availability(self, agent_metrics: AgentMetrics, requirements: Dict[str, Any]) -> bool:
        """Check if agent has required resources available"""
        for resource, required_amount in requirements.items():
            if resource in agent_metrics.resource_usage:
                available = 1.0 - agent_metrics.resource_usage[resource]
                if available < required_amount:
                    return False
        return True
    
    async def _select_optimal_agent(self, task: TaskRequest, suitable_agents: List[str]) -> RoutingDecision:
        """Select optimal agent using configured load balancing algorithm"""
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return await self._round_robin_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return await self._least_connections_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return await self._least_response_time_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            return await self._resource_based_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.PERFORMANCE_BASED:
            return await self._performance_based_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.CAPABILITY_WEIGHTED:
            return await self._capability_weighted_selection(task, suitable_agents)
        elif self.algorithm == LoadBalancingAlgorithm.HYBRID_INTELLIGENT:
            return await self._hybrid_intelligent_selection(task, suitable_agents)
        else:
            # Default to round robin
            return await self._round_robin_selection(task, suitable_agents)
    
    async def _round_robin_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Simple round-robin selection"""
        key = f"rr_{task.type}"
        self.round_robin_counters[key] = (self.round_robin_counters[key] + 1) % len(agents)
        selected = agents[self.round_robin_counters[key]]
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.5,
            reasoning="Round-robin selection",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=0.5
        )
    
    async def _weighted_round_robin_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Weighted round-robin based on agent performance"""
        weights = []
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if metrics:
                weight = metrics.health_score * (1 - metrics.current_load) * metrics.success_rate
                weights.append(weight)
            else:
                weights.append(0.1)  # Low weight for unknown agents
        
        total_weight = sum(weights)
        if total_weight == 0:
            return await self._round_robin_selection(task, agents)
        
        # Weighted random selection
        import random
        target = random.uniform(0, total_weight)
        current = 0
        for i, weight in enumerate(weights):
            current += weight
            if current >= target:
                selected = agents[i]
                break
        else:
            selected = agents[-1]
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.7,
            reasoning="Weighted round-robin based on performance metrics",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=weights[agents.index(selected)] / max(weights)
        )
    
    async def _least_connections_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Select agent with least active tasks"""
        min_connections = float('inf')
        selected = agents[0]
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if metrics and metrics.active_tasks < min_connections:
                min_connections = metrics.active_tasks
                selected = agent_id
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.8,
            reasoning=f"Least connections selection ({min_connections} active tasks)",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=1.0 - (min_connections / 10.0)  # Assuming max 10 concurrent tasks
        )
    
    async def _least_response_time_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Select agent with lowest average response time"""
        best_time = float('inf')
        selected = agents[0]
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if metrics and metrics.avg_response_time < best_time:
                best_time = metrics.avg_response_time
                selected = agent_id
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.75,
            reasoning=f"Least response time selection ({best_time:.2f}s avg)",
            alternative_agents=agents[:3],
            expected_completion_time=best_time,
            load_balance_score=1.0 - min(best_time / 10.0, 1.0)  # Normalize to 0-1
        )
    
    async def _resource_based_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Select agent based on resource availability"""
        best_score = -1
        selected = agents[0]
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if not metrics:
                continue
            
            # Calculate resource availability score
            resource_score = 0
            for resource, usage in metrics.resource_usage.items():
                resource_score += (1.0 - usage)
            resource_score /= len(metrics.resource_usage) if metrics.resource_usage else 1
            
            if resource_score > best_score:
                best_score = resource_score
                selected = agent_id
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.8,
            reasoning=f"Resource-based selection (score: {best_score:.2f})",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=best_score
        )
    
    async def _performance_based_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Select agent based on overall performance metrics"""
        best_score = -1
        selected = agents[0]
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if not metrics:
                continue
            
            # Composite performance score
            performance_score = (
                metrics.health_score * 0.3 +
                metrics.success_rate * 0.4 +
                (1 - metrics.current_load) * 0.2 +
                min(1.0, 10.0 / max(metrics.avg_response_time, 0.1)) * 0.1
            )
            
            if performance_score > best_score:
                best_score = performance_score
                selected = agent_id
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.85,
            reasoning=f"Performance-based selection (score: {best_score:.2f})",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=best_score
        )
    
    async def _capability_weighted_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Select agent based on capability-specific performance"""
        best_score = -1
        selected = agents[0]
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if not metrics:
                continue
            
            # Calculate capability-weighted score
            capability_score = 0
            if task.capabilities_required:
                for capability in task.capabilities_required:
                    if capability in metrics.capabilities_performance:
                        capability_score += metrics.capabilities_performance[capability]
                capability_score /= len(task.capabilities_required)
            else:
                capability_score = sum(metrics.capabilities_performance.values()) / len(metrics.capabilities_performance)
            
            # Combine with general performance
            combined_score = capability_score * 0.7 + metrics.health_score * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                selected = agent_id
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=0.9,
            reasoning=f"Capability-weighted selection (score: {best_score:.2f})",
            alternative_agents=agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=best_score
        )
    
    async def _hybrid_intelligent_selection(self, task: TaskRequest, agents: List[str]) -> RoutingDecision:
        """Advanced hybrid selection using multiple factors"""
        scores = {}
        
        for agent_id in agents:
            metrics = self.agent_metrics.get(agent_id)
            if not metrics:
                scores[agent_id] = 0
                continue
            
            # Multi-factor scoring
            health_score = metrics.health_score
            load_score = 1 - metrics.current_load
            success_score = metrics.success_rate
            response_score = min(1.0, 10.0 / max(metrics.avg_response_time, 0.1))
            
            # Capability match score
            capability_score = 1.0
            if task.capabilities_required:
                matched_capabilities = 0
                for capability in task.capabilities_required:
                    if capability in metrics.capabilities_performance:
                        matched_capabilities += metrics.capabilities_performance[capability]
                capability_score = matched_capabilities / len(task.capabilities_required)
            
            # Resource availability score
            resource_score = 1.0
            if task.resource_requirements:
                resource_penalties = 0
                for resource, required in task.resource_requirements.items():
                    if resource in metrics.resource_usage:
                        available = 1.0 - metrics.resource_usage[resource]
                        if available < required:
                            resource_penalties += (required - available)
                resource_score = max(0, 1.0 - resource_penalties)
            
            # Priority adjustment
            priority_weight = 1.0 + (task.priority.value - 1) * 0.1
            
            # Time urgency factor
            urgency_factor = 1.0
            if task.deadline:
                time_left = (task.deadline - datetime.now()).total_seconds()
                if time_left < 3600:  # Less than 1 hour
                    urgency_factor = 1.5
                elif time_left < 7200:  # Less than 2 hours
                    urgency_factor = 1.3
            
            # Composite score calculation
            composite_score = (
                health_score * 0.2 +
                load_score * 0.25 +
                success_score * 0.15 +
                response_score * 0.15 +
                capability_score * 0.2 +
                resource_score * 0.05
            ) * priority_weight * urgency_factor
            
            scores[agent_id] = composite_score
        
        # Select agent with highest score
        selected = max(scores, key=scores.get)
        confidence = min(scores[selected], 1.0)
        
        # Sort agents by score for alternatives
        sorted_agents = sorted(agents, key=lambda x: scores.get(x, 0), reverse=True)
        
        return RoutingDecision(
            selected_agent=selected,
            confidence=confidence,
            reasoning=f"Hybrid intelligent selection (score: {scores[selected]:.3f})",
            alternative_agents=sorted_agents[:3],
            expected_completion_time=task.estimated_duration,
            load_balance_score=scores[selected]
        )
    
    async def update_agent_metrics(self, agent_id: str, metrics: AgentMetrics):
        """Update agent metrics for routing decisions"""
        metrics.last_updated = datetime.now()
        self.agent_metrics[agent_id] = metrics
        
        # Store in Redis
        await self.redis_client.hset(
            "agent_metrics",
            agent_id,
            json.dumps(metrics.__dict__, default=str)
        )
    
    async def remove_agent(self, agent_id: str):
        """Remove agent from routing consideration"""
        if agent_id in self.agent_metrics:
            del self.agent_metrics[agent_id]
        
        await self.redis_client.hdel("agent_metrics", agent_id)
    
    async def set_load_balancing_algorithm(self, algorithm: LoadBalancingAlgorithm):
        """Set the load balancing algorithm"""
        self.algorithm = algorithm
        logger.info(f"Load balancing algorithm changed to: {algorithm.value}")
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_size": self.task_queue.size(),
            "is_empty": self.task_queue.is_empty(),
            "next_task_priority": self.task_queue.peek_task().priority.value if not self.task_queue.is_empty() else None,
            "algorithm": self.algorithm.value,
            "metrics": self.metrics
        }
    
    async def get_routing_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent routing history"""
        return list(self.routing_history)[-limit:]
    
    async def _metrics_updater(self):
        """Background task to update routing metrics"""
        while True:
            try:
                # Store current metrics
                await self.redis_client.hset(
                    "router_metrics",
                    "current",
                    json.dumps(self.metrics)
                )
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Metrics updater error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_analyzer(self):
        """Analyze routing performance and optimize"""
        while True:
            try:
                # Analyze recent routing decisions
                if len(self.routing_history) >= 50:
                    await self._analyze_routing_performance()
                
                await asyncio.sleep(60)  # Analyze every minute
                
            except Exception as e:
                logger.error(f"Performance analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_routing_performance(self):
        """Analyze routing performance and suggest optimizations"""
        recent_decisions = list(self.routing_history)[-100:]
        
        # Calculate success rates by algorithm
        algorithm_performance = defaultdict(list)
        for decision in recent_decisions:
            algorithm_performance[self.algorithm.value].append(decision['confidence'])
        
        # Log performance insights
        if self.algorithm.value in algorithm_performance:
            avg_confidence = sum(algorithm_performance[self.algorithm.value]) / len(algorithm_performance[self.algorithm.value])
            logger.info(f"Current algorithm {self.algorithm.value} avg confidence: {avg_confidence:.3f}")
    
    async def stop(self):
        """Stop the task router"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Task router stopped")

# Singleton instance
task_router = IntelligentTaskRouter()
