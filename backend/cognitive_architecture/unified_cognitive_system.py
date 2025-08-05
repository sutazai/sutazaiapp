"""
Unified Cognitive Architecture for SutazAI Multi-Agent System
=============================================================

This module implements a human-like cognitive architecture that coordinates
69+ AI agents with shared working memory, attention mechanisms, episodic
memory, and metacognitive monitoring.

Architecture Components:
1. Working Memory System - Short-term information processing
2. Episodic Memory - Experience and event storage
3. Semantic Memory - Knowledge representation
4. Attention Mechanism - Priority and focus management
5. Executive Control - Agent orchestration and decision making
6. Metacognitive Monitor - Self-awareness and performance tracking
7. Learning System - Adaptation and improvement mechanisms
8. Reasoning Engine - Multi-modal reasoning capabilities
"""

import asyncio
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
import uuid
import json
import numpy as np
from abc import ABC, abstractmethod

from ..knowledge_graph.manager import get_knowledge_graph_manager
from ..knowledge_graph.query_engine import QueryEngine
from ..ai_agents.core.agent_registry import get_agent_registry
from ..app.core.agi_brain import ReasoningType, TaskComplexity

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in the cognitive system"""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SENSORY = "sensory"


class AttentionMode(Enum):
    """Attention allocation modes"""
    FOCUSED = "focused"      # Single task focus
    DIVIDED = "divided"      # Multiple tasks
    SELECTIVE = "selective"  # Filtering relevant information
    SUSTAINED = "sustained"  # Long-term monitoring
    EXECUTIVE = "executive"  # High-level control


class CognitiveState(Enum):
    """Overall cognitive system state"""
    IDLE = "idle"
    PROCESSING = "processing"
    REASONING = "reasoning"
    LEARNING = "learning"
    CREATING = "creating"
    REFLECTING = "reflecting"
    COORDINATING = "coordinating"


@dataclass
class MemoryItem:
    """Individual memory item with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.WORKING
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5
    decay_rate: float = 0.1
    associations: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    source_agents: List[str] = field(default_factory=list)
    
    def access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.importance = min(1.0, self.importance + 0.05)
    
    def get_activation_strength(self) -> float:
        """Calculate current activation strength"""
        time_decay = (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
        base_activation = self.importance * (self.access_count + 1)
        return base_activation * np.exp(-self.decay_rate * time_decay)


@dataclass
class AttentionFocus:
    """Current attention focus with resource allocation"""
    task_id: str
    agents: List[str]
    priority: float
    mode: AttentionMode
    allocated_resources: float
    start_time: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def duration(self) -> timedelta:
        return datetime.utcnow() - self.start_time


@dataclass
class ReasoningChain:
    """Chain of reasoning with intermediate steps"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    steps: List[Dict[str, Any]] = field(default_factory=list)
    premises: List[Any] = field(default_factory=list)
    conclusions: List[Any] = field(default_factory=list)
    confidence: float = 0.0
    agents_involved: Set[str] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    def add_step(self, step_type: str, content: Any, agent: str, confidence: float = 0.8):
        """Add a reasoning step"""
        self.steps.append({
            "step_type": step_type,
            "content": content,
            "agent": agent,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        })
        self.agents_involved.add(agent)
    
    def complete(self, conclusion: Any, confidence: float):
        """Complete the reasoning chain"""
        self.conclusions.append(conclusion)
        self.confidence = confidence
        self.end_time = datetime.utcnow()


class WorkingMemory:
    """Working memory system with limited capacity and decay"""
    
    def __init__(self, capacity: int = 7, chunk_size: int = 4):
        self.capacity = capacity  # Miller's 7±2 rule
        self.chunk_size = chunk_size
        self.items: deque[MemoryItem] = deque(maxlen=capacity)
        self.chunks: Dict[str, List[MemoryItem]] = {}
        self.attention_weights: Dict[str, float] = {}
        
    def add(self, item: MemoryItem) -> bool:
        """Add item to working memory"""
        # Check if we need to make room
        if len(self.items) >= self.capacity:
            # Remove least activated item
            min_activation = float('inf')
            min_item = None
            for existing in self.items:
                activation = existing.get_activation_strength()
                if activation < min_activation:
                    min_activation = activation
                    min_item = existing
            
            if min_item and item.importance > min_item.importance:
                self.items.remove(min_item)
            else:
                return False
        
        self.items.append(item)
        self._update_attention_weights()
        return True
    
    def retrieve(self, query: Dict[str, Any]) -> List[MemoryItem]:
        """Retrieve relevant items based on query"""
        relevant_items = []
        
        for item in self.items:
            relevance = self._calculate_relevance(item, query)
            if relevance > 0.3:
                item.access()
                relevant_items.append((relevance, item))
        
        # Sort by relevance
        relevant_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in relevant_items]
    
    def chunk(self, items: List[MemoryItem]) -> str:
        """Create a chunk from multiple items"""
        if len(items) > self.chunk_size:
            items = items[:self.chunk_size]
        
        chunk_id = str(uuid.uuid4())
        self.chunks[chunk_id] = items
        
        # Create a summary representation
        chunk_item = MemoryItem(
            content={"type": "chunk", "items": [item.id for item in items]},
            memory_type=MemoryType.WORKING,
            importance=np.mean([item.importance for item in items])
        )
        
        self.add(chunk_item)
        return chunk_id
    
    def _calculate_relevance(self, item: MemoryItem, query: Dict[str, Any]) -> float:
        """Calculate relevance score between item and query"""
        relevance = 0.0
        
        # Check content similarity
        if isinstance(item.content, dict) and isinstance(query, dict):
            common_keys = set(item.content.keys()) & set(query.keys())
            relevance += len(common_keys) / max(len(item.content), len(query))
        
        # Check context similarity
        if "context" in query and item.context:
            context_similarity = len(set(item.context.keys()) & set(query["context"].keys()))
            relevance += context_similarity * 0.2
        
        # Boost for recent items
        recency = 1.0 / (1.0 + (datetime.utcnow() - item.timestamp).total_seconds() / 3600)
        relevance += recency * 0.3
        
        return min(1.0, relevance)
    
    def _update_attention_weights(self):
        """Update attention weights for all items"""
        total_activation = sum(item.get_activation_strength() for item in self.items)
        
        if total_activation > 0:
            for item in self.items:
                self.attention_weights[item.id] = item.get_activation_strength() / total_activation


class EpisodicMemory:
    """Episodic memory for storing experiences and events"""
    
    def __init__(self, max_episodes: int = 10000):
        self.max_episodes = max_episodes
        self.episodes: deque[MemoryItem] = deque(maxlen=max_episodes)
        self.episode_index: Dict[str, List[MemoryItem]] = defaultdict(list)
        self.temporal_index: Dict[datetime, List[MemoryItem]] = defaultdict(list)
        
    def store_episode(self, content: Any, context: Dict[str, Any], 
                     agents: List[str], importance: float = 0.5) -> str:
        """Store a new episode"""
        episode = MemoryItem(
            content=content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            context=context,
            source_agents=agents
        )
        
        self.episodes.append(episode)
        
        # Index by context
        for key, value in context.items():
            if isinstance(value, str):
                self.episode_index[f"{key}:{value}"].append(episode)
        
        # Index by time
        time_key = episode.timestamp.replace(second=0, microsecond=0)
        self.temporal_index[time_key].append(episode)
        
        return episode.id
    
    def recall(self, cues: Dict[str, Any], time_range: Optional[Tuple[datetime, datetime]] = None) -> List[MemoryItem]:
        """Recall episodes based on cues"""
        candidates = set()
        
        # Search by cues
        for key, value in cues.items():
            index_key = f"{key}:{value}"
            if index_key in self.episode_index:
                candidates.update(self.episode_index[index_key])
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            candidates = {
                ep for ep in candidates
                if start_time <= ep.timestamp <= end_time
            }
        
        # Sort by relevance and recency
        scored_episodes = []
        for episode in candidates:
            score = self._calculate_recall_score(episode, cues)
            scored_episodes.append((score, episode))
        
        scored_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # Update access counts
        for _, episode in scored_episodes[:10]:
            episode.access()
        
        return [ep for _, ep in scored_episodes[:10]]
    
    def consolidate(self):
        """Consolidate memories, strengthening important ones"""
        # Calculate average importance
        avg_importance = np.mean([ep.importance for ep in self.episodes])
        
        # Strengthen frequently accessed memories
        for episode in self.episodes:
            if episode.access_count > 5:
                episode.importance = min(1.0, episode.importance * 1.1)
            elif episode.importance < avg_importance * 0.5:
                episode.decay_rate *= 1.2
    
    def _calculate_recall_score(self, episode: MemoryItem, cues: Dict[str, Any]) -> float:
        """Calculate recall score for an episode"""
        score = 0.0
        
        # Context match
        for key, value in cues.items():
            if key in episode.context and episode.context[key] == value:
                score += 0.3
        
        # Importance
        score += episode.importance * 0.3
        
        # Recency
        time_diff = (datetime.utcnow() - episode.timestamp).total_seconds() / 86400  # days
        recency_score = np.exp(-time_diff / 30)  # 30-day half-life
        score += recency_score * 0.2
        
        # Access frequency
        score += min(1.0, episode.access_count / 10) * 0.2
        
        return score


class AttentionMechanism:
    """Attention mechanism for managing cognitive focus"""
    
    def __init__(self, max_concurrent_focus: int = 3):
        self.max_concurrent_focus = max_concurrent_focus
        self.current_focus: List[AttentionFocus] = []
        self.attention_history: deque[AttentionFocus] = deque(maxlen=100)
        self.resource_pool: float = 1.0
        self.mode: AttentionMode = AttentionMode.SELECTIVE
        
    def allocate_attention(self, task_id: str, agents: List[str], 
                         priority: float, mode: AttentionMode = AttentionMode.FOCUSED) -> Optional[AttentionFocus]:
        """Allocate attention to a task"""
        required_resources = self._calculate_required_resources(len(agents), priority, mode)
        
        # Check if we have enough resources
        if required_resources > self.resource_pool:
            # Try to free up resources
            self._rebalance_attention()
            
            if required_resources > self.resource_pool:
                return None
        
        # Create new focus
        focus = AttentionFocus(
            task_id=task_id,
            agents=agents,
            priority=priority,
            mode=mode,
            allocated_resources=required_resources
        )
        
        self.current_focus.append(focus)
        self.resource_pool -= required_resources
        
        # Limit concurrent focus
        if len(self.current_focus) > self.max_concurrent_focus:
            self._suspend_lowest_priority()
        
        return focus
    
    def release_attention(self, task_id: str):
        """Release attention from a task"""
        for focus in self.current_focus:
            if focus.task_id == task_id:
                self.resource_pool += focus.allocated_resources
                self.current_focus.remove(focus)
                self.attention_history.append(focus)
                break
    
    def get_attention_distribution(self) -> Dict[str, float]:
        """Get current attention distribution"""
        distribution = {}
        
        for focus in self.current_focus:
            weight = focus.allocated_resources * focus.priority
            for agent in focus.agents:
                if agent not in distribution:
                    distribution[agent] = 0.0
                distribution[agent] += weight / len(focus.agents)
        
        # Normalize
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}
        
        return distribution
    
    def _calculate_required_resources(self, num_agents: int, priority: float, mode: AttentionMode) -> float:
        """Calculate required resources for attention allocation"""
        base_cost = {
            AttentionMode.FOCUSED: 0.6,
            AttentionMode.DIVIDED: 0.3,
            AttentionMode.SELECTIVE: 0.4,
            AttentionMode.SUSTAINED: 0.5,
            AttentionMode.EXECUTIVE: 0.8
        }
        
        cost = base_cost.get(mode, 0.5)
        cost *= (1 + np.log(num_agents)) / 3  # Scale with number of agents
        cost *= priority
        
        return min(1.0, cost)
    
    def _rebalance_attention(self):
        """Rebalance attention resources"""
        if not self.current_focus:
            return
        
        # Calculate total priority
        total_priority = sum(focus.priority for focus in self.current_focus)
        
        # Redistribute resources proportionally
        total_resources = sum(focus.allocated_resources for focus in self.current_focus) + self.resource_pool
        
        for focus in self.current_focus:
            new_allocation = (focus.priority / total_priority) * total_resources * 0.9  # Keep 10% reserve
            self.resource_pool += focus.allocated_resources - new_allocation
            focus.allocated_resources = new_allocation
    
    def _suspend_lowest_priority(self):
        """Suspend the lowest priority focus"""
        if not self.current_focus:
            return
        
        lowest = min(self.current_focus, key=lambda f: f.priority)
        self.release_attention(lowest.task_id)


class ExecutiveControl:
    """Executive control system for high-level coordination"""
    
    def __init__(self, agent_registry, knowledge_graph_manager):
        self.agent_registry = agent_registry
        self.knowledge_graph = knowledge_graph_manager
        self.active_goals: List[Dict[str, Any]] = []
        self.task_queue: deque[Dict[str, Any]] = deque()
        self.execution_plans: Dict[str, List[Dict[str, Any]]] = {}
        self.inhibition_rules: List[Callable] = []
        
    async def plan_execution(self, goal: Dict[str, Any], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create execution plan for a goal"""
        plan_id = str(uuid.uuid4())
        
        # Decompose goal into tasks
        tasks = await self._decompose_goal(goal)
        
        # Select agents for each task
        agent_assignments = []
        for task in tasks:
            agents = await self._select_agents_for_task(task, constraints)
            agent_assignments.append({
                "task": task,
                "agents": agents,
                "dependencies": task.get("dependencies", []),
                "priority": task.get("priority", 0.5)
            })
        
        # Order tasks by dependencies
        ordered_plan = self._order_by_dependencies(agent_assignments)
        
        self.execution_plans[plan_id] = ordered_plan
        return ordered_plan
    
    async def coordinate_agents(self, task: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Coordinate multiple agents on a task"""
        coordination_id = str(uuid.uuid4())
        
        # Create coordination structure
        coordination = {
            "id": coordination_id,
            "task": task,
            "agents": agents,
            "status": "initializing",
            "start_time": datetime.utcnow(),
            "results": {},
            "communications": []
        }
        
        # Assign roles
        roles = self._assign_agent_roles(task, agents)
        
        # Create shared context
        shared_context = {
            "task_id": task.get("id", coordination_id),
            "goal": task.get("goal"),
            "constraints": task.get("constraints", {}),
            "roles": roles,
            "coordination_id": coordination_id
        }
        
        # Execute with coordination
        results = await self._execute_coordinated_task(agents, task, shared_context, roles)
        
        coordination["results"] = results
        coordination["status"] = "completed"
        coordination["end_time"] = datetime.utcnow()
        
        return coordination
    
    def add_inhibition_rule(self, rule: Callable[[Dict[str, Any]], bool]):
        """Add rule to inhibit certain behaviors"""
        self.inhibition_rules.append(rule)
    
    def should_inhibit(self, action: Dict[str, Any]) -> bool:
        """Check if an action should be inhibited"""
        for rule in self.inhibition_rules:
            if rule(action):
                return True
        return False
    
    async def _decompose_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose high-level goal into tasks"""
        tasks = []
        
        goal_type = goal.get("type", "general")
        
        if goal_type == "complex_analysis":
            tasks.extend([
                {"type": "data_gathering", "priority": 0.9},
                {"type": "initial_analysis", "priority": 0.8, "dependencies": ["data_gathering"]},
                {"type": "deep_analysis", "priority": 0.7, "dependencies": ["initial_analysis"]},
                {"type": "synthesis", "priority": 0.6, "dependencies": ["deep_analysis"]},
                {"type": "report_generation", "priority": 0.5, "dependencies": ["synthesis"]}
            ])
        elif goal_type == "creative_solution":
            tasks.extend([
                {"type": "brainstorming", "priority": 0.9},
                {"type": "concept_development", "priority": 0.8, "dependencies": ["brainstorming"]},
                {"type": "feasibility_analysis", "priority": 0.7, "dependencies": ["concept_development"]},
                {"type": "refinement", "priority": 0.6, "dependencies": ["feasibility_analysis"]}
            ])
        else:
            # Generic decomposition
            tasks.append({"type": "analysis", "priority": 0.8})
            tasks.append({"type": "execution", "priority": 0.7, "dependencies": ["analysis"]})
            tasks.append({"type": "validation", "priority": 0.6, "dependencies": ["execution"]})
        
        # Add goal-specific details to each task
        for task in tasks:
            task["goal_id"] = goal.get("id", str(uuid.uuid4()))
            task["context"] = goal.get("context", {})
        
        return tasks
    
    async def _select_agents_for_task(self, task: Dict[str, Any], constraints: Dict[str, Any]) -> List[str]:
        """Select appropriate agents for a task"""
        task_type = task.get("type", "general")
        required_capabilities = []
        
        # Map task types to capabilities
        capability_mapping = {
            "data_gathering": ["data_extraction", "web_search", "api_integration"],
            "analysis": ["data_analysis", "pattern_recognition", "statistical_analysis"],
            "creative": ["creative_generation", "ideation", "design"],
            "coding": ["code_generation", "debugging", "testing"],
            "communication": ["natural_language", "summarization", "translation"]
        }
        
        # Get required capabilities
        for task_key, capabilities in capability_mapping.items():
            if task_key in task_type:
                required_capabilities.extend(capabilities)
        
        # Query agent registry
        if self.agent_registry:
            available_agents = await self.agent_registry.find_agents_by_capabilities(required_capabilities)
            
            # Filter by constraints
            if "max_agents" in constraints:
                available_agents = available_agents[:constraints["max_agents"]]
            
            return [agent.agent_id for agent in available_agents]
        
        return []
    
    def _order_by_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order tasks by dependencies using topological sort"""
        # Build dependency graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        task_map = {task["task"].get("type"): task for task in tasks}
        
        for task in tasks:
            task_type = task["task"].get("type")
            for dep in task.get("dependencies", []):
                graph[dep].append(task_type)
                in_degree[task_type] += 1
        
        # Topological sort
        queue = deque([task["task"].get("type") for task in tasks if in_degree[task["task"].get("type")] == 0])
        ordered = []
        
        while queue:
            current = queue.popleft()
            ordered.append(task_map[current])
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return ordered
    
    def _assign_agent_roles(self, task: Dict[str, Any], agents: List[str]) -> Dict[str, str]:
        """Assign specific roles to agents"""
        roles = {}
        
        if len(agents) == 1:
            roles[agents[0]] = "executor"
        elif len(agents) == 2:
            roles[agents[0]] = "lead"
            roles[agents[1]] = "support"
        else:
            roles[agents[0]] = "coordinator"
            for i, agent in enumerate(agents[1:], 1):
                if i <= len(agents) // 3:
                    roles[agent] = "analyst"
                elif i <= 2 * len(agents) // 3:
                    roles[agent] = "executor"
                else:
                    roles[agent] = "validator"
        
        return roles
    
    async def _execute_coordinated_task(self, agents: List[str], task: Dict[str, Any], 
                                      context: Dict[str, Any], roles: Dict[str, str]) -> Dict[str, Any]:
        """Execute task with agent coordination"""
        results = {
            "task_id": task.get("id", str(uuid.uuid4())),
            "agent_results": {},
            "coordination_summary": {},
            "success": True
        }
        
        # Simulate coordinated execution
        # In production, this would actually dispatch to agents
        for agent in agents:
            role = roles.get(agent, "participant")
            results["agent_results"][agent] = {
                "role": role,
                "status": "completed",
                "output": f"Task executed by {agent} as {role}"
            }
        
        results["coordination_summary"] = {
            "total_agents": len(agents),
            "completion_time": datetime.utcnow().isoformat(),
            "coordination_effectiveness": 0.85
        }
        
        return results


class MetacognitiveMonitor:
    """Metacognitive monitoring for self-awareness and improvement"""
    
    def __init__(self):
        self.performance_history: deque[Dict[str, Any]] = deque(maxlen=1000)
        self.learning_insights: List[Dict[str, Any]] = []
        self.confidence_calibration: Dict[str, float] = {}
        self.strategy_effectiveness: Dict[str, float] = {}
        self.error_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def monitor_performance(self, task: Dict[str, Any], result: Dict[str, Any], 
                          predicted_confidence: float, actual_success: bool):
        """Monitor task performance and update metrics"""
        performance = {
            "task_id": task.get("id", str(uuid.uuid4())),
            "task_type": task.get("type", "unknown"),
            "predicted_confidence": predicted_confidence,
            "actual_success": actual_success,
            "timestamp": datetime.utcnow(),
            "duration": result.get("duration", 0),
            "resources_used": result.get("resources_used", {})
        }
        
        self.performance_history.append(performance)
        
        # Update confidence calibration
        task_type = task.get("type", "general")
        if task_type not in self.confidence_calibration:
            self.confidence_calibration[task_type] = []
        
        self.confidence_calibration[task_type].append({
            "predicted": predicted_confidence,
            "actual": 1.0 if actual_success else 0.0
        })
        
        # Analyze for patterns
        if not actual_success:
            self._analyze_failure(task, result)
    
    def reflect_on_performance(self) -> Dict[str, Any]:
        """Reflect on recent performance and generate insights"""
        if len(self.performance_history) < 10:
            return {"status": "insufficient_data"}
        
        recent_performance = list(self.performance_history)[-100:]
        
        # Calculate metrics
        success_rate = sum(1 for p in recent_performance if p["actual_success"]) / len(recent_performance)
        
        # Confidence calibration analysis
        calibration_scores = {}
        for task_type, calibrations in self.confidence_calibration.items():
            if len(calibrations) >= 5:
                predicted = [c["predicted"] for c in calibrations[-20:]]
                actual = [c["actual"] for c in calibrations[-20:]]
                calibration_error = np.mean(np.abs(np.array(predicted) - np.array(actual)))
                calibration_scores[task_type] = 1.0 - calibration_error
        
        # Strategy effectiveness
        strategy_scores = self._evaluate_strategies()
        
        # Generate insights
        insights = {
            "overall_success_rate": success_rate,
            "confidence_calibration": calibration_scores,
            "strategy_effectiveness": strategy_scores,
            "improvement_areas": self._identify_improvement_areas(),
            "recommended_adjustments": self._recommend_adjustments()
        }
        
        self.learning_insights.append({
            "timestamp": datetime.utcnow(),
            "insights": insights
        })
        
        return insights
    
    def update_strategy_effectiveness(self, strategy: str, success: bool, context: Dict[str, Any]):
        """Update effectiveness metrics for a strategy"""
        if strategy not in self.strategy_effectiveness:
            self.strategy_effectiveness[strategy] = {
                "successes": 0,
                "failures": 0,
                "contexts": []
            }
        
        if success:
            self.strategy_effectiveness[strategy]["successes"] += 1
        else:
            self.strategy_effectiveness[strategy]["failures"] += 1
        
        self.strategy_effectiveness[strategy]["contexts"].append(context)
    
    def _analyze_failure(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Analyze failure patterns"""
        failure_type = result.get("error_type", "unknown")
        
        self.error_patterns[failure_type].append({
            "task": task,
            "result": result,
            "timestamp": datetime.utcnow()
        })
        
        # Look for patterns
        if len(self.error_patterns[failure_type]) >= 3:
            # Check for common factors
            recent_failures = self.error_patterns[failure_type][-5:]
            common_factors = self._find_common_factors(recent_failures)
            
            if common_factors:
                logger.warning(f"Detected failure pattern in {failure_type}: {common_factors}")
    
    def _evaluate_strategies(self) -> Dict[str, float]:
        """Evaluate effectiveness of different strategies"""
        scores = {}
        
        for strategy, data in self.strategy_effectiveness.items():
            total = data["successes"] + data["failures"]
            if total > 0:
                scores[strategy] = data["successes"] / total
        
        return scores
    
    def _identify_improvement_areas(self) -> List[str]:
        """Identify areas needing improvement"""
        areas = []
        
        # Low success rate task types
        for task_type, calibrations in self.confidence_calibration.items():
            if len(calibrations) >= 5:
                recent_success = np.mean([c["actual"] for c in calibrations[-10:]])
                if recent_success < 0.6:
                    areas.append(f"Low success rate in {task_type} tasks")
        
        # Poor confidence calibration
        for task_type, score in self.confidence_calibration.items():
            if isinstance(score, float) and score < 0.7:
                areas.append(f"Poor confidence calibration for {task_type}")
        
        # Recurring errors
        for error_type, patterns in self.error_patterns.items():
            if len(patterns) >= 5:
                areas.append(f"Recurring {error_type} errors")
        
        return areas
    
    def _recommend_adjustments(self) -> List[Dict[str, Any]]:
        """Recommend adjustments based on analysis"""
        recommendations = []
        
        # Strategy adjustments
        for strategy, score in self.strategy_effectiveness.items():
            if isinstance(score, dict):
                effectiveness = score["successes"] / (score["successes"] + score["failures"]) if (score["successes"] + score["failures"]) > 0 else 0
                if effectiveness < 0.5:
                    recommendations.append({
                        "type": "strategy_adjustment",
                        "strategy": strategy,
                        "recommendation": "Consider alternative approach",
                        "current_effectiveness": effectiveness
                    })
        
        # Confidence adjustments
        for task_type, calibrations in self.confidence_calibration.items():
            if len(calibrations) >= 10:
                recent = calibrations[-10:]
                avg_predicted = np.mean([c["predicted"] for c in recent])
                avg_actual = np.mean([c["actual"] for c in recent])
                
                if avg_predicted - avg_actual > 0.2:
                    recommendations.append({
                        "type": "confidence_adjustment",
                        "task_type": task_type,
                        "recommendation": "Reduce confidence predictions",
                        "adjustment_factor": 0.8
                    })
                elif avg_actual - avg_predicted > 0.2:
                    recommendations.append({
                        "type": "confidence_adjustment",
                        "task_type": task_type,
                        "recommendation": "Increase confidence predictions",
                        "adjustment_factor": 1.2
                    })
        
        return recommendations
    
    def _find_common_factors(self, failures: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common factors in failures"""
        common_factors = {}
        
        # Check for common task types
        task_types = [f["task"].get("type") for f in failures]
        if len(set(task_types)) == 1:
            common_factors["task_type"] = task_types[0]
        
        # Check for common contexts
        contexts = [f["task"].get("context", {}) for f in failures]
        common_keys = set(contexts[0].keys()) if contexts else set()
        for ctx in contexts[1:]:
            common_keys &= set(ctx.keys())
        
        if common_keys:
            common_factors["context_keys"] = list(common_keys)
        
        return common_factors


class LearningSystem:
    """Adaptive learning system for continuous improvement"""
    
    def __init__(self, knowledge_graph_manager):
        self.knowledge_graph = knowledge_graph_manager
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.skill_levels: Dict[str, float] = defaultdict(lambda: 0.5)
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.1
        
    async def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from a completed experience"""
        task_type = experience.get("task_type", "general")
        success = experience.get("success", False)
        
        # Update skill level
        current_skill = self.skill_levels[task_type]
        if success:
            self.skill_levels[task_type] = min(1.0, current_skill + self.learning_rate)
        else:
            self.skill_levels[task_type] = max(0.0, current_skill - self.learning_rate * 0.5)
        
        # Extract patterns
        pattern = self._extract_pattern(experience)
        if pattern:
            self.learned_patterns[task_type].append(pattern)
            
            # Store in knowledge graph
            if self.knowledge_graph:
                await self._store_pattern_in_graph(pattern, task_type)
        
        # Adjust learning parameters
        self._adjust_learning_parameters(experience)
    
    def apply_learned_knowledge(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned knowledge to improve task execution"""
        task_type = task.get("type", "general")
        
        # Get relevant patterns
        relevant_patterns = self._find_relevant_patterns(task, task_type)
        
        # Generate recommendations
        recommendations = {
            "skill_level": self.skill_levels[task_type],
            "confidence_modifier": self._calculate_confidence_modifier(task_type),
            "suggested_strategies": self._suggest_strategies(relevant_patterns),
            "avoid_patterns": self._identify_failure_patterns(task_type),
            "exploration_suggested": np.random.random() < self.exploration_rate
        }
        
        return recommendations
    
    def consolidate_learning(self):
        """Consolidate and generalize learned patterns"""
        for task_type, patterns in self.learned_patterns.items():
            if len(patterns) >= 10:
                # Find common successful patterns
                successful_patterns = [p for p in patterns if p.get("success", False)]
                
                if len(successful_patterns) >= 5:
                    # Extract common elements
                    generalized = self._generalize_patterns(successful_patterns)
                    
                    # Replace specific patterns with generalized ones
                    self.learned_patterns[task_type] = [generalized] + patterns[-5:]
    
    def _extract_pattern(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract learnable pattern from experience"""
        pattern = {
            "context": experience.get("context", {}),
            "approach": experience.get("approach", {}),
            "success": experience.get("success", False),
            "duration": experience.get("duration", 0),
            "resources": experience.get("resources_used", {}),
            "timestamp": datetime.utcnow()
        }
        
        # Only keep patterns with meaningful information
        if pattern["context"] or pattern["approach"]:
            return pattern
        
        return None
    
    def _find_relevant_patterns(self, task: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Find patterns relevant to the current task"""
        relevant = []
        
        for pattern in self.learned_patterns[task_type]:
            relevance = self._calculate_pattern_relevance(pattern, task)
            if relevance > 0.5:
                relevant.append((relevance, pattern))
        
        # Sort by relevance
        relevant.sort(key=lambda x: x[0], reverse=True)
        
        return [pattern for _, pattern in relevant[:5]]
    
    def _calculate_pattern_relevance(self, pattern: Dict[str, Any], task: Dict[str, Any]) -> float:
        """Calculate relevance between pattern and task"""
        relevance = 0.0
        
        # Context similarity
        pattern_context = pattern.get("context", {})
        task_context = task.get("context", {})
        
        if pattern_context and task_context:
            common_keys = set(pattern_context.keys()) & set(task_context.keys())
            if common_keys:
                matches = sum(1 for k in common_keys if pattern_context[k] == task_context[k])
                relevance += matches / len(common_keys) * 0.5
        
        # Success weight
        if pattern.get("success", False):
            relevance += 0.3
        
        # Recency weight
        pattern_age = (datetime.utcnow() - pattern.get("timestamp", datetime.utcnow())).total_seconds() / 86400
        recency = np.exp(-pattern_age / 30)  # 30-day half-life
        relevance += recency * 0.2
        
        return relevance
    
    def _calculate_confidence_modifier(self, task_type: str) -> float:
        """Calculate confidence modifier based on skill level"""
        skill = self.skill_levels[task_type]
        
        # Higher skill -> higher confidence
        # But not too high to avoid overconfidence
        return 0.7 + (skill * 0.25)
    
    def _suggest_strategies(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest strategies based on successful patterns"""
        strategies = []
        
        # Group successful patterns
        successful = [p for p in patterns if p.get("success", False)]
        
        for pattern in successful[:3]:
            strategy = {
                "approach": pattern.get("approach", {}),
                "expected_success_rate": 0.7 + (0.1 * len(successful) / len(patterns)) if patterns else 0.7,
                "resources_needed": pattern.get("resources", {})
            }
            strategies.append(strategy)
        
        return strategies
    
    def _identify_failure_patterns(self, task_type: str) -> List[Dict[str, Any]]:
        """Identify patterns to avoid"""
        failures = []
        
        for pattern in self.learned_patterns[task_type]:
            if not pattern.get("success", False):
                failures.append({
                    "context": pattern.get("context", {}),
                    "approach": pattern.get("approach", {}),
                    "failure_reason": pattern.get("error", "unknown")
                })
        
        return failures[-3:]  # Return most recent failures
    
    def _generalize_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generalize from multiple patterns"""
        generalized = {
            "type": "generalized",
            "based_on": len(patterns),
            "success_rate": sum(1 for p in patterns if p.get("success", False)) / len(patterns),
            "common_context": {},
            "common_approach": {},
            "average_duration": np.mean([p.get("duration", 0) for p in patterns])
        }
        
        # Find common context elements
        all_contexts = [p.get("context", {}) for p in patterns]
        if all_contexts:
            common_keys = set(all_contexts[0].keys())
            for ctx in all_contexts[1:]:
                common_keys &= set(ctx.keys())
            
            for key in common_keys:
                values = [ctx[key] for ctx in all_contexts]
                if len(set(values)) == 1:  # All same value
                    generalized["common_context"][key] = values[0]
        
        return generalized
    
    def _adjust_learning_parameters(self, experience: Dict[str, Any]):
        """Adjust learning parameters based on experience"""
        if experience.get("success", False):
            # Successful experience - can reduce exploration
            self.exploration_rate = max(0.05, self.exploration_rate * 0.99)
        else:
            # Failed experience - increase exploration
            self.exploration_rate = min(0.3, self.exploration_rate * 1.01)
        
        # Adjust learning rate based on confidence
        if "confidence" in experience:
            confidence_error = abs(experience["confidence"] - (1.0 if experience["success"] else 0.0))
            if confidence_error > 0.3:
                # Large error - increase learning rate
                self.learning_rate = min(0.1, self.learning_rate * 1.05)
            else:
                # Small error - decrease learning rate
                self.learning_rate = max(0.001, self.learning_rate * 0.95)
    
    async def _store_pattern_in_graph(self, pattern: Dict[str, Any], task_type: str):
        """Store learned pattern in knowledge graph"""
        if not self.knowledge_graph:
            return
        
        try:
            query_engine = self.knowledge_graph.get_query_engine()
            if query_engine:
                # Create pattern node
                await query_engine.create_node({
                    "type": "LearnedPattern",
                    "task_type": task_type,
                    "success_rate": 1.0 if pattern.get("success", False) else 0.0,
                    "context": json.dumps(pattern.get("context", {})),
                    "approach": json.dumps(pattern.get("approach", {})),
                    "timestamp": pattern.get("timestamp", datetime.utcnow()).isoformat()
                })
        except Exception as e:
            logger.error(f"Failed to store pattern in graph: {e}")


class UnifiedCognitiveSystem:
    """
    Main cognitive architecture that integrates all components
    """
    
    def __init__(self, agent_registry=None, knowledge_graph_manager=None):
        # Core components
        self.working_memory = WorkingMemory(capacity=7)
        self.episodic_memory = EpisodicMemory(max_episodes=10000)
        self.attention = AttentionMechanism(max_concurrent_focus=3)
        self.executive_control = ExecutiveControl(agent_registry, knowledge_graph_manager)
        self.metacognitive_monitor = MetacognitiveMonitor()
        self.learning_system = LearningSystem(knowledge_graph_manager)
        
        # System state
        self.cognitive_state = CognitiveState.IDLE
        self.active_reasoning_chains: Dict[str, ReasoningChain] = {}
        self.agent_registry = agent_registry
        self.knowledge_graph = knowledge_graph_manager
        
        # Performance metrics
        self.metrics = {
            "total_tasks_processed": 0,
            "successful_tasks": 0,
            "average_response_time": 0.0,
            "memory_utilization": 0.0,
            "attention_efficiency": 0.0
        }
        
        self.logger = logging.getLogger("UnifiedCognitiveSystem")
    
    async def process_task(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for task processing with full cognitive capabilities
        """
        start_time = time.time()
        task_id = task.get("id", str(uuid.uuid4()))
        
        try:
            # Update cognitive state
            self.cognitive_state = CognitiveState.PROCESSING
            
            # Store in working memory
            task_memory = MemoryItem(
                content=task,
                memory_type=MemoryType.WORKING,
                importance=task.get("priority", 0.5),
                context=context or {}
            )
            self.working_memory.add(task_memory)
            
            # Recall relevant episodic memories
            memory_cues = {
                "task_type": task.get("type", "general"),
                "domain": task.get("domain", "general")
            }
            relevant_memories = self.episodic_memory.recall(memory_cues)
            
            # Apply learned knowledge
            learning_recommendations = self.learning_system.apply_learned_knowledge(task)
            
            # Plan execution
            constraints = {
                "max_agents": task.get("max_agents", 5),
                "time_limit": task.get("time_limit", 3600),
                "resource_limit": task.get("resource_limit", 1.0)
            }
            
            execution_plan = await self.executive_control.plan_execution(
                {"type": task.get("type", "general"), "goal": task.get("goal")},
                constraints
            )
            
            # Allocate attention
            selected_agents = []
            for step in execution_plan:
                selected_agents.extend(step.get("agents", []))
            
            attention_focus = self.attention.allocate_attention(
                task_id,
                list(set(selected_agents)),
                task.get("priority", 0.5),
                AttentionMode.FOCUSED
            )
            
            if not attention_focus:
                return {
                    "status": "failed",
                    "error": "Insufficient attention resources",
                    "task_id": task_id
                }
            
            # Create reasoning chain
            self.cognitive_state = CognitiveState.REASONING
            reasoning_chain = ReasoningChain(
                reasoning_type=ReasoningType(task.get("reasoning_type", "deductive"))
            )
            self.active_reasoning_chains[task_id] = reasoning_chain
            
            # Execute plan with coordination
            results = []
            for step in execution_plan:
                step_result = await self.executive_control.coordinate_agents(
                    step["task"],
                    step["agents"]
                )
                
                reasoning_chain.add_step(
                    step["task"].get("type", "unknown"),
                    step_result,
                    step["agents"][0] if step["agents"] else "system",
                    confidence=0.8
                )
                
                results.append(step_result)
            
            # Synthesize results
            final_result = self._synthesize_results(results, task, learning_recommendations)
            
            # Complete reasoning chain
            reasoning_chain.complete(
                final_result,
                confidence=learning_recommendations.get("confidence_modifier", 0.8)
            )
            
            # Store episode
            episode_content = {
                "task": task,
                "result": final_result,
                "reasoning_chain": reasoning_chain.id,
                "agents_used": selected_agents
            }
            
            self.episodic_memory.store_episode(
                episode_content,
                context or {},
                selected_agents,
                importance=task.get("priority", 0.5)
            )
            
            # Monitor performance
            success = final_result.get("success", True)
            self.metacognitive_monitor.monitor_performance(
                task,
                final_result,
                learning_recommendations.get("confidence_modifier", 0.8),
                success
            )
            
            # Learn from experience
            self.cognitive_state = CognitiveState.LEARNING
            await self.learning_system.learn_from_experience({
                "task_type": task.get("type", "general"),
                "success": success,
                "context": context or {},
                "approach": {"agents": selected_agents, "reasoning": reasoning_chain.reasoning_type.value},
                "duration": time.time() - start_time,
                "resources_used": {"agents": len(selected_agents)}
            })
            
            # Release attention
            self.attention.release_attention(task_id)
            
            # Update metrics
            self._update_metrics(success, time.time() - start_time)
            
            # Return comprehensive result
            return {
                "task_id": task_id,
                "status": "completed" if success else "failed",
                "result": final_result,
                "reasoning_chain_id": reasoning_chain.id,
                "confidence": learning_recommendations.get("confidence_modifier", 0.8),
                "agents_used": selected_agents,
                "execution_time": time.time() - start_time,
                "memory_references": [m.id for m in relevant_memories],
                "learning_applied": learning_recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            
            # Store failure episode
            self.episodic_memory.store_episode(
                {"task": task, "error": str(e)},
                context or {},
                [],
                importance=0.8
            )
            
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
        
        finally:
            # Clean up reasoning chain
            if task_id in self.active_reasoning_chains:
                del self.active_reasoning_chains[task_id]
            
            # Return to idle state
            self.cognitive_state = CognitiveState.IDLE
    
    async def reflect(self) -> Dict[str, Any]:
        """
        Perform metacognitive reflection on system performance
        """
        self.cognitive_state = CognitiveState.REFLECTING
        
        try:
            # Get performance insights
            performance_insights = self.metacognitive_monitor.reflect_on_performance()
            
            # Consolidate episodic memories
            self.episodic_memory.consolidate()
            
            # Consolidate learning
            self.learning_system.consolidate_learning()
            
            # Analyze attention patterns
            attention_stats = {
                "current_focus": len(self.attention.current_focus),
                "attention_distribution": self.attention.get_attention_distribution(),
                "resource_utilization": 1.0 - self.attention.resource_pool
            }
            
            # Working memory analysis
            working_memory_stats = {
                "utilization": len(self.working_memory.items) / self.working_memory.capacity,
                "average_activation": np.mean([item.get_activation_strength() for item in self.working_memory.items]) if self.working_memory.items else 0
            }
            
            reflection_result = {
                "timestamp": datetime.utcnow().isoformat(),
                "performance_insights": performance_insights,
                "attention_analysis": attention_stats,
                "working_memory_analysis": working_memory_stats,
                "system_metrics": self.metrics,
                "cognitive_state": self.cognitive_state.value,
                "active_reasoning_chains": len(self.active_reasoning_chains)
            }
            
            # Store reflection as episodic memory
            self.episodic_memory.store_episode(
                {"type": "reflection", "insights": reflection_result},
                {"system": "metacognitive"},
                ["metacognitive_monitor"],
                importance=0.9
            )
            
            return reflection_result
            
        finally:
            self.cognitive_state = CognitiveState.IDLE
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """
        Get current cognitive system state
        """
        return {
            "state": self.cognitive_state.value,
            "working_memory": {
                "items": len(self.working_memory.items),
                "capacity": self.working_memory.capacity,
                "chunks": len(self.working_memory.chunks)
            },
            "episodic_memory": {
                "episodes": len(self.episodic_memory.episodes),
                "indices": len(self.episodic_memory.episode_index)
            },
            "attention": {
                "current_focus": [
                    {
                        "task_id": focus.task_id,
                        "agents": focus.agents,
                        "priority": focus.priority,
                        "duration": focus.duration().total_seconds()
                    }
                    for focus in self.attention.current_focus
                ],
                "resource_pool": self.attention.resource_pool,
                "mode": self.attention.mode.value
            },
            "active_reasoning": len(self.active_reasoning_chains),
            "learning": {
                "skill_levels": dict(self.learning_system.skill_levels),
                "exploration_rate": self.learning_system.exploration_rate,
                "learning_rate": self.learning_system.learning_rate
            },
            "metrics": self.metrics
        }
    
    def _synthesize_results(self, results: List[Dict[str, Any]], task: Dict[str, Any], 
                          learning: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize results from multiple execution steps
        """
        synthesis = {
            "success": all(r.get("status") == "completed" for r in results),
            "steps_completed": len(results),
            "final_output": None,
            "insights": [],
            "confidence": learning.get("confidence_modifier", 0.8)
        }
        
        # Combine outputs
        outputs = []
        for result in results:
            if "results" in result and "final_output" in result["results"]:
                outputs.append(result["results"]["final_output"])
        
        if outputs:
            synthesis["final_output"] = " ".join(str(o) for o in outputs if o)
        
        # Extract insights
        for result in results:
            if "results" in result and "intermediate_insights" in result["results"]:
                synthesis["insights"].extend(result["results"]["intermediate_insights"])
        
        return synthesis
    
    def _update_metrics(self, success: bool, execution_time: float):
        """
        Update system performance metrics
        """
        self.metrics["total_tasks_processed"] += 1
        
        if success:
            self.metrics["successful_tasks"] += 1
        
        # Update average response time
        n = self.metrics["total_tasks_processed"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = ((n - 1) * current_avg + execution_time) / n
        
        # Update memory utilization
        self.metrics["memory_utilization"] = len(self.working_memory.items) / self.working_memory.capacity
        
        # Update attention efficiency
        if self.attention.current_focus:
            self.metrics["attention_efficiency"] = 1.0 - self.attention.resource_pool
        else:
            self.metrics["attention_efficiency"] = 0.0


# Global instance
_cognitive_system: Optional[UnifiedCognitiveSystem] = None


def get_cognitive_system() -> Optional[UnifiedCognitiveSystem]:
    """Get the global cognitive system instance"""
    return _cognitive_system


def initialize_cognitive_system(agent_registry=None, knowledge_graph_manager=None) -> UnifiedCognitiveSystem:
    """Initialize the cognitive system"""
    global _cognitive_system
    _cognitive_system = UnifiedCognitiveSystem(agent_registry, knowledge_graph_manager)
    return _cognitive_system