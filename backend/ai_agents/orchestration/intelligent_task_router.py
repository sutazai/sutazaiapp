"""
Intelligent Task Routing and Delegation System
==============================================

This module implements an advanced AI-powered task routing system that intelligently
delegates tasks to optimal agent combinations based on capabilities, performance,
workload, and contextual requirements.

Key Features:
- AI-powered agent selection
- Dynamic load balancing
- Performance-based routing
- Context-aware task decomposition
- Resource optimization
- Failure recovery and rerouting
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import uuid

from .multi_agent_workflow_system import (
    AgentProfile, Task, TaskPriority, AgentCapability,
    MultiAgentWorkflowSystem
)
from ..protocols.enhanced_agent_communication import (
    EnhancedAgentCommunication, Message, MessageFactory,
    CommunicationPattern, MessagePriority
)

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Task routing strategies"""
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_MATCHED = "capability_matched"
    RESOURCE_OPTIMIZED = "resource_optimized"
    CONTEXTUAL = "contextual"
    LEARNING_BASED = "learning_based"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    CRITICAL = 4


@dataclass
class RoutingDecision:
    """Represents a routing decision"""
    task_id: str
    selected_agents: List[str]
    strategy_used: RoutingStrategy
    confidence_score: float
    reasoning: str
    estimated_duration: Optional[float] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    fallback_agents: List[str] = field(default_factory=list)


@dataclass
class AgentPerformanceMetrics:
    """Detailed agent performance metrics"""
    agent_id: str
    success_rate: float = 1.0
    average_response_time: float = 1.0
    current_load: int = 0
    reliability_score: float = 1.0
    specialization_scores: Dict[AgentCapability, float] = field(default_factory=dict)
    recent_tasks: deque = field(default_factory=lambda: deque(maxlen=50))
    resource_efficiency: float = 1.0
    collaboration_score: float = 1.0
    learning_rate: float = 0.1
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TaskContext:
    """Contextual information for task routing"""
    task_id: str
    domain: str
    urgency: float
    complexity: TaskComplexity
    dependencies: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    historical_similar_tasks: List[str] = field(default_factory=list)
    required_quality_level: float = 0.8
    deadline: Optional[datetime] = None


class IntelligentTaskRouter:
    """
    Advanced AI-powered task routing system with learning capabilities
    """
    
    def __init__(self, 
                 workflow_system: MultiAgentWorkflowSystem,
                 communication_system: EnhancedAgentCommunication):
        self.workflow_system = workflow_system
        self.communication_system = communication_system
        
        # Performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        
        # Task history for learning
        self.task_history: Dict[str, Dict[str, Any]] = {}
        self.routing_decisions: Dict[str, RoutingDecision] = {}
        
        # Learning components
        self.capability_matrix = {}  # Agent x Capability performance matrix
        self.collaboration_matrix = {}  # Agent x Agent collaboration scores
        self.task_patterns = {}  # Task pattern recognition
        
        # Configuration
        self.routing_config = {
            "default_strategy": RoutingStrategy.CONTEXTUAL,
            "max_agents_per_task": 5,
            "min_confidence_threshold": 0.6,
            "performance_weight": 0.3,
            "load_weight": 0.2,
            "capability_weight": 0.3,
            "collaboration_weight": 0.2,
            "learning_rate": 0.1,
            "retry_attempts": 3
        }
        
        # Cache for quick lookups
        self._agent_cache = {}
        self._task_similarity_cache = {}
        
        self.running = False
        self._background_tasks: List[asyncio.Task] = []
    
    async def initialize(self):
        """Initialize the intelligent task router"""
        logger.info("Initializing Intelligent Task Router...")
        
        # Initialize agent metrics for all known agents
        await self._initialize_agent_metrics()
        
        # Load historical data
        await self._load_historical_data()
        
        # Start background services
        self._background_tasks = [
            asyncio.create_task(self._performance_updater()),
            asyncio.create_task(self._learning_engine()),
            asyncio.create_task(self._cache_optimizer()),
            asyncio.create_task(self._health_monitor())
        ]
        
        self.running = True
        logger.info("Intelligent Task Router initialized")
    
    async def shutdown(self):
        """Shutdown the router"""
        logger.info("Shutting down Intelligent Task Router...")
        self.running = False
        
        for task in self._background_tasks:
            task.cancel()
        
        # Save learning data
        await self._save_learning_data()
        
        logger.info("Intelligent Task Router shutdown complete")
    
    # ==================== Core Routing Methods ====================
    
    async def route_task(self, 
                        task: Task,
                        context: Optional[TaskContext] = None,
                        strategy: Optional[RoutingStrategy] = None) -> RoutingDecision:
        """
        Intelligently route a task to optimal agents
        """
        # Generate context if not provided
        if not context:
            context = await self._generate_task_context(task)
        
        # Determine routing strategy
        if not strategy:
            strategy = await self._select_routing_strategy(task, context)
        
        # Get candidate agents
        candidates = await self._get_candidate_agents(task, context)
        
        if not candidates:
            raise ValueError(f"No suitable agents found for task {task.id}")
        
        # Score and rank agents
        scored_agents = await self._score_agents(task, candidates, context, strategy)
        
        # Select optimal agent combination
        selected_agents = await self._select_agent_combination(
            task, scored_agents, context, strategy
        )
        
        # Calculate confidence and reasoning
        confidence, reasoning = await self._calculate_confidence(
            task, selected_agents, scored_agents, context
        )
        
        # Create routing decision
        decision = RoutingDecision(
            task_id=task.id,
            selected_agents=selected_agents,
            strategy_used=strategy,
            confidence_score=confidence,
            reasoning=reasoning,
            estimated_duration=await self._estimate_duration(task, selected_agents),
            resource_requirements=await self._calculate_resource_requirements(
                task, selected_agents
            ),
            fallback_agents=await self._select_fallback_agents(
                scored_agents, selected_agents
            )
        )
        
        # Store decision for learning
        self.routing_decisions[task.id] = decision
        
        logger.info(f"Routed task {task.id} to agents {selected_agents} "
                   f"(confidence: {confidence:.2f})")
        
        return decision
    
    async def reroute_task(self, 
                          task_id: str,
                          failure_reason: str) -> Optional[RoutingDecision]:
        """
        Reroute a failed task to alternative agents
        """
        if task_id not in self.routing_decisions:
            logger.error(f"No routing decision found for task {task_id}")
            return None
        
        original_decision = self.routing_decisions[task_id]
        
        # Update agent performance based on failure
        await self._update_failure_metrics(original_decision.selected_agents, failure_reason)
        
        # Get the original task
        task = await self.workflow_system.get_task_status(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return None
        
        # Generate new context with failure information
        context = await self._generate_task_context(task)
        context.historical_similar_tasks.append(task_id)
        
        # Try fallback agents first
        if original_decision.fallback_agents:
            selected_agents = original_decision.fallback_agents[:2]  # Limit fallbacks
            
            decision = RoutingDecision(
                task_id=task_id,
                selected_agents=selected_agents,
                strategy_used=RoutingStrategy.PERFORMANCE_BASED,
                confidence_score=0.7,  # Lower confidence for fallback
                reasoning=f"Fallback routing due to: {failure_reason}"
            )
            
            self.routing_decisions[f"{task_id}_retry"] = decision
            return decision
        
        # Full rerouting with different strategy
        new_strategy = await self._select_fallback_strategy(original_decision.strategy_used)
        return await self.route_task(task, context, new_strategy)
    
    # ==================== Agent Scoring and Selection ====================
    
    async def _get_candidate_agents(self, task: Task, context: TaskContext) -> List[str]:
        """Get candidate agents based on capabilities"""
        candidates = set()
        
        # Find agents with required capabilities
        for capability in task.requirements:
            capable_agents = self.workflow_system.get_agents_by_capability(capability)
            for agent in capable_agents:
                if agent.status == "healthy":
                    candidates.add(agent.id)
        
        # Filter by context requirements
        filtered_candidates = []
        for agent_id in candidates:
            if await self._meets_context_requirements(agent_id, context):
                filtered_candidates.append(agent_id)
        
        return filtered_candidates
    
    async def _score_agents(self, 
                           task: Task, 
                           candidates: List[str],
                           context: TaskContext,
                           strategy: RoutingStrategy) -> List[Tuple[str, float]]:
        """Score candidate agents based on multiple factors"""
        scored_agents = []
        
        for agent_id in candidates:
            score = await self._calculate_agent_score(agent_id, task, context, strategy)
            scored_agents.append((agent_id, score))
        
        # Sort by score (highest first)
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_agents
    
    async def _calculate_agent_score(self,
                                   agent_id: str,
                                   task: Task,
                                   context: TaskContext,
                                   strategy: RoutingStrategy) -> float:
        """Calculate comprehensive agent score"""
        if agent_id not in self.agent_metrics:
            await self._initialize_agent_metric(agent_id)
        
        metrics = self.agent_metrics[agent_id]
        base_score = 0.0
        
        # Performance score (0-100)
        performance_score = (
            metrics.success_rate * 30 +
            (1 / max(metrics.average_response_time, 0.1)) * 20 +
            metrics.reliability_score * 30 +
            metrics.resource_efficiency * 20
        )
        
        # Capability match score (0-100)
        capability_score = await self._calculate_capability_score(agent_id, task)
        
        # Load balancing score (0-100)
        load_score = max(0, 100 - (metrics.current_load * 10))
        
        # Collaboration score (0-100)
        collaboration_score = metrics.collaboration_score * 100
        
        # Context relevance score (0-100)
        context_score = await self._calculate_context_score(agent_id, context)
        
        # Learning-based adjustment (-20 to +20)
        learning_adjustment = await self._calculate_learning_adjustment(agent_id, task)
        
        # Weight scores based on strategy
        if strategy == RoutingStrategy.PERFORMANCE_BASED:
            base_score = performance_score * 0.6 + capability_score * 0.4
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            base_score = load_score * 0.5 + performance_score * 0.3 + capability_score * 0.2
        elif strategy == RoutingStrategy.CAPABILITY_MATCHED:
            base_score = capability_score * 0.6 + performance_score * 0.4
        elif strategy == RoutingStrategy.RESOURCE_OPTIMIZED:
            resource_score = metrics.resource_efficiency * 100
            base_score = resource_score * 0.4 + performance_score * 0.3 + load_score * 0.3
        elif strategy == RoutingStrategy.CONTEXTUAL:
            base_score = (
                context_score * 0.3 +
                capability_score * 0.25 +
                performance_score * 0.25 +
                collaboration_score * 0.2
            )
        elif strategy == RoutingStrategy.LEARNING_BASED:
            base_score = (
                performance_score * 0.4 +
                capability_score * 0.3 +
                collaboration_score * 0.2 +
                context_score * 0.1
            )
        
        # Apply learning adjustment
        final_score = base_score + learning_adjustment
        
        # Ensure score is within bounds
        return max(0, min(100, final_score))
    
    async def _select_agent_combination(self,
                                      task: Task,
                                      scored_agents: List[Tuple[str, float]],
                                      context: TaskContext,
                                      strategy: RoutingStrategy) -> List[str]:
        """Select optimal combination of agents"""
        # Determine number of agents needed
        num_agents = await self._determine_agent_count(task, context)
        
        if num_agents == 1:
            # Single agent selection
            return [scored_agents[0][0]]
        
        # Multi-agent selection with collaboration consideration
        selected = []
        remaining_agents = scored_agents.copy()
        
        # Select primary agent (highest score)
        primary_agent = remaining_agents.pop(0)[0]
        selected.append(primary_agent)
        
        # Select additional agents considering collaboration
        for _ in range(num_agents - 1):
            if not remaining_agents:
                break
            
            best_collaborator = None
            best_score = -1
            
            for agent_id, base_score in remaining_agents:
                # Calculate collaboration bonus
                collaboration_bonus = await self._calculate_collaboration_bonus(
                    agent_id, selected
                )
                
                # Avoid capability overlap penalty
                overlap_penalty = await self._calculate_overlap_penalty(
                    agent_id, selected, task
                )
                
                combined_score = base_score + collaboration_bonus - overlap_penalty
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_collaborator = agent_id
            
            if best_collaborator:
                selected.append(best_collaborator)
                remaining_agents = [
                    (aid, score) for aid, score in remaining_agents 
                    if aid != best_collaborator
                ]
        
        return selected
    
    # ==================== Context and Intelligence ====================
    
    async def _generate_task_context(self, task: Task) -> TaskContext:
        """Generate contextual information for a task"""
        # Analyze task description for domain and complexity
        domain = await self._classify_task_domain(task)
        complexity = await self._assess_task_complexity(task)
        urgency = self._calculate_urgency(task)
        
        # Find similar historical tasks
        similar_tasks = await self._find_similar_tasks(task)
        
        return TaskContext(
            task_id=task.id,
            domain=domain,
            urgency=urgency,
            complexity=complexity,
            historical_similar_tasks=similar_tasks,
            required_quality_level=0.8,  # Default quality requirement
            deadline=task.constraints.get("deadline")
        )
    
    async def _select_routing_strategy(self, 
                                     task: Task, 
                                     context: TaskContext) -> RoutingStrategy:
        """Intelligently select routing strategy"""
        # Critical tasks -> Performance-based
        if task.priority == TaskPriority.CRITICAL:
            return RoutingStrategy.PERFORMANCE_BASED
        
        # High complexity -> Contextual
        if context.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            return RoutingStrategy.CONTEXTUAL
        
        # High urgency -> Load-balanced
        if context.urgency > 0.8:
            return RoutingStrategy.LOAD_BALANCED
        
        # Learning opportunities -> Learning-based
        if len(context.historical_similar_tasks) > 5:
            return RoutingStrategy.LEARNING_BASED
        
        # Default to contextual
        return RoutingStrategy.CONTEXTUAL
    
    async def _calculate_capability_score(self, agent_id: str, task: Task) -> float:
        """Calculate how well agent capabilities match task requirements"""
        if agent_id not in self.workflow_system.agents:
            return 0.0
        
        agent = self.workflow_system.agents[agent_id]
        agent_caps = agent.capabilities
        required_caps = task.requirements
        
        if not required_caps:
            return 100.0
        
        # Calculate intersection and specialization scores
        matching_caps = agent_caps.intersection(required_caps)
        match_ratio = len(matching_caps) / len(required_caps)
        
        # Get specialization scores for matching capabilities
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            specialization_bonus = 0.0
            
            for cap in matching_caps:
                spec_score = metrics.specialization_scores.get(cap, 0.5)
                specialization_bonus += spec_score
            
            if matching_caps:
                specialization_bonus /= len(matching_caps)
        else:
            specialization_bonus = 0.5
        
        # Combine match ratio and specialization
        capability_score = (match_ratio * 70) + (specialization_bonus * 30)
        
        return min(100, capability_score)
    
    async def _calculate_context_score(self, agent_id: str, context: TaskContext) -> float:
        """Calculate context relevance score"""
        score = 50.0  # Base score
        
        # Domain expertise
        if agent_id in self.agent_metrics:
            # This would be enhanced with actual domain tracking
            score += 20
        
        # Historical performance on similar tasks
        if context.historical_similar_tasks:
            similar_performance = await self._get_historical_performance(
                agent_id, context.historical_similar_tasks
            )
            score += similar_performance * 30
        
        # Deadline pressure handling
        if context.deadline:
            time_pressure_score = await self._calculate_time_pressure_score(
                agent_id, context.deadline
            )
            score += time_pressure_score
        
        return min(100, score)
    
    # ==================== Learning and Adaptation ====================
    
    async def _learning_engine(self):
        """Background learning engine"""
        while self.running:
            try:
                # Update capability matrix
                await self._update_capability_matrix()
                
                # Update collaboration matrix
                await self._update_collaboration_matrix()
                
                # Learn task patterns
                await self._learn_task_patterns()
                
                # Adjust routing parameters
                await self._adjust_routing_parameters()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Learning engine error: {e}")
    
    async def _update_capability_matrix(self):
        """Update agent capability performance matrix"""
        for agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            
            # Analyze recent tasks to update specialization scores
            for task_info in list(metrics.recent_tasks):
                if 'capabilities_used' in task_info and 'success' in task_info:
                    success = task_info['success']
                    for capability in task_info['capabilities_used']:
                        current_score = metrics.specialization_scores.get(capability, 0.5)
                        
                        # Update using learning rate
                        if success:
                            new_score = current_score + (metrics.learning_rate * (1.0 - current_score))
                        else:
                            new_score = current_score - (metrics.learning_rate * current_score)
                        
                        metrics.specialization_scores[capability] = max(0.1, min(1.0, new_score))
    
    async def _update_collaboration_matrix(self):
        """Update agent collaboration effectiveness matrix"""
        # Analyze multi-agent task outcomes
        for task_id, decision in self.routing_decisions.items():
            if len(decision.selected_agents) > 1:
                task_outcome = await self._get_task_outcome(task_id)
                if task_outcome:
                    success = task_outcome.get('success', False)
                    
                    # Update collaboration scores between agent pairs
                    for i, agent1 in enumerate(decision.selected_agents):
                        for agent2 in decision.selected_agents[i+1:]:
                            key = f"{agent1}:{agent2}"
                            if key not in self.collaboration_matrix:
                                self.collaboration_matrix[key] = 0.5
                            
                            current_score = self.collaboration_matrix[key]
                            if success:
                                new_score = current_score + (0.1 * (1.0 - current_score))
                            else:
                                new_score = current_score - (0.1 * current_score)
                            
                            self.collaboration_matrix[key] = max(0.1, min(1.0, new_score))
    
    # ==================== Background Services ====================
    
    async def _performance_updater(self):
        """Update agent performance metrics"""
        while self.running:
            try:
                for agent_id in self.workflow_system.agents:
                    await self._update_agent_performance(agent_id)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance updater error: {e}")
    
    async def _update_agent_performance(self, agent_id: str):
        """Update performance metrics for a specific agent"""
        if agent_id not in self.agent_metrics:
            await self._initialize_agent_metric(agent_id)
        
        metrics = self.agent_metrics[agent_id]
        
        # Update current load
        current_tasks = [
            t for t in self.workflow_system.active_tasks.values()
            if agent_id in t.assigned_agents
        ]
        metrics.current_load = len(current_tasks)
        
        # Update success rate and response time from recent tasks
        if metrics.recent_tasks:
            successful_tasks = [t for t in metrics.recent_tasks if t.get('success', False)]
            metrics.success_rate = len(successful_tasks) / len(metrics.recent_tasks)
            
            response_times = [t.get('response_time', 1.0) for t in metrics.recent_tasks]
            metrics.average_response_time = sum(response_times) / len(response_times)
        
        metrics.last_updated = datetime.now()
    
    async def _health_monitor(self):
        """Monitor router health and performance"""
        while self.running:
            try:
                # Check for stale metrics
                current_time = datetime.now()
                for agent_id, metrics in self.agent_metrics.items():
                    if current_time - metrics.last_updated > timedelta(minutes=10):
                        logger.warning(f"Stale metrics for agent {agent_id}")
                
                # Log routing statistics
                total_decisions = len(self.routing_decisions)
                if total_decisions > 0:
                    strategies_used = defaultdict(int)
                    avg_confidence = 0.0
                    
                    for decision in self.routing_decisions.values():
                        strategies_used[decision.strategy_used.value] += 1
                        avg_confidence += decision.confidence_score
                    
                    avg_confidence /= total_decisions
                    
                    logger.info(f"Router stats - Total decisions: {total_decisions}, "
                              f"Avg confidence: {avg_confidence:.2f}, "
                              f"Strategies: {dict(strategies_used)}")
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    # ==================== Helper Methods ====================
    
    async def _initialize_agent_metrics(self):
        """Initialize metrics for all known agents"""
        for agent_id in self.workflow_system.agents:
            await self._initialize_agent_metric(agent_id)
    
    async def _initialize_agent_metric(self, agent_id: str):
        """Initialize metrics for a single agent"""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(agent_id=agent_id)
    
    async def _determine_agent_count(self, task: Task, context: TaskContext) -> int:
        """Determine optimal number of agents for a task"""
        base_count = 1
        
        # Increase based on complexity
        if context.complexity == TaskComplexity.COMPLEX:
            base_count = 2
        elif context.complexity == TaskComplexity.CRITICAL:
            base_count = 3
        
        # Increase for high priority tasks
        if task.priority == TaskPriority.CRITICAL:
            base_count += 1
        
        # Increase for tasks with many requirements
        if len(task.requirements) > 3:
            base_count += 1
        
        return min(base_count, self.routing_config["max_agents_per_task"])
    
    async def _calculate_confidence(self,
                                  task: Task,
                                  selected_agents: List[str],
                                  scored_agents: List[Tuple[str, float]],
                                  context: TaskContext) -> Tuple[float, str]:
        """Calculate confidence score and reasoning"""
        if not selected_agents:
            return 0.0, "No agents selected"
        
        # Base confidence from agent scores
        selected_scores = [
            score for agent_id, score in scored_agents 
            if agent_id in selected_agents
        ]
        avg_score = sum(selected_scores) / len(selected_scores) if selected_scores else 0
        base_confidence = avg_score / 100
        
        # Adjust based on various factors
        confidence_adjustments = []
        
        # Capability coverage
        total_caps = len(task.requirements)
        covered_caps = 0
        for agent_id in selected_agents:
            if agent_id in self.workflow_system.agents:
                agent_caps = self.workflow_system.agents[agent_id].capabilities
                covered_caps += len(agent_caps.intersection(task.requirements))
        
        coverage_ratio = min(1.0, covered_caps / total_caps) if total_caps > 0 else 1.0
        confidence_adjustments.append(f"capability coverage: {coverage_ratio:.2f}")
        
        # Historical performance
        historical_bonus = 0.0
        if context.historical_similar_tasks:
            # This would check performance on similar tasks
            historical_bonus = 0.1
            confidence_adjustments.append("historical performance bonus")
        
        # Multi-agent coordination penalty
        coordination_penalty = 0.0
        if len(selected_agents) > 1:
            coordination_penalty = 0.1
            confidence_adjustments.append("multi-agent coordination penalty")
        
        # Calculate final confidence
        final_confidence = (
            base_confidence * coverage_ratio + 
            historical_bonus - 
            coordination_penalty
        )
        final_confidence = max(0.0, min(1.0, final_confidence))
        
        # Generate reasoning
        reasoning = (
            f"Selected {len(selected_agents)} agents based on "
            f"{context.complexity.name.lower()} task complexity. "
            f"Adjustments: {', '.join(confidence_adjustments)}"
        )
        
        return final_confidence, reasoning
    
    async def _meets_context_requirements(self, agent_id: str, context: TaskContext) -> bool:
        """Check if agent meets context-specific requirements"""
        # For now, basic checks - can be enhanced with domain-specific logic
        if context.urgency > 0.9:
            # High urgency requires agents with good response times
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                return metrics.average_response_time < 2.0
        
        return True
    
    async def _calculate_collaboration_bonus(self, 
                                           agent_id: str, 
                                           selected_agents: List[str]) -> float:
        """Calculate collaboration bonus for agent"""
        bonus = 0.0
        
        for selected_agent in selected_agents:
            key1 = f"{agent_id}:{selected_agent}"
            key2 = f"{selected_agent}:{agent_id}"
            
            collab_score = (
                self.collaboration_matrix.get(key1, 0.5) +
                self.collaboration_matrix.get(key2, 0.5)
            ) / 2
            
            bonus += (collab_score - 0.5) * 10  # Convert to bonus points
        
        return bonus / len(selected_agents) if selected_agents else 0
    
    async def _calculate_overlap_penalty(self,
                                       agent_id: str,
                                       selected_agents: List[str],
                                       task: Task) -> float:
        """Calculate penalty for capability overlap"""
        if agent_id not in self.workflow_system.agents:
            return 0.0
        
        agent_caps = self.workflow_system.agents[agent_id].capabilities
        
        overlap_count = 0
        total_selected_caps = set()
        
        for selected_id in selected_agents:
            if selected_id in self.workflow_system.agents:
                selected_caps = self.workflow_system.agents[selected_id].capabilities
                total_selected_caps.update(selected_caps)
                overlap_count += len(agent_caps.intersection(selected_caps))
        
        # Penalty for excessive overlap
        if overlap_count > len(task.requirements):
            return (overlap_count - len(task.requirements)) * 5
        
        return 0.0
    
    async def _classify_task_domain(self, task: Task) -> str:
        """Classify task domain based on description and type"""
        # Simple keyword-based classification - can be enhanced with ML
        description = task.description.lower()
        
        if any(word in description for word in ['code', 'programming', 'development']):
            return 'development'
        elif any(word in description for word in ['security', 'vulnerability', 'pentest']):
            return 'security'
        elif any(word in description for word in ['document', 'text', 'analysis']):
            return 'document_processing'
        elif any(word in description for word in ['web', 'browser', 'automation']):
            return 'web_automation'
        elif any(word in description for word in ['financial', 'money', 'trading']):
            return 'financial'
        else:
            return 'general'
    
    async def _assess_task_complexity(self, task: Task) -> TaskComplexity:
        """Assess task complexity"""
        complexity_score = 0
        
        # Based on requirements count
        complexity_score += len(task.requirements)
        
        # Based on description length and keywords
        description = task.description.lower()
        if len(description) > 200:
            complexity_score += 1
        
        complex_keywords = ['integrate', 'optimize', 'analyze', 'complex', 'advanced']
        complexity_score += sum(1 for word in complex_keywords if word in description)
        
        # Based on priority
        if task.priority == TaskPriority.CRITICAL:
            complexity_score += 2
        elif task.priority == TaskPriority.HIGH:
            complexity_score += 1
        
        # Map to enum
        if complexity_score <= 2:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 4:
            return TaskComplexity.MODERATE
        elif complexity_score <= 6:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.CRITICAL
    
    def _calculate_urgency(self, task: Task) -> float:
        """Calculate task urgency (0.0 to 1.0)"""
        # Map priority to urgency
        priority_map = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.8,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.3,
            TaskPriority.BACKGROUND: 0.1
        }
        
        return priority_map.get(task.priority, 0.5)
    
    async def _find_similar_tasks(self, task: Task) -> List[str]:
        """Find historically similar tasks"""
        # Simple similarity based on type and capabilities
        similar_tasks = []
        
        for historical_task_id, task_info in self.task_history.items():
            if (task_info.get('type') == task.type and
                len(set(task_info.get('requirements', [])).intersection(task.requirements)) > 0):
                similar_tasks.append(historical_task_id)
        
        return similar_tasks[-10:]  # Return last 10 similar tasks
    
    async def _estimate_duration(self, task: Task, selected_agents: List[str]) -> float:
        """Estimate task duration in seconds"""
        # Base estimation based on task complexity and agent performance
        base_duration = 300.0  # 5 minutes base
        
        # Adjust based on agent performance
        if selected_agents:
            total_response_time = 0
            for agent_id in selected_agents:
                if agent_id in self.agent_metrics:
                    total_response_time += self.agent_metrics[agent_id].average_response_time
                else:
                    total_response_time += 1.0
            
            avg_response_time = total_response_time / len(selected_agents)
            base_duration *= avg_response_time
        
        # Adjust for multi-agent coordination overhead
        if len(selected_agents) > 1:
            base_duration *= 1.3  # 30% overhead for coordination
        
        return base_duration
    
    async def _calculate_resource_requirements(self, 
                                             task: Task, 
                                             selected_agents: List[str]) -> Dict[str, float]:
        """Calculate total resource requirements"""
        total_requirements = {'memory_mb': 0, 'cpu_cores': 0}
        
        for agent_id in selected_agents:
            if agent_id in self.workflow_system.agents:
                agent = self.workflow_system.agents[agent_id]
                req = agent.resource_requirements
                total_requirements['memory_mb'] += req.get('memory_mb', 0)
                total_requirements['cpu_cores'] += req.get('cpu_cores', 0)
        
        return total_requirements
    
    async def _select_fallback_agents(self,
                                    scored_agents: List[Tuple[str, float]],
                                    selected_agents: List[str]) -> List[str]:
        """Select fallback agents"""
        fallbacks = []
        
        for agent_id, score in scored_agents:
            if agent_id not in selected_agents and len(fallbacks) < 3:
                fallbacks.append(agent_id)
        
        return fallbacks


# ==================== Router Factory ====================

def create_intelligent_router(workflow_system: MultiAgentWorkflowSystem,
                             communication_system: EnhancedAgentCommunication) -> IntelligentTaskRouter:
    """Factory function to create configured router"""
    router = IntelligentTaskRouter(workflow_system, communication_system)
    
    # Configure based on system resources
    router.routing_config.update({
        "max_agents_per_task": 3,  # Limit for resource constraints
        "min_confidence_threshold": 0.7,
        "performance_weight": 0.4,
        "capability_weight": 0.35,
        "load_weight": 0.15,
        "collaboration_weight": 0.1
    })
    
    return router


# ==================== Example Usage ====================

async def example_router_usage():
    """Example of using the intelligent task router"""
    from .multi_agent_workflow_system import MultiAgentWorkflowSystem
    from ..protocols.enhanced_agent_communication import EnhancedAgentCommunication
    
    # Initialize systems
    workflow_system = MultiAgentWorkflowSystem()
    communication_system = EnhancedAgentCommunication()
    router = create_intelligent_router(workflow_system, communication_system)
    
    await workflow_system.initialize()
    await communication_system.initialize()
    await router.initialize()
    
    # Create a sample task
    task = Task(
        id="sample_task",
        type="code_generation",
        description="Create a secure REST API with authentication",
        priority=TaskPriority.HIGH,
        requirements={AgentCapability.CODE_GENERATION, AgentCapability.SECURITY_SCANNING},
        payload={"framework": "FastAPI", "database": "PostgreSQL"}
    )
    
    # Route the task
    decision = await router.route_task(task)
    print(f"Task routed to: {decision.selected_agents}")
    print(f"Confidence: {decision.confidence_score:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    
    # Simulate task completion and update metrics
    # This would normally be done by the workflow system
    
    await router.shutdown()
    await communication_system.shutdown()
    await workflow_system.shutdown()


if __name__ == "__main__":
    asyncio.run(example_router_usage())