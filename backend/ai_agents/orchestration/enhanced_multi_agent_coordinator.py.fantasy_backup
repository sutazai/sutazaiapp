"""
Enhanced Multi-Agent Coordinator for SutazAI AGI/ASI Platform
============================================================

Advanced coordination system that integrates with the autonomous system controller
to provide sophisticated multi-agent collaboration patterns, dynamic load balancing,
and intelligent task distribution with self-optimization capabilities.

Features:
- Dynamic agent discovery and capability matching
- Intelligent task decomposition and distribution
- Real-time collaboration orchestration
- Performance-based agent selection
- Adaptive load balancing
- Failure recovery and self-healing
- Cross-agent knowledge sharing
- Emergent behavior detection
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import networkx as nx
import redis.asyncio as redis
import httpx
from concurrent.futures import ThreadPoolExecutor

# Import base classes
from .multi_agent_workflow_system import (
    MultiAgentWorkflowSystem, AgentProfile, Task, TaskPriority,
    AgentCapability, MessageType
)
from ..reasoning.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType

logger = logging.getLogger(__name__)


class CoordinationPattern(Enum):
    """Multi-agent coordination patterns"""
    PARALLEL = "parallel"           # Independent parallel execution
    SEQUENTIAL = "sequential"       # Sequential handoff
    HIERARCHICAL = "hierarchical"   # Master-worker pattern
    COLLABORATIVE = "collaborative" # Peer-to-peer collaboration
    PIPELINE = "pipeline"          # Data processing pipeline
    SWARM = "swarm"               # Swarm intelligence
    DEMOCRATIC = "democratic"      # Consensus-based decisions


class AgentRole(Enum):
    """Roles agents can play in coordination"""
    COORDINATOR = "coordinator"     # Orchestrates other agents
    WORKER = "worker"              # Executes assigned tasks
    SPECIALIST = "specialist"       # Domain expert
    MONITOR = "monitor"            # Observes and reports
    MEDIATOR = "mediator"          # Resolves conflicts
    OPTIMIZER = "optimizer"        # Improves performance


@dataclass
class CollaborationGraph:
    """Graph representing agent collaboration relationships"""
    nodes: Dict[str, Dict[str, Any]]  # Agent nodes with metadata
    edges: Dict[str, Dict[str, Any]]  # Collaboration relationships
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CoordinationContext:
    """Context for multi-agent coordination"""
    task_id: str
    coordination_pattern: CoordinationPattern
    participating_agents: List[str]
    agent_roles: Dict[str, AgentRole]
    communication_channels: Dict[str, str]
    shared_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    start_time: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class CollaborationSession:
    """Active collaboration session between agents"""
    id: str
    task: Task
    context: CoordinationContext
    messages: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


@dataclass
class AgentPerformanceProfile:
    """Detailed performance profile for an agent"""
    agent_id: str
    success_rate: float
    average_response_time: float
    resource_efficiency: float
    collaboration_score: float
    specialization_scores: Dict[str, float]
    recent_tasks: deque = field(default_factory=lambda: deque(maxlen=100))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_updated: datetime = field(default_factory=datetime.now)


class EnhancedMultiAgentCoordinator:
    """
    Enhanced multi-agent coordination system with advanced AI capabilities
    """
    
    def __init__(self, 
                 workflow_system: MultiAgentWorkflowSystem,
                 reasoning_engine: Optional[AdvancedReasoningEngine] = None):
        self.workflow_system = workflow_system
        self.reasoning_engine = reasoning_engine or AdvancedReasoningEngine()
        
        # Coordination state
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.collaboration_graph = CollaborationGraph({}, {})
        self.coordination_patterns = {}
        
        # Performance tracking
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.coordination_metrics = {
            "successful_collaborations": 0,
            "failed_collaborations": 0,
            "average_coordination_time": 0.0,
            "efficiency_improvements": 0.0
        }
        
        # Learning and adaptation
        self.coordination_history: deque = deque(maxlen=10000)
        self.learned_patterns: Dict[str, Dict[str, Any]] = {}
        self.optimization_suggestions: List[Dict[str, Any]] = []
        
        # Dynamic capabilities
        self.capability_matrix: Dict[str, Dict[str, float]] = {}
        self.agent_compatibility: Dict[Tuple[str, str], float] = {}
        
        # Communication infrastructure
        self.redis_client: Optional[redis.Redis] = None
        self.message_queue = asyncio.Queue()
        self.broadcast_channels: Dict[str, asyncio.Queue] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("Enhanced Multi-Agent Coordinator initialized")
    
    async def initialize(self):
        """Initialize the coordination system"""
        logger.info("🚀 Initializing Enhanced Multi-Agent Coordinator...")
        
        # Connect to Redis
        self.redis_client = await redis.from_url("redis://redis:6379")
        
        # Build initial collaboration graph
        await self._build_collaboration_graph()
        
        # Load coordination patterns
        await self._load_coordination_patterns()
        
        # Initialize performance profiles
        await self._initialize_performance_profiles()
        
        # Start background services
        await self._start_background_services()
        
        self.running = True
        logger.info("✅ Enhanced Multi-Agent Coordinator ready")
    
    async def shutdown(self):
        """Shutdown the coordination system"""
        logger.info("🛑 Shutting down Enhanced Multi-Agent Coordinator...")
        
        self.running = False
        
        # Complete active sessions
        for session in self.active_sessions.values():
            await self._complete_session(session.id, "system_shutdown")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Enhanced Multi-Agent Coordinator shutdown complete")
    
    # ==================== Coordination Orchestration ====================
    
    async def coordinate_task(self, 
                            task: Task,
                            preferred_pattern: Optional[CoordinationPattern] = None,
                            constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Main coordination entry point - orchestrate multi-agent task execution
        """
        session_id = str(uuid.uuid4())
        logger.info(f"🎭 Starting coordination session: {session_id}")
        
        try:
            # Analyze task requirements
            task_analysis = await self._analyze_task_requirements(task)
            
            # Determine optimal coordination pattern
            if not preferred_pattern:
                preferred_pattern = await self._select_coordination_pattern(task, task_analysis)
            
            # Select and assign agents
            agent_assignments = await self._select_and_assign_agents(
                task, preferred_pattern, task_analysis, constraints
            )
            
            # Create coordination context
            context = CoordinationContext(
                task_id=task.id,
                coordination_pattern=preferred_pattern,
                participating_agents=list(agent_assignments.keys()),
                agent_roles=agent_assignments,
                communication_channels=await self._setup_communication_channels(agent_assignments),
                shared_state={},
                performance_metrics={}
            )
            
            # Create collaboration session
            session = CollaborationSession(
                id=session_id,
                task=task,
                context=context
            )
            
            self.active_sessions[session_id] = session
            
            # Execute coordination based on pattern
            result = await self._execute_coordination_pattern(session)
            
            # Complete session
            await self._complete_session(session_id, "success")
            
            logger.info(f"✅ Coordination completed successfully: {session_id}")
            return result
        
        except Exception as e:
            logger.error(f"❌ Coordination failed: {e}")
            await self._complete_session(session_id, "error", str(e))
            raise
    
    async def _analyze_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Analyze task to understand coordination requirements"""
        analysis = {
            "complexity": "medium",
            "estimated_duration": 300,  # seconds
            "resource_requirements": {},
            "interdependencies": [],
            "parallelization_potential": 0.5,
            "communication_intensity": "medium",
            "failure_tolerance": "medium"
        }
        
        # Use reasoning engine for deeper analysis
        reasoning_result = await self.reasoning_engine.reason(
            query=f"Analyze task coordination requirements: {task.description}",
            context={
                "task_type": task.type,
                "priority": task.priority.value,
                "requirements": [cap.value for cap in task.requirements]
            },
            reasoning_type=ReasoningType.STRATEGIC
        )
        
        if reasoning_result.overall_confidence > 0.6:
            # Extract insights from reasoning
            analysis["ai_insights"] = reasoning_result.final_conclusion
            analysis["confidence"] = reasoning_result.overall_confidence
        
        # Analyze based on requirements
        requirement_count = len(task.requirements)
        if requirement_count > 3:
            analysis["complexity"] = "high"
            analysis["parallelization_potential"] = 0.8
        elif requirement_count < 2:
            analysis["complexity"] = "low"
            analysis["parallelization_potential"] = 0.2
        
        # Estimate duration based on task type and complexity
        base_time = 180  # 3 minutes base
        complexity_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}
        analysis["estimated_duration"] = base_time * complexity_multiplier[analysis["complexity"]]
        
        return analysis
    
    async def _select_coordination_pattern(self, 
                                         task: Task,
                                         analysis: Dict[str, Any]) -> CoordinationPattern:
        """Select optimal coordination pattern for the task"""
        
        # Rule-based pattern selection
        if analysis["parallelization_potential"] > 0.7:
            if len(task.requirements) > 4:
                return CoordinationPattern.HIERARCHICAL
            else:
                return CoordinationPattern.PARALLEL
        
        elif analysis["communication_intensity"] == "high":
            return CoordinationPattern.COLLABORATIVE
        
        elif "sequential" in task.description.lower() or "pipeline" in task.description.lower():
            return CoordinationPattern.PIPELINE
        
        elif analysis["complexity"] == "high":
            return CoordinationPattern.HIERARCHICAL
        
        else:
            return CoordinationPattern.PARALLEL
    
    async def _select_and_assign_agents(self, 
                                       task: Task,
                                       pattern: CoordinationPattern,
                                       analysis: Dict[str, Any],
                                       constraints: Optional[Dict[str, Any]] = None) -> Dict[str, AgentRole]:
        """Select agents and assign roles based on coordination pattern"""
        
        assignments = {}
        
        # Get candidate agents for each capability
        candidate_agents = {}
        for capability in task.requirements:
            agents = self.workflow_system.get_agents_by_capability(capability)
            candidate_agents[capability] = agents
        
        if pattern == CoordinationPattern.HIERARCHICAL:
            # Select coordinator and workers
            coordinator = await self._select_best_coordinator(candidate_agents)
            if coordinator:
                assignments[coordinator.id] = AgentRole.COORDINATOR
            
            # Select workers for each capability
            for capability, agents in candidate_agents.items():
                best_worker = await self._select_best_agent_for_capability(agents, capability)
                if best_worker and best_worker.id != coordinator.id:
                    assignments[best_worker.id] = AgentRole.WORKER
        
        elif pattern == CoordinationPattern.COLLABORATIVE:
            # Select peers with good collaboration scores
            all_candidates = []
            for agents in candidate_agents.values():
                all_candidates.extend(agents)
            
            # Remove duplicates
            unique_candidates = {agent.id: agent for agent in all_candidates}.values()
            
            # Select based on collaboration compatibility
            selected_agents = await self._select_collaborative_team(list(unique_candidates), task)
            
            for agent in selected_agents:
                assignments[agent.id] = AgentRole.SPECIALIST
        
        elif pattern == CoordinationPattern.PIPELINE:
            # Create pipeline stages
            capabilities = list(task.requirements)
            for i, capability in enumerate(capabilities):
                agents = candidate_agents[capability]
                best_agent = await self._select_best_agent_for_capability(agents, capability)
                if best_agent:
                    role = AgentRole.COORDINATOR if i == 0 else AgentRole.WORKER
                    assignments[best_agent.id] = role
        
        else:  # PARALLEL or other patterns
            # Select best agent for each capability
            for capability, agents in candidate_agents.items():
                best_agent = await self._select_best_agent_for_capability(agents, capability)
                if best_agent:
                    assignments[best_agent.id] = AgentRole.WORKER
        
        return assignments
    
    # ==================== Pattern-Specific Execution ====================
    
    async def _execute_coordination_pattern(self, session: CollaborationSession) -> Dict[str, Any]:
        """Execute coordination based on the selected pattern"""
        
        pattern = session.context.coordination_pattern
        
        if pattern == CoordinationPattern.HIERARCHICAL:
            return await self._execute_hierarchical_coordination(session)
        elif pattern == CoordinationPattern.COLLABORATIVE:
            return await self._execute_collaborative_coordination(session)
        elif pattern == CoordinationPattern.PIPELINE:
            return await self._execute_pipeline_coordination(session)
        elif pattern == CoordinationPattern.PARALLEL:
            return await self._execute_parallel_coordination(session)
        elif pattern == CoordinationPattern.SWARM:
            return await self._execute_swarm_coordination(session)
        else:
            return await self._execute_default_coordination(session)
    
    async def _execute_hierarchical_coordination(self, session: CollaborationSession) -> Dict[str, Any]:
        """Execute hierarchical coordination pattern"""
        logger.info(f"👑 Executing hierarchical coordination: {session.id}")
        
        # Find coordinator
        coordinator_id = None
        worker_ids = []
        
        for agent_id, role in session.context.agent_roles.items():
            if role == AgentRole.COORDINATOR:
                coordinator_id = agent_id
            else:
                worker_ids.append(agent_id)
        
        if not coordinator_id:
            raise ValueError("No coordinator assigned for hierarchical pattern")
        
        # Phase 1: Task decomposition by coordinator
        decomposition_task = {
            "type": "task_decomposition",
            "description": f"Decompose task: {session.task.description}",
            "payload": {
                "original_task": session.task.payload,
                "available_workers": worker_ids,
                "worker_capabilities": {
                    wid: [cap.value for cap in self.workflow_system.agents[wid].capabilities]
                    for wid in worker_ids if wid in self.workflow_system.agents
                }
            }
        }
        
        decomposition_result = await self._execute_agent_task(coordinator_id, decomposition_task)
        session.results["decomposition"] = decomposition_result
        
        # Phase 2: Distribute subtasks to workers
        subtasks = decomposition_result.get("subtasks", [])
        worker_tasks = []
        
        for i, subtask in enumerate(subtasks):
            worker_id = worker_ids[i % len(worker_ids)]  # Round-robin assignment
            worker_task = asyncio.create_task(
                self._execute_agent_task(worker_id, subtask)
            )
            worker_tasks.append((worker_id, worker_task))
        
        # Wait for worker completion
        worker_results = {}
        for worker_id, worker_task in worker_tasks:
            try:
                result = await worker_task
                worker_results[worker_id] = result
            except Exception as e:
                logger.error(f"Worker {worker_id} failed: {e}")
                worker_results[worker_id] = {"error": str(e)}
        
        session.results["worker_results"] = worker_results
        
        # Phase 3: Result aggregation by coordinator
        aggregation_task = {
            "type": "result_aggregation",
            "description": "Aggregate worker results",
            "payload": {
                "worker_results": worker_results,
                "original_task": session.task.payload
            }
        }
        
        final_result = await self._execute_agent_task(coordinator_id, aggregation_task)
        session.results["final_result"] = final_result
        
        return {
            "coordination_pattern": "hierarchical",
            "coordinator": coordinator_id,
            "workers": worker_ids,
            "result": final_result,
            "performance_metrics": await self._calculate_session_metrics(session)
        }
    
    async def _execute_collaborative_coordination(self, session: CollaborationSession) -> Dict[str, Any]:
        """Execute collaborative coordination pattern"""
        logger.info(f"🤝 Executing collaborative coordination: {session.id}")
        
        agents = list(session.context.agent_roles.keys())
        
        # Create shared workspace
        shared_workspace = {
            "task": session.task.payload,
            "contributions": {},
            "decisions": [],
            "consensus_threshold": 0.7
        }
        
        # Phase 1: Initial contributions from all agents
        initial_tasks = []
        for agent_id in agents:
            contrib_task = {
                "type": "collaborative_contribution",
                "description": f"Contribute to collaborative task: {session.task.description}",
                "payload": {
                    "shared_workspace": shared_workspace,
                    "agent_role": "contributor",
                    "other_agents": [aid for aid in agents if aid != agent_id]
                }
            }
            initial_tasks.append((agent_id, asyncio.create_task(
                self._execute_agent_task(agent_id, contrib_task)
            )))
        
        # Collect initial contributions
        for agent_id, task in initial_tasks:
            try:
                contribution = await task
                shared_workspace["contributions"][agent_id] = contribution
            except Exception as e:
                logger.error(f"Agent {agent_id} contribution failed: {e}")
        
        # Phase 2: Iterative collaboration
        collaboration_rounds = 3
        for round_num in range(collaboration_rounds):
            round_tasks = []
            
            for agent_id in agents:
                collab_task = {
                    "type": "collaborative_iteration",
                    "description": f"Collaborate on task (round {round_num + 1})",
                    "payload": {
                        "shared_workspace": shared_workspace,
                        "round": round_num + 1,
                        "agent_role": "collaborator"
                    }
                }
                round_tasks.append((agent_id, asyncio.create_task(
                    self._execute_agent_task(agent_id, collab_task)
                )))
            
            # Process round results
            round_results = {}
            for agent_id, task in round_tasks:
                try:
                    result = await task
                    round_results[agent_id] = result
                    # Update shared workspace
                    if "workspace_update" in result:
                        shared_workspace.update(result["workspace_update"])
                except Exception as e:
                    logger.error(f"Agent {agent_id} collaboration failed: {e}")
            
            session.results[f"round_{round_num}"] = round_results
            
            # Check for consensus
            consensus_score = await self._calculate_consensus(round_results)
            if consensus_score >= shared_workspace["consensus_threshold"]:
                logger.info(f"Consensus reached in round {round_num + 1}")
                break
        
        # Phase 3: Final synthesis
        synthesis_result = await self._synthesize_collaborative_results(shared_workspace, agents)
        session.results["synthesis"] = synthesis_result
        
        return {
            "coordination_pattern": "collaborative",
            "participants": agents,
            "consensus_score": consensus_score,
            "result": synthesis_result,
            "performance_metrics": await self._calculate_session_metrics(session)
        }
    
    async def _execute_pipeline_coordination(self, session: CollaborationSession) -> Dict[str, Any]:
        """Execute pipeline coordination pattern"""
        logger.info(f"🔄 Executing pipeline coordination: {session.id}")
        
        # Order agents in pipeline sequence
        pipeline_agents = await self._order_agents_for_pipeline(session)
        
        current_data = session.task.payload
        pipeline_results = []
        
        for i, agent_id in enumerate(pipeline_agents):
            stage_task = {
                "type": "pipeline_stage",
                "description": f"Pipeline stage {i + 1}: {session.task.description}",
                "payload": {
                    "input_data": current_data,
                    "stage_number": i + 1,
                    "total_stages": len(pipeline_agents),
                    "next_agent": pipeline_agents[i + 1] if i + 1 < len(pipeline_agents) else None
                }
            }
            
            try:
                stage_result = await self._execute_agent_task(agent_id, stage_task)
                pipeline_results.append({
                    "agent": agent_id,
                    "stage": i + 1,
                    "result": stage_result
                })
                
                # Pass output to next stage
                current_data = stage_result.get("output", current_data)
                
            except Exception as e:
                logger.error(f"Pipeline stage {i + 1} failed: {e}")
                pipeline_results.append({
                    "agent": agent_id,
                    "stage": i + 1,
                    "error": str(e)
                })
                break
        
        session.results["pipeline_results"] = pipeline_results
        
        return {
            "coordination_pattern": "pipeline",
            "pipeline_sequence": pipeline_agents,
            "stages_completed": len(pipeline_results),
            "final_output": current_data,
            "performance_metrics": await self._calculate_session_metrics(session)
        }
    
    # ==================== Performance and Learning ====================
    
    async def _update_performance_profiles(self, session: CollaborationSession):
        """Update performance profiles based on session results"""
        
        session_duration = (datetime.now() - session.context.start_time).total_seconds()
        success = session.results.get("success", False)
        
        for agent_id in session.context.participating_agents:
            if agent_id not in self.agent_profiles:
                self.agent_profiles[agent_id] = AgentPerformanceProfile(
                    agent_id=agent_id,
                    success_rate=0.0,
                    average_response_time=0.0,
                    resource_efficiency=0.0,
                    collaboration_score=0.0,
                    specialization_scores={}
                )
            
            profile = self.agent_profiles[agent_id]
            
            # Update success rate
            profile.recent_tasks.append({
                "session_id": session.id,
                "success": success,
                "duration": session_duration,
                "timestamp": datetime.now()
            })
            
            recent_successes = sum(1 for task in profile.recent_tasks if task["success"])
            profile.success_rate = recent_successes / len(profile.recent_tasks)
            
            # Update response time
            profile.average_response_time = np.mean([
                task["duration"] for task in profile.recent_tasks
            ])
            
            # Update collaboration score based on role and interaction
            role = session.context.agent_roles.get(agent_id, AgentRole.WORKER)
            if role == AgentRole.COORDINATOR:
                collaboration_bonus = 0.1 if success else -0.05
            else:
                collaboration_bonus = 0.05 if success else -0.02
            
            profile.collaboration_score = max(0.0, min(1.0, 
                profile.collaboration_score + collaboration_bonus))
    
    async def _learn_coordination_patterns(self, session: CollaborationSession):
        """Learn from coordination session to improve future performance"""
        
        pattern_key = f"{session.context.coordination_pattern.value}_{len(session.context.participating_agents)}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "pattern": session.context.coordination_pattern.value,
                "agent_count": len(session.context.participating_agents),
                "success_count": 0,
                "total_count": 0,
                "average_duration": 0.0,
                "best_configurations": []
            }
        
        pattern_data = self.learned_patterns[pattern_key]
        pattern_data["total_count"] += 1
        
        session_duration = (datetime.now() - session.context.start_time).total_seconds()
        success = session.results.get("success", False)
        
        if success:
            pattern_data["success_count"] += 1
            
            # Store successful configuration
            config = {
                "agents": list(session.context.agent_roles.keys()),
                "roles": {k: v.value for k, v in session.context.agent_roles.items()},
                "duration": session_duration,
                "metrics": session.context.performance_metrics
            }
            
            pattern_data["best_configurations"].append(config)
            
            # Keep only top 10 configurations
            pattern_data["best_configurations"] = sorted(
                pattern_data["best_configurations"],
                key=lambda x: x["duration"]
            )[:10]
        
        # Update average duration
        total_duration = pattern_data["average_duration"] * (pattern_data["total_count"] - 1)
        pattern_data["average_duration"] = (total_duration + session_duration) / pattern_data["total_count"]
    
    # ==================== Helper Methods ====================
    
    async def _execute_agent_task(self, agent_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task on a specific agent"""
        
        if agent_id not in self.workflow_system.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        agent = self.workflow_system.agents[agent_id]
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.url}:{agent.port}/execute",
                    json=task_config,
                    timeout=300.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise RuntimeError(f"Agent returned status {response.status_code}")
        
        except Exception as e:
            logger.error(f"Failed to execute task on agent {agent_id}: {e}")
            raise
    
    async def _select_best_agent_for_capability(self, 
                                              agents: List[AgentProfile],
                                              capability: AgentCapability) -> Optional[AgentProfile]:
        """Select the best agent for a specific capability"""
        
        if not agents:
            return None
        
        scored_agents = []
        
        for agent in agents:
            if agent.id in self.agent_profiles:
                profile = self.agent_profiles[agent.id]
                
                # Calculate composite score
                capability_score = profile.specialization_scores.get(capability.value, 0.5)
                performance_score = (profile.success_rate + profile.resource_efficiency) / 2
                collaboration_score = profile.collaboration_score
                
                # Current load penalty
                current_load = self.workflow_system.agent_workloads.get(agent.id, 0)
                load_penalty = min(0.5, current_load * 0.1)
                
                total_score = (
                    capability_score * 0.4 +
                    performance_score * 0.3 +
                    collaboration_score * 0.2 +
                    (1.0 - load_penalty) * 0.1
                )
                
                scored_agents.append((agent, total_score))
            else:
                # New agent - give moderate score
                scored_agents.append((agent, 0.5))
        
        # Return agent with highest score
        best_agent, _ = max(scored_agents, key=lambda x: x[1])
        return best_agent
    
    async def _calculate_session_metrics(self, session: CollaborationSession) -> Dict[str, float]:
        """Calculate performance metrics for a coordination session"""
        
        duration = (datetime.now() - session.context.start_time).total_seconds()
        agent_count = len(session.context.participating_agents)
        
        metrics = {
            "duration": duration,
            "agent_count": agent_count,
            "efficiency": agent_count / max(1, duration / 60),  # agents per minute
            "success": 1.0 if session.results.get("success", False) else 0.0
        }
        
        # Add pattern-specific metrics
        pattern = session.context.coordination_pattern
        if pattern == CoordinationPattern.HIERARCHICAL:
            metrics["coordination_overhead"] = len(session.results.get("worker_results", {})) / max(1, duration)
        elif pattern == CoordinationPattern.COLLABORATIVE:
            consensus_rounds = len([k for k in session.results.keys() if k.startswith("round_")])
            metrics["consensus_efficiency"] = 1.0 / max(1, consensus_rounds)
        
        return metrics
    
    async def _build_collaboration_graph(self):
        """Build graph of agent collaboration relationships"""
        
        agents = self.workflow_system.agents
        graph = nx.DiGraph()
        
        # Add agent nodes
        for agent_id, agent in agents.items():
            graph.add_node(agent_id, 
                          name=agent.name,
                          capabilities=[cap.value for cap in agent.capabilities],
                          health=self.workflow_system.agent_health_scores.get(agent_id, 100))
        
        # Add collaboration edges based on compatibility
        for agent1_id in agents:
            for agent2_id in agents:
                if agent1_id != agent2_id:
                    compatibility = await self._calculate_agent_compatibility(agent1_id, agent2_id)
                    if compatibility > 0.3:  # Threshold for viable collaboration
                        graph.add_edge(agent1_id, agent2_id, weight=compatibility)
        
        self.collaboration_graph.graph = graph
        self.collaboration_graph.last_updated = datetime.now()
    
    async def _calculate_agent_compatibility(self, agent1_id: str, agent2_id: str) -> float:
        """Calculate compatibility score between two agents"""
        
        if (agent1_id, agent2_id) in self.agent_compatibility:
            return self.agent_compatibility[(agent1_id, agent2_id)]
        
        agent1 = self.workflow_system.agents.get(agent1_id)
        agent2 = self.workflow_system.agents.get(agent2_id)
        
        if not agent1 or not agent2:
            return 0.0
        
        # Capability complementarity
        cap1 = set(agent1.capabilities)
        cap2 = set(agent2.capabilities)
        
        overlap = len(cap1.intersection(cap2)) / max(1, len(cap1.union(cap2)))
        complementarity = 1.0 - overlap  # Higher score for different capabilities
        
        # Performance compatibility
        profile1 = self.agent_profiles.get(agent1_id)
        profile2 = self.agent_profiles.get(agent2_id)
        
        if profile1 and profile2:
            perf_diff = abs(profile1.success_rate - profile2.success_rate)
            performance_compatibility = 1.0 - perf_diff  # Similar performance is better
        else:
            performance_compatibility = 0.5
        
        # Collaboration history
        collaboration_history = 0.5  # Default neutral score
        
        # Calculate final compatibility
        compatibility = (
            complementarity * 0.5 +
            performance_compatibility * 0.3 +
            collaboration_history * 0.2
        )
        
        # Cache result
        self.agent_compatibility[(agent1_id, agent2_id)] = compatibility
        self.agent_compatibility[(agent2_id, agent1_id)] = compatibility
        
        return compatibility
    
    # ==================== Background Services ====================
    
    async def _start_background_services(self):
        """Start background monitoring and optimization services"""
        
        self.background_tasks = [
            asyncio.create_task(self._performance_monitor()),
            asyncio.create_task(self._optimization_engine()),
            asyncio.create_task(self._collaboration_analyzer()),
            asyncio.create_task(self._pattern_learner())
        ]
        
        logger.info("🚀 Background coordination services started")
    
    async def _performance_monitor(self):
        """Monitor coordination performance and detect issues"""
        
        while self.running:
            try:
                # Update performance profiles
                for session in list(self.active_sessions.values()):
                    if session.is_active:
                        await self._update_session_metrics(session)
                
                # Detect performance issues
                issues = await self._detect_performance_issues()
                
                for issue in issues:
                    logger.warning(f"⚠️ Coordination issue detected: {issue['type']}")
                    await self._handle_performance_issue(issue)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_engine(self):
        """Continuously optimize coordination strategies"""
        
        while self.running:
            try:
                # Analyze recent coordination patterns
                optimizations = await self._identify_optimization_opportunities()
                
                for optimization in optimizations:
                    await self._apply_optimization(optimization)
                    logger.info(f"🔧 Applied coordination optimization: {optimization['type']}")
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization engine error: {e}")
                await asyncio.sleep(300)
    
    # Additional methods would be implemented here for:
    # - Session completion and cleanup
    # - Communication channel management
    # - Advanced pattern-specific coordination logic
    # - Consensus calculation
    # - Result synthesis
    # - And many more supporting functions
    
    async def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get coordination system statistics"""
        
        active_sessions_count = len(self.active_sessions)
        total_sessions = len(self.coordination_history)
        
        if total_sessions > 0:
            success_rate = sum(1 for session in self.coordination_history 
                             if session.get("success", False)) / total_sessions
            avg_duration = np.mean([session.get("duration", 0) 
                                   for session in self.coordination_history])
        else:
            success_rate = 0.0
            avg_duration = 0.0
        
        return {
            "active_sessions": active_sessions_count,
            "total_sessions": total_sessions,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "learned_patterns": len(self.learned_patterns),
            "agent_profiles": len(self.agent_profiles),
            "coordination_patterns": {
                pattern.value: len([s for s in self.coordination_history 
                                  if s.get("pattern") == pattern.value])
                for pattern in CoordinationPattern
            },
            "performance_metrics": self.coordination_metrics
        }


# ==================== Example Usage ====================

async def example_coordination():
    """Example of enhanced multi-agent coordination"""
    
    from .multi_agent_workflow_system import MultiAgentWorkflowSystem, Task, TaskPriority
    
    # Initialize systems
    workflow_system = MultiAgentWorkflowSystem()
    await workflow_system.initialize()
    
    coordinator = EnhancedMultiAgentCoordinator(workflow_system)
    await coordinator.initialize()
    
    # Create a complex task requiring coordination
    task = Task(
        id="complex_analysis_task",
        type="comprehensive_analysis",
        description="Analyze system performance and provide optimization recommendations",
        priority=TaskPriority.HIGH,
        requirements={
            AgentCapability.CODE_ANALYSIS,
            AgentCapability.SECURITY_SCANNING,
            AgentCapability.RESOURCE_OPTIMIZATION,
            AgentCapability.TESTING_VALIDATION
        },
        payload={
            "target_system": "/opt/sutazaiapp",
            "analysis_depth": "comprehensive",
            "optimization_goals": ["performance", "security", "maintainability"]
        }
    )
    
    # Execute coordinated task
    result = await coordinator.coordinate_task(
        task=task,
        preferred_pattern=CoordinationPattern.HIERARCHICAL
    )
    
    print(f"Coordination Result: {result}")
    
    # Get statistics
    stats = await coordinator.get_coordination_statistics()
    print(f"Coordination Statistics: {stats}")
    
    # Shutdown
    await coordinator.shutdown()
    await workflow_system.shutdown()


if __name__ == "__main__":
    asyncio.run(example_coordination())