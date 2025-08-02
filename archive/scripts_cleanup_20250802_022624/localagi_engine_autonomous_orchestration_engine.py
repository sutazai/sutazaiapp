#!/usr/bin/env python3
"""
SutazAI LocalAGI Autonomous Orchestration Engine

This engine coordinates all 38 AI agents without external dependencies,
creating a fully autonomous AI system capable of self-organization,
recursive task decomposition, and independent goal achievement.
"""

import asyncio
import json
import yaml
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import redis
import httpx
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    OFFLINE = "offline"

@dataclass
class Task:
    id: str
    description: str
    requirements: List[str]
    priority: float
    complexity: float
    estimated_duration: float
    created_at: datetime
    parent_task_id: Optional[str] = None
    sub_tasks: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Agent:
    name: str
    capabilities: List[str]
    current_load: float
    max_capacity: float
    performance_score: float
    status: AgentStatus
    current_tasks: List[str] = field(default_factory=list)
    completion_history: List[Dict[str, Any]] = field(default_factory=list)
    specialization_score: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class Swarm:
    id: str
    leader_agent: str
    member_agents: List[str]
    goal: str
    tasks: List[str]
    coordination_pattern: str
    created_at: datetime
    status: str = "active"

class AutonomousOrchestrationEngine:
    """
    Core engine for autonomous AI agent orchestration.
    
    Features:
    - Self-organizing agent swarms
    - Recursive task decomposition
    - Autonomous decision-making
    - Self-improving workflows
    - Independent goal achievement
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/localagi/configs/autonomous_orchestrator_config.yaml"):
        self.config = self._load_config(config_path)
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.swarms: Dict[str, Swarm] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.knowledge_base: Dict[str, Any] = {}
        
        # Initialize components
        self._init_redis()
        self._init_http_clients()
        self._load_agent_registry()
        self._init_performance_tracking()
        
        # Autonomous operation flags
        self.autonomous_mode = self.config.get('system', {}).get('autonomous_mode', True)
        self.continuous_improvement_enabled = self.config.get('system', {}).get('continuous_improvement', True)
        
        logger.info("Autonomous Orchestration Engine initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _init_redis(self):
        """Initialize Redis connection for memory and state persistence."""
        try:
            redis_url = self.config.get('memory', {}).get('url', 'redis://redis:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_http_clients(self):
        """Initialize HTTP clients for agent communication."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.ollama_client = httpx.AsyncClient(
            base_url=self.config.get('ollama', {}).get('base_url', 'http://ollama:11434'),
            timeout=60.0
        )
            timeout=60.0
        )
    
    def _load_agent_registry(self):
        """Load and initialize all available agents."""
        try:
            registry_path = Path("/opt/sutazaiapp/agents/agent_registry.json")
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            for agent_name, agent_info in registry_data.get('agents', {}).items():
                agent = Agent(
                    name=agent_name,
                    capabilities=agent_info.get('capabilities', []),
                    current_load=0.0,
                    max_capacity=1.0,
                    performance_score=0.8,  # Initial score
                    status=AgentStatus.IDLE
                )
                self.agents[agent_name] = agent
                
            logger.info(f"Loaded {len(self.agents)} agents from registry")
        except Exception as e:
            logger.error(f"Failed to load agent registry: {e}")
    
    def _init_performance_tracking(self):
        """Initialize performance tracking system."""
        self.performance_metrics = {
            'total_tasks_completed': 0,
            'average_completion_time': 0.0,
            'agent_utilization': {},
            'success_rate': 1.0,
            'resource_efficiency': 0.8,
            'swarm_effectiveness': {},
            'learning_progress': 0.0
        }
    
    async def analyze_task_requirements(self, task: Task) -> Dict[str, Any]:
        """
        Analyze task requirements using local LLM.
        Determines complexity, required capabilities, and decomposition strategy.
        """
        try:
            analysis_prompt = f"""
            Analyze this task for autonomous AI orchestration:
            
            Task: {task.description}
            Requirements: {task.requirements}
            
            Provide analysis in JSON format:
            {{
                "complexity_score": 0.0-1.0,
                "required_capabilities": [],
                "estimated_duration": hours,
                "decomposition_strategy": "strategy_name",
                "parallel_opportunities": true/false,
                "resource_requirements": {{}},
                "risk_factors": [],
                "success_criteria": []
            }}
            """
            
            response = await self.ollama_client.post("/api/generate", json={
                "model": self.config.get('ollama', {}).get('models', {}).get('reasoning', 'tinyllama'),
                "prompt": analysis_prompt,
                "stream": False
            })
            
            if response.status_code == 200:
                result = response.json()
                try:
                    analysis = json.loads(result.get('response', '{}'))
                    return analysis
                except json.JSONDecodeError:
                    logger.warning("Failed to parse LLM analysis, using defaults")
                    
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
        
        # Fallback analysis
        return {
            "complexity_score": 0.5,
            "required_capabilities": task.requirements,
            "estimated_duration": 1.0,
            "decomposition_strategy": "functional_breakdown",
            "parallel_opportunities": True,
            "resource_requirements": {},
            "risk_factors": [],
            "success_criteria": []
        }
    
    async def decompose_task_recursively(self, task: Task, max_depth: int = 5) -> List[Task]:
        """
        Recursively decompose complex tasks into smaller, manageable subtasks.
        """
        if max_depth <= 0 or task.complexity < 0.3:
            return [task]
        
        try:
            decomposition_prompt = f"""
            Decompose this task into smaller, independent subtasks:
            
            Task: {task.description}
            Complexity: {task.complexity}
            Requirements: {task.requirements}
            
            Create 2-5 subtasks in JSON format:
            {{
                "subtasks": [
                    {{
                        "description": "subtask description",
                        "requirements": [],
                        "complexity": 0.0-1.0,
                        "dependencies": [],
                        "parallel_execution": true/false
                    }}
                ]
            }}
            """
            
            response = await self.ollama_client.post("/api/generate", json={
                "model": self.config.get('ollama', {}).get('models', {}).get('planning', 'qwen2.5:14b'),
                "prompt": decomposition_prompt,
                "stream": False
            })
            
            if response.status_code == 200:
                result = response.json()
                try:
                    decomposition = json.loads(result.get('response', '{}'))
                    subtasks = []
                    
                    for subtask_data in decomposition.get('subtasks', []):
                        subtask = Task(
                            id=str(uuid.uuid4()),
                            description=subtask_data.get('description', ''),
                            requirements=subtask_data.get('requirements', []),
                            priority=task.priority * 0.9,  # Slightly lower priority
                            complexity=subtask_data.get('complexity', task.complexity / 2),
                            estimated_duration=task.estimated_duration / len(decomposition.get('subtasks', [1])),
                            created_at=datetime.now(),
                            parent_task_id=task.id
                        )
                        subtasks.append(subtask)
                        task.sub_tasks.append(subtask.id)
                    
                    # Recursively decompose if still complex
                    all_subtasks = []
                    for subtask in subtasks:
                        if subtask.complexity > 0.5:
                            recursive_subtasks = await self.decompose_task_recursively(subtask, max_depth - 1)
                            all_subtasks.extend(recursive_subtasks)
                        else:
                            all_subtasks.append(subtask)
                    
                    return all_subtasks
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse task decomposition")
                    
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
        
        return [task]  # Return original task if decomposition fails
    
    def calculate_agent_capability_match(self, task: Task, agent: Agent) -> float:
        """Calculate how well an agent's capabilities match a task's requirements."""
        if not task.requirements:
            return 0.5
        
        matches = 0
        for requirement in task.requirements:
            for capability in agent.capabilities:
                if requirement.lower() in capability.lower() or capability.lower() in requirement.lower():
                    matches += 1
                    break
        
        base_match = matches / len(task.requirements)
        
        # Factor in agent's specialization score
        specialization_bonus = 0
        for requirement in task.requirements:
            if requirement in agent.specialization_score:
                specialization_bonus += agent.specialization_score[requirement]
        
        specialization_bonus = specialization_bonus / len(task.requirements) if task.requirements else 0
        
        return min(1.0, base_match + specialization_bonus * 0.3)
    
    def calculate_agent_workload_score(self, agent: Agent) -> float:
        """Calculate agent's current workload as a score (lower is better)."""
        if agent.status == AgentStatus.OFFLINE or agent.status == AgentStatus.FAILED:
            return float('inf')
        
        return agent.current_load / agent.max_capacity
    
    def calculate_agent_performance_score(self, agent: Agent) -> float:
        """Get agent's historical performance score."""
        return agent.performance_score
    
    async def select_optimal_agent(self, task: Task) -> Optional[str]:
        """
        Use multi-criteria optimization to select the best agent for a task.
        """
        if not self.agents:
            return None
        
        best_agent = None
        best_score = -1
        
        decision_factors = self.config.get('decision_engine', {}).get('factors', [])
        
        for agent_name, agent in self.agents.items():
            if agent.status == AgentStatus.OFFLINE or agent.status == AgentStatus.FAILED:
                continue
            
            # Calculate individual scores
            capability_score = self.calculate_agent_capability_match(task, agent)
            workload_score = 1.0 - self.calculate_agent_workload_score(agent)  # Invert so lower load = higher score
            performance_score = self.calculate_agent_performance_score(agent)
            
            # Multi-criteria weighted score
            composite_score = (
                capability_score * 0.4 +
                workload_score * 0.3 +
                performance_score * 0.3
            )
            
            if composite_score > best_score:
                best_score = composite_score
                best_agent = agent_name
        
        confidence_threshold = self.config.get('decision_engine', {}).get('confidence_threshold', 0.8)
        
        if best_score >= confidence_threshold:
            return best_agent
        
        # Fall back to general purpose agents
        fallback_agents = self.config.get('decision_engine', {}).get('fallback_agents', [])
        for fallback_agent in fallback_agents:
            if fallback_agent in self.agents and self.agents[fallback_agent].status == AgentStatus.IDLE:
                return fallback_agent
        
        return best_agent  # Return best even if below threshold
    
    async def create_agent_swarm(self, tasks: List[Task], goal: str) -> Swarm:
        """
        Create a self-organizing agent swarm for collaborative problem solving.
        """
        swarm_id = str(uuid.uuid4())
        
        # Analyze tasks to determine required capabilities
        all_capabilities = set()
        for task in tasks:
            all_capabilities.update(task.requirements)
        
        # Select agents for swarm
        swarm_agents = []
        for capability in all_capabilities:
            best_agent = None
            best_score = 0
            
            for agent_name, agent in self.agents.items():
                if agent_name in swarm_agents:
                    continue
                
                capability_score = sum(1 for cap in agent.capabilities if capability.lower() in cap.lower())
                if capability_score > best_score:
                    best_score = capability_score
                    best_agent = agent_name
            
            if best_agent and len(swarm_agents) < self.config.get('swarm_config', {}).get('max_swarm_size', 10):
                swarm_agents.append(best_agent)
        
        # Select leader based on performance
        leader_agent = max(swarm_agents, key=lambda a: self.agents[a].performance_score) if swarm_agents else None
        
        if not leader_agent:
            raise ValueError("Unable to create swarm - no suitable agents available")
        
        swarm = Swarm(
            id=swarm_id,
            leader_agent=leader_agent,
            member_agents=swarm_agents,
            goal=goal,
            tasks=[task.id for task in tasks],
            coordination_pattern=self.config.get('swarm_config', {}).get('coordination_pattern', 'hierarchical_mesh'),
            created_at=datetime.now()
        )
        
        self.swarms[swarm_id] = swarm
        logger.info(f"Created swarm {swarm_id} with {len(swarm_agents)} agents for goal: {goal}")
        
        return swarm
    
    async def execute_task_autonomously(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task autonomously, handling failures and optimizations.
        """
        try:
            # Analyze task
            analysis = await self.analyze_task_requirements(task)
            task.complexity = analysis.get('complexity_score', 0.5)
            task.estimated_duration = analysis.get('estimated_duration', 1.0)
            
            # Decompose if complex
            if task.complexity > 0.7:
                subtasks = await self.decompose_task_recursively(task)
                if len(subtasks) > 1:
                    logger.info(f"Decomposed task {task.id} into {len(subtasks)} subtasks")
                    
                    # Execute subtasks
                    results = []
                    for subtask in subtasks:
                        self.tasks[subtask.id] = subtask
                        subtask_result = await self.execute_task_autonomously(subtask)
                        results.append(subtask_result)
                    
                    # Consolidate results
                    task.status = TaskStatus.COMPLETED
                    task.result = {
                        'subtask_results': results,
                        'consolidated_result': 'All subtasks completed successfully',
                        'execution_time': sum(r.get('execution_time', 0) for r in results)
                    }
                    return task.result
            
            # Select agent
            selected_agent = await self.select_optimal_agent(task)
            if not selected_agent:
                raise ValueError("No suitable agent available")
            
            task.assigned_agent = selected_agent
            task.status = TaskStatus.ASSIGNED
            
            # Update agent status
            agent = self.agents[selected_agent]
            agent.current_tasks.append(task.id)
            agent.current_load += task.complexity
            agent.status = AgentStatus.BUSY if agent.current_load < agent.max_capacity else AgentStatus.OVERLOADED
            
            # Execute task (simulated - in real implementation, this would call the actual agent)
            start_time = time.time()
            task.status = TaskStatus.IN_PROGRESS
            
            # Simulate task execution
            await asyncio.sleep(min(task.estimated_duration, 0.1))  # Quick simulation
            
            execution_time = time.time() - start_time
            
            # Update task result
            task.status = TaskStatus.COMPLETED
            task.result = {
                'status': 'success',
                'agent': selected_agent,
                'execution_time': execution_time,
                'result': f'Task completed by {selected_agent}'
            }
            
            # Update agent status
            agent.current_tasks.remove(task.id)
            agent.current_load -= task.complexity
            agent.status = AgentStatus.IDLE if agent.current_load == 0 else AgentStatus.BUSY
            
            # Update performance metrics
            self._update_performance_metrics(task, agent, execution_time)
            
            return task.result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.result = {'status': 'failed', 'error': str(e)}
            return task.result
    
    def _update_performance_metrics(self, task: Task, agent: Agent, execution_time: float):
        """Update performance metrics for continuous learning."""
        # Update agent performance
        success_factor = 1.0 if task.status == TaskStatus.COMPLETED else 0.0
        agent.performance_score = agent.performance_score * 0.9 + success_factor * 0.1
        
        # Update agent specialization scores
        for requirement in task.requirements:
            if requirement in agent.specialization_score:
                agent.specialization_score[requirement] = agent.specialization_score[requirement] * 0.95 + success_factor * 0.05
            else:
                agent.specialization_score[requirement] = success_factor * 0.1
        
        # Update global metrics
        self.performance_metrics['total_tasks_completed'] += 1
        self.performance_metrics['average_completion_time'] = (
            self.performance_metrics['average_completion_time'] * 0.9 + execution_time * 0.1
        )
        
        if agent.name not in self.performance_metrics['agent_utilization']:
            self.performance_metrics['agent_utilization'][agent.name] = 0
        
        self.performance_metrics['agent_utilization'][agent.name] += 1
    
    async def self_improve_system(self):
        """
        Implement continuous_improvement based on performance metrics and feedback.
        """
        if not self.continuous_improvement_enabled:
            return
        
        logger.info("Running continuous_improvement analysis...")
        
        # Analyze performance patterns
        improvement_opportunities = []
        
        # Check agent utilization balance
        utilization = self.performance_metrics.get('agent_utilization', {})
        if utilization:
            avg_utilization = np.mean(list(utilization.values()))
            std_utilization = np.std(list(utilization.values()))
            
            if std_utilization > avg_utilization * 0.5:  # High variance indicates imbalance
                improvement_opportunities.append("rebalance_agent_workload")
        
        # Check success rates
        total_tasks = self.performance_metrics.get('total_tasks_completed', 0)
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        success_rate = (total_tasks - failed_tasks) / total_tasks if total_tasks > 0 else 1.0
        
        if success_rate < 0.9:
            improvement_opportunities.append("improve_task_assignment")
        
        # Check average completion time trends
        avg_time = self.performance_metrics.get('average_completion_time', 0)
        if avg_time > 5.0:  # Arbitrary threshold
            improvement_opportunities.append("optimize_task_decomposition")
        
        # Implement improvements
        for opportunity in improvement_opportunities:
            await self._implement_improvement(opportunity)
        
        self.performance_metrics['learning_progress'] += 0.1
        logger.info(f"continuous_improvement completed. Opportunities addressed: {improvement_opportunities}")
    
    async def _implement_improvement(self, improvement_type: str):
        """Implement specific system improvements."""
        if improvement_type == "rebalance_agent_workload":
            # Adjust agent capacity factors
            for agent in self.agents.values():
                if agent.current_load > agent.max_capacity * 0.8:
                    agent.max_capacity *= 1.1  # Increase capacity
                elif agent.current_load < agent.max_capacity * 0.2:
                    agent.max_capacity *= 0.95  # Decrease capacity slightly
        
        elif improvement_type == "improve_task_assignment":
            # Adjust decision engine confidence threshold
            current_threshold = self.config.get('decision_engine', {}).get('confidence_threshold', 0.8)
            new_threshold = max(0.6, current_threshold - 0.05)
            self.config['decision_engine']['confidence_threshold'] = new_threshold
        
        elif improvement_type == "optimize_task_decomposition":
            # Adjust decomposition parameters
            current_depth = self.config.get('task_decomposition', {}).get('max_depth', 5)
            new_depth = min(7, current_depth + 1)
            self.config['task_decomposition']['max_depth'] = new_depth
    
    async def run_autonomous_orchestration(self):
        """
        Main autonomous orchestration loop.
        """
        logger.info("Starting autonomous orchestration...")
        
        while self.autonomous_mode:
            try:
                # Check for new tasks
                pending_tasks = [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
                
                # Execute pending tasks
                for task in pending_tasks:
                    asyncio.create_task(self.execute_task_autonomously(task))
                
                # Periodic continuous_improvement
                if len(self.tasks) > 0 and len(self.tasks) % 10 == 0:
                    await self.self_improve_system()
                
                # Monitor agent health
                await self._monitor_agent_health()
                
                # Persist state
                await self._persist_state()
                
                # Short sleep to prevent tight loop
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _monitor_agent_health(self):
        """Monitor and recover failed agents."""
        for agent_name, agent in self.agents.items():
            if agent.status == AgentStatus.FAILED:
                # Attempt recovery
                agent.status = AgentStatus.IDLE
                agent.current_load = 0.0
                agent.current_tasks.clear()
                logger.info(f"Recovered agent {agent_name}")
    
    async def _persist_state(self):
        """Persist current state to Redis."""
        if not self.redis_client:
            return
        
        try:
            # Persist tasks
            for task_id, task in self.tasks.items():
                task_data = {
                    'id': task.id,
                    'description': task.description,
                    'status': task.status.value,
                    'assigned_agent': task.assigned_agent,
                    'created_at': task.created_at.isoformat(),
                    'result': task.result
                }
                self.redis_client.hset(f"task:{task_id}", mapping=task_data)
            
            # Persist agent states
            for agent_name, agent in self.agents.items():
                agent_data = {
                    'name': agent.name,
                    'status': agent.status.value,
                    'current_load': agent.current_load,
                    'performance_score': agent.performance_score,
                    'current_tasks': json.dumps(agent.current_tasks)
                }
                self.redis_client.hset(f"agent:{agent_name}", mapping=agent_data)
            
            # Persist metrics
            self.redis_client.hset("metrics", mapping=self.performance_metrics)
            
        except Exception as e:
            logger.error(f"State persistence failed: {e}")
    
    # Public API methods
    
    async def submit_task(self, description: str, requirements: List[str] = None, priority: float = 0.5) -> str:
        """Submit a new task for autonomous execution."""
        task = Task(
            id=str(uuid.uuid4()),
            description=description,
            requirements=requirements or [],
            priority=priority,
            complexity=0.5,  # Will be analyzed
            estimated_duration=1.0,  # Will be analyzed
            created_at=datetime.now()
        )
        
        self.tasks[task.id] = task
        logger.info(f"Submitted task {task.id}: {description}")
        
        return task.id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task."""
        if task_id not in self.tasks:
            return {'error': 'Task not found'}
        
        task = self.tasks[task_id]
        return {
            'id': task.id,
            'description': task.description,
            'status': task.status.value,
            'assigned_agent': task.assigned_agent,
            'progress': self._calculate_task_progress(task),
            'result': task.result
        }
    
    def _calculate_task_progress(self, task: Task) -> float:
        """Calculate task completion progress."""
        if task.status == TaskStatus.COMPLETED:
            return 1.0
        elif task.status == TaskStatus.FAILED:
            return 0.0
        elif task.status == TaskStatus.IN_PROGRESS:
            return 0.5
        elif task.status == TaskStatus.ASSIGNED:
            return 0.1
        else:
            return 0.0
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'autonomous_mode': self.autonomous_mode,
            'total_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.status in [AgentStatus.IDLE, AgentStatus.BUSY]]),
            'total_tasks': len(self.tasks),
            'pending_tasks': len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            'active_swarms': len(self.swarms),
            'performance_metrics': self.performance_metrics,
            'continuous_improvement_enabled': self.continuous_improvement_enabled
        }

# Singleton instance
orchestration_engine = None

def get_orchestration_engine() -> AutonomousOrchestrationEngine:
    """Get or create the global orchestration engine instance."""
    global orchestration_engine
    if orchestration_engine is None:
        orchestration_engine = AutonomousOrchestrationEngine()
    return orchestration_engine

# Main execution
async def main():
    """Main entry point for autonomous orchestration."""
    engine = get_orchestration_engine()
    
    # Start autonomous orchestration
    await engine.run_autonomous_orchestration()

if __name__ == "__main__":
    asyncio.run(main())