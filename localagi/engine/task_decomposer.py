#!/usr/bin/env python3
"""
SutazAI Recursive Task Decomposition and Delegation Engine

This engine breaks down complex tasks into smaller, manageable subtasks
and autonomously delegates them to appropriate agents. It uses AI-powered
analysis to determine optimal decomposition strategies and creates
hierarchical task execution trees.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import numpy as np
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)

class DecompositionStrategy(Enum):
    FUNCTIONAL = "functional_breakdown"
    DOMAIN = "domain_separation"
    DEPENDENCY = "dependency_analysis"
    PARALLEL = "parallel_opportunities"
    COMPLEXITY = "complexity_reduction"
    CAPABILITY = "capability_matching"
    TEMPORAL = "temporal_sequencing"
    HIERARCHICAL = "hierarchical_structure"

class TaskType(Enum):
    ATOMIC = "atomic"
    COMPOSITE = "composite"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"

class TaskPriority(Enum):
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    DEFERRED = 0.2

@dataclass
class TaskDependency:
    task_id: str
    dependency_id: str
    dependency_type: str  # 'prerequisite', 'resource', 'data', 'agent'
    strength: float  # 0.0 to 1.0
    optional: bool = False

@dataclass
class DecomposedTask:
    id: str
    parent_id: Optional[str]
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    complexity: float
    estimated_duration: float
    required_capabilities: List[str]
    required_resources: Dict[str, Any]
    dependencies: List[TaskDependency]
    constraints: Dict[str, Any]
    success_criteria: List[str]
    decomposition_strategy: Optional[DecompositionStrategy]
    subtasks: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecompositionResult:
    original_task_id: str
    task_tree: Dict[str, DecomposedTask]
    execution_plan: List[List[str]]  # Stages of parallel execution
    total_estimated_duration: float
    decomposition_depth: int
    strategy_used: DecompositionStrategy
    confidence_score: float
    optimization_opportunities: List[str]

class RecursiveTaskDecomposer:
    """
    Advanced task decomposition engine with AI-powered analysis.
    """
    
    def __init__(self, orchestration_engine):
        self.orchestration_engine = orchestration_engine
        
        # Decomposition state
        self.task_trees: Dict[str, Dict[str, DecomposedTask]] = {}
        self.execution_plans: Dict[str, List[List[str]]] = {}
        self.decomposition_history: List[DecompositionResult] = []
        
        # Learning and optimization
        self.strategy_performance: Dict[DecompositionStrategy, float] = {
            strategy: 0.5 for strategy in DecompositionStrategy
        }
        self.complexity_patterns: Dict[str, float] = {}
        self.capability_mappings: Dict[str, List[str]] = {}
        
        # Configuration
        self.max_decomposition_depth = 7
        self.min_task_size = 0.05  # Minimum relative complexity
        self.max_subtasks_per_level = 8
        self.parallel_threshold = 0.3  # Tasks below this complexity can be parallelized
        
        logger.info("Recursive Task Decomposer initialized")
    
    async def decompose_task_recursively(self, 
                                       task_description: str, 
                                       requirements: List[str] = None,
                                       constraints: Dict[str, Any] = None,
                                       max_depth: Optional[int] = None) -> DecompositionResult:
        """
        Recursively decompose a complex task into manageable subtasks.
        """
        logger.info(f"Starting recursive decomposition: {task_description}")
        
        # Create root task
        root_task = DecomposedTask(
            id=str(uuid.uuid4()),
            parent_id=None,
            title=task_description[:50] + "..." if len(task_description) > 50 else task_description,
            description=task_description,
            task_type=TaskType.COMPOSITE,
            priority=TaskPriority.HIGH,
            complexity=1.0,  # Will be analyzed
            estimated_duration=0.0,  # Will be calculated
            required_capabilities=requirements or [],
            required_resources={},
            dependencies=[],
            constraints=constraints or {},
            success_criteria=[],
            decomposition_strategy=None
        )
        
        # Analyze task complexity and requirements
        analysis = await self._analyze_task_complexity(root_task)
        root_task.complexity = analysis['complexity']
        root_task.estimated_duration = analysis['estimated_duration']
        root_task.required_capabilities = analysis['required_capabilities']
        root_task.success_criteria = analysis['success_criteria']
        
        # Determine optimal decomposition strategy
        strategy = await self._select_decomposition_strategy(root_task)
        root_task.decomposition_strategy = strategy
        
        # Perform recursive decomposition
        task_tree = {root_task.id: root_task}
        max_depth = max_depth or self.max_decomposition_depth
        
        await self._decompose_recursive(root_task, task_tree, strategy, 0, max_depth)
        
        # Generate execution plan
        execution_plan = await self._generate_execution_plan(task_tree, root_task.id)
        
        # Calculate total duration
        total_duration = await self._calculate_total_duration(task_tree, execution_plan)
        
        # Identify optimization opportunities
        optimizations = await self._identify_optimizations(task_tree, execution_plan)
        
        # Create decomposition result
        result = DecompositionResult(
            original_task_id=root_task.id,
            task_tree=task_tree,
            execution_plan=execution_plan,
            total_estimated_duration=total_duration,
            decomposition_depth=self._calculate_tree_depth(task_tree, root_task.id),
            strategy_used=strategy,
            confidence_score=self._calculate_decomposition_confidence(task_tree),
            optimization_opportunities=optimizations
        )
        
        # Store results
        self.task_trees[root_task.id] = task_tree
        self.execution_plans[root_task.id] = execution_plan
        self.decomposition_history.append(result)
        
        logger.info(f"Task decomposition completed: {len(task_tree)} tasks, {len(execution_plan)} stages")
        return result
    
    async def _analyze_task_complexity(self, task: DecomposedTask) -> Dict[str, Any]:
        """
        Use AI to analyze task complexity and requirements.
        """
        try:
            analysis_prompt = f"""
            Analyze this task for decomposition planning:
            
            Task: {task.description}
            Current Requirements: {task.required_capabilities}
            Constraints: {task.constraints}
            
            Provide analysis in JSON format:
            {{
                "complexity": 0.0-1.0,
                "estimated_duration": hours,
                "required_capabilities": ["capability1", "capability2"],
                "success_criteria": ["criterion1", "criterion2"],
                "decomposition_hints": ["hint1", "hint2"],
                "risk_factors": ["risk1", "risk2"],
                "resource_requirements": {{}},
                "parallel_potential": true/false,
                "domain_areas": ["domain1", "domain2"]
            }}
            """
            
            response = await self.orchestration_engine.ollama_client.post("/api/generate", json={
                "model": self.orchestration_engine.config.get('ollama', {}).get('models', {}).get('reasoning', 'deepseek-r1:8b'),
                "prompt": analysis_prompt,
                "stream": False
            })
            
            if response.status_code == 200:
                result = response.json()
                try:
                    analysis = json.loads(result.get('response', '{}'))
                    
                    # Validate and normalize
                    return {
                        'complexity': max(0.1, min(1.0, analysis.get('complexity', 0.5))),
                        'estimated_duration': max(0.1, analysis.get('estimated_duration', 1.0)),
                        'required_capabilities': analysis.get('required_capabilities', task.required_capabilities),
                        'success_criteria': analysis.get('success_criteria', ['Task completed successfully']),
                        'decomposition_hints': analysis.get('decomposition_hints', []),
                        'risk_factors': analysis.get('risk_factors', []),
                        'resource_requirements': analysis.get('resource_requirements', {}),
                        'parallel_potential': analysis.get('parallel_potential', False),
                        'domain_areas': analysis.get('domain_areas', [])
                    }
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse task analysis, using defaults")
                    
        except Exception as e:
            logger.error(f"Task analysis failed: {e}")
        
        # Fallback analysis
        return {
            'complexity': 0.7,
            'estimated_duration': 2.0,
            'required_capabilities': task.required_capabilities,
            'success_criteria': ['Task completed successfully'],
            'decomposition_hints': [],
            'risk_factors': [],
            'resource_requirements': {},
            'parallel_potential': True,
            'domain_areas': []
        }
    
    async def _select_decomposition_strategy(self, task: DecomposedTask) -> DecompositionStrategy:
        """
        Select the optimal decomposition strategy based on task characteristics.
        """
        strategy_scores = {}
        
        # Analyze task characteristics
        has_clear_functions = any(word in task.description.lower() 
                                for word in ['create', 'build', 'implement', 'design', 'analyze'])
        has_domains = len(task.required_capabilities) > 3
        has_complexity = task.complexity > 0.7
        has_dependencies = 'depend' in task.description.lower() or 'require' in task.description.lower()
        
        # Score strategies based on characteristics
        for strategy in DecompositionStrategy:
            base_performance = self.strategy_performance[strategy]
            
            if strategy == DecompositionStrategy.FUNCTIONAL:
                # Good for tasks with clear functional components
                score = base_performance * (1.2 if has_clear_functions else 0.8)
            elif strategy == DecompositionStrategy.DOMAIN:
                # Good for multi-domain tasks
                score = base_performance * (1.3 if has_domains else 0.7)
            elif strategy == DecompositionStrategy.COMPLEXITY:
                # Good for highly complex tasks
                score = base_performance * (1.4 if has_complexity else 0.6)
            elif strategy == DecompositionStrategy.DEPENDENCY:
                # Good for tasks with clear dependencies
                score = base_performance * (1.3 if has_dependencies else 0.8)
            elif strategy == DecompositionStrategy.PARALLEL:
                # Good for tasks that can be parallelized
                score = base_performance * (1.2 if task.complexity < 0.8 else 0.9)
            else:
                score = base_performance
            
            strategy_scores[strategy] = score
        
        # Select best strategy
        selected_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Selected decomposition strategy: {selected_strategy.value}")
        return selected_strategy
    
    async def _decompose_recursive(self, 
                                 parent_task: DecomposedTask, 
                                 task_tree: Dict[str, DecomposedTask],
                                 strategy: DecompositionStrategy,
                                 current_depth: int,
                                 max_depth: int):
        """
        Recursively decompose a task using the specified strategy.
        """
        if current_depth >= max_depth:
            logger.info(f"Reached maximum decomposition depth for task {parent_task.id}")
            return
        
        if parent_task.complexity < self.min_task_size:
            logger.info(f"Task {parent_task.id} too small to decompose further")
            parent_task.task_type = TaskType.ATOMIC
            return
        
        # Generate subtasks using AI
        subtasks = await self._generate_subtasks(parent_task, strategy)
        
        if not subtasks or len(subtasks) < 2:
            logger.info(f"No meaningful decomposition found for task {parent_task.id}")
            parent_task.task_type = TaskType.ATOMIC
            return
        
        # Limit number of subtasks
        if len(subtasks) > self.max_subtasks_per_level:
            subtasks = subtasks[:self.max_subtasks_per_level]
            logger.info(f"Limited subtasks to {self.max_subtasks_per_level} for task {parent_task.id}")
        
        # Add subtasks to tree
        for subtask in subtasks:
            subtask.parent_id = parent_task.id
            parent_task.subtasks.append(subtask.id)
            task_tree[subtask.id] = subtask
        
        # Analyze dependencies between subtasks
        await self._analyze_subtask_dependencies(subtasks, task_tree)
        
        # Recursively decompose complex subtasks
        for subtask in subtasks:
            if subtask.complexity > self.min_task_size * 2:  # Worth decomposing further
                await self._decompose_recursive(subtask, task_tree, strategy, current_depth + 1, max_depth)
    
    async def _generate_subtasks(self, parent_task: DecomposedTask, strategy: DecompositionStrategy) -> List[DecomposedTask]:
        """
        Generate subtasks using AI based on the decomposition strategy.
        """
        try:
            strategy_prompts = {
                DecompositionStrategy.FUNCTIONAL: "Break down by functional components and features",
                DecompositionStrategy.DOMAIN: "Break down by domain expertise and specialization areas",
                DecompositionStrategy.COMPLEXITY: "Break down from complex to simple, reducing cognitive load",
                DecompositionStrategy.DEPENDENCY: "Break down by dependency chains and prerequisites",
                DecompositionStrategy.PARALLEL: "Break down to maximize parallel execution opportunities",
                DecompositionStrategy.CAPABILITY: "Break down by required agent capabilities",
                DecompositionStrategy.TEMPORAL: "Break down by time sequence and phases",
                DecompositionStrategy.HIERARCHICAL: "Break down by abstraction levels"
            }
            
            decomposition_prompt = f"""
            Decompose this task using {strategy.value} strategy:
            
            Parent Task: {parent_task.description}
            Complexity: {parent_task.complexity}
            Strategy: {strategy_prompts.get(strategy, 'Break down into logical subtasks')}
            Required Capabilities: {parent_task.required_capabilities}
            Constraints: {parent_task.constraints}
            
            Generate 2-6 subtasks in JSON format:
            {{
                "subtasks": [
                    {{
                        "title": "Subtask title",
                        "description": "Detailed description",
                        "complexity": 0.0-1.0,
                        "estimated_duration": hours,
                        "required_capabilities": ["capability1"],
                        "task_type": "atomic|composite|parallel|sequential",
                        "priority": "critical|high|medium|low",
                        "success_criteria": ["criterion1"],
                        "dependencies_on_siblings": [],
                        "resources_needed": {{}}
                    }}
                ]
            }}
            """
            
            response = await self.orchestration_engine.ollama_client.post("/api/generate", json={
                "model": self.orchestration_engine.config.get('ollama', {}).get('models', {}).get('planning', 'qwen2.5:14b'),
                "prompt": decomposition_prompt,
                "stream": False
            })
            
            if response.status_code == 200:
                result = response.json()
                try:
                    decomposition_data = json.loads(result.get('response', '{}'))
                    subtasks = []
                    
                    total_child_complexity = 0
                    
                    for i, subtask_data in enumerate(decomposition_data.get('subtasks', [])):
                        # Parse task type
                        task_type_str = subtask_data.get('task_type', 'atomic').lower()
                        task_type = TaskType.ATOMIC
                        for tt in TaskType:
                            if tt.value in task_type_str:
                                task_type = tt
                                break
                        
                        # Parse priority
                        priority_str = subtask_data.get('priority', 'medium').lower()
                        priority = TaskPriority.MEDIUM
                        for p in TaskPriority:
                            if p.name.lower() in priority_str:
                                priority = p
                                break
                        
                        subtask_complexity = max(0.05, min(0.95, subtask_data.get('complexity', parent_task.complexity / 3)))
                        total_child_complexity += subtask_complexity
                        
                        subtask = DecomposedTask(
                            id=str(uuid.uuid4()),
                            parent_id=parent_task.id,
                            title=subtask_data.get('title', f'Subtask {i+1}'),
                            description=subtask_data.get('description', ''),
                            task_type=task_type,
                            priority=priority,
                            complexity=subtask_complexity,
                            estimated_duration=max(0.1, subtask_data.get('estimated_duration', parent_task.estimated_duration / 4)),
                            required_capabilities=subtask_data.get('required_capabilities', []),
                            required_resources=subtask_data.get('resources_needed', {}),
                            dependencies=[],
                            constraints=parent_task.constraints.copy(),
                            success_criteria=subtask_data.get('success_criteria', ['Subtask completed']),
                            decomposition_strategy=strategy,
                            metadata={
                                'dependencies_on_siblings': subtask_data.get('dependencies_on_siblings', []),
                                'original_index': i
                            }
                        )
                        
                        subtasks.append(subtask)
                    
                    # Normalize complexities so they don't exceed parent
                    if total_child_complexity > parent_task.complexity:
                        normalization_factor = parent_task.complexity * 0.9 / total_child_complexity
                        for subtask in subtasks:
                            subtask.complexity *= normalization_factor
                    
                    return subtasks
                    
                except json.JSONDecodeError:
                    logger.warning("Failed to parse subtask generation")
                    
        except Exception as e:
            logger.error(f"Subtask generation failed: {e}")
        
        # Fallback: create simple breakdown
        num_subtasks = min(4, max(2, int(parent_task.complexity * 5)))
        fallback_subtasks = []
        
        for i in range(num_subtasks):
            fallback_subtasks.append(DecomposedTask(
                id=str(uuid.uuid4()),
                parent_id=parent_task.id,
                title=f"Subtask {i+1}",
                description=f"Part {i+1} of {parent_task.title}",
                task_type=TaskType.ATOMIC,
                priority=parent_task.priority,
                complexity=parent_task.complexity / num_subtasks,
                estimated_duration=parent_task.estimated_duration / num_subtasks,
                required_capabilities=parent_task.required_capabilities,
                required_resources={},
                dependencies=[],
                constraints=parent_task.constraints.copy(),
                success_criteria=[f"Subtask {i+1} completed"],
                decomposition_strategy=strategy
            ))
        
        return fallback_subtasks
    
    async def _analyze_subtask_dependencies(self, subtasks: List[DecomposedTask], task_tree: Dict[str, DecomposedTask]):
        """
        Analyze and establish dependencies between subtasks.
        """
        for subtask in subtasks:
            sibling_dependencies = subtask.metadata.get('dependencies_on_siblings', [])
            
            for dep_name in sibling_dependencies:
                # Find matching sibling by title similarity
                for other_subtask in subtasks:
                    if other_subtask.id != subtask.id:
                        # Simple text matching for dependencies
                        if (dep_name.lower() in other_subtask.title.lower() or 
                            other_subtask.title.lower() in dep_name.lower()):
                            
                            dependency = TaskDependency(
                                task_id=subtask.id,
                                dependency_id=other_subtask.id,
                                dependency_type='prerequisite',
                                strength=0.8,
                                optional=False
                            )
                            
                            subtask.dependencies.append(dependency)
                            break
        
        # Analyze implicit dependencies based on capabilities and resources
        for i, subtask1 in enumerate(subtasks):
            for j, subtask2 in enumerate(subtasks):
                if i != j:
                    # Check if subtask2's outputs might be needed by subtask1
                    implicit_dependency = self._detect_implicit_dependency(subtask1, subtask2)
                    if implicit_dependency:
                        dependency = TaskDependency(
                            task_id=subtask1.id,
                            dependency_id=subtask2.id,
                            dependency_type=implicit_dependency['type'],
                            strength=implicit_dependency['strength'],
                            optional=implicit_dependency['optional']
                        )
                        subtask1.dependencies.append(dependency)
    
    def _detect_implicit_dependency(self, task1: DecomposedTask, task2: DecomposedTask) -> Optional[Dict[str, Any]]:
        """
        Detect implicit dependencies between tasks based on their characteristics.
        """
        # Resource dependencies
        if task1.required_resources and task2.required_resources:
            common_resources = set(task1.required_resources.keys()) & set(task2.required_resources.keys())
            if common_resources:
                return {
                    'type': 'resource',
                    'strength': 0.6,
                    'optional': True
                }
        
        # Capability dependencies (if one task produces what another needs)
        task1_keywords = set(re.findall(r'\b\w+\b', task1.description.lower()))
        task2_keywords = set(re.findall(r'\b\w+\b', task2.description.lower()))
        
        # Simple heuristic: if task2 creates/builds something task1 uses/needs
        creation_words = {'create', 'build', 'generate', 'produce', 'develop', 'make'}
        usage_words = {'use', 'utilize', 'implement', 'deploy', 'configure', 'apply'}
        
        task2_creates = bool(creation_words & task2_keywords)
        task1_uses = bool(usage_words & task1_keywords)
        
        if task2_creates and task1_uses:
            # Check for overlapping concepts
            overlap = len(task1_keywords & task2_keywords)
            if overlap > 2:
                return {
                    'type': 'data',
                    'strength': min(0.8, overlap * 0.2),
                    'optional': False
                }
        
        return None
    
    async def _generate_execution_plan(self, task_tree: Dict[str, DecomposedTask], root_id: str) -> List[List[str]]:
        """
        Generate an execution plan that respects dependencies and maximizes parallelism.
        """
        # Get all atomic tasks (leaf nodes)
        atomic_tasks = [task for task in task_tree.values() if task.task_type == TaskType.ATOMIC]
        
        # Build dependency graph
        dependency_graph = defaultdict(set)
        reverse_dependencies = defaultdict(set)  # Tasks that depend on this task
        
        for task in atomic_tasks:
            for dep in task.dependencies:
                if dep.dependency_id in task_tree:
                    dependency_graph[task.id].add(dep.dependency_id)
                    reverse_dependencies[dep.dependency_id].add(task.id)
        
        # Topological sort with parallelization
        execution_plan = []
        completed_tasks = set()
        remaining_tasks = {task.id for task in atomic_tasks}
        
        while remaining_tasks:
            # Find tasks with no unfulfilled dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                dependencies = dependency_graph[task_id]
                if dependencies.issubset(completed_tasks):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Deadlock or circular dependency - break it
                logger.warning("Circular dependency detected, breaking cycle")
                ready_tasks = [next(iter(remaining_tasks))]
            
            # Sort ready tasks by priority and complexity
            ready_tasks.sort(key=lambda tid: (
                -task_tree[tid].priority.value,  # Higher priority first
                -task_tree[tid].complexity  # More complex tasks first
            ))
            
            execution_plan.append(ready_tasks)
            completed_tasks.update(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return execution_plan
    
    async def _calculate_total_duration(self, task_tree: Dict[str, DecomposedTask], execution_plan: List[List[str]]) -> float:
        """
        Calculate the total execution duration considering parallelism.
        """
        total_duration = 0.0
        
        for stage in execution_plan:
            # Duration of this stage is the maximum duration among parallel tasks
            stage_durations = [task_tree[task_id].estimated_duration for task_id in stage]
            stage_duration = max(stage_durations) if stage_durations else 0.0
            total_duration += stage_duration
        
        return total_duration
    
    async def _identify_optimizations(self, task_tree: Dict[str, DecomposedTask], execution_plan: List[List[str]]) -> List[str]:
        """
        Identify potential optimizations in the decomposition and execution plan.
        """
        optimizations = []
        
        # Check for load balancing opportunities
        stage_loads = []
        for stage in execution_plan:
            total_complexity = sum(task_tree[task_id].complexity for task_id in stage)
            stage_loads.append(total_complexity)
        
        if stage_loads:
            load_variance = np.var(stage_loads)
            if load_variance > 0.1:
                optimizations.append("Load balancing: Some execution stages are significantly more complex than others")
        
        # Check for over-decomposition
        atomic_tasks = [task for task in task_tree.values() if task.task_type == TaskType.ATOMIC]
        very_small_tasks = [task for task in atomic_tasks if task.complexity < 0.1]
        
        if len(very_small_tasks) > len(atomic_tasks) * 0.3:
            optimizations.append("Over-decomposition: Many tasks are very small and could be consolidated")
        
        # Check for capability clustering opportunities
        capability_groups = defaultdict(list)
        for task in atomic_tasks:
            key = tuple(sorted(task.required_capabilities))
            capability_groups[key].append(task.id)
        
        large_groups = [group for group in capability_groups.values() if len(group) > 3]
        if large_groups:
            optimizations.append("Capability clustering: Some tasks with similar capabilities could be grouped")
        
        # Check for dependency optimization
        total_dependencies = sum(len(task.dependencies) for task in task_tree.values())
        if total_dependencies > len(task_tree) * 0.5:
            optimizations.append("Dependency optimization: High number of dependencies may indicate over-coupling")
        
        # Check for parallel opportunities
        single_task_stages = [stage for stage in execution_plan if len(stage) == 1]
        if len(single_task_stages) > len(execution_plan) * 0.6:
            optimizations.append("Parallelization: Many stages have only one task, parallel opportunities may be missed")
        
        return optimizations
    
    def _calculate_tree_depth(self, task_tree: Dict[str, DecomposedTask], root_id: str) -> int:
        """
        Calculate the maximum depth of the task tree.
        """
        def get_depth(task_id: str) -> int:
            task = task_tree[task_id]
            if not task.subtasks:
                return 1
            return 1 + max(get_depth(subtask_id) for subtask_id in task.subtasks)
        
        return get_depth(root_id)
    
    def _calculate_decomposition_confidence(self, task_tree: Dict[str, DecomposedTask]) -> float:
        """
        Calculate confidence score for the decomposition quality.
        """
        factors = []
        
        # Factor 1: Balanced complexity distribution
        atomic_tasks = [task for task in task_tree.values() if task.task_type == TaskType.ATOMIC]
        if atomic_tasks:
            complexities = [task.complexity for task in atomic_tasks]
            complexity_variance = np.var(complexities)
            balance_score = max(0, 1.0 - complexity_variance * 2)
            factors.append(balance_score)
        
        # Factor 2: Appropriate decomposition depth
        depth = self._calculate_tree_depth(task_tree, next(iter(task_tree.keys())))
        depth_score = max(0, 1.0 - abs(depth - 3) * 0.1)  # Optimal depth around 3
        factors.append(depth_score)
        
        # Factor 3: Dependency reasonableness
        total_dependencies = sum(len(task.dependencies) for task in task_tree.values())
        dependency_ratio = total_dependencies / len(task_tree)
        dependency_score = max(0, 1.0 - abs(dependency_ratio - 0.3) * 2)  # Optimal ratio around 0.3
        factors.append(dependency_score)
        
        # Factor 4: Capability distribution
        all_capabilities = set()
        for task in task_tree.values():
            all_capabilities.update(task.required_capabilities)
        
        capability_coverage = len(all_capabilities) / max(1, len(self.orchestration_engine.agents))
        coverage_score = min(1.0, capability_coverage * 2)
        factors.append(coverage_score)
        
        return np.mean(factors) if factors else 0.5
    
    async def optimize_execution_plan(self, task_tree_id: str) -> Dict[str, Any]:
        """
        Optimize an existing execution plan for better performance.
        """
        if task_tree_id not in self.task_trees:
            raise ValueError(f"Task tree {task_tree_id} not found")
        
        task_tree = self.task_trees[task_tree_id]
        current_plan = self.execution_plans[task_tree_id]
        
        logger.info(f"Optimizing execution plan for task tree {task_tree_id}")
        
        # Try different optimization strategies
        optimizations = {}
        
        # 1. Load balancing optimization
        balanced_plan = await self._optimize_load_balancing(task_tree, current_plan)
        optimizations['load_balanced'] = {
            'plan': balanced_plan,
            'duration': await self._calculate_total_duration(task_tree, balanced_plan)
        }
        
        # 2. Critical path optimization
        critical_path_plan = await self._optimize_critical_path(task_tree, current_plan)
        optimizations['critical_path'] = {
            'plan': critical_path_plan,
            'duration': await self._calculate_total_duration(task_tree, critical_path_plan)
        }
        
        # 3. Resource optimization
        resource_optimized_plan = await self._optimize_resource_usage(task_tree, current_plan)
        optimizations['resource_optimized'] = {
            'plan': resource_optimized_plan,
            'duration': await self._calculate_total_duration(task_tree, resource_optimized_plan)
        }
        
        # Select best optimization
        current_duration = await self._calculate_total_duration(task_tree, current_plan)
        best_optimization = None
        best_improvement = 0
        
        for opt_name, opt_data in optimizations.items():
            improvement = (current_duration - opt_data['duration']) / current_duration
            if improvement > best_improvement:
                best_improvement = improvement
                best_optimization = opt_name
        
        result = {
            'original_duration': current_duration,
            'optimizations': optimizations,
            'best_optimization': best_optimization,
            'improvement': best_improvement
        }
        
        # Apply best optimization if significant improvement
        if best_improvement > 0.1:  # 10% improvement threshold
            self.execution_plans[task_tree_id] = optimizations[best_optimization]['plan']
            logger.info(f"Applied {best_optimization} optimization: {best_improvement:.1%} improvement")
        
        return result
    
    async def _optimize_load_balancing(self, task_tree: Dict[str, DecomposedTask], current_plan: List[List[str]]) -> List[List[str]]:
        """
        Optimize execution plan for better load balancing across stages.
        """
        # Get all atomic tasks with their complexities
        all_tasks = []
        for stage in current_plan:
            for task_id in stage:
                all_tasks.append((task_id, task_tree[task_id].complexity))
        
        # Sort by complexity (descending)
        all_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Redistribute tasks across stages with load balancing
        target_stages = len(current_plan)
        stage_loads = [0.0] * target_stages
        optimized_plan = [[] for _ in range(target_stages)]
        
        # Build dependency constraints
        dependencies = {}
        for task_id, _ in all_tasks:
            task = task_tree[task_id]
            dependencies[task_id] = {dep.dependency_id for dep in task.dependencies if dep.dependency_id in task_tree}
        
        # Assign tasks to stages while respecting dependencies
        task_to_stage = {}
        
        for task_id, complexity in all_tasks:
            # Find earliest possible stage based on dependencies
            min_stage = 0
            for dep_id in dependencies[task_id]:
                if dep_id in task_to_stage:
                    min_stage = max(min_stage, task_to_stage[dep_id] + 1)
            
            # Find stage with minimum load from min_stage onwards
            best_stage = min_stage
            for stage_idx in range(min_stage, target_stages):
                if stage_loads[stage_idx] < stage_loads[best_stage]:
                    best_stage = stage_idx
            
            # Assign task to best stage
            optimized_plan[best_stage].append(task_id)
            stage_loads[best_stage] += complexity
            task_to_stage[task_id] = best_stage
        
        # Remove empty stages
        optimized_plan = [stage for stage in optimized_plan if stage]
        
        return optimized_plan
    
    async def _optimize_critical_path(self, task_tree: Dict[str, DecomposedTask], current_plan: List[List[str]]) -> List[List[str]]:
        """
        Optimize execution plan by prioritizing critical path tasks.
        """
        # Calculate task priorities based on longest path to completion
        task_priorities = {}
        
        def calculate_priority(task_id: str, visited: Set[str] = None) -> float:
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0  # Avoid cycles
            
            if task_id in task_priorities:
                return task_priorities[task_id]
            
            task = task_tree[task_id]
            visited.add(task_id)
            
            # Base priority is task duration
            priority = task.estimated_duration
            
            # Add maximum priority of dependent tasks
            max_dependent_priority = 0
            for other_task in task_tree.values():
                for dep in other_task.dependencies:
                    if dep.dependency_id == task_id:
                        dependent_priority = calculate_priority(other_task.id, visited.copy())
                        max_dependent_priority = max(max_dependent_priority, dependent_priority)
            
            priority += max_dependent_priority
            task_priorities[task_id] = priority
            return priority
        
        # Calculate priorities for all tasks
        for task_id in task_tree:
            calculate_priority(task_id)
        
        # Rebuild execution plan prioritizing critical path
        all_atomic_tasks = [task_id for task_id, task in task_tree.items() if task.task_type == TaskType.ATOMIC]
        
        # Sort by priority (descending)
        all_atomic_tasks.sort(key=lambda tid: task_priorities.get(tid, 0), reverse=True)
        
        # Create new plan respecting dependencies
        optimized_plan = []
        completed_tasks = set()
        remaining_tasks = set(all_atomic_tasks)
        
        # Build dependency map
        dependencies = {}
        for task_id in all_atomic_tasks:
            task = task_tree[task_id]
            dependencies[task_id] = {dep.dependency_id for dep in task.dependencies if dep.dependency_id in task_tree}
        
        while remaining_tasks:
            # Find ready tasks (no unfulfilled dependencies)
            ready_tasks = []
            for task_id in remaining_tasks:
                if dependencies[task_id].issubset(completed_tasks):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Break cycle by selecting highest priority remaining task
                ready_tasks = [max(remaining_tasks, key=lambda tid: task_priorities.get(tid, 0))]
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda tid: task_priorities.get(tid, 0), reverse=True)
            
            optimized_plan.append(ready_tasks)
            completed_tasks.update(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return optimized_plan
    
    async def _optimize_resource_usage(self, task_tree: Dict[str, DecomposedTask], current_plan: List[List[str]]) -> List[List[str]]:
        """
        Optimize execution plan for better resource utilization.
        """
        # Group tasks by required capabilities
        capability_groups = defaultdict(list)
        for task_id, task in task_tree.items():
            if task.task_type == TaskType.ATOMIC:
                cap_key = tuple(sorted(task.required_capabilities))
                capability_groups[cap_key].append(task_id)
        
        # Try to schedule tasks with similar capabilities together
        optimized_plan = []
        scheduled_tasks = set()
        
        # Build dependency constraints
        dependencies = {}
        for task_id, task in task_tree.items():
            if task.task_type == TaskType.ATOMIC:
                dependencies[task_id] = {dep.dependency_id for dep in task.dependencies if dep.dependency_id in task_tree}
        
        # Schedule by capability groups
        while len(scheduled_tasks) < len([t for t in task_tree.values() if t.task_type == TaskType.ATOMIC]):
            stage_tasks = []
            
            # For each capability group, find ready tasks
            for cap_key, task_ids in capability_groups.items():
                available_tasks = [tid for tid in task_ids if tid not in scheduled_tasks]
                
                for task_id in available_tasks:
                    # Check if dependencies are satisfied
                    if dependencies[task_id].issubset(scheduled_tasks):
                        stage_tasks.append(task_id)
                        break  # One task per capability group per stage
            
            if not stage_tasks:
                # If no tasks are ready, break dependency cycle
                remaining_tasks = [tid for tid, task in task_tree.items() 
                                 if task.task_type == TaskType.ATOMIC and tid not in scheduled_tasks]
                if remaining_tasks:
                    stage_tasks = [remaining_tasks[0]]
            
            if stage_tasks:
                optimized_plan.append(stage_tasks)
                scheduled_tasks.update(stage_tasks)
        
        return optimized_plan
    
    async def learn_from_execution_feedback(self, task_tree_id: str, execution_results: Dict[str, Any]):
        """
        Learn from execution results to improve future decompositions.
        """
        if task_tree_id not in self.task_trees:
            logger.warning(f"Task tree {task_tree_id} not found for learning")
            return
        
        task_tree = self.task_trees[task_tree_id]
        root_task = next(task for task in task_tree.values() if task.parent_id is None)
        strategy_used = root_task.decomposition_strategy
        
        # Calculate success metrics
        total_tasks = len([t for t in task_tree.values() if t.task_type == TaskType.ATOMIC])
        successful_tasks = execution_results.get('successful_tasks', 0)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        actual_duration = execution_results.get('actual_duration', 0)
        estimated_duration = execution_results.get('estimated_duration', 1)
        duration_accuracy = 1.0 - abs(actual_duration - estimated_duration) / estimated_duration
        
        # Update strategy performance
        if strategy_used:
            performance_score = (success_rate * 0.6 + duration_accuracy * 0.4)
            current_performance = self.strategy_performance[strategy_used]
            self.strategy_performance[strategy_used] = (
                current_performance * 0.8 + performance_score * 0.2
            )
        
        # Learn complexity patterns
        for task in task_tree.values():
            if task.task_type == TaskType.ATOMIC:
                task_key = f"{len(task.required_capabilities)}_{task.complexity:.1f}"
                actual_complexity = execution_results.get('task_complexities', {}).get(task.id, task.complexity)
                
                if task_key in self.complexity_patterns:
                    self.complexity_patterns[task_key] = (
                        self.complexity_patterns[task_key] * 0.8 + actual_complexity * 0.2
                    )
                else:
                    self.complexity_patterns[task_key] = actual_complexity
        
        # Update capability mappings
        for task in task_tree.values():
            if task.assigned_agent and task.task_type == TaskType.ATOMIC:
                for capability in task.required_capabilities:
                    if capability not in self.capability_mappings:
                        self.capability_mappings[capability] = []
                    
                    if task.assigned_agent not in self.capability_mappings[capability]:
                        self.capability_mappings[capability].append(task.assigned_agent)
        
        logger.info(f"Learning completed for task tree {task_tree_id}: success_rate={success_rate:.2f}, strategy_performance updated")
    
    def get_decomposer_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the task decomposer.
        """
        return {
            'total_decompositions': len(self.decomposition_history),
            'active_task_trees': len(self.task_trees),
            'strategy_performance': {strategy.value: score for strategy, score in self.strategy_performance.items()},
            'learned_patterns': len(self.complexity_patterns),
            'capability_mappings': {cap: len(agents) for cap, agents in self.capability_mappings.items()},
            'configuration': {
                'max_decomposition_depth': self.max_decomposition_depth,
                'min_task_size': self.min_task_size,
                'max_subtasks_per_level': self.max_subtasks_per_level,
                'parallel_threshold': self.parallel_threshold
            },
            'recent_decompositions': [
                {
                    'task_id': result.original_task_id,
                    'strategy': result.strategy_used.value,
                    'depth': result.decomposition_depth,
                    'confidence': result.confidence_score,
                    'total_tasks': len(result.task_tree),
                    'estimated_duration': result.total_estimated_duration
                }
                for result in self.decomposition_history[-5:]
            ]
        }