#!/usr/bin/env python3
"""
Main Brain Orchestrator using LangGraph
Implements the AGI/ASI control flow with self-improvement capabilities
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from brain_state import BrainState, TaskStatus, AgentType, BrainConfig
from ..memory.vector_memory import VectorMemory
from ..agents.agent_router import AgentRouter
from ..evaluator.quality_evaluator import QualityEvaluator
from ..improver.self_improver import SelfImprover
from .universal_learning_machine import UniversalLearningMachine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrainOrchestrator:
    """Main orchestrator for the AGI/ASI Brain system"""
    
    def __init__(self, config: BrainConfig):
        self.config = config
        self.memory = VectorMemory(config)
        self.router = AgentRouter(config)
        self.evaluator = QualityEvaluator(config)
        self.improver = SelfImprover(config)
        
        # Initialize Universal Learning Machine
        self.ulm = UniversalLearningMachine(config)
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
        
        # Track active sessions
        self.active_sessions: Dict[str, BrainState] = {}
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the Brain"""
        workflow = StateGraph(BrainState)
        
        # Add nodes for each phase
        workflow.add_node("perceive", self._perceive)
        workflow.add_node("plan", self._plan)
        workflow.add_node("execute", self._execute)
        workflow.add_node("evaluate", self._evaluate)
        workflow.add_node("improve", self._improve)
        workflow.add_node("learn", self._learn)
        workflow.add_node("finalize", self._finalize)
        
        # Define the flow
        workflow.set_entry_point("perceive")
        workflow.add_edge("perceive", "plan")
        workflow.add_edge("plan", "execute")
        workflow.add_edge("execute", "evaluate")
        
        # Conditional edge for improvement
        workflow.add_conditional_edges(
            "evaluate",
            self._should_improve,
            {
                True: "improve",
                False: "learn"
            }
        )
        
        workflow.add_edge("improve", "learn")
        workflow.add_edge("learn", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    async def _perceive(self, state: BrainState) -> BrainState:
        """Perception phase: Process input and retrieve relevant memories"""
        logger.info(f"🧠 Perceiving input: {state['user_input'][:100]}...")
        
        state['status'] = TaskStatus.PLANNING
        state['current_step'] = 1
        
        # Retrieve relevant memories
        memories = await self.memory.search(
            query=state['user_input'],
            top_k=10,
            include_metadata=True
        )
        state['retrieved_memories'] = memories
        
        # Analyze input type and complexity
        input_analysis = await self._analyze_input(state['user_input'])
        state['task_plan'] = input_analysis['task_breakdown']
        
        logger.info(f"📊 Retrieved {len(memories)} relevant memories")
        return state
    
    async def _plan(self, state: BrainState) -> BrainState:
        """Planning phase: Select agents and allocate resources"""
        logger.info("📋 Planning agent execution strategy...")
        
        # Select optimal agents based on task
        selected_agents = await self.router.select_agents(
            task_plan=state['task_plan'],
            available_resources=self._get_available_resources(),
            memories=state['retrieved_memories']
        )
        
        state['selected_agents'] = selected_agents['agents']
        state['resource_allocation'] = selected_agents['resources']
        
        logger.info(f"🎯 Selected {len(state['selected_agents'])} agents for execution")
        return state
    
    async def _execute(self, state: BrainState) -> BrainState:
        """Execution phase: Run selected agents in parallel"""
        logger.info("⚡ Executing agent tasks...")
        state['status'] = TaskStatus.EXECUTING
        
        # Prepare agent tasks
        agent_tasks = []
        for agent_type in state['selected_agents']:
            task = self._create_agent_task(
                agent_type=agent_type,
                user_input=state['user_input'],
                task_plan=state['task_plan'],
                memories=state['retrieved_memories'],
                resources=state['resource_allocation'].get(agent_type.value, {})
            )
            agent_tasks.append(task)
        
        # Execute agents in parallel with resource limits
        results = await self._execute_parallel_with_limits(
            agent_tasks,
            max_concurrent=self.config['max_concurrent_agents']
        )
        
        state['agent_results'] = results
        state['current_step'] = 2
        
        logger.info(f"✅ Completed {len(results)} agent executions")
        return state
    
    async def _evaluate(self, state: BrainState) -> BrainState:
        """Evaluation phase: Score results and determine quality"""
        logger.info("🎯 Evaluating results quality...")
        state['status'] = TaskStatus.EVALUATING
        
        # Evaluate each agent result
        quality_scores = {}
        for result in state['agent_results']:
            score = await self.evaluator.evaluate_result(
                result=result,
                original_input=state['user_input'],
                expected_output=state.get('expected_output')
            )
            quality_scores[result['agent']] = score
        
        # Calculate overall score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        
        state['quality_scores'] = quality_scores
        state['overall_score'] = overall_score
        state['needs_improvement'] = overall_score < self.config['improvement_threshold']
        
        logger.info(f"📊 Overall quality score: {overall_score:.2f}")
        return state
    
    async def _improve(self, state: BrainState) -> BrainState:
        """Improvement phase: Generate patches to improve the system"""
        logger.info("🔧 Generating improvement patches...")
        state['status'] = TaskStatus.IMPROVING
        
        # Analyze failures and generate improvements
        improvements = await self.improver.analyze_and_improve(
            state=state,
            min_score=self.config['min_quality_score']
        )
        
        state['improvement_suggestions'] = improvements['suggestions']
        state['patches'] = improvements['patches']
        
        # Apply improvements if auto-improve is enabled
        if self.config['auto_improve'] and improvements['patches']:
            await self._apply_improvements(improvements['patches'])
        
        logger.info(f"💡 Generated {len(improvements['patches'])} improvement patches")
        return state
    
    async def _learn(self, state: BrainState) -> BrainState:
        """Learning phase: Update models and memory based on experience"""
        logger.info("🎓 Learning from experience...")
        
        # Store successful patterns in memory
        new_memories = []
        for result in state['agent_results']:
            if result['success'] and result.get('quality_score', 0) > 0.8:
                memory_entry = {
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now(),
                    'content': f"Task: {state['user_input']}\nSolution: {result['output']}",
                    'metadata': {
                        'agent': result['agent'],
                        'score': result.get('quality_score', 0),
                        'execution_time': result['execution_time']
                    },
                    'agent_source': result['agent']
                }
                new_memories.append(memory_entry)
        
        # Store memories
        if new_memories:
            await self.memory.store_batch(new_memories)
            state['new_memories'] = new_memories
        
        # Extract learned patterns
        patterns = await self._extract_patterns(state)
        state['learned_patterns'] = patterns
        
        logger.info(f"📚 Stored {len(new_memories)} new memories")
        return state
    
    async def _finalize(self, state: BrainState) -> BrainState:
        """Finalization phase: Prepare output and cleanup"""
        logger.info("🎁 Finalizing response...")
        state['status'] = TaskStatus.COMPLETED
        
        # Aggregate best results
        best_result = max(
            state['agent_results'],
            key=lambda r: r.get('quality_score', 0) if r['success'] else -1
        )
        
        state['final_output'] = best_result['output']
        state['confidence'] = best_result.get('quality_score', 0.5)
        
        # Clean up resources
        await self._cleanup_resources(state)
        
        logger.info("✨ Brain cycle completed successfully")
        return state
    
    def _should_improve(self, state: BrainState) -> bool:
        """Determine if improvement phase is needed"""
        return state['needs_improvement']
    
    async def _analyze_input(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to create task breakdown"""
        # This would use an LLM to break down the task
        # For now, return a simple structure
        return {
            'task_breakdown': [
                {
                    'description': user_input,
                    'complexity': 'medium',
                    'required_capabilities': ['reasoning', 'coding']
                }
            ]
        }
    
    def _get_available_resources(self) -> Dict[str, float]:
        """Get current available system resources"""
        import psutil
        
        return {
            'cpu_percent': 100 - psutil.cpu_percent(),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_available': self.config['gpu_memory_gb'] > 0
        }
    
    async def _create_agent_task(
        self,
        agent_type: AgentType,
        user_input: str,
        task_plan: List[Dict[str, Any]],
        memories: List[Dict[str, Any]],
        resources: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create a task for a specific agent"""
        return {
            'agent_type': agent_type,
            'input': user_input,
            'task_plan': task_plan,
            'context': {
                'memories': memories,
                'resources': resources
            }
        }
    
    async def _execute_parallel_with_limits(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int
    ) -> List[Dict[str, Any]]:
        """Execute tasks in parallel with concurrency limits"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_limit(task):
            async with semaphore:
                return await self.router.execute_agent(task)
        
        results = await asyncio.gather(
            *[run_with_limit(task) for task in tasks],
            return_exceptions=True
        )
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'agent': tasks[i]['agent_type'].value,
                    'task_id': str(uuid.uuid4()),
                    'output': None,
                    'success': False,
                    'error': str(result),
                    'execution_time': 0,
                    'resources_used': {}
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _apply_improvements(self, patches: List[Dict[str, Any]]):
        """Apply improvement patches to the system"""
        logger.info(f"🔨 Applying {len(patches)} improvement patches...")
        
        # This would integrate with git and create PRs
        # For now, log the patches
        for patch in patches:
            logger.info(f"  - {patch['description']}")
    
    async def _extract_patterns(self, state: BrainState) -> List[Dict[str, Any]]:
        """Extract reusable patterns from successful executions"""
        patterns = []
        
        for result in state['agent_results']:
            if result['success'] and result.get('quality_score', 0) > 0.9:
                pattern = {
                    'input_type': 'general',  # Would be classified by LLM
                    'agent': result['agent'],
                    'execution_time': result['execution_time'],
                    'quality_score': result.get('quality_score', 0),
                    'resource_efficiency': self._calculate_efficiency(result)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _calculate_efficiency(self, result: Dict[str, Any]) -> float:
        """Calculate resource efficiency score"""
        resources = result.get('resources_used', {})
        time_factor = 1 / (1 + result['execution_time'] / 60)  # Normalize to minutes
        resource_factor = 1 / (1 + sum(resources.values()) / 100)
        
        return (time_factor + resource_factor) / 2
    
    async def _cleanup_resources(self, state: BrainState):
        """Clean up resources after execution"""
        # Release model locks, clear caches, etc.
        pass
    
    async def process(self, user_input: str) -> Dict[str, Any]:
        """Main entry point to process a user request"""
        # Initialize state
        initial_state = {
            'request_id': str(uuid.uuid4()),
            'user_input': user_input,
            'timestamp': datetime.now(),
            'status': TaskStatus.PENDING,
            'current_step': 0,
            'gpu_available': self.config['gpu_memory_gb'] > 0,
            'memory_usage': 0.0,
            'active_models': [],
            'error_log': [],
            'agent_results': [],
            'retrieved_memories': [],
            'new_memories': [],
            'quality_scores': {},
            'overall_score': 0.0,
            'needs_improvement': False,
            'improvement_suggestions': [],
            'patches': [],
            'model_adaptations': {},
            'learned_patterns': [],
            'final_output': None,
            'output_format': 'text',
            'confidence': 0.0
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": initial_state['request_id']}}
        final_state = await self.app.ainvoke(initial_state, config)
        
        return {
            'request_id': final_state['request_id'],
            'output': final_state['final_output'],
            'confidence': final_state['confidence'],
            'execution_time': (datetime.now() - final_state['timestamp']).total_seconds(),
            'agents_used': [r['agent'] for r in final_state['agent_results']],
            'improvements_suggested': len(final_state['improvement_suggestions'])
        }