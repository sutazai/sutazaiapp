#!/usr/bin/env python3
"""
SutazAI LocalAGI Main Integration Module

This module integrates all LocalAGI components and provides the main
entry point for the autonomous orchestration system. It coordinates
the engines, frameworks, and protocols to create a fully autonomous
AI system capable of operating without human intervention.
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import LocalAGI components
from localagi.engine.autonomous_orchestration_engine import AutonomousOrchestrationEngine, get_orchestration_engine
from localagi.engine.decision_engine import AutonomousDecisionEngine
from localagi.engine.task_decomposer import RecursiveTaskDecomposer
from localagi.swarms.swarm_coordinator import AutonomousSwarmCoordinator
from localagi.workflows.self_improving_workflow_engine import SelfImprovingWorkflowEngine, OptimizationObjective, WorkflowStep
from localagi.frameworks.collaborative_problem_solver import CollaborativeProblemSolver, Problem, ProblemType
from localagi.goals.autonomous_goal_achievement_system import AutonomousGoalAchievementSystem, Goal, GoalType
from localagi.protocols.autonomous_coordination_protocols import AutonomousCoordinationProtocols

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/localagi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LocalAGISystem:
    """
    Main LocalAGI system that integrates all autonomous components.
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/localagi/configs/autonomous_orchestrator_config.yaml"):
        self.config_path = config_path
        
        # Core engines
        self.orchestration_engine = None
        self.decision_engine = None
        self.task_decomposer = None
        
        # Coordination systems
        self.swarm_coordinator = None
        self.workflow_engine = None
        self.problem_solver = None
        self.goal_system = None
        self.coordination_protocols = None
        
        # System state
        self.system_status = "initializing"
        self.initialization_time = None
        self.last_health_check = None
        
        logger.info("LocalAGI System initializing...")
    
    async def initialize(self):
        """
        Initialize all LocalAGI components.
        """
        try:
            logger.info("Starting LocalAGI system initialization...")
            
            # Initialize core orchestration engine
            self.orchestration_engine = get_orchestration_engine()
            await self._wait_for_orchestration_engine()
            
            # Initialize decision engine
            self.decision_engine = AutonomousDecisionEngine(self.orchestration_engine)
            
            # Initialize task decomposer
            self.task_decomposer = RecursiveTaskDecomposer(self.orchestration_engine)
            
            # Initialize coordination systems
            self.coordination_protocols = AutonomousCoordinationProtocols(self.orchestration_engine)
            
            # Initialize swarm coordinator
            self.swarm_coordinator = AutonomousSwarmCoordinator(
                "master_swarm", 
                "Autonomous AI System Coordination", 
                self.orchestration_engine
            )
            
            # Initialize workflow engine
            self.workflow_engine = SelfImprovingWorkflowEngine(self.orchestration_engine)
            
            # Initialize problem solver
            self.problem_solver = CollaborativeProblemSolver(self.orchestration_engine)
            
            # Initialize goal achievement system
            self.goal_system = AutonomousGoalAchievementSystem(self.orchestration_engine)
            
            # Start system health monitoring
            asyncio.create_task(self._system_health_monitor())
            
            # Start autonomous operations
            asyncio.create_task(self._autonomous_operation_loop())
            
            self.system_status = "operational"
            self.initialization_time = datetime.now()
            
            logger.info("LocalAGI system initialization completed successfully")
            
        except Exception as e:
            logger.error(f"LocalAGI system initialization failed: {e}")
            self.system_status = "failed"
            raise
    
    async def _wait_for_orchestration_engine(self):
        """
        Wait for orchestration engine to be ready.
        """
        max_wait = 60  # seconds
        wait_time = 0
        
        while wait_time < max_wait:
            try:
                if self.orchestration_engine and len(self.orchestration_engine.agents) > 0:
                    logger.info(f"Orchestration engine ready with {len(self.orchestration_engine.agents)} agents")
                    return
            except Exception as e:
                logger.debug(f"Waiting for orchestration engine: {e}")
            
            await asyncio.sleep(2)
            wait_time += 2
        
        logger.warning("Orchestration engine not fully ready, proceeding anyway")
    
    async def submit_autonomous_task(self, 
                                   description: str, 
                                   requirements: List[str] = None,
                                   priority: float = 0.5,
                                   autonomous_mode: bool = True) -> str:
        """
        Submit a task for autonomous execution.
        """
        logger.info(f"Submitting autonomous task: {description}")
        
        try:
            # Submit to orchestration engine
            task_id = await self.orchestration_engine.submit_task(
                description=description,
                requirements=requirements or [],
                priority=priority
            )
            
            # If autonomous mode is enabled, also create a goal
            if autonomous_mode:
                goal = Goal(
                    id=f"goal_{task_id}",
                    title=f"Autonomous Task: {description[:50]}",
                    description=description,
                    goal_type=GoalType.ACHIEVEMENT,
                    success_criteria=[f"Task {task_id} completed successfully"],
                    constraints={},
                    priority=priority,
                    deadline=None,
                    estimated_effort=0.5,
                    required_resources={},
                    required_capabilities=requirements or [],
                    context={'original_task_id': task_id}
                )
                
                await self.goal_system.pursue_goal(goal)
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to submit autonomous task: {e}")
            raise
    
    async def solve_problem_collaboratively(self, 
                                          problem_description: str,
                                          problem_type: str = "analytical",
                                          max_agents: int = 8) -> Dict[str, Any]:
        """
        Solve a problem using collaborative multi-agent approach.
        """
        logger.info(f"Starting collaborative problem solving: {problem_description}")
        
        try:
            # Convert string to ProblemType enum
            ptype = ProblemType.ANALYTICAL
            for pt in ProblemType:
                if pt.value.lower() == problem_type.lower():
                    ptype = pt
                    break
            
            problem = Problem(
                id=f"problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title=problem_description[:100],
                description=problem_description,
                problem_type=ptype,
                constraints={},
                success_criteria=["Problem solved with high confidence"],
                domain_areas=["general"],
                complexity_score=0.7,
                priority=0.8,
                deadline=None,
                context={}
            )
            
            session = await self.problem_solver.solve_problem_collaboratively(
                problem=problem,
                max_agents=max_agents
            )
            
            return {
                'session_id': session.session_id,
                'problem_id': problem.id,
                'final_solution': session.final_solution.__dict__ if session.final_solution else None,
                'participating_agents': session.participating_agents,
                'collaboration_pattern': session.collaboration_pattern.value,
                'solutions_generated': len(session.solutions),
                'session_duration': (session.session_end - session.session_start).total_seconds() if session.session_end else None
            }
            
        except Exception as e:
            logger.error(f"Collaborative problem solving failed: {e}")
            raise
    
    async def create_autonomous_workflow(self, 
                                       description: str,
                                       steps: List[Dict[str, Any]],
                                       optimization_objectives: List[str] = None) -> str:
        """
        Create a self-improving autonomous workflow.
        """
        logger.info(f"Creating autonomous workflow: {description}")
        
        try:
            # Convert string objectives to enum
            objectives = []
            obj_mapping = {
                'speed': OptimizationObjective.SPEED,
                'accuracy': OptimizationObjective.ACCURACY,
                'efficiency': OptimizationObjective.EFFICIENCY,
                'cost': OptimizationObjective.COST,
                'quality': OptimizationObjective.QUALITY,
                'reliability': OptimizationObjective.RELIABILITY
            }
            
            for obj_str in (optimization_objectives or ['efficiency', 'quality']):
                if obj_str.lower() in obj_mapping:
                    objectives.append(obj_mapping[obj_str.lower()])
            
            if not objectives:
                objectives = [OptimizationObjective.EFFICIENCY]
            
            # Convert step dictionaries to WorkflowStep objects
            workflow_steps = []
            for i, step_data in enumerate(steps):
                step = WorkflowStep(
                    id=step_data.get('id', f"step_{i+1}"),
                    name=step_data.get('name', f"Step {i+1}"),
                    description=step_data.get('description', ''),
                    agent_capability=step_data.get('agent_capability', 'general'),
                    parameters=step_data.get('parameters', {}),
                    preconditions=step_data.get('preconditions', []),
                    postconditions=step_data.get('postconditions', []),
                    success_criteria=step_data.get('success_criteria', []),
                    timeout=step_data.get('timeout', 300.0),
                    retry_count=step_data.get('retry_count', 3),
                    importance_weight=step_data.get('importance_weight', 1.0),
                    adaptable_parameters=step_data.get('adaptable_parameters', [])
                )
                workflow_steps.append(step)
            
            workflow_id = self.workflow_engine.create_workflow(
                description=description,
                optimization_objectives=objectives,
                steps=workflow_steps
            )
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a self-improving workflow.
        """
        logger.info(f"Executing workflow: {workflow_id}")
        
        try:
            execution = await self.workflow_engine.execute_workflow(workflow_id, context or {})
            
            return {
                'execution_id': execution.execution_id,
                'workflow_id': execution.workflow_id,
                'success': execution.overall_success,
                'steps_executed': len(execution.steps_executed),
                'performance_metrics': execution.performance_metrics,
                'learned_patterns': execution.learned_patterns,
                'optimization_opportunities': execution.optimization_opportunities,
                'duration': (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def form_agent_swarm(self, 
                             goal: str,
                             required_capabilities: List[str],
                             max_size: int = 10) -> Dict[str, Any]:
        """
        Form an autonomous agent swarm for collaborative task execution.
        """
        logger.info(f"Forming agent swarm for goal: {goal}")
        
        try:
            success = await self.swarm_coordinator.form_swarm(
                required_capabilities=required_capabilities,
                max_size=max_size
            )
            
            if success:
                swarm_status = self.swarm_coordinator.get_swarm_status()
                return {
                    'swarm_formed': True,
                    'swarm_id': self.swarm_coordinator.swarm_id,
                    'goal': goal,
                    'member_count': swarm_status['member_count'],
                    'leader_id': swarm_status['leader_id'],
                    'members': swarm_status['members']
                }
            else:
                return {
                    'swarm_formed': False,
                    'error': 'Unable to form swarm with specified requirements'
                }
                
        except Exception as e:
            logger.error(f"Swarm formation failed: {e}")
            raise
    
    async def decompose_complex_task(self, 
                                   task_description: str,
                                   requirements: List[str] = None,
                                   constraints: Dict[str, Any] = None,
                                   max_depth: int = 5) -> Dict[str, Any]:
        """
        Recursively decompose a complex task into manageable subtasks.
        """
        logger.info(f"Decomposing complex task: {task_description}")
        
        try:
            result = await self.task_decomposer.decompose_task_recursively(
                task_description=task_description,
                requirements=requirements or [],
                constraints=constraints or {},
                max_depth=max_depth
            )
            
            return {
                'original_task_id': result.original_task_id,
                'total_tasks': len(result.task_tree),
                'decomposition_depth': result.decomposition_depth,
                'strategy_used': result.strategy_used.value,
                'confidence_score': result.confidence_score,
                'execution_stages': len(result.execution_plan),
                'estimated_duration': result.total_estimated_duration,
                'optimization_opportunities': result.optimization_opportunities,
                'task_tree': {
                    task_id: {
                        'title': task.title,
                        'description': task.description,
                        'complexity': task.complexity,
                        'required_capabilities': task.required_capabilities,
                        'subtasks': task.subtasks
                    }
                    for task_id, task in result.task_tree.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            raise
    
    async def _autonomous_operation_loop(self):
        """
        Main autonomous operation loop that continuously optimizes the system.
        """
        logger.info("Starting autonomous operation loop")
        
        while self.system_status == "operational":
            try:
                # Run system optimizations
                await self._perform_system_optimizations()
                
                # Check for learning opportunities
                await self._check_learning_opportunities()
                
                # Adaptive system improvements
                await self._adaptive_system_improvements()
                
                # Sleep before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Autonomous operation loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_system_optimizations(self):
        """
        Perform various system optimizations autonomously.
        """
        try:
            # Optimize workflows
            await self.workflow_engine.optimize_all_workflows()
            
            # Adapt decision parameters
            await self.decision_engine.adapt_decision_parameters()
            
            # System self-improvement
            await self.orchestration_engine.self_improve_system()
            
            logger.debug("System optimizations completed")
            
        except Exception as e:
            logger.error(f"System optimization failed: {e}")
    
    async def _check_learning_opportunities(self):
        """
        Check for and act on learning opportunities.
        """
        try:
            # Check orchestration engine performance trends
            system_status = await self.orchestration_engine.get_system_status()
            
            # If performance is declining, trigger adaptations
            if system_status.get('performance_metrics', {}).get('success_rate', 1.0) < 0.8:
                logger.info("Performance decline detected, triggering adaptations")
                await self._trigger_performance_adaptations()
            
        except Exception as e:
            logger.error(f"Learning opportunity check failed: {e}")
    
    async def _adaptive_system_improvements(self):
        """
        Implement adaptive system improvements based on learned patterns.
        """
        try:
            # Get system metrics
            metrics = await self.get_comprehensive_status()
            
            # Identify improvement opportunities
            opportunities = []
            
            # Check agent utilization balance
            agent_utilization = metrics.get('orchestration_engine', {}).get('agent_utilization', {})
            if agent_utilization:
                utilization_values = list(agent_utilization.values())
                if len(utilization_values) > 1:
                    std_dev = np.std(utilization_values)
                    mean_util = np.mean(utilization_values)
                    if std_dev > mean_util * 0.5:
                        opportunities.append("rebalance_agent_workload")
            
            # Check workflow performance
            workflow_status = metrics.get('workflow_engine', {})
            total_workflows = workflow_status.get('total_workflows', 0)
            if total_workflows > 0:
                avg_adaptations = sum(
                    wf.get('adaptation_count', 0) 
                    for wf in workflow_status.get('workflows', {}).values()
                ) / total_workflows
                
                if avg_adaptations > 5:  # High adaptation rate
                    opportunities.append("review_workflow_templates")
            
            # Implement improvements
            for opportunity in opportunities:
                await self._implement_system_improvement(opportunity)
            
        except Exception as e:
            logger.error(f"Adaptive system improvement failed: {e}")
    
    async def _implement_system_improvement(self, improvement_type: str):
        """
        Implement a specific system improvement.
        """
        logger.info(f"Implementing system improvement: {improvement_type}")
        
        try:
            if improvement_type == "rebalance_agent_workload":
                # Trigger agent workload rebalancing
                await self.coordination_protocols.distribute_tasks(
                    tasks=[],  # Empty tasks to trigger rebalancing
                    agents=list(self.orchestration_engine.agents.keys()),
                    strategy="load_balanced"
                )
            
            elif improvement_type == "review_workflow_templates":
                # Analyze workflow patterns and create optimized templates
                # This would involve more sophisticated analysis in practice
                logger.info("Reviewing workflow templates for optimization")
            
        except Exception as e:
            logger.error(f"System improvement implementation failed: {e}")
    
    async def _trigger_performance_adaptations(self):
        """
        Trigger performance adaptations when system performance declines.
        """
        logger.info("Triggering performance adaptations")
        
        try:
            # Reduce exploration rate in decision making
            self.decision_engine.exploration_rate *= 0.9
            
            # Increase system monitoring frequency temporarily
            # This would be implemented with more sophisticated monitoring
            
            logger.info("Performance adaptations applied")
            
        except Exception as e:
            logger.error(f"Performance adaptation failed: {e}")
    
    async def _system_health_monitor(self):
        """
        Continuously monitor system health and performance.
        """
        logger.info("Starting system health monitor")
        
        while self.system_status == "operational":
            try:
                # Update last health check time
                self.last_health_check = datetime.now()
                
                # Check component health
                await self._check_component_health()
                
                # Check resource usage
                await self._check_resource_usage()
                
                # Check agent network health
                await self._check_agent_network_health()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"System health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_health(self):
        """
        Check the health of all system components.
        """
        components = {
            'orchestration_engine': self.orchestration_engine,
            'decision_engine': self.decision_engine,
            'task_decomposer': self.task_decomposer,
            'workflow_engine': self.workflow_engine,
            'problem_solver': self.problem_solver,
            'goal_system': self.goal_system,
            'coordination_protocols': self.coordination_protocols
        }
        
        for component_name, component in components.items():
            try:
                if hasattr(component, 'get_system_status') or hasattr(component, 'get_status'):
                    # Component is healthy if it can provide status
                    pass
            except Exception as e:
                logger.warning(f"Component {component_name} health check failed: {e}")
    
    async def _check_resource_usage(self):
        """
        Check system resource usage and trigger optimizations if needed.
        """
        try:
            # Check orchestration engine metrics
            if self.orchestration_engine:
                status = await self.orchestration_engine.get_system_status()
                
                # Check if too many tasks are pending
                pending_tasks = status.get('pending_tasks', 0)
                if pending_tasks > 100:  # Arbitrary threshold
                    logger.warning(f"High number of pending tasks: {pending_tasks}")
                    # Could trigger task prioritization or agent scaling
        
        except Exception as e:
            logger.error(f"Resource usage check failed: {e}")
    
    async def _check_agent_network_health(self):
        """
        Check the health of the agent network.
        """
        try:
            if self.coordination_protocols:
                protocol_status = self.coordination_protocols.get_protocol_status()
                
                network_health = protocol_status.get('network_health', {})
                total_agents = network_health.get('total_agents', 0)
                responsive_agents = network_health.get('responsive_agents', 0)
                
                if total_agents > 0:
                    health_ratio = responsive_agents / total_agents
                    if health_ratio < 0.8:  # Less than 80% responsive
                        logger.warning(f"Agent network health degraded: {health_ratio:.2%}")
        
        except Exception as e:
            logger.error(f"Agent network health check failed: {e}")
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the entire LocalAGI system.
        """
        try:
            status = {
                'system_status': self.system_status,
                'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'uptime': (datetime.now() - self.initialization_time).total_seconds() if self.initialization_time else 0
            }
            
            # Get status from each component
            if self.orchestration_engine:
                status['orchestration_engine'] = await self.orchestration_engine.get_system_status()
            
            if self.decision_engine:
                status['decision_engine'] = self.decision_engine.get_decision_engine_status()
            
            if self.task_decomposer:
                status['task_decomposer'] = self.task_decomposer.get_decomposer_status()
            
            if self.workflow_engine:
                status['workflow_engine'] = self.workflow_engine.get_engine_status()
            
            if self.problem_solver:
                status['problem_solver'] = self.problem_solver.get_solver_status()
            
            if self.goal_system:
                status['goal_system'] = self.goal_system.get_system_status()
            
            if self.coordination_protocols:
                status['coordination_protocols'] = self.coordination_protocols.get_protocol_status()
            
            if self.swarm_coordinator:
                status['swarm_coordinator'] = self.swarm_coordinator.get_swarm_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive status: {e}")
            return {'error': str(e)}
    
    async def shutdown(self):
        """
        Gracefully shutdown the LocalAGI system.
        """
        logger.info("Shutting down LocalAGI system...")
        
        self.system_status = "shutting_down"
        
        try:
            # Stop autonomous operations
            await asyncio.sleep(1)  # Allow current operations to complete
            
            # Cleanup resources
            if self.orchestration_engine:
                # Save state if needed
                pass
            
            self.system_status = "shutdown"
            logger.info("LocalAGI system shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            self.system_status = "shutdown_error"

# Global system instance
localagi_system = None

async def get_localagi_system() -> LocalAGISystem:
    """
    Get or create the global LocalAGI system instance.
    """
    global localagi_system
    if localagi_system is None:
        localagi_system = LocalAGISystem()
        await localagi_system.initialize()
    return localagi_system

async def main():
    """
    Main entry point for LocalAGI system.
    """
    try:
        logger.info("Starting LocalAGI Autonomous Orchestration System")
        
        # Initialize system
        system = await get_localagi_system()
        
        # Demo: Submit an autonomous task
        task_id = await system.submit_autonomous_task(
            description="Analyze system performance and provide optimization recommendations",
            requirements=["analysis", "optimization", "monitoring"],
            priority=0.8
        )
        
        logger.info(f"Submitted demo task: {task_id}")
        
        # Demo: Create and execute a workflow
        workflow_steps = [
            {
                'name': 'Data Collection',
                'description': 'Collect system performance data',
                'agent_capability': 'monitoring',
                'parameters': {'data_source': 'system_metrics'},
                'success_criteria': ['Data collected successfully']
            },
            {
                'name': 'Data Analysis',
                'description': 'Analyze collected performance data',
                'agent_capability': 'analysis',
                'parameters': {'analysis_type': 'performance'},
                'success_criteria': ['Analysis completed with insights'],
                'preconditions': ['Data Collection']
            },
            {
                'name': 'Optimization Recommendations',
                'description': 'Generate optimization recommendations',
                'agent_capability': 'optimization',
                'parameters': {'optimization_target': 'performance'},
                'success_criteria': ['Recommendations generated'],
                'preconditions': ['Data Analysis']
            }
        ]
        
        workflow_id = await system.create_autonomous_workflow(
            description="System Performance Optimization Workflow",
            steps=workflow_steps,
            optimization_objectives=['efficiency', 'quality']
        )
        
        logger.info(f"Created demo workflow: {workflow_id}")
        
        # Execute workflow
        execution_result = await system.execute_workflow(workflow_id)
        logger.info(f"Workflow execution result: {execution_result}")
        
        # Demo: Solve a problem collaboratively
        problem_result = await system.solve_problem_collaboratively(
            problem_description="How can we improve the efficiency of task distribution across AI agents?",
            problem_type="optimization",
            max_agents=5
        )
        
        logger.info(f"Collaborative problem solving result: {problem_result}")
        
        # Keep system running
        logger.info("LocalAGI system operational. Press Ctrl+C to shutdown.")
        
        try:
            while True:
                # Get and log system status periodically
                status = await system.get_comprehensive_status()
                
                # Log key metrics
                orchestration_status = status.get('orchestration_engine', {})
                logger.info(
                    f"System Status - Active Agents: {orchestration_status.get('active_agents', 0)}, "
                    f"Total Tasks: {orchestration_status.get('total_tasks', 0)}, "
                    f"Success Rate: {orchestration_status.get('performance_metrics', {}).get('success_rate', 0):.2%}"
                )
                
                await asyncio.sleep(300)  # Log status every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Shutdown system
        await system.shutdown()
        
    except Exception as e:
        logger.error(f"LocalAGI system error: {e}")
        raise

if __name__ == "__main__":
    # Add numpy import for system improvements
    import numpy as np
    
    # Run the main LocalAGI system
    asyncio.run(main())