#!/usr/bin/env python3
"""
SutazAI Self-Improving Workflow Engine

This engine creates workflows that learn from execution results,
adapt their behavior based on performance feedback, and continuously
optimize themselves for better outcomes. It implements machine learning
approaches to workflow optimization and autonomous system evolution.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
from collections import defaultdict, deque
import pickle
import math
import random

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    LEARNING = "learning"
    ADAPTING = "adapting"
    COMPLETED = "completed"
    FAILED = "failed"
    OPTIMIZING = "optimizing"

class LearningMode(Enum):
    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META = "meta"

class OptimizationObjective(Enum):
    SPEED = "speed"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    COST = "cost"
    QUALITY = "quality"
    RELIABILITY = "reliability"
    MULTI_OBJECTIVE = "multi_objective"

@dataclass
class WorkflowStep:
    id: str
    name: str
    description: str
    agent_capability: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    postconditions: List[str]
    success_criteria: List[str]
    timeout: float
    retry_count: int
    importance_weight: float
    adaptable_parameters: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    start_time: datetime
    end_time: Optional[datetime]
    steps_executed: List[str]
    step_results: Dict[str, Any]
    overall_success: bool
    performance_metrics: Dict[str, float]
    learned_patterns: Dict[str, Any]
    optimization_opportunities: List[str]
    context_data: Dict[str, Any]

@dataclass
class PerformanceFeedback:
    execution_id: str
    step_id: str
    metric_name: str
    metric_value: float
    target_value: Optional[float]
    improvement_suggestion: Optional[str]
    confidence: float
    timestamp: datetime

class SelfImprovingWorkflow:
    """
    A workflow that learns and adapts from its execution history.
    """
    
    def __init__(self, workflow_id: str, description: str, optimization_objectives: List[OptimizationObjective]):
        self.workflow_id = workflow_id
        self.description = description
        self.optimization_objectives = optimization_objectives
        
        # Workflow structure
        self.steps: Dict[str, WorkflowStep] = {}
        self.step_dependencies: Dict[str, List[str]] = {}
        self.execution_order: List[List[str]] = []  # Stages of parallel execution
        
        # Learning and adaptation
        self.execution_history: List[WorkflowExecution] = []
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.learned_optimizations: Dict[str, Any] = {}
        self.adaptation_rules: List[Callable] = []
        
        # Machine learning models
        self.performance_predictors: Dict[str, Any] = {}
        self.parameter_optimizers: Dict[str, Any] = {}
        self.pattern_detectors: Dict[str, Any] = {}
        
        # State
        self.state = WorkflowState.INITIALIZING
        self.version = 1
        self.adaptation_count = 0
        self.last_optimization = datetime.now()
        
        # Configuration
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05  # Minimum improvement to trigger adaptation
        self.min_executions_for_learning = 10
        self.max_history_size = 1000
        
        logger.info(f"Self-improving workflow {workflow_id} initialized")
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps[step.id] = step
        self.step_dependencies[step.id] = step.preconditions.copy()
        self._recalculate_execution_order()
    
    def _recalculate_execution_order(self) -> None:
        """Recalculate the optimal execution order based on dependencies."""
        # Topological sort with parallelization
        execution_order = []
        completed_steps = set()
        remaining_steps = set(self.steps.keys())
        
        while remaining_steps:
            # Find steps with no unfulfilled dependencies
            ready_steps = []
            for step_id in remaining_steps:
                dependencies = set(self.step_dependencies[step_id])
                if dependencies.issubset(completed_steps):
                    ready_steps.append(step_id)
            
            if not ready_steps:
                # Circular dependency - break it
                ready_steps = [next(iter(remaining_steps))]
                logger.warning(f"Circular dependency detected in workflow {self.workflow_id}")
            
            execution_order.append(ready_steps)
            completed_steps.update(ready_steps)
            remaining_steps -= set(ready_steps)
        
        self.execution_order = execution_order
    
    async def execute(self, orchestration_engine, context: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute the workflow and collect performance data."""
        execution_id = str(uuid.uuid4())
        context = context or {}
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=self.workflow_id,
            start_time=datetime.now(),
            end_time=None,
            steps_executed=[],
            step_results={},
            overall_success=True,
            performance_metrics={},
            learned_patterns={},
            optimization_opportunities=[],
            context_data=context
        )
        
        self.state = WorkflowState.EXECUTING
        logger.info(f"Executing workflow {self.workflow_id} (execution {execution_id})")
        
        try:
            # Execute steps in order
            for stage in self.execution_order:
                stage_results = await self._execute_stage(stage, orchestration_engine, execution, context)
                
                # Check if any step failed
                if not all(result.get('success', False) for result in stage_results.values()):
                    execution.overall_success = False
                    failed_steps = [step_id for step_id, result in stage_results.items() 
                                  if not result.get('success', False)]
                    logger.warning(f"Workflow stage failed: {failed_steps}")
                    
                    # Decide whether to continue or abort
                    critical_failed = any(self.steps[step_id].importance_weight > 0.8 
                                        for step_id in failed_steps)
                    if critical_failed:
                        break
            
            execution.end_time = datetime.now()
            execution.performance_metrics = self._calculate_performance_metrics(execution)
            
            # Store execution history
            self.execution_history.append(execution)
            if len(self.execution_history) > self.max_history_size:
                self.execution_history = self.execution_history[-self.max_history_size:]
            
            # Trigger learning and adaptation
            await self._process_execution_feedback(execution)
            
            self.state = WorkflowState.COMPLETED if execution.overall_success else WorkflowState.FAILED
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.overall_success = False
            execution.end_time = datetime.now()
            self.state = WorkflowState.FAILED
        
        return execution
    
    async def _execute_stage(self, 
                           stage_steps: List[str], 
                           orchestration_engine, 
                           execution: WorkflowExecution,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a stage of parallel steps."""
        stage_results = {}
        
        # Execute steps in parallel
        tasks = []
        for step_id in stage_steps:
            task = self._execute_step(step_id, orchestration_engine, execution, context)
            tasks.append(task)
        
        # Wait for all steps to complete
        step_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, step_id in enumerate(stage_steps):
            result = step_results[i]
            if isinstance(result, Exception):
                logger.error(f"Step {step_id} failed: {result}")
                stage_results[step_id] = {
                    'success': False,
                    'error': str(result),
                    'execution_time': 0,
                    'metrics': {}
                }
            else:
                stage_results[step_id] = result
                execution.step_results[step_id] = result
                execution.steps_executed.append(step_id)
        
        return stage_results
    
    async def _execute_step(self, 
                          step_id: str, 
                          orchestration_engine, 
                          execution: WorkflowExecution,
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step = self.steps[step_id]
        start_time = datetime.now()
        
        try:
            # Create task for orchestration engine
            task = orchestration_engine.Task(
                id=str(uuid.uuid4()),
                description=f"{step.name}: {step.description}",
                requirements=[step.agent_capability],
                priority=step.importance_weight,
                complexity=0.5,  # Will be analyzed by orchestration engine
                estimated_duration=step.timeout / 3600,  # Convert to hours
                created_at=start_time,
                metadata={
                    'workflow_id': self.workflow_id,
                    'execution_id': execution.execution_id,
                    'step_id': step_id,
                    'parameters': step.parameters
                }
            )
            
            # Execute through orchestration engine
            result = await orchestration_engine.execute_task_autonomously(task)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract performance metrics
            metrics = {
                'execution_time': execution_time,
                'success': result.get('status') == 'success',
                'agent_used': result.get('agent'),
                'complexity_actual': result.get('complexity', 0.5)
            }
            
            # Store performance history for the step
            step.performance_history.append({
                'execution_id': execution.execution_id,
                'timestamp': start_time,
                'metrics': metrics,
                'parameters': step.parameters.copy(),
                'context': context.copy()
            })
            
            # Limit history size
            if len(step.performance_history) > 100:
                step.performance_history = step.performance_history[-100:]
            
            return {
                'success': metrics['success'],
                'result': result,
                'execution_time': execution_time,
                'metrics': metrics
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Step {step_id} execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'metrics': {'execution_time': execution_time, 'success': False}
            }
    
    def _calculate_performance_metrics(self, execution: WorkflowExecution) -> Dict[str, float]:
        """Calculate overall performance metrics for the execution."""
        metrics = {}
        
        # Time metrics
        if execution.end_time and execution.start_time:
            total_time = (execution.end_time - execution.start_time).total_seconds()
            metrics['total_execution_time'] = total_time
            
            # Calculate step time efficiency
            step_times = [result.get('execution_time', 0) for result in execution.step_results.values()]
            if step_times:
                metrics['average_step_time'] = np.mean(step_times)
                metrics['max_step_time'] = max(step_times)
                metrics['step_time_variance'] = np.var(step_times)
        
        # Success metrics
        successful_steps = sum(1 for result in execution.step_results.values() 
                             if result.get('success', False))
        total_steps = len(execution.step_results)
        if total_steps > 0:
            metrics['success_rate'] = successful_steps / total_steps
        
        # Quality metrics (based on optimization objectives)
        for objective in self.optimization_objectives:
            if objective == OptimizationObjective.SPEED:
                metrics['speed_score'] = 1.0 / max(1.0, metrics.get('total_execution_time', 1.0))
            elif objective == OptimizationObjective.EFFICIENCY:
                if 'success_rate' in metrics and 'total_execution_time' in metrics:
                    metrics['efficiency_score'] = metrics['success_rate'] / max(1.0, metrics['total_execution_time'])
            elif objective == OptimizationObjective.RELIABILITY:
                metrics['reliability_score'] = metrics.get('success_rate', 0.0)
        
        # Composite score
        objective_scores = []
        for objective in self.optimization_objectives:
            score_key = f"{objective.value}_score"
            if score_key in metrics:
                objective_scores.append(metrics[score_key])
        
        if objective_scores:
            metrics['composite_score'] = np.mean(objective_scores)
        
        return metrics
    
    async def _process_execution_feedback(self, execution: WorkflowExecution) -> None:
        """Process execution results for learning and adaptation."""
        self.state = WorkflowState.LEARNING
        
        # Update performance trends
        for metric_name, metric_value in execution.performance_metrics.items():
            self.performance_trends[metric_name].append(metric_value)
        
        # Detect patterns in performance
        await self._detect_performance_patterns(execution)
        
        # Learn parameter optimizations
        await self._learn_parameter_optimizations(execution)
        
        # Check if adaptation is needed
        if await self._should_adapt():
            await self._adapt_workflow()
        
        self.state = WorkflowState.READY
    
    async def _detect_performance_patterns(self, execution: WorkflowExecution) -> None:
        """Detect patterns in performance data."""
        if len(self.execution_history) < 5:
            return  # Need more data
        
        patterns = {}
        
        # Trend analysis
        for metric_name, values in self.performance_trends.items():
            if len(values) >= 5:
                recent_values = list(values)[-5:]
                older_values = list(values)[-10:-5] if len(values) >= 10 else []
                
                if older_values:
                    recent_avg = np.mean(recent_values)
                    older_avg = np.mean(older_values)
                    trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                    
                    patterns[f"{metric_name}_trend"] = trend
                    
                    if abs(trend) > 0.1:  # Significant trend
                        direction = "improving" if trend > 0 else "degrading"
                        patterns[f"{metric_name}_trend_significant"] = direction
        
        # Context correlation analysis
        context_factors = []
        performance_values = []
        
        for past_execution in self.execution_history[-20:]:  # Last 20 executions
            if 'composite_score' in past_execution.performance_metrics:
                context_factors.append(list(past_execution.context_data.values()))
                performance_values.append(past_execution.performance_metrics['composite_score'])
        
        if len(context_factors) > 5 and all(len(cf) == len(context_factors[0]) for cf in context_factors):
            # Simple correlation analysis
            try:
                context_array = np.array(context_factors)
                performance_array = np.array(performance_values)
                
                for i in range(context_array.shape[1]):
                    correlation = np.corrcoef(context_array[:, i], performance_array)[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > 0.3:
                        context_key = list(execution.context_data.keys())[i] if i < len(execution.context_data) else f"context_{i}"
                        patterns[f"context_correlation_{context_key}"] = correlation
            except Exception as e:
                logger.warning(f"Context correlation analysis failed: {e}")
        
        execution.learned_patterns = patterns
        
        # Store significant patterns
        for pattern_name, pattern_value in patterns.items():
            if abs(pattern_value) > 0.2:  # Significant pattern
                self.pattern_detectors[pattern_name] = pattern_value
    
    async def _learn_parameter_optimizations(self, execution: WorkflowExecution) -> None:
        """Learn optimal parameters for workflow steps."""
        if len(self.execution_history) < self.min_executions_for_learning:
            return
        
        # For each step with adaptable parameters
        for step_id, step in self.steps.items():
            if not step.adaptable_parameters or len(step.performance_history) < 5:
                continue
            
            # Analyze parameter-performance relationships
            parameter_performance = defaultdict(list)
            
            for history_entry in step.performance_history[-50:]:  # Last 50 executions
                performance_score = history_entry['metrics'].get('success', 0) * 0.5
                if 'execution_time' in history_entry['metrics']:
                    # Lower execution time is better (inverted)
                    time_score = 1.0 / max(1.0, history_entry['metrics']['execution_time'])
                    performance_score += time_score * 0.5
                
                for param_name in step.adaptable_parameters:
                    if param_name in history_entry['parameters']:
                        param_value = history_entry['parameters'][param_name]
                        parameter_performance[param_name].append((param_value, performance_score))
            
            # Find optimal parameter values
            for param_name, value_performance_pairs in parameter_performance.items():
                if len(value_performance_pairs) >= 5:
                    # Simple optimization: find value with best average performance
                    value_groups = defaultdict(list)
                    for value, performance in value_performance_pairs:
                        value_groups[value].append(performance)
                    
                    best_value = None
                    best_avg_performance = -1
                    
                    for value, performances in value_groups.items():
                        avg_performance = np.mean(performances)
                        if avg_performance > best_avg_performance:
                            best_avg_performance = avg_performance
                            best_value = value
                    
                    if best_value is not None:
                        optimization_key = f"{step_id}_{param_name}"
                        self.learned_optimizations[optimization_key] = {
                            'parameter': param_name,
                            'optimal_value': best_value,
                            'performance_improvement': best_avg_performance,
                            'confidence': min(1.0, len(value_performance_pairs) / 20),
                            'learned_at': datetime.now()
                        }
    
    async def _should_adapt(self) -> bool:
        """Determine if the workflow should adapt based on performance trends."""
        if len(self.execution_history) < self.min_executions_for_learning:
            return False
        
        # Check if enough time has passed since last optimization
        time_since_optimization = datetime.now() - self.last_optimization
        if time_since_optimization < timedelta(hours=1):
            return False
        
        # Check performance trends
        if 'composite_score' in self.performance_trends:
            recent_scores = list(self.performance_trends['composite_score'])[-10:]
            if len(recent_scores) >= 5:
                recent_avg = np.mean(recent_scores[-5:])
                older_avg = np.mean(recent_scores[:5]) if len(recent_scores) >= 10 else recent_avg
                
                # Adapt if performance is declining
                if recent_avg < older_avg - self.adaptation_threshold:
                    logger.info(f"Performance declining: {recent_avg:.3f} vs {older_avg:.3f}")
                    return True
        
        # Check if we have learned significant optimizations
        confident_optimizations = [
            opt for opt in self.learned_optimizations.values()
            if opt['confidence'] > 0.7 and opt['performance_improvement'] > 0.1
        ]
        
        if len(confident_optimizations) > 0:
            logger.info(f"Found {len(confident_optimizations)} confident optimizations")
            return True
        
        return False
    
    async def _adapt_workflow(self) -> None:
        """Adapt the workflow based on learned optimizations."""
        self.state = WorkflowState.ADAPTING
        logger.info(f"Adapting workflow {self.workflow_id}")
        
        adaptations_made = 0
        
        # Apply parameter optimizations
        for optimization_key, optimization in self.learned_optimizations.items():
            if optimization['confidence'] > 0.7:
                step_id, param_name = optimization_key.rsplit('_', 1)
                
                if step_id in self.steps and param_name in self.steps[step_id].adaptable_parameters:
                    old_value = self.steps[step_id].parameters.get(param_name)
                    new_value = optimization['optimal_value']
                    
                    if old_value != new_value:
                        self.steps[step_id].parameters[param_name] = new_value
                        self.steps[step_id].optimization_hints[f"adapted_{param_name}"] = {
                            'old_value': old_value,
                            'new_value': new_value,
                            'expected_improvement': optimization['performance_improvement'],
                            'adapted_at': datetime.now()
                        }
                        
                        adaptations_made += 1
                        logger.info(f"Adapted {step_id}.{param_name}: {old_value} -> {new_value}")
        
        # Structural adaptations based on patterns
        await self._apply_structural_adaptations()
        
        # Update workflow version and counters
        if adaptations_made > 0:
            self.version += 1
            self.adaptation_count += 1
            self.last_optimization = datetime.now()
            
            # Clear learned optimizations that were applied
            applied_optimizations = [
                key for key, opt in self.learned_optimizations.items()
                if opt['confidence'] > 0.7
            ]
            for key in applied_optimizations:
                del self.learned_optimizations[key]
        
        self.state = WorkflowState.READY
        logger.info(f"Workflow adaptation completed: {adaptations_made} changes made")
    
    async def _apply_structural_adaptations(self) -> None:
        """Apply structural changes to the workflow based on learned patterns."""
        # Reorder steps based on performance patterns
        if len(self.execution_history) >= 20:
            # Analyze step execution patterns
            step_performance = defaultdict(list)
            
            for execution in self.execution_history[-20:]:
                for step_id, result in execution.step_results.items():
                    if result.get('success', False):
                        execution_time = result.get('execution_time', 0)
                        step_performance[step_id].append(execution_time)
            
            # Identify consistently slow steps
            slow_steps = []
            for step_id, times in step_performance.items():
                if len(times) >= 5:
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    if avg_time > 10 and std_time / avg_time < 0.5:  # Consistently slow
                        slow_steps.append((step_id, avg_time))
            
            # Consider moving slow steps earlier to start them sooner
            if slow_steps:
                slow_steps.sort(key=lambda x: x[1], reverse=True)  # Slowest first
                
                for step_id, _ in slow_steps:
                    # Try to reduce dependencies to allow earlier execution
                    current_deps = self.step_dependencies[step_id]
                    essential_deps = await self._identify_essential_dependencies(step_id)
                    
                    if len(essential_deps) < len(current_deps):
                        self.step_dependencies[step_id] = essential_deps
                        self._recalculate_execution_order()
                        logger.info(f"Reduced dependencies for slow step {step_id}")
    
    async def _identify_essential_dependencies(self, step_id: str) -> List[str]:
        """Identify which dependencies are truly essential for a step."""
        step = self.steps[step_id]
        current_deps = self.step_dependencies[step_id]
        
        # For now, use simple heuristics - in practice would use more sophisticated analysis
        essential_deps = []
        
        for dep_id in current_deps:
            if dep_id in self.steps:
                dep_step = self.steps[dep_id]
                
                # Consider dependency essential if:
                # 1. It produces data this step needs
                # 2. It has high importance weight
                # 3. Historical data shows failures when dependency is missing
                
                if (dep_step.importance_weight > 0.7 or
                    any(keyword in dep_step.name.lower() for keyword in ['create', 'setup', 'initialize']) or
                    any(keyword in step.name.lower() for keyword in ['use', 'process', 'analyze'])):
                    essential_deps.append(dep_id)
        
        return essential_deps
    
    def add_performance_feedback(self, feedback: PerformanceFeedback) -> None:
        """Add external performance feedback for learning."""
        execution = next((e for e in self.execution_history if e.execution_id == feedback.execution_id), None)
        if not execution:
            logger.warning(f"Execution {feedback.execution_id} not found for feedback")
            return
        
        # Store feedback for the specific step
        if feedback.step_id in self.steps:
            step = self.steps[feedback.step_id]
            
            # Find the corresponding performance history entry
            for history_entry in step.performance_history:
                if history_entry['execution_id'] == feedback.execution_id:
                    if 'external_feedback' not in history_entry:
                        history_entry['external_feedback'] = []
                    
                    history_entry['external_feedback'].append({
                        'metric_name': feedback.metric_name,
                        'metric_value': feedback.metric_value,
                        'target_value': feedback.target_value,
                        'improvement_suggestion': feedback.improvement_suggestion,
                        'confidence': feedback.confidence,
                        'timestamp': feedback.timestamp
                    })
                    break
        
        # Update learning based on feedback
        if feedback.target_value and feedback.metric_value:
            performance_gap = abs(feedback.metric_value - feedback.target_value) / feedback.target_value
            if performance_gap > 0.1:  # Significant gap
                # Create or update optimization hint
                optimization_key = f"{feedback.step_id}_{feedback.metric_name}"
                if optimization_key not in self.learned_optimizations:
                    self.learned_optimizations[optimization_key] = {
                        'parameter': feedback.metric_name,
                        'target_value': feedback.target_value,
                        'current_gap': performance_gap,
                        'confidence': feedback.confidence,
                        'improvement_suggestion': feedback.improvement_suggestion,
                        'learned_at': datetime.now()
                    }
        
        logger.info(f"Added performance feedback for execution {feedback.execution_id}")
    
    async def predict_performance(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """Predict workflow performance for given context."""
        if len(self.execution_history) < 5:
            return {'confidence': 0.0}
        
        predictions = {}
        context = context or {}
        
        # Simple prediction based on historical averages with context similarity
        similar_executions = []
        
        for execution in self.execution_history[-50:]:  # Last 50 executions
            similarity = self._calculate_context_similarity(context, execution.context_data)
            if similarity > 0.5:  # Similar enough
                similar_executions.append((execution, similarity))
        
        if not similar_executions:
            # Fall back to general historical average
            similar_executions = [(execution, 1.0) for execution in self.execution_history[-20:]]
        
        # Weighted average based on similarity
        for metric_name in ['total_execution_time', 'success_rate', 'composite_score']:
            values = []
            weights = []
            
            for execution, similarity in similar_executions:
                if metric_name in execution.performance_metrics:
                    values.append(execution.performance_metrics[metric_name])
                    weights.append(similarity)
            
            if values:
                if weights:
                    predictions[metric_name] = np.average(values, weights=weights)
                else:
                    predictions[metric_name] = np.mean(values)
        
        predictions['confidence'] = min(1.0, len(similar_executions) / 10)
        return predictions
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts."""
        if not context1 or not context2:
            return 0.5  # Neutral similarity for empty contexts
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2), 1)
                    similarity = 1.0 - abs(val1 - val2) / max_val
                    similarities.append(max(0, similarity))
            elif isinstance(val1, str) and isinstance(val2, str):
                # String similarity (simple)
                if val1 == val2:
                    similarities.append(1.0)
                else:
                    similarities.append(0.3)  # Partial similarity for different strings
            else:
                # Generic equality
                similarities.append(1.0 if val1 == val2 else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status."""
        recent_executions = self.execution_history[-10:] if self.execution_history else []
        
        return {
            'workflow_id': self.workflow_id,
            'description': self.description,
            'version': self.version,
            'state': self.state.value,
            'adaptation_count': self.adaptation_count,
            'total_executions': len(self.execution_history),
            'steps_count': len(self.steps),
            'optimization_objectives': [obj.value for obj in self.optimization_objectives],
            'learned_optimizations': len(self.learned_optimizations),
            'performance_trends': {
                name: {
                    'current': list(values)[-1] if values else None,
                    'trend': list(values)[-5:] if len(values) >= 5 else list(values),
                    'count': len(values)
                }
                for name, values in self.performance_trends.items()
            },
            'recent_performance': [
                {
                    'execution_id': exec.execution_id,
                    'success': exec.overall_success,
                    'duration': (exec.end_time - exec.start_time).total_seconds() if exec.end_time else None,
                    'composite_score': exec.performance_metrics.get('composite_score')
                }
                for exec in recent_executions
            ],
            'adaptation_hints': len(self.learned_optimizations),
            'last_optimization': self.last_optimization.isoformat()
        }

class SelfImprovingWorkflowEngine:
    """
    Engine that manages multiple self-improving workflows.
    """
    
    def __init__(self, orchestration_engine):
        self.orchestration_engine = orchestration_engine
        self.workflows: Dict[str, SelfImprovingWorkflow] = {}
        self.global_learning_patterns: Dict[str, Any] = {}
        self.cross_workflow_optimizations: Dict[str, Any] = {}
        
        logger.info("Self-Improving Workflow Engine initialized")
    
    def create_workflow(self, 
                       description: str, 
                       optimization_objectives: List[OptimizationObjective],
                       steps: List[WorkflowStep] = None) -> str:
        """Create a new self-improving workflow."""
        workflow_id = str(uuid.uuid4())
        
        workflow = SelfImprovingWorkflow(workflow_id, description, optimization_objectives)
        
        if steps:
            for step in steps:
                workflow.add_step(step)
        
        workflow.state = WorkflowState.READY
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id}: {description}")
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> WorkflowExecution:
        """Execute a workflow and return the execution result."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution = await workflow.execute(self.orchestration_engine, context)
        
        # Cross-workflow learning
        await self._cross_workflow_learning(workflow, execution)
        
        return execution
    
    async def _cross_workflow_learning(self, workflow: SelfImprovingWorkflow, execution: WorkflowExecution):
        """Learn patterns that apply across multiple workflows."""
        # Identify patterns that might benefit other workflows
        for pattern_name, pattern_value in execution.learned_patterns.items():
            if abs(pattern_value) > 0.3:  # Significant pattern
                if pattern_name not in self.global_learning_patterns:
                    self.global_learning_patterns[pattern_name] = []
                
                self.global_learning_patterns[pattern_name].append({
                    'workflow_id': workflow.workflow_id,
                    'pattern_value': pattern_value,
                    'context': execution.context_data,
                    'performance': execution.performance_metrics.get('composite_score', 0),
                    'timestamp': execution.start_time
                })
        
        # Apply global patterns to other workflows if applicable
        for other_workflow_id, other_workflow in self.workflows.items():
            if other_workflow_id != workflow.workflow_id:
                await self._apply_global_patterns(other_workflow, self.global_learning_patterns)
    
    async def _apply_global_patterns(self, workflow: SelfImprovingWorkflow, global_patterns: Dict[str, Any]):
        """Apply global learning patterns to a workflow."""
        for pattern_name, pattern_instances in global_patterns.items():
            if len(pattern_instances) >= 3:  # Enough evidence
                # Check if pattern is consistently beneficial
                beneficial_instances = [p for p in pattern_instances if p['performance'] > 0.7]
                
                if len(beneficial_instances) / len(pattern_instances) > 0.6:  # Majority beneficial
                    # Try to apply pattern to this workflow
                    if 'context_correlation' in pattern_name:
                        # Context-based optimizations
                        context_key = pattern_name.split('_')[-1]
                        avg_correlation = np.mean([p['pattern_value'] for p in beneficial_instances])
                        
                        workflow.pattern_detectors[pattern_name] = avg_correlation
                        logger.info(f"Applied global pattern {pattern_name} to workflow {workflow.workflow_id}")
    
    def add_workflow_feedback(self, workflow_id: str, feedback: PerformanceFeedback):
        """Add performance feedback to a specific workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].add_performance_feedback(feedback)
    
    async def optimize_all_workflows(self):
        """Run optimization across all workflows."""
        logger.info("Running global workflow optimization")
        
        for workflow in self.workflows.values():
            if workflow.state == WorkflowState.READY and len(workflow.execution_history) >= 5:
                try:
                    if await workflow._should_adapt():
                        await workflow._adapt_workflow()
                except Exception as e:
                    logger.error(f"Workflow optimization failed for {workflow.workflow_id}: {e}")
        
        # Global cross-workflow optimizations
        await self._perform_cross_workflow_optimizations()
    
    async def _perform_cross_workflow_optimizations(self):
        """Perform optimizations that benefit multiple workflows."""
        # Find common optimization patterns
        common_patterns = defaultdict(list)
        
        for workflow in self.workflows.values():
            for opt_key, optimization in workflow.learned_optimizations.items():
                pattern_key = opt_key.split('_')[-1]  # Parameter name
                common_patterns[pattern_key].append(optimization)
        
        # Apply common optimizations to workflows that haven't learned them yet
        for pattern_key, optimizations in common_patterns.items():
            if len(optimizations) >= 2:  # Found in multiple workflows
                avg_improvement = np.mean([opt['performance_improvement'] for opt in optimizations])
                avg_confidence = np.mean([opt['confidence'] for opt in optimizations])
                
                if avg_improvement > 0.1 and avg_confidence > 0.6:
                    # This is a generally beneficial optimization
                    self.cross_workflow_optimizations[pattern_key] = {
                        'average_improvement': avg_improvement,
                        'confidence': avg_confidence,
                        'discovered_at': datetime.now(),
                        'source_workflows': len(optimizations)
                    }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        total_executions = sum(len(w.execution_history) for w in self.workflows.values())
        total_adaptations = sum(w.adaptation_count for w in self.workflows.values())
        
        return {
            'total_workflows': len(self.workflows),
            'total_executions': total_executions,
            'total_adaptations': total_adaptations,
            'global_patterns': len(self.global_learning_patterns),
            'cross_workflow_optimizations': len(self.cross_workflow_optimizations),
            'workflows': {
                wf_id: workflow.get_workflow_status()
                for wf_id, workflow in self.workflows.items()
            }
        }