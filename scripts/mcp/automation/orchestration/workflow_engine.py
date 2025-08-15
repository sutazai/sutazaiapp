#!/usr/bin/env python3
"""
MCP Workflow Engine

Advanced workflow definition and execution engine with DAG support,
parallel execution, conditional logic, and comprehensive error handling.
Enables complex multi-step automation workflows for MCP operations.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 11:52:00 UTC
Version: 1.0.0
"""

import asyncio
import uuid
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class StepType(Enum):
    """Workflow step types."""
    ACTION = "action"          # Execute an action
    CONDITION = "condition"    # Conditional branch
    PARALLEL = "parallel"      # Parallel execution
    LOOP = "loop"             # Loop execution
    WAIT = "wait"             # Wait/delay
    NOTIFICATION = "notification"  # Send notification
    VALIDATION = "validation"  # Validate state
    ROLLBACK = "rollback"     # Rollback action


@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    id: str
    name: str
    type: StepType
    action: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    retry_policy: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    on_failure: Optional[str] = None  # Step to execute on failure
    on_success: Optional[str] = None  # Step to execute on success
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    version: str
    description: str
    steps: List[WorkflowStep]
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    max_retries: int = 3
    rollback_on_failure: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate workflow definition."""
        errors = []
        
        # Check for duplicate step IDs
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
            
        # Check dependencies exist
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.id} has invalid dependency: {dep}")
                    
        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")
            
        return errors
        
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            step = next((s for s in self.steps if s.id == step_id), None)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
                        
            rec_stack.remove(step_id)
            return False
            
        for step in self.steps:
            if step.id not in visited:
                if has_cycle(step.id):
                    return True
                    
        return False


@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    context: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    current_step: Optional[str] = None
    completed_steps: Set[str] = field(default_factory=set)
    failed_steps: Set[str] = field(default_factory=set)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()


class WorkflowEngine:
    """
    Advanced workflow execution engine.
    
    Manages workflow definitions, executes workflows with DAG support,
    handles parallel execution, and provides comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize workflow engine."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # Workflow storage
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Action handlers
        self.action_handlers: Dict[str, Callable] = {}
        self.condition_evaluators: Dict[str, Callable] = {}
        
        # Event handler
        self.event_handler: Optional[Callable] = None
        
        # Execution control
        self._execution_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Load default workflows
        self._load_default_workflows()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.workflow_engine")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _load_default_workflows(self) -> None:
        """Load default workflow definitions."""
        # MCP Update Check Workflow
        self.workflows["mcp-update-check"] = WorkflowDefinition(
            id="mcp-update-check",
            name="MCP Update Check",
            version="1.0.0",
            description="Check and apply MCP server updates",
            steps=[
                WorkflowStep(
                    id="check-updates",
                    name="Check for Updates",
                    type=StepType.ACTION,
                    action="check_mcp_updates",
                    timeout=60
                ),
                WorkflowStep(
                    id="validate-updates",
                    name="Validate Updates",
                    type=StepType.VALIDATION,
                    action="validate_updates",
                    dependencies=["check-updates"],
                    conditions=[{"updates_available": True}]
                ),
                WorkflowStep(
                    id="backup-current",
                    name="Backup Current Version",
                    type=StepType.ACTION,
                    action="backup_mcp_servers",
                    dependencies=["validate-updates"]
                ),
                WorkflowStep(
                    id="apply-updates",
                    name="Apply Updates",
                    type=StepType.ACTION,
                    action="apply_mcp_updates",
                    dependencies=["backup-current"],
                    retry_policy={"max_attempts": 3, "delay": 10}
                ),
                WorkflowStep(
                    id="verify-updates",
                    name="Verify Updates",
                    type=StepType.VALIDATION,
                    action="verify_mcp_updates",
                    dependencies=["apply-updates"]
                ),
                WorkflowStep(
                    id="notify-completion",
                    name="Notify Completion",
                    type=StepType.NOTIFICATION,
                    action="send_notification",
                    dependencies=["verify-updates"],
                    parameters={"type": "update_complete"}
                )
            ],
            rollback_on_failure=True,
            timeout=600
        )
        
        # MCP Cleanup Workflow
        self.workflows["mcp-cleanup"] = WorkflowDefinition(
            id="mcp-cleanup",
            name="MCP Cleanup",
            version="1.0.0",
            description="Clean up MCP artifacts and old versions",
            steps=[
                WorkflowStep(
                    id="analyze-artifacts",
                    name="Analyze Artifacts",
                    type=StepType.ACTION,
                    action="analyze_mcp_artifacts"
                ),
                WorkflowStep(
                    id="validate-cleanup",
                    name="Validate Cleanup Safety",
                    type=StepType.VALIDATION,
                    action="validate_cleanup_safety",
                    dependencies=["analyze-artifacts"]
                ),
                WorkflowStep(
                    id="cleanup-versions",
                    name="Cleanup Old Versions",
                    type=StepType.ACTION,
                    action="cleanup_old_versions",
                    dependencies=["validate-cleanup"],
                    parameters={"dry_run": False}
                ),
                WorkflowStep(
                    id="cleanup-artifacts",
                    name="Cleanup Artifacts",
                    type=StepType.ACTION,
                    action="cleanup_artifacts",
                    dependencies=["cleanup-versions"]
                ),
                WorkflowStep(
                    id="audit-cleanup",
                    name="Audit Cleanup",
                    type=StepType.ACTION,
                    action="audit_cleanup_operation",
                    dependencies=["cleanup-artifacts"]
                )
            ],
            timeout=300
        )
        
        # Service Recovery Workflow
        self.workflows["service-recovery"] = WorkflowDefinition(
            id="service-recovery",
            name="Service Recovery",
            version="1.0.0",
            description="Recover failed MCP services",
            steps=[
                WorkflowStep(
                    id="identify-failure",
                    name="Identify Failure",
                    type=StepType.ACTION,
                    action="identify_service_failure"
                ),
                WorkflowStep(
                    id="attempt-restart",
                    name="Attempt Restart",
                    type=StepType.ACTION,
                    action="restart_service",
                    dependencies=["identify-failure"],
                    retry_policy={"max_attempts": 3, "delay": 5}
                ),
                WorkflowStep(
                    id="verify-recovery",
                    name="Verify Recovery",
                    type=StepType.VALIDATION,
                    action="verify_service_health",
                    dependencies=["attempt-restart"]
                ),
                WorkflowStep(
                    id="escalate-failure",
                    name="Escalate if Failed",
                    type=StepType.NOTIFICATION,
                    action="escalate_failure",
                    dependencies=["verify-recovery"],
                    conditions=[{"recovery_failed": True}]
                )
            ],
            timeout=180
        )
        
    async def initialize(self) -> None:
        """Initialize workflow engine."""
        self.logger.info("Initializing workflow engine...")
        
        # Register default action handlers
        self._register_default_handlers()
        
        # Load custom workflows from configuration
        await self._load_custom_workflows()
        
        self.logger.info(f"Loaded {len(self.workflows)} workflows")
        
    async def _load_custom_workflows(self) -> None:
        """Load custom workflow definitions from files."""
        workflows_dir = Path("/opt/sutazaiapp/scripts/mcp/automation/workflows")
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob("*.json"):
                try:
                    with open(workflow_file) as f:
                        workflow_data = json.load(f)
                        workflow = self._parse_workflow(workflow_data)
                        self.workflows[workflow.id] = workflow
                        self.logger.info(f"Loaded workflow: {workflow.name}")
                except Exception as e:
                    self.logger.error(f"Failed to load workflow {workflow_file}: {e}")
                    
    def _parse_workflow(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow definition from dictionary."""
        steps = []
        for step_data in data.get("steps", []):
            step = WorkflowStep(
                id=step_data["id"],
                name=step_data["name"],
                type=StepType[step_data["type"].upper()],
                action=step_data.get("action"),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                conditions=step_data.get("conditions", []),
                retry_policy=step_data.get("retry_policy"),
                timeout=step_data.get("timeout"),
                on_failure=step_data.get("on_failure"),
                on_success=step_data.get("on_success"),
                metadata=step_data.get("metadata", {})
            )
            steps.append(step)
            
        return WorkflowDefinition(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            steps=steps,
            triggers=data.get("triggers", []),
            parameters=data.get("parameters", {}),
            timeout=data.get("timeout"),
            max_retries=data.get("max_retries", 3),
            rollback_on_failure=data.get("rollback_on_failure", True),
            metadata=data.get("metadata", {})
        )
        
    def _register_default_handlers(self) -> None:
        """Register default action handlers."""
        # These would be replaced with actual implementations
        self.register_action("check_mcp_updates", self._default_action_handler)
        self.register_action("validate_updates", self._default_action_handler)
        self.register_action("backup_mcp_servers", self._default_action_handler)
        self.register_action("apply_mcp_updates", self._default_action_handler)
        self.register_action("verify_mcp_updates", self._default_action_handler)
        self.register_action("send_notification", self._default_action_handler)
        self.register_action("analyze_mcp_artifacts", self._default_action_handler)
        self.register_action("validate_cleanup_safety", self._default_action_handler)
        self.register_action("cleanup_old_versions", self._default_action_handler)
        self.register_action("cleanup_artifacts", self._default_action_handler)
        self.register_action("audit_cleanup_operation", self._default_action_handler)
        self.register_action("identify_service_failure", self._default_action_handler)
        self.register_action("restart_service", self._default_action_handler)
        self.register_action("verify_service_health", self._default_action_handler)
        self.register_action("escalate_failure", self._default_action_handler)
        
    async def _default_action_handler(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default action handler for testing."""
        self.logger.debug(f"Executing action with context: {context}")
        return {"status": "success", "message": "Action completed"}
        
    def register_action(self, name: str, handler: Callable) -> None:
        """Register an action handler."""
        self.action_handlers[name] = handler
        self.logger.debug(f"Registered action handler: {name}")
        
    def register_condition(self, name: str, evaluator: Callable) -> None:
        """Register a condition evaluator."""
        self.condition_evaluators[name] = evaluator
        self.logger.debug(f"Registered condition evaluator: {name}")
        
    def set_event_handler(self, handler: Callable) -> None:
        """Set event handler for workflow events."""
        self.event_handler = handler
        
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID."""
        return self.workflows.get(workflow_id)
        
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows."""
        return [
            {
                "id": w.id,
                "name": w.name,
                "version": w.version,
                "description": w.description,
                "steps": len(w.steps)
            }
            for w in self.workflows.values()
        ]
        
    async def execute(
        self,
        workflow: Union[str, WorkflowDefinition],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a workflow."""
        try:
            # Get workflow definition
            if isinstance(workflow, str):
                workflow_def = self.workflows.get(workflow)
                if not workflow_def:
                    raise ValueError(f"Unknown workflow: {workflow}")
            else:
                workflow_def = workflow
                
            # Validate workflow
            errors = workflow_def.validate()
            if errors:
                raise ValueError(f"Invalid workflow: {errors}")
                
            # Create execution instance
            execution = WorkflowExecution(
                id=str(uuid.uuid4()),
                workflow_id=workflow_def.id,
                workflow_name=workflow_def.name,
                status=WorkflowStatus.RUNNING,
                context=context or {},
                started_at=datetime.now(timezone.utc)
            )
            
            self.executions[execution.id] = execution
            
            # Execute workflow
            self.logger.info(f"Starting workflow execution: {execution.id}")
            
            # Create execution task
            task = asyncio.create_task(
                self._execute_workflow(workflow_def, execution)
            )
            self._execution_tasks[execution.id] = task
            
            # Wait for completion or timeout
            timeout = workflow_def.timeout or 3600
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                execution.status = WorkflowStatus.TIMEOUT
                execution.error = f"Workflow timed out after {timeout} seconds"
                task.cancel()
                
            # Return execution result
            return {
                "execution_id": execution.id,
                "status": execution.status.value,
                "execution_time": execution.execution_time,
                "completed_steps": list(execution.completed_steps),
                "failed_steps": list(execution.failed_steps),
                "results": execution.step_results,
                "error": execution.error
            }
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            raise
            
    async def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow steps."""
        try:
            # Build execution plan
            plan = self._build_execution_plan(workflow)
            
            # Execute steps according to plan
            for step_group in plan:
                # Execute parallel steps
                tasks = []
                for step in step_group:
                    if self._should_execute_step(step, execution):
                        execution.current_step = step.id
                        task = asyncio.create_task(
                            self._execute_step(step, execution)
                        )
                        tasks.append((step, task))
                        
                # Wait for all parallel steps to complete
                for step, task in tasks:
                    try:
                        await task
                        execution.completed_steps.add(step.id)
                    except Exception as e:
                        self.logger.error(f"Step {step.id} failed: {e}")
                        execution.failed_steps.add(step.id)
                        execution.step_results[step.id] = {"error": str(e)}
                        
                        # Handle failure
                        if step.on_failure:
                            failure_step = next(
                                (s for s in workflow.steps if s.id == step.on_failure),
                                None
                            )
                            if failure_step:
                                await self._execute_step(failure_step, execution)
                                
                        # Check if we should continue
                        if workflow.rollback_on_failure:
                            await self._rollback_workflow(workflow, execution)
                            execution.status = WorkflowStatus.FAILED
                            return
                            
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            
            # Publish completion event
            if self.event_handler:
                await self.event_handler({
                    "type": "workflow_completed",
                    "workflow_id": workflow.id,
                    "execution_id": execution.id,
                    "status": execution.status.value,
                    "execution_time": execution.execution_time
                })
                
        except Exception as e:
            self.logger.error(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            
    def _build_execution_plan(
        self,
        workflow: WorkflowDefinition
    ) -> List[List[WorkflowStep]]:
        """Build execution plan with parallel step groups."""
        plan = []
        executed = set()
        
        while len(executed) < len(workflow.steps):
            # Find steps that can be executed
            ready = []
            for step in workflow.steps:
                if step.id not in executed:
                    # Check if dependencies are satisfied
                    if all(dep in executed for dep in step.dependencies):
                        ready.append(step)
                        
            if not ready:
                # No steps ready, might be circular dependency
                break
                
            # Add ready steps to plan
            plan.append(ready)
            executed.update(step.id for step in ready)
            
        return plan
        
    def _should_execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> bool:
        """Check if step should be executed based on conditions."""
        if not step.conditions:
            return True
            
        # Evaluate conditions
        for condition in step.conditions:
            if not self._evaluate_condition(condition, execution):
                self.logger.debug(f"Step {step.id} skipped due to condition")
                return False
                
        return True
        
    def _evaluate_condition(
        self,
        condition: Dict[str, Any],
        execution: WorkflowExecution
    ) -> bool:
        """Evaluate a condition."""
        # Simple condition evaluation
        for key, value in condition.items():
            if key in execution.context:
                if execution.context[key] != value:
                    return False
            elif key in execution.step_results:
                if execution.step_results[key] != value:
                    return False
                    
        return True
        
    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a single workflow step."""
        self.logger.debug(f"Executing step: {step.id}")
        
        try:
            # Handle different step types
            if step.type == StepType.ACTION:
                await self._execute_action_step(step, execution)
            elif step.type == StepType.VALIDATION:
                await self._execute_validation_step(step, execution)
            elif step.type == StepType.PARALLEL:
                await self._execute_parallel_step(step, execution)
            elif step.type == StepType.WAIT:
                await self._execute_wait_step(step, execution)
            elif step.type == StepType.NOTIFICATION:
                await self._execute_notification_step(step, execution)
            elif step.type == StepType.CONDITION:
                await self._execute_condition_step(step, execution)
            elif step.type == StepType.LOOP:
                await self._execute_loop_step(step, execution)
                
            # Handle success action
            if step.on_success:
                success_step = next(
                    (s for s in execution.workflow.steps if s.id == step.on_success),
                    None
                )
                if success_step:
                    await self._execute_step(success_step, execution)
                    
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            
            # Handle retry
            if step.retry_policy:
                retries = step.retry_policy.get("max_attempts", 1)
                delay = step.retry_policy.get("delay", 0)
                
                for attempt in range(retries - 1):
                    await asyncio.sleep(delay)
                    try:
                        await self._execute_step(step, execution)
                        return
                    except:
                        continue
                        
            raise
            
    async def _execute_action_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute an action step."""
        if step.action and step.action in self.action_handlers:
            handler = self.action_handlers[step.action]
            
            # Prepare context
            context = {
                **execution.context,
                **step.parameters,
                "step_id": step.id,
                "execution_id": execution.id
            }
            
            # Execute action
            result = await handler(context)
            execution.step_results[step.id] = result
        else:
            raise ValueError(f"Unknown action: {step.action}")
            
    async def _execute_validation_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a validation step."""
        # Similar to action but with validation logic
        await self._execute_action_step(step, execution)
        
    async def _execute_parallel_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute parallel sub-steps."""
        # Would execute multiple actions in parallel
        pass
        
    async def _execute_wait_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a wait step."""
        wait_time = step.parameters.get("duration", 1)
        await asyncio.sleep(wait_time)
        execution.step_results[step.id] = {"waited": wait_time}
        
    async def _execute_notification_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a notification step."""
        # Send notification through event handler
        if self.event_handler:
            await self.event_handler({
                "type": "notification",
                "step_id": step.id,
                "execution_id": execution.id,
                "parameters": step.parameters
            })
            
    async def _execute_condition_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a conditional branch step."""
        # Evaluate condition and update execution path
        pass
        
    async def _execute_loop_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> None:
        """Execute a loop step."""
        # Execute action multiple times
        iterations = step.parameters.get("iterations", 1)
        for i in range(iterations):
            await self._execute_action_step(step, execution)
            
    async def _rollback_workflow(
        self,
        workflow: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Rollback workflow on failure."""
        self.logger.warning(f"Rolling back workflow: {execution.id}")
        
        # Find rollback steps
        for step in workflow.steps:
            if step.type == StepType.ROLLBACK:
                await self._execute_step(step, execution)
                
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        if execution_id in self._execution_tasks:
            task = self._execution_tasks[execution_id]
            task.cancel()
            
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now(timezone.utc)
                
            return True
        return False
        
    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution details."""
        return self.executions.get(execution_id)
        
    async def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflow executions."""
        active = []
        for execution in self.executions.values():
            if execution.status in [WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
                active.append({
                    "id": execution.id,
                    "workflow": execution.workflow_name,
                    "status": execution.status.value,
                    "current_step": execution.current_step,
                    "execution_time": execution.execution_time
                })
        return active
        
    async def get_pending_workflows(self) -> List[str]:
        """Get list of pending workflow execution IDs."""
        return [
            e.id for e in self.executions.values()
            if e.status == WorkflowStatus.PENDING
        ]
        
    async def can_execute(self, workflow_id: str) -> bool:
        """Check if workflow can be executed."""
        # Check resource availability, dependencies, etc.
        return True
        
    async def execute_by_id(self, execution_id: str) -> None:
        """Execute a pending workflow by execution ID."""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            workflow = self.workflows.get(execution.workflow_id)
            if workflow:
                await self._execute_workflow(workflow, execution)
                
    async def shutdown(self) -> None:
        """Shutdown workflow engine."""
        self.logger.info("Shutting down workflow engine...")
        
        # Cancel all running executions
        for task in self._execution_tasks.values():
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)
        
        self.logger.info("Workflow engine shutdown complete")