"""
SutazAI Workflow Execution Engine
Advanced workflow orchestration with dependency management, parallel execution,
error handling, and workflow optimization.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict, deque
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"

class NodeStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class NodeType(Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    LOOP = "loop"
    SUBWORKFLOW = "subworkflow"

@dataclass
class WorkflowNode:
    id: str
    name: str
    type: NodeType
    agent_type: str
    task_definition: Dict[str, Any]
    dependencies: List[str]
    status: NodeStatus = NodeStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

@dataclass
class ConditionalBranch:
    condition: str
    target_nodes: List[str]
    else_nodes: List[str] = None

@dataclass
class LoopConfiguration:
    condition: str
    max_iterations: int = 100
    iteration_delay: float = 0.0

@dataclass
class Workflow:
    id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    status: WorkflowStatus
    nodes: Dict[str, WorkflowNode]
    execution_graph: nx.DiGraph
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    parallel_limit: int = 5
    timeout: Optional[int] = None

@dataclass
class WorkflowExecution:
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    current_nodes: Set[str]
    completed_nodes: Set[str]
    failed_nodes: Set[str]
    execution_context: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None

class WorkflowEngine:
    """
    Advanced workflow execution engine with support for:
    - Complex dependency management
    - Parallel and sequential execution
    - Conditional branching
    - Loop constructs
    - Error handling and retry logic
    - Workflow optimization
    - Real-time monitoring
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Workflow storage
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        
        # Execution state
        self.running_workflows: Set[str] = set()
        self.workflow_queues: Dict[str, deque] = defaultdict(deque)
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Configuration
        self.max_concurrent_workflows = 10
        self.default_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30
        
        # Metrics
        self.metrics = {
            "workflows_executed": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "avg_execution_time": 0.0,
            "active_workflows": 0,
            "nodes_executed": 0
        }
    
    async def initialize(self):
        """Initialize the workflow engine"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start background tasks
            asyncio.create_task(self._workflow_executor())
            asyncio.create_task(self._workflow_monitor())
            asyncio.create_task(self._metrics_collector())
            
            logger.info("Workflow engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Workflow engine initialization failed: {e}")
            raise
    
    async def create_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Create a new workflow from definition"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Parse workflow definition
            workflow = await self._parse_workflow_definition(workflow_id, workflow_definition)
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow)
            if not validation_result["valid"]:
                raise ValueError(f"Workflow validation failed: {validation_result['errors']}")
            
            # Store workflow
            self.workflows[workflow_id] = workflow
            
            # Persist to Redis
            await self.redis_client.hset(
                "workflows",
                workflow_id,
                json.dumps(asdict(workflow), default=str)
            )
            
            logger.info(f"Workflow created: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise
    
    async def _parse_workflow_definition(self, workflow_id: str, definition: Dict[str, Any]) -> Workflow:
        """Parse workflow definition into Workflow object"""
        workflow = Workflow(
            id=workflow_id,
            name=definition.get("name", f"Workflow_{workflow_id[:8]}"),
            description=definition.get("description", ""),
            created_by=definition.get("created_by", "system"),
            created_at=datetime.now(),
            status=WorkflowStatus.PENDING,
            nodes={},
            execution_graph=nx.DiGraph(),
            metadata=definition.get("metadata", {}),
            parallel_limit=definition.get("parallel_limit", 5),
            timeout=definition.get("timeout", self.default_timeout)
        )
        
        # Parse nodes
        for node_def in definition.get("nodes", []):
            node = WorkflowNode(
                id=node_def["id"],
                name=node_def.get("name", node_def["id"]),
                type=NodeType(node_def.get("type", "task")),
                agent_type=node_def.get("agent_type", "general"),
                task_definition=node_def.get("task", {}),
                dependencies=node_def.get("dependencies", []),
                max_retries=node_def.get("max_retries", 3),
                timeout=node_def.get("timeout"),
                metadata=node_def.get("metadata", {})
            )
            
            workflow.nodes[node.id] = node
            workflow.execution_graph.add_node(node.id)
        
        # Add dependencies to graph
        for node in workflow.nodes.values():
            for dep in node.dependencies:
                if dep in workflow.nodes:
                    workflow.execution_graph.add_edge(dep, node.id)
        
        return workflow
    
    async def _validate_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Validate workflow for correctness"""
        errors = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(workflow.execution_graph):
            errors.append("Workflow contains cycles")
        
        # Check node references
        for node in workflow.nodes.values():
            for dep in node.dependencies:
                if dep not in workflow.nodes:
                    errors.append(f"Node {node.id} references non-existent dependency {dep}")
        
        # Check for disconnected components
        if workflow.nodes and not nx.is_weakly_connected(workflow.execution_graph):
            errors.append("Workflow has disconnected components")
        
        # Validate node types and configurations
        for node in workflow.nodes.values():
            if node.type == NodeType.CONDITION and "condition" not in node.task_definition:
                errors.append(f"Condition node {node.id} missing condition definition")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def execute_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> str:
        """Start workflow execution"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.FAILED]:
                raise ValueError(f"Workflow {workflow_id} is not in executable state")
            
            # Create execution instance
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                workflow_id=workflow_id,
                execution_id=execution_id,
                status=WorkflowStatus.RUNNING,
                current_nodes=set(),
                completed_nodes=set(),
                failed_nodes=set(),
                execution_context=context or {},
                start_time=datetime.now()
            )
            
            self.executions[execution_id] = execution
            
            # Update workflow status
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Add to execution queue
            self.workflow_queues[workflow_id].append(execution_id)
            self.running_workflows.add(workflow_id)
            
            # Emit event
            await self._emit_event("workflow_started", {
                "workflow_id": workflow_id,
                "execution_id": execution_id
            })
            
            self.metrics["workflows_executed"] += 1
            self.metrics["active_workflows"] = len(self.running_workflows)
            
            logger.info(f"Workflow execution started: {workflow_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Workflow execution failed to start: {e}")
            raise
    
    async def _workflow_executor(self):
        """Main workflow execution loop"""
        while True:
            try:
                # Process active workflows
                for workflow_id in list(self.running_workflows):
                    await self._process_workflow(workflow_id)
                
                await asyncio.sleep(1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Workflow executor error: {e}")
                await asyncio.sleep(5)
    
    async def _process_workflow(self, workflow_id: str):
        """Process a single workflow execution"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow or workflow_id not in self.workflow_queues:
                return
            
            queue = self.workflow_queues[workflow_id]
            if not queue:
                return
            
            execution_id = queue[0]  # Process first execution
            execution = self.executions.get(execution_id)
            if not execution:
                queue.popleft()
                return
            
            # Check timeout
            if workflow.timeout:
                elapsed = (datetime.now() - execution.start_time).total_seconds()
                if elapsed > workflow.timeout:
                    await self._timeout_workflow(workflow, execution)
                    return
            
            # Find ready nodes
            ready_nodes = await self._find_ready_nodes(workflow, execution)
            
            # Execute ready nodes (up to parallel limit)
            executing_count = len(execution.current_nodes)
            available_slots = workflow.parallel_limit - executing_count
            
            for node_id in ready_nodes[:available_slots]:
                await self._execute_node(workflow, execution, node_id)
            
            # Check if workflow is complete
            if await self._is_workflow_complete(workflow, execution):
                await self._complete_workflow(workflow, execution)
            
        except Exception as e:
            logger.error(f"Workflow processing error for {workflow_id}: {e}")
            workflow = self.workflows.get(workflow_id)
            execution = self.executions.get(self.workflow_queues[workflow_id][0])
            if workflow and execution:
                await self._fail_workflow(workflow, execution, str(e))
    
    async def _find_ready_nodes(self, workflow: Workflow, execution: WorkflowExecution) -> List[str]:
        """Find nodes that are ready to execute"""
        ready_nodes = []
        
        for node_id, node in workflow.nodes.items():
            # Skip if already processed or currently executing
            if (node_id in execution.completed_nodes or 
                node_id in execution.failed_nodes or 
                node_id in execution.current_nodes):
                continue
            
            # Check if all dependencies are completed
            dependencies_met = all(
                dep in execution.completed_nodes 
                for dep in node.dependencies
            )
            
            if dependencies_met:
                # Additional checks based on node type
                if await self._is_node_ready(workflow, execution, node):
                    ready_nodes.append(node_id)
        
        return ready_nodes
    
    async def _is_node_ready(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode) -> bool:
        """Check if a specific node is ready to execute"""
        if node.type == NodeType.CONDITION:
            # Evaluate condition
            condition = node.task_definition.get("condition", "true")
            return await self._evaluate_condition(condition, execution.execution_context)
        elif node.type == NodeType.LOOP:
            # Check loop condition
            loop_config = node.task_definition.get("loop", {})
            condition = loop_config.get("condition", "false")
            return await self._evaluate_condition(condition, execution.execution_context)
        else:
            return True
    
    async def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string"""
        try:
            # Simple condition evaluation (in production, use safer evaluation)
            # This is a simplified version - in production, use a proper expression evaluator
            return True  # Placeholder implementation
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _execute_node(self, workflow: Workflow, execution: WorkflowExecution, node_id: str):
        """Execute a single workflow node"""
        try:
            node = workflow.nodes[node_id]
            node.status = NodeStatus.RUNNING
            node.started_at = datetime.now()
            execution.current_nodes.add(node_id)
            
            # Emit event
            await self._emit_event("node_started", {
                "workflow_id": workflow.id,
                "execution_id": execution.execution_id,
                "node_id": node_id
            })
            
            # Execute based on node type
            if node.type == NodeType.TASK:
                await self._execute_task_node(workflow, execution, node)
            elif node.type == NodeType.CONDITION:
                await self._execute_condition_node(workflow, execution, node)
            elif node.type == NodeType.PARALLEL:
                await self._execute_parallel_node(workflow, execution, node)
            elif node.type == NodeType.SEQUENTIAL:
                await self._execute_sequential_node(workflow, execution, node)
            elif node.type == NodeType.LOOP:
                await self._execute_loop_node(workflow, execution, node)
            elif node.type == NodeType.SUBWORKFLOW:
                await self._execute_subworkflow_node(workflow, execution, node)
            
        except Exception as e:
            logger.error(f"Node execution failed: {e}")
            await self._fail_node(workflow, execution, node_id, str(e))
    
    async def _execute_task_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a task node"""
        try:
            # Prepare task data
            task_data = {
                "task_id": str(uuid.uuid4()),
                "node_id": node.id,
                "task_type": node.task_definition.get("type", "general"),
                "description": node.task_definition.get("description", ""),
                "input_data": node.task_definition.get("input", {}),
                "context": execution.execution_context
            }
            
            # Send task to agent orchestrator via message bus
            # This would integrate with the task router we created earlier
            from .message_bus import message_bus
            
            message_id = await message_bus.send_task_assignment(
                agent_id=node.assigned_agent or "any",
                task_data=task_data
            )
            
            # Store message ID for tracking
            node.metadata = node.metadata or {}
            node.metadata["message_id"] = message_id
            
            # For now, simulate completion (in production, wait for actual response)
            await asyncio.sleep(1)  # Simulate task execution time
            
            # Simulate successful completion
            result = {
                "success": True,
                "output": f"Task {node.id} completed successfully",
                "execution_time": 1.0
            }
            
            await self._complete_node(workflow, execution, node.id, result)
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _execute_condition_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a conditional node"""
        try:
            condition = node.task_definition.get("condition", "true")
            result = await self._evaluate_condition(condition, execution.execution_context)
            
            # Update execution context with condition result
            execution.execution_context[f"{node.id}_result"] = result
            
            await self._complete_node(workflow, execution, node.id, {"condition_result": result})
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _execute_parallel_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a parallel node (container for parallel execution)"""
        try:
            # Parallel nodes are handled by the main execution logic
            # This is just a placeholder for completion
            await self._complete_node(workflow, execution, node.id, {"parallel_completed": True})
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _execute_sequential_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a sequential node"""
        try:
            # Sequential execution is handled by dependencies
            await self._complete_node(workflow, execution, node.id, {"sequential_completed": True})
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _execute_loop_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a loop node"""
        try:
            loop_config = node.task_definition.get("loop", {})
            max_iterations = loop_config.get("max_iterations", 100)
            condition = loop_config.get("condition", "false")
            
            iteration_count = 0
            while (iteration_count < max_iterations and 
                   await self._evaluate_condition(condition, execution.execution_context)):
                
                # Execute loop body (placeholder)
                iteration_count += 1
                
                # Update context
                execution.execution_context[f"{node.id}_iteration"] = iteration_count
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
            
            result = {
                "iterations_completed": iteration_count,
                "loop_terminated": True
            }
            
            await self._complete_node(workflow, execution, node.id, result)
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _execute_subworkflow_node(self, workflow: Workflow, execution: WorkflowExecution, node: WorkflowNode):
        """Execute a subworkflow node"""
        try:
            subworkflow_id = node.task_definition.get("subworkflow_id")
            if not subworkflow_id:
                raise ValueError("Subworkflow ID not specified")
            
            # Start subworkflow execution
            sub_execution_id = await self.execute_workflow(
                subworkflow_id, 
                execution.execution_context.copy()
            )
            
            # Store reference
            node.metadata = node.metadata or {}
            node.metadata["subworkflow_execution"] = sub_execution_id
            
            # For now, mark as completed (in production, wait for actual completion)
            result = {
                "subworkflow_execution_id": sub_execution_id,
                "status": "started"
            }
            
            await self._complete_node(workflow, execution, node.id, result)
            
        except Exception as e:
            await self._fail_node(workflow, execution, node.id, str(e))
    
    async def _complete_node(self, workflow: Workflow, execution: WorkflowExecution, node_id: str, result: Any):
        """Mark a node as completed"""
        node = workflow.nodes[node_id]
        node.status = NodeStatus.COMPLETED
        node.completed_at = datetime.now()
        node.result = result
        
        # Update execution state
        execution.current_nodes.discard(node_id)
        execution.completed_nodes.add(node_id)
        
        # Update workflow progress
        workflow.progress = len(execution.completed_nodes) / len(workflow.nodes)
        
        # Emit event
        await self._emit_event("node_completed", {
            "workflow_id": workflow.id,
            "execution_id": execution.execution_id,
            "node_id": node_id,
            "result": result,
            "progress": workflow.progress
        })
        
        self.metrics["nodes_executed"] += 1
        
        logger.debug(f"Node completed: {node_id}")
    
    async def _fail_node(self, workflow: Workflow, execution: WorkflowExecution, node_id: str, error: str):
        """Mark a node as failed"""
        node = workflow.nodes[node_id]
        node.error = error
        
        # Check if we should retry
        if node.retry_count < node.max_retries:
            node.retry_count += 1
            node.status = NodeStatus.RETRYING
            execution.current_nodes.discard(node_id)
            
            logger.warning(f"Node {node_id} failed, retrying ({node.retry_count}/{node.max_retries})")
            
            # Retry after a delay
            await asyncio.sleep(2 ** node.retry_count)  # Exponential backoff
            return
        
        # Mark as failed
        node.status = NodeStatus.FAILED
        node.completed_at = datetime.now()
        
        # Update execution state
        execution.current_nodes.discard(node_id)
        execution.failed_nodes.add(node_id)
        
        # Emit event
        await self._emit_event("node_failed", {
            "workflow_id": workflow.id,
            "execution_id": execution.execution_id,
            "node_id": node_id,
            "error": error
        })
        
        # Check if workflow should fail
        if self._should_fail_workflow(workflow, execution):
            await self._fail_workflow(workflow, execution, f"Node {node_id} failed: {error}")
        
        logger.error(f"Node failed: {node_id} - {error}")
    
    def _should_fail_workflow(self, workflow: Workflow, execution: WorkflowExecution) -> bool:
        """Determine if workflow should fail based on failed nodes"""
        # For now, fail if any node fails (can be made configurable)
        return len(execution.failed_nodes) > 0
    
    async def _is_workflow_complete(self, workflow: Workflow, execution: WorkflowExecution) -> bool:
        """Check if workflow execution is complete"""
        total_nodes = len(workflow.nodes)
        processed_nodes = len(execution.completed_nodes) + len(execution.failed_nodes)
        currently_executing = len(execution.current_nodes)
        
        return processed_nodes == total_nodes and currently_executing == 0
    
    async def _complete_workflow(self, workflow: Workflow, execution: WorkflowExecution):
        """Complete workflow execution"""
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now()
        workflow.progress = 1.0
        
        execution.status = WorkflowStatus.COMPLETED
        execution.end_time = datetime.now()
        
        # Calculate execution time
        execution_time = (execution.end_time - execution.start_time).total_seconds()
        
        # Update metrics
        self.metrics["workflows_completed"] += 1
        current_avg = self.metrics["avg_execution_time"]
        completed_count = self.metrics["workflows_completed"]
        self.metrics["avg_execution_time"] = (
            (current_avg * (completed_count - 1) + execution_time) / completed_count
        )
        
        # Clean up
        self.running_workflows.discard(workflow.id)
        if workflow.id in self.workflow_queues:
            self.workflow_queues[workflow.id].popleft()
            if not self.workflow_queues[workflow.id]:
                del self.workflow_queues[workflow.id]
        
        self.metrics["active_workflows"] = len(self.running_workflows)
        
        # Emit event
        await self._emit_event("workflow_completed", {
            "workflow_id": workflow.id,
            "execution_id": execution.execution_id,
            "execution_time": execution_time
        })
        
        logger.info(f"Workflow completed: {workflow.id}")
    
    async def _fail_workflow(self, workflow: Workflow, execution: WorkflowExecution, error: str):
        """Fail workflow execution"""
        workflow.status = WorkflowStatus.FAILED
        workflow.completed_at = datetime.now()
        workflow.error_message = error
        
        execution.status = WorkflowStatus.FAILED
        execution.end_time = datetime.now()
        
        # Update metrics
        self.metrics["workflows_failed"] += 1
        
        # Clean up
        self.running_workflows.discard(workflow.id)
        if workflow.id in self.workflow_queues:
            self.workflow_queues[workflow.id].popleft()
            if not self.workflow_queues[workflow.id]:
                del self.workflow_queues[workflow.id]
        
        self.metrics["active_workflows"] = len(self.running_workflows)
        
        # Emit event
        await self._emit_event("workflow_failed", {
            "workflow_id": workflow.id,
            "execution_id": execution.execution_id,
            "error": error
        })
        
        logger.error(f"Workflow failed: {workflow.id} - {error}")
    
    async def _timeout_workflow(self, workflow: Workflow, execution: WorkflowExecution):
        """Handle workflow timeout"""
        await self._fail_workflow(workflow, execution, "Workflow execution timed out")
    
    async def _workflow_monitor(self):
        """Monitor workflow executions"""
        while True:
            try:
                # Check for stalled workflows
                current_time = datetime.now()
                
                for execution in self.executions.values():
                    if execution.status == WorkflowStatus.RUNNING:
                        elapsed = (current_time - execution.start_time).total_seconds()
                        workflow = self.workflows.get(execution.workflow_id)
                        
                        if workflow and workflow.timeout and elapsed > workflow.timeout:
                            await self._timeout_workflow(workflow, execution)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Workflow monitor error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _metrics_collector(self):
        """Collect workflow metrics"""
        while True:
            try:
                # Store metrics in Redis
                await self.redis_client.hset(
                    "workflow_metrics",
                    "current",
                    json.dumps(self.metrics)
                )
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Workflow metrics collector error: {e}")
                await asyncio.sleep(10)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit workflow event"""
        event_data = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Publish to Redis
        await self.redis_client.publish("workflow_events", json.dumps(event_data))
        
        # Call registered handlers
        for handler in self.event_handlers.get(event_type, []):
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    # Public API methods
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": workflow.progress,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "error_message": workflow.error_message,
            "node_count": len(workflow.nodes),
            "completed_nodes": sum(1 for n in workflow.nodes.values() if n.status == NodeStatus.COMPLETED),
            "failed_nodes": sum(1 for n in workflow.nodes.values() if n.status == NodeStatus.FAILED),
            "running_nodes": sum(1 for n in workflow.nodes.values() if n.status == NodeStatus.RUNNING)
        }
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                return False
            
            if workflow.status != WorkflowStatus.RUNNING:
                return False
            
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            
            # Clean up execution
            self.running_workflows.discard(workflow_id)
            if workflow_id in self.workflow_queues:
                del self.workflow_queues[workflow_id]
            
            # Emit event
            await self._emit_event("workflow_cancelled", {
                "workflow_id": workflow_id
            })
            
            logger.info(f"Workflow cancelled: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow {workflow_id}: {e}")
            return False
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow or workflow.status != WorkflowStatus.RUNNING:
                return False
            
            workflow.status = WorkflowStatus.PAUSED
            
            # Emit event
            await self._emit_event("workflow_paused", {
                "workflow_id": workflow_id
            })
            
            logger.info(f"Workflow paused: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause workflow {workflow_id}: {e}")
            return False
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow or workflow.status != WorkflowStatus.PAUSED:
                return False
            
            workflow.status = WorkflowStatus.RUNNING
            self.running_workflows.add(workflow_id)
            
            # Emit event
            await self._emit_event("workflow_resumed", {
                "workflow_id": workflow_id
            })
            
            logger.info(f"Workflow resumed: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get workflow engine metrics"""
        return {
            **self.metrics,
            "workflow_details": {
                "total_workflows": len(self.workflows),
                "running_workflows": len(self.running_workflows),
                "queued_executions": sum(len(q) for q in self.workflow_queues.values())
            }
        }
    
    async def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
    
    async def stop(self):
        """Stop the workflow engine"""
        # Cancel all running workflows
        for workflow_id in list(self.running_workflows):
            await self.cancel_workflow(workflow_id)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Workflow engine stopped")

# Singleton instance
workflow_engine = WorkflowEngine()