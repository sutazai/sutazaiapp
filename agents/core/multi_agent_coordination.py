"""
Multi-Agent Coordination Patterns Implementation
Rule 14 Compliant - Advanced Orchestration Patterns

Implements Sequential, Parallel, and Event-Driven coordination patterns
for sophisticated multi-agent workflows.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Inter-agent message types."""
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    HANDOFF = "handoff"
    CHECKPOINT = "checkpoint"
    HEARTBEAT = "heartbeat"
    CONTROL = "control"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str
    type: MessageType
    sender: str
    receiver: str
    payload: Any
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowContext:
    """Shared context for workflow execution."""
    workflow_id: str
    state: WorkflowState = WorkflowState.PENDING
    shared_data: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class SequentialCoordinator:
    """
    Sequential Coordination Pattern Implementation.
    
    Implements waterfall workflows, pipeline processing, approval gates,
    and knowledge transfer between agents.
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    async def execute_waterfall(self, agents: List[str], tasks: List[Dict[str, Any]], 
                               context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute agents in strict sequential order (waterfall pattern).
        
        Args:
            agents: List of agent names
            tasks: List of task specifications
            context: Workflow context
            
        Returns:
            Execution result with outputs from all stages
        """
        context.state = WorkflowState.RUNNING
        context.start_time = datetime.now()
        results = []
        
        try:
            for i, (agent, task) in enumerate(zip(agents, tasks)):
                logger.info(f"Waterfall stage {i+1}/{len(agents)}: {agent}")
                
                # Create checkpoint before execution
                checkpoint = self._create_checkpoint(context, f"stage_{i+1}")
                context.checkpoints.append(checkpoint)
                
                # Execute agent task
                result = await self._execute_agent_task(agent, task, context)
                
                if result["status"] == "failed":
                    context.state = WorkflowState.FAILED
                    context.errors.append({
                        "stage": i + 1,
                        "agent": agent,
                        "error": result.get("error", "Unknown error"),
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                
                # Knowledge transfer to next agent
                if i < len(agents) - 1:
                    await self._transfer_knowledge(agent, agents[i+1], result["output"], context)
                
                results.append(result)
                
                # Update shared context
                context.shared_data[f"stage_{i+1}_output"] = result["output"]
            
            if context.state != WorkflowState.FAILED:
                context.state = WorkflowState.COMPLETED
            
        except Exception as e:
            logger.error(f"Waterfall execution error: {str(e)}")
            context.state = WorkflowState.FAILED
            context.errors.append({
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat()
            })
        
        finally:
            context.end_time = datetime.now()
            self._record_execution(context, results)
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "results": results,
            "duration": (context.end_time - context.start_time).total_seconds() if context.end_time else 0,
            "checkpoints": context.checkpoints,
            "errors": context.errors
        }
    
    async def execute_pipeline(self, agents: List[str], data_stream: Any,
                              context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute pipeline processing with streaming data flow.
        
        Args:
            agents: List of agent names forming the pipeline
            data_stream: Input data stream
            context: Workflow context
            
        Returns:
            Pipeline execution results
        """
        context.state = WorkflowState.RUNNING
        context.start_time = datetime.now()
        pipeline_output = data_stream
        stage_outputs = []
        
        try:
            for i, agent in enumerate(agents):
                logger.info(f"Pipeline stage {i+1}/{len(agents)}: {agent}")
                
                # Process data through agent
                stage_result = await self._process_pipeline_stage(
                    agent, pipeline_output, context, stage_num=i+1
                )
                
                if stage_result["status"] == "failed":
                    context.state = WorkflowState.FAILED
                    break
                
                # Pass output to next stage
                pipeline_output = stage_result["output"]
                stage_outputs.append(stage_result)
                
                # Update metrics
                context.metrics[f"stage_{i+1}_latency"] = stage_result.get("latency", 0)
            
            if context.state != WorkflowState.FAILED:
                context.state = WorkflowState.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {str(e)}")
            context.state = WorkflowState.FAILED
            context.errors.append({"error": str(e)})
        
        finally:
            context.end_time = datetime.now()
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "final_output": pipeline_output,
            "stage_outputs": stage_outputs,
            "metrics": context.metrics
        }
    
    async def execute_with_approval_gates(self, agents: List[str], tasks: List[Dict[str, Any]],
                                         approval_func: Callable, context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute workflow with approval gates between stages.
        
        Args:
            agents: List of agent names
            tasks: List of task specifications
            approval_func: Function to check approval
            context: Workflow context
            
        Returns:
            Execution results with approval history
        """
        context.state = WorkflowState.RUNNING
        results = []
        approvals = []
        
        try:
            for i, (agent, task) in enumerate(zip(agents, tasks)):
                # Check approval gate
                if i > 0:  # No approval needed for first stage
                    approval = await approval_func(context, results[-1])
                    approvals.append({
                        "stage": i,
                        "approved": approval["approved"],
                        "reason": approval.get("reason", ""),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if not approval["approved"]:
                        logger.warning(f"Approval denied at stage {i}")
                        context.state = WorkflowState.PAUSED
                        break
                
                # Execute stage
                result = await self._execute_agent_task(agent, task, context)
                results.append(result)
                
                if result["status"] == "failed":
                    context.state = WorkflowState.FAILED
                    break
            
            if context.state == WorkflowState.RUNNING:
                context.state = WorkflowState.COMPLETED
        
        except Exception as e:
            logger.error(f"Approval gate workflow error: {str(e)}")
            context.state = WorkflowState.FAILED
            context.errors.append({"error": str(e)})
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "results": results,
            "approvals": approvals
        }
    
    async def _execute_agent_task(self, agent: str, task: Dict[str, Any], 
                                 context: WorkflowContext) -> Dict[str, Any]:
        """Execute a single agent task."""
        start_time = datetime.now()
        
        try:
            # Simulate agent execution (replace with actual agent call)
            await asyncio.sleep(0.5)  # Simulated processing time
            
            output = {
                "agent": agent,
                "task": task.get("name", "unnamed"),
                "result": f"Completed by {agent}",
                "data": task.get("input_data", {})
            }
            
            return {
                "status": "success",
                "output": output,
                "duration": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
    async def _transfer_knowledge(self, from_agent: str, to_agent: str, 
                                 knowledge: Any, context: WorkflowContext):
        """Transfer knowledge between agents."""
        transfer_record = {
            "from": from_agent,
            "to": to_agent,
            "knowledge_size": len(str(knowledge)),
            "timestamp": datetime.now().isoformat()
        }
        
        context.shared_data.setdefault("knowledge_transfers", []).append(transfer_record)
        logger.info(f"Knowledge transferred from {from_agent} to {to_agent}")
    
    async def _process_pipeline_stage(self, agent: str, input_data: Any,
                                     context: WorkflowContext, stage_num: int) -> Dict[str, Any]:
        """Process a single pipeline stage."""
        start_time = datetime.now()
        
        try:
            # Simulate data transformation
            await asyncio.sleep(0.3)
            
            output = {
                "stage": stage_num,
                "agent": agent,
                "transformed_data": f"{input_data}_processed_by_{agent}"
            }
            
            return {
                "status": "success",
                "output": output,
                "latency": (datetime.now() - start_time).total_seconds() * 1000  # ms
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "latency": (datetime.now() - start_time).total_seconds() * 1000
            }
    
    def _create_checkpoint(self, context: WorkflowContext, name: str) -> Dict[str, Any]:
        """Create a workflow checkpoint."""
        return {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "state": context.state.value,
            "data_snapshot": dict(context.shared_data)
        }
    
    def _record_execution(self, context: WorkflowContext, results: List[Dict[str, Any]]):
        """Record execution history."""
        self.execution_history.append({
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "start_time": context.start_time.isoformat() if context.start_time else None,
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "results_count": len(results),
            "errors_count": len(context.errors)
        })


class ParallelCoordinator:
    """
    Parallel Coordination Pattern Implementation.
    
    Implements scatter-gather, load balancing, resource pooling,
    and conflict resolution for parallel agent execution.
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, Set[str]] = {}
        self.resource_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
    async def execute_scatter_gather(self, agents: List[str], task: Dict[str, Any],
                                    context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute scatter-gather pattern: distribute task to all agents, gather results.
        
        Args:
            agents: List of agent names
            task: Task to scatter
            context: Workflow context
            
        Returns:
            Gathered results from all agents
        """
        context.state = WorkflowState.RUNNING
        context.start_time = datetime.now()
        
        # Scatter phase
        futures = []
        for agent in agents:
            future = self.executor.submit(self._execute_parallel_task, agent, task, context)
            futures.append((agent, future))
        
        # Gather phase
        results = []
        errors = []
        
        for agent, future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
                logger.info(f"Gathered result from {agent}")
            except Exception as e:
                error_info = {
                    "agent": agent,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                logger.error(f"Agent {agent} failed: {str(e)}")
        
        # Consolidate results
        consolidated = self._consolidate_results(results)
        
        context.state = WorkflowState.COMPLETED if not errors else WorkflowState.FAILED
        context.end_time = datetime.now()
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "pattern": "scatter_gather",
            "agents_count": len(agents),
            "successful_count": len(results),
            "failed_count": len(errors),
            "consolidated_result": consolidated,
            "individual_results": results,
            "errors": errors,
            "duration": (context.end_time - context.start_time).total_seconds()
        }
    
    async def execute_with_load_balancing(self, agents: List[str], tasks: List[Dict[str, Any]],
                                         context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute tasks with intelligent load balancing across agents.
        
        Args:
            agents: List of available agents
            tasks: List of tasks to execute
            context: Workflow context
            
        Returns:
            Load-balanced execution results
        """
        context.state = WorkflowState.RUNNING
        agent_loads = {agent: 0 for agent in agents}
        task_queue = deque(tasks)
        results = []
        
        while task_queue:
            # Find least loaded agent
            agent = min(agent_loads, key=agent_loads.get)
            task = task_queue.popleft()
            
            # Assign task to agent
            logger.info(f"Assigning task to {agent} (load: {agent_loads[agent]})")
            
            # Execute asynchronously
            result = await self._execute_with_load_tracking(agent, task, agent_loads)
            results.append(result)
        
        context.state = WorkflowState.COMPLETED
        
        # Calculate load distribution metrics
        load_metrics = {
            "min_load": min(agent_loads.values()),
            "max_load": max(agent_loads.values()),
            "avg_load": sum(agent_loads.values()) / len(agent_loads),
            "distribution": agent_loads
        }
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "pattern": "load_balanced",
            "tasks_count": len(tasks),
            "agents_count": len(agents),
            "results": results,
            "load_metrics": load_metrics
        }
    
    async def execute_with_resource_pooling(self, agents: List[str], tasks: List[Dict[str, Any]],
                                           resource_pool: Dict[str, int], 
                                           context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute with shared resource pool management.
        
        Args:
            agents: List of agent names
            tasks: List of tasks
            resource_pool: Available resources
            context: Workflow context
            
        Returns:
            Execution results with resource utilization
        """
        context.state = WorkflowState.RUNNING
        available_resources = dict(resource_pool)
        resource_allocations = []
        results = []
        
        for task in tasks:
            # Wait for resources to become available
            allocated = await self._allocate_resources(task, available_resources)
            
            if allocated:
                # Find available agent
                agent = await self._find_available_agent(agents)
                
                # Execute with allocated resources
                result = await self._execute_with_resources(agent, task, allocated)
                results.append(result)
                
                # Release resources
                await self._release_resources(allocated, available_resources)
                
                resource_allocations.append({
                    "task": task.get("name", "unnamed"),
                    "agent": agent,
                    "resources": allocated,
                    "timestamp": datetime.now().isoformat()
                })
        
        context.state = WorkflowState.COMPLETED
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "pattern": "resource_pooled",
            "results": results,
            "resource_allocations": resource_allocations,
            "resource_utilization": self._calculate_utilization(resource_allocations, resource_pool)
        }
    
    def _execute_parallel_task(self, agent: str, task: Dict[str, Any], 
                              context: WorkflowContext) -> Dict[str, Any]:
        """Execute task in parallel (synchronous for thread pool)."""
        import time
        start_time = datetime.now()
        
        try:
            # Simulate agent execution
            time.sleep(0.5)
            
            return {
                "agent": agent,
                "status": "success",
                "output": f"Task completed by {agent}",
                "duration": (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            return {
                "agent": agent,
                "status": "failed",
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
    def _consolidate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate results from multiple agents."""
        if not results:
            return {}
        
        # Simple consolidation - can be customized
        successful = [r for r in results if r.get("status") == "success"]
        
        return {
            "total_results": len(results),
            "successful": len(successful),
            "consolidated_output": [r.get("output") for r in successful],
            "average_duration": sum(r.get("duration", 0) for r in results) / len(results) if results else 0
        }
    
    async def _execute_with_load_tracking(self, agent: str, task: Dict[str, Any],
                                         agent_loads: Dict[str, int]) -> Dict[str, Any]:
        """Execute task with load tracking."""
        agent_loads[agent] += 1
        
        try:
            # Simulate execution
            await asyncio.sleep(0.3)
            
            result = {
                "agent": agent,
                "task": task.get("name", "unnamed"),
                "status": "success"
            }
            
        finally:
            agent_loads[agent] -= 1
        
        return result
    
    async def _allocate_resources(self, task: Dict[str, Any], 
                                 available: Dict[str, int]) -> Optional[Dict[str, int]]:
        """Allocate resources for task."""
        required = task.get("resources", {"cpu": 10, "memory": 20})
        
        # Check if resources are available
        for resource, amount in required.items():
            if available.get(resource, 0) < amount:
                return None
        
        # Allocate resources
        allocated = {}
        for resource, amount in required.items():
            available[resource] -= amount
            allocated[resource] = amount
        
        return allocated
    
    async def _release_resources(self, allocated: Dict[str, int], 
                                available: Dict[str, int]):
        """Release allocated resources back to pool."""
        for resource, amount in allocated.items():
            available[resource] += amount
    
    async def _find_available_agent(self, agents: List[str]) -> str:
        """Find an available agent."""
        # Simple round-robin selection
        import random
        return random.choice(agents)
    
    async def _execute_with_resources(self, agent: str, task: Dict[str, Any],
                                     resources: Dict[str, int]) -> Dict[str, Any]:
        """Execute task with allocated resources."""
        await asyncio.sleep(0.5)  # Simulate execution
        
        return {
            "agent": agent,
            "task": task.get("name", "unnamed"),
            "resources_used": resources,
            "status": "success"
        }
    
    def _calculate_utilization(self, allocations: List[Dict], 
                              pool: Dict[str, int]) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        if not allocations:
            return {}
        
        total_used = defaultdict(int)
        for allocation in allocations:
            for resource, amount in allocation.get("resources", {}).items():
                total_used[resource] += amount
        
        utilization = {}
        for resource, total in pool.items():
            if total > 0:
                utilization[resource] = (total_used[resource] / total) * 100
        
        return utilization


class EventDrivenCoordinator:
    """
    Event-Driven Coordination Pattern Implementation.
    
    Implements publish-subscribe, message queuing, state management,
    and circuit breakers for event-driven agent coordination.
    """
    
    def __init__(self):
        self.event_bus = defaultdict(list)  # event_type -> list of subscribers
        self.message_queue = queue.Queue()
        self.state_store: Dict[str, Any] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.event_history: deque = deque(maxlen=1000)
        
    def subscribe(self, event_type: str, agent: str, handler: Callable):
        """Subscribe an agent to an event type."""
        self.event_bus[event_type].append({
            "agent": agent,
            "handler": handler
        })
        logger.info(f"Agent {agent} subscribed to {event_type}")
    
    async def publish(self, event_type: str, payload: Any, context: WorkflowContext) -> Dict[str, Any]:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event
            payload: Event payload
            context: Workflow context
            
        Returns:
            Publication results
        """
        subscribers = self.event_bus.get(event_type, [])
        
        if not subscribers:
            logger.warning(f"No subscribers for event {event_type}")
            return {"subscribers": 0, "delivered": 0}
        
        delivered = 0
        failures = []
        
        for subscriber in subscribers:
            try:
                # Check circuit breaker
                if self._is_circuit_open(subscriber["agent"]):
                    logger.warning(f"Circuit breaker open for {subscriber['agent']}")
                    continue
                
                # Deliver event
                await subscriber["handler"](payload, context)
                delivered += 1
                
                # Record successful delivery
                self._record_event({
                    "type": event_type,
                    "agent": subscriber["agent"],
                    "status": "delivered",
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                failures.append({
                    "agent": subscriber["agent"],
                    "error": str(e)
                })
                
                # Trip circuit breaker if needed
                self._handle_failure(subscriber["agent"])
        
        return {
            "event_type": event_type,
            "subscribers": len(subscribers),
            "delivered": delivered,
            "failures": failures
        }
    
    async def execute_with_message_queue(self, agents: List[str], messages: List[AgentMessage],
                                        context: WorkflowContext) -> Dict[str, Any]:
        """
        Execute workflow using message queue for agent communication.
        
        Args:
            agents: List of agent names
            messages: Initial messages
            context: Workflow context
            
        Returns:
            Message processing results
        """
        context.state = WorkflowState.RUNNING
        
        # Initialize message queue
        for msg in messages:
            self.message_queue.put(msg)
        
        processed_messages = []
        agent_handlers = {agent: self._create_message_handler(agent) for agent in agents}
        
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get(timeout=1)
                
                # Route message to appropriate agent
                if message.receiver in agent_handlers:
                    result = await agent_handlers[message.receiver](message, context)
                    processed_messages.append({
                        "message_id": message.id,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Generate follow-up messages if needed
                    if result.get("follow_up"):
                        for follow_up_msg in result["follow_up"]:
                            self.message_queue.put(follow_up_msg)
                
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Message processing error: {str(e)}")
                context.errors.append({"error": str(e)})
        
        context.state = WorkflowState.COMPLETED
        
        return {
            "workflow_id": context.workflow_id,
            "state": context.state.value,
            "pattern": "message_queue",
            "messages_processed": len(processed_messages),
            "results": processed_messages
        }
    
    async def manage_state(self, key: str, value: Any, operation: str = "set") -> Any:
        """
        Centralized state management for agent coordination.
        
        Args:
            key: State key
            value: State value
            operation: Operation type (set, get, update, delete)
            
        Returns:
            Operation result
        """
        if operation == "set":
            self.state_store[key] = value
            return value
        elif operation == "get":
            return self.state_store.get(key)
        elif operation == "update":
            if key in self.state_store:
                if isinstance(self.state_store[key], dict) and isinstance(value, dict):
                    self.state_store[key].update(value)
                else:
                    self.state_store[key] = value
            return self.state_store.get(key)
        elif operation == "delete":
            return self.state_store.pop(key, None)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _create_message_handler(self, agent: str) -> Callable:
        """Create a message handler for an agent."""
        async def handler(message: AgentMessage, context: WorkflowContext) -> Dict[str, Any]:
            # Simulate message processing
            await asyncio.sleep(0.2)
            
            result = {
                "agent": agent,
                "message_type": message.type.value,
                "processed": True
            }
            
            # Generate follow-up messages for complex workflows
            if message.type == MessageType.TASK:
                result["follow_up"] = [
                    AgentMessage(
                        id=hashlib.md5(f"{message.id}_result".encode()).hexdigest()[:8],
                        type=MessageType.RESULT,
                        sender=agent,
                        receiver=message.sender,
                        payload={"task_completed": True},
                        correlation_id=message.id
                    )
                ]
            
            return result
        
        return handler
    
    def _is_circuit_open(self, agent: str) -> bool:
        """Check if circuit breaker is open for an agent."""
        if agent not in self.circuit_breakers:
            self.circuit_breakers[agent] = CircuitBreaker(agent)
        
        return self.circuit_breakers[agent].is_open()
    
    def _handle_failure(self, agent: str):
        """Handle agent failure for circuit breaker."""
        if agent not in self.circuit_breakers:
            self.circuit_breakers[agent] = CircuitBreaker(agent)
        
        self.circuit_breakers[agent].record_failure()
    
    def _record_event(self, event: Dict[str, Any]):
        """Record event in history."""
        self.event_history.append(event)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, failure_threshold: int = 3, timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                if (datetime.now() - self.last_failure_time).seconds > self.timeout:
                    self.state = "half_open"
                    return False
            return True
        return False
    
    def record_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened for {self.name}")
    
    def record_success(self):
        """Record a success."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info(f"Circuit breaker closed for {self.name}")


class MultiAgentOrchestrator:
    """
    Master orchestrator combining all coordination patterns.
    
    Provides unified interface for sequential, parallel, and event-driven
    multi-agent workflow execution.
    """
    
    def __init__(self):
        self.sequential = SequentialCoordinator()
        self.parallel = ParallelCoordinator()
        self.event_driven = EventDrivenCoordinator()
        self.workflows: Dict[str, WorkflowContext] = {}
        
    async def execute_workflow(self, workflow_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a complete multi-agent workflow based on specification.
        
        Args:
            workflow_spec: Complete workflow specification
            
        Returns:
            Workflow execution results
        """
        workflow_id = workflow_spec.get("id", hashlib.md5(str(workflow_spec).encode()).hexdigest()[:8])
        context = WorkflowContext(workflow_id=workflow_id)
        self.workflows[workflow_id] = context
        
        pattern = workflow_spec.get("coordination_pattern", "sequential")
        agents = workflow_spec.get("agents", [])
        tasks = workflow_spec.get("tasks", [])
        
        logger.info(f"Executing workflow {workflow_id} with pattern {pattern}")
        
        try:
            if pattern == "sequential":
                result = await self.sequential.execute_waterfall(agents, tasks, context)
            elif pattern == "parallel":
                result = await self.parallel.execute_scatter_gather(agents, tasks[0], context)
            elif pattern == "pipeline":
                result = await self.sequential.execute_pipeline(agents, tasks[0], context)
            elif pattern == "event_driven":
                messages = [AgentMessage(
                    id=f"msg_{i}",
                    type=MessageType.TASK,
                    sender="orchestrator",
                    receiver=agent,
                    payload=task
                ) for i, (agent, task) in enumerate(zip(agents, tasks))]
                result = await self.event_driven.execute_with_message_queue(agents, messages, context)
            else:
                # Hybrid pattern - combine multiple patterns
                result = await self._execute_hybrid(workflow_spec, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            context.state = WorkflowState.FAILED
            context.errors.append({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return {
                "workflow_id": workflow_id,
                "state": "failed",
                "error": str(e)
            }
    
    async def _execute_hybrid(self, workflow_spec: Dict[str, Any], 
                            context: WorkflowContext) -> Dict[str, Any]:
        """Execute hybrid coordination pattern."""
        stages = workflow_spec.get("stages", [])
        results = []
        
        for stage in stages:
            pattern = stage.get("pattern", "sequential")
            agents = stage.get("agents", [])
            tasks = stage.get("tasks", [])
            
            if pattern == "sequential":
                stage_result = await self.sequential.execute_waterfall(agents, tasks, context)
            elif pattern == "parallel":
                stage_result = await self.parallel.execute_scatter_gather(agents, tasks[0], context)
            else:
                stage_result = {"error": f"Unknown pattern: {pattern}"}
            
            results.append(stage_result)
        
        return {
            "workflow_id": context.workflow_id,
            "pattern": "hybrid",
            "stages": len(stages),
            "results": results
        }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        context = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "state": context.state.value,
            "start_time": context.start_time.isoformat() if context.start_time else None,
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "checkpoints": len(context.checkpoints),
            "errors": len(context.errors),
            "shared_data_keys": list(context.shared_data.keys())
        }
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].state = WorkflowState.PAUSED
            logger.info(f"Workflow {workflow_id} paused")
            return True
        return False
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id in self.workflows:
            if self.workflows[workflow_id].state == WorkflowState.PAUSED:
                self.workflows[workflow_id].state = WorkflowState.RUNNING
                logger.info(f"Workflow {workflow_id} resumed")
                return True
        return False
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].state = WorkflowState.CANCELLED
            self.workflows[workflow_id].end_time = datetime.now()
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        return False