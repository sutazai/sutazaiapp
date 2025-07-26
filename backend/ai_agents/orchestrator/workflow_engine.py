"""
Workflow Engine
Manages task workflows and dependencies
"""

import logging
import uuid
import asyncio
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Status of workflow tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class WorkflowStatus(Enum):
    """Status of workflows"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    workflow_id: str
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Workflow:
    """Workflow definition and execution state"""
    workflow_id: str
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    priority: int = 5
    result: Optional[Dict[str, Any]] = None

class WorkflowEngine:
    """Engine for executing workflows with task dependencies"""
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.running = False
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        
    def start(self):
        """Start the workflow engine"""
        self.running = True
        logger.info("Workflow engine started")
        
    def stop(self):
        """Stop the workflow engine"""
        self.running = False
        
        # Cancel all running executions
        for task in self.execution_tasks.values():
            task.cancel()
            
        logger.info("Workflow engine stopped")
        
    def create_workflow(self, name: str, description: str = "") -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description
        )
        
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow_id
        
    def add_task(self, workflow_id: str, agent_id: str, task_type: str,
                 parameters: Dict[str, Any], dependencies: Optional[List[str]] = None) -> str:
        """Add a task to a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status not in [WorkflowStatus.CREATED, WorkflowStatus.PAUSED]:
            raise ValueError(f"Cannot add tasks to workflow in status {workflow.status}")
            
        task_id = str(uuid.uuid4())
        dependencies = dependencies or []
        
        # Validate dependencies exist
        for dep_id in dependencies:
            if dep_id not in workflow.tasks:
                raise ValueError(f"Dependency task {dep_id} not found in workflow")
                
        task = WorkflowTask(
            task_id=task_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            task_type=task_type,
            parameters=parameters,
            dependencies=dependencies
        )
        
        workflow.tasks[task_id] = task
        
        logger.info(f"Added task {task_id} to workflow {workflow_id}")
        return task_id
        
    def execute_workflow(self, workflow_id: str, async_execution: bool = True) -> Optional[Dict[str, Any]]:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.CREATED:
            raise ValueError(f"Workflow {workflow_id} is not in created state")
            
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        if async_execution:
            # Start async execution
            task = asyncio.create_task(self._execute_workflow_async(workflow))
            self.execution_tasks[workflow_id] = task
            return None
        else:
            # Synchronous execution (simplified)
            return self._execute_workflow_sync(workflow)
            
    async def _execute_workflow_async(self, workflow: Workflow):
        """Execute workflow asynchronously"""
        try:
            logger.info(f"Starting async execution of workflow {workflow.workflow_id}")
            
            while True:
                # Find ready tasks (dependencies completed)
                ready_tasks = self._get_ready_tasks(workflow)
                
                if not ready_tasks:
                    # Check if workflow is complete
                    if all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                           for task in workflow.tasks.values()):
                        break
                    else:
                        # Wait for running tasks to complete
                        await asyncio.sleep(1)
                        continue
                        
                # Execute ready tasks
                task_executions = []
                for task in ready_tasks:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    task_executions.append(self._execute_task(task))
                    
                # Wait for current batch to complete
                await asyncio.gather(*task_executions, return_exceptions=True)
                
            # Determine final workflow status
            failed_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.FAILED]
            cancelled_tasks = [t for t in workflow.tasks.values() if t.status == TaskStatus.CANCELLED]
            
            if cancelled_tasks:
                workflow.status = WorkflowStatus.CANCELLED
            elif failed_tasks:
                workflow.status = WorkflowStatus.FAILED
            else:
                workflow.status = WorkflowStatus.COMPLETED
                
            workflow.completed_at = datetime.now()
            
            # Collect results
            workflow.result = {
                "task_results": {tid: task.result for tid, task in workflow.tasks.items() if task.result},
                "execution_summary": {
                    "total_tasks": len(workflow.tasks),
                    "completed_tasks": len([t for t in workflow.tasks.values() if t.status == TaskStatus.COMPLETED]),
                    "failed_tasks": len(failed_tasks),
                    "cancelled_tasks": len(cancelled_tasks)
                }
            }
            
            logger.info(f"Workflow {workflow.workflow_id} completed with status {workflow.status}")
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow.workflow_id}: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            
        finally:
            # Clean up execution tracking
            if workflow.workflow_id in self.execution_tasks:
                del self.execution_tasks[workflow.workflow_id]
                
    def _execute_workflow_sync(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute workflow synchronously (simplified)"""
        completed_tasks = []
        
        # Simple sequential execution for sync mode
        for task in workflow.tasks.values():
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                # Simulate task execution
                task.result = {
                    "status": "completed",
                    "message": f"Task {task.task_type} executed by agent {task.agent_id}",
                    "parameters": task.parameters
                }
                
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                completed_tasks.append(task.task_id)
                
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now()
        
        return {
            "workflow_id": workflow.workflow_id,
            "completed_tasks": completed_tasks,
            "status": workflow.status.value
        }
        
    async def _execute_task(self, task: WorkflowTask):
        """Execute individual task"""
        try:
            # Simulate task execution with the assigned agent
            await asyncio.sleep(0.1)  # Simulate processing time
            
            task.result = {
                "status": "completed",
                "agent_id": task.agent_id,
                "task_type": task.task_type,
                "parameters": task.parameters,
                "execution_time": "0.1s",
                "timestamp": datetime.now().isoformat()
            }
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            logger.debug(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count + 1})")
                
    def _get_ready_tasks(self, workflow: Workflow) -> List[WorkflowTask]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        
        for task in workflow.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_completed = all(
                workflow.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in workflow.tasks
            )
            
            if dependencies_completed:
                ready_tasks.append(task)
                
        return ready_tasks
        
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and details"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        task_statuses = {}
        for task_id, task in workflow.tasks.items():
            task_statuses[task_id] = {
                "status": task.status.value,
                "agent_id": task.agent_id,
                "task_type": task.task_type,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error": task.error
            }
            
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "total_tasks": len(workflow.tasks),
            "task_statuses": task_statuses,
            "result": workflow.result
        }
        
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status not in [WorkflowStatus.RUNNING, WorkflowStatus.CREATED]:
            return False
            
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()
        
        # Cancel all pending/running tasks
        for task in workflow.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
        # Cancel async execution if running
        if workflow_id in self.execution_tasks:
            self.execution_tasks[workflow_id].cancel()
            del self.execution_tasks[workflow_id]
            
        logger.info(f"Cancelled workflow {workflow_id}")
        return True
        
    def get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows summary"""
        return [
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "status": wf.status.value,
                "task_count": len(wf.tasks),
                "created_at": wf.created_at.isoformat()
            }
            for wf in self.workflows.values()
        ]