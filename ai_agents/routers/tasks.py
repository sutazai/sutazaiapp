"""
Agent Task Router

This module provides REST API endpoints for managing agent tasks and task execution.
"""

import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

from ..agent_manager import AgentManager
from ai_agents.dependencies import (
    get_agent_manager,
    get_agent_communication,
    get_interaction_manager,
    get_workflow_engine,
    get_memory_manager,
    get_shared_memory_manager,
    get_health_check,
)
from ai_agents.utils.performance_metrics import PerformanceMetrics
from ai_agents.utils.startup import initialize_agent_system

router = APIRouter(prefix="/tasks", tags=["tasks"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class TaskRequest(BaseModel):
    """Task request model."""

    agent_id: str
    task_type: str
    parameters: Optional[Dict[str, Any]] = None
    priority: Optional[int] = 0
    timeout: Optional[int] = None


class TaskResponse(BaseModel):
    """Task response model."""

    task_id: str
    agent_id: str
    task_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskStatus(BaseModel):
    """Task status model."""

    status: str
    progress: Optional[float] = None
    current_step: Optional[str] = None
    estimated_time: Optional[float] = None


@router.on_event("startup")
def startup_event():
    """Initialize agent manager and other components on startup."""
    logger.info("Initializing agent system...")
    try:
        # Initialize all core components
        initialize_agent_system()
        logger.info("Agent system initialized successfully.")
    except Exception as e:
        logger.exception(f"Fatal error during agent system initialization: {e}")
        # Depending on severity, might want to exit or prevent app start


@router.post("/submit", response_model=TaskResponse)
async def submit_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    agent_manager: AgentManager = Depends(get_agent_manager),
) -> TaskResponse:
    """
    Submit a new task for execution.

    Args:
        request: Task request details
        background_tasks: FastAPI background tasks
        agent_manager: Agent manager instance

    Returns:
        TaskResponse: Created task details
    """
    try:
        if request.agent_id not in agent_manager.agents:
            raise HTTPException(
                status_code=404, detail=f"Agent {request.agent_id} not found"
            )

        agent = agent_manager.agents[request.agent_id]
        if not agent.is_capable(request.task_type):
            raise HTTPException(
                status_code=400,
                detail=f"Agent {request.agent_id} does not support task type {request.task_type}",
            )

        # Create task
        task_id = str(uuid.uuid4())
        task = TaskResponse(
            task_id=task_id,
            agent_id=request.agent_id,
            task_type=request.task_type,
            status="pending",
            created_at=datetime.utcnow(),
        )

        # Store task
        agent_manager.task_queue[task_id] = {
            "task": task,
            "parameters": request.parameters or {},
            "priority": request.priority,
            "timeout": request.timeout,
        }

        # Schedule task execution
        background_tasks.add_task(
            agent_manager.execute_task, task_id, request.parameters or {}, request.timeout
        )

        return task
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str, agent_manager: AgentManager = Depends(get_agent_manager)
) -> TaskResponse:
    """
    Get status of a specific task.

    Args:
        task_id: Task ID to get status for
        agent_manager: Agent manager instance

    Returns:
        TaskResponse: Task status and details
    """
    try:
        if task_id not in agent_manager.task_queue:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return agent_manager.task_queue[task_id]["task"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent/{agent_id}", response_model=List[TaskResponse])
async def get_agent_tasks(
    agent_id: str,
    status: Optional[str] = None,
    agent_manager: AgentManager = Depends(get_agent_manager),
) -> List[TaskResponse]:
    """
    Get all tasks for a specific agent.

    Args:
        agent_id: Agent ID to get tasks for
        status: Optional status filter
        agent_manager: Agent manager instance

    Returns:
        List[TaskResponse]: List of tasks
    """
    try:
        if agent_id not in agent_manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        tasks = [
            task_data["task"]
            for task_data in agent_manager.task_queue.values()
            if task_data["task"].agent_id == agent_id
        ]

        if status:
            tasks = [task for task in tasks if task.status == status]

        return tasks
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str, agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Cancel a specific task.

    Args:
        task_id: Task ID to cancel
        agent_manager: Agent manager instance

    Returns:
        Dict[str, str]: Cancellation status
    """
    try:
        if task_id not in agent_manager.task_queue:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        task_data = agent_manager.task_queue[task_id]
        task = task_data["task"]

        if task.status in ["completed", "failed"]:
            raise HTTPException(
                status_code=400, detail=f"Cannot cancel task in status {task.status}"
            )

        # Cancel task
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()

        return {
            "status": "success",
            "message": f"Task {task_id} cancelled successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queue/status", response_model=Dict[str, Any])
async def get_queue_status(
    agent_manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """
    Get current task queue status.

    Args:
        agent_manager: Agent manager instance

    Returns:
        Dict[str, Any]: Queue status
    """
    try:
        return {
            "total_tasks": len(agent_manager.task_queue),
            "pending_tasks": len(
                [
                    task
                    for task in agent_manager.task_queue.values()
                    if task["task"].status == "pending"
                ]
            ),
            "running_tasks": len(
                [
                    task
                    for task in agent_manager.task_queue.values()
                    if task["task"].status == "running"
                ]
            ),
            "completed_tasks": len(
                [
                    task
                    for task in agent_manager.task_queue.values()
                    if task["task"].status == "completed"
                ]
            ),
            "failed_tasks": len(
                [
                    task
                    for task in agent_manager.task_queue.values()
                    if task["task"].status == "failed"
                ]
            ),
            "cancelled_tasks": len(
                [
                    task
                    for task in agent_manager.task_queue.values()
                    if task["task"].status == "cancelled"
                ]
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/queue/clear")
async def clear_task_queue(
    status: Optional[str] = None, agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Clear tasks from the queue.

    Args:
        status: Optional status filter for clearing
        agent_manager: Agent manager instance

    Returns:
        Dict[str, str]: Clear operation status
    """
    try:
        if status:
            # Clear tasks with specific status
            tasks_to_remove = [
                task_id
                for task_id, task_data in agent_manager.task_queue.items()
                if task_data["task"].status == status
            ]
            for task_id in tasks_to_remove:
                del agent_manager.task_queue[task_id]
        else:
            # Clear all tasks
            agent_manager.task_queue.clear()

        return {"status": "success", "message": "Task queue cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
