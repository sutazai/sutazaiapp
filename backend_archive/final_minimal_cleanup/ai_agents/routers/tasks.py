"""
Agent Task Router

This module provides REST API endpoints for managing agent tasks and task execution.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.utils.startup import initialize_agent_system

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskInput(BaseModel):
    agent_id: str
    task_type: str
    parameters: Dict[str, Any]
    # Add other relevant fields like priority, dependencies etc.


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str # Consider using an Enum
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str


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


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_task(
    task_input: TaskInput,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> TaskResponse:
    """Submit a new task to a specific agent."""
    try:
        # Assuming AgentManager has execute_task or similar method for async execution
        if not hasattr(agent_manager, 'execute_task'):
            raise HTTPException(status_code=501, detail="Task execution not implemented")

        # AgentManager.execute_task might return immediately with a task ID
        # or handle async execution internally.
        # Adjust based on actual AgentManager implementation.
        task_info = agent_manager.execute_task(
            agent_id=task_input.agent_id,
            task=task_input.dict() # Pass task details
        )
        # Assume task_info is a dict like {"task_id": "xyz", "status": "PENDING"}
        return TaskResponse(**task_info)

    except ValueError as e: # e.g., Agent not found
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error submitting task to agent {task_input.agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit task")


@router.get("/{task_id}", response_model=TaskStatusResponse)
def get_task_status(
    task_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> TaskStatusResponse:
    """Get the status and result (if available) of a specific task."""
    try:
        # Assuming AgentManager has get_task_status method
        if not hasattr(agent_manager, 'get_task_status'):
            raise HTTPException(status_code=501, detail="get_task_status not implemented")

        status = agent_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        # Adapt the returned dict/object to TaskStatusResponse
        return TaskStatusResponse(**status)

    except ValueError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting status for task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve task status")


@router.post("/{task_id}/cancel")
def cancel_task(
    task_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Cancel a running or pending task."""
    try:
        # Assuming AgentManager has cancel_task method
        if not hasattr(agent_manager, 'cancel_task'):
            raise HTTPException(status_code=501, detail="cancel_task not implemented")

        success = agent_manager.cancel_task(task_id)
        if not success:
             # Could be because task doesn't exist or is already completed/cancelled
             raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")

        return {"status": "success", "message": f"Task {task_id} cancellation requested"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel task")


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
