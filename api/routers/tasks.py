from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from uuid import uuid4

from agents import agent_orchestrator
from agents.base_agent import Task, TaskPriority, AgentStatus
from api.auth import get_current_user, require_admin
from api.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def list_tasks(
    status: Optional[str] = None,
    agent_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """List tasks with optional filtering."""
    try:
        # Get tasks from orchestrator queue and completed tasks
        tasks = []
        
        # Add pending tasks from orchestrator
        if hasattr(agent_orchestrator, 'task_queue'):
            for task in list(agent_orchestrator.task_queue.queue):
                if isinstance(task, Task):
                    task_dict = {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "agent_type": task.agent_type,
                        "parameters": task.parameters,
                        "created_at": task.created_at.isoformat(),
                        "result": getattr(task, 'result', None),
                        "error": getattr(task, 'error', None)
                    }
                    tasks.append(task_dict)
        
        # Add completed tasks from orchestrator history
        if hasattr(agent_orchestrator, 'completed_tasks'):
            for task in agent_orchestrator.completed_tasks:
                if isinstance(task, Task):
                    task_dict = {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "agent_type": task.agent_type,
                        "parameters": task.parameters,
                        "created_at": task.created_at.isoformat(),
                        "completed_at": getattr(task, 'completed_at', datetime.utcnow()).isoformat(),
                        "result": getattr(task, 'result', None),
                        "error": getattr(task, 'error', None)
                    }
                    tasks.append(task_dict)
        
        # Apply filters
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        if agent_type:
            tasks = [t for t in tasks if t["agent_type"] == agent_type]
        
        # Apply pagination
        total = len(tasks)
        tasks = tasks[offset:offset + limit]
        
        await db_manager.log_system_event(
            "info", "tasks", "Listed tasks",
            {"user": current_user.get("username"), "count": len(tasks), "total": total}
        )
        
        return {
            "tasks": tasks,
            "total": total,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{task_id}")
async def get_task(task_id: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get details about a specific task."""
    try:
        # Search in pending tasks
        if hasattr(agent_orchestrator, 'task_queue'):
            for task in list(agent_orchestrator.task_queue.queue):
                if isinstance(task, Task) and task.id == task_id:
                    return {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "agent_type": task.agent_type,
                        "parameters": task.parameters,
                        "created_at": task.created_at.isoformat(),
                        "result": getattr(task, 'result', None),
                        "error": getattr(task, 'error', None)
                    }
        
        # Search in completed tasks
        if hasattr(agent_orchestrator, 'completed_tasks'):
            for task in agent_orchestrator.completed_tasks:
                if isinstance(task, Task) and task.id == task_id:
                    return {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "agent_type": task.agent_type,
                        "parameters": task.parameters,
                        "created_at": task.created_at.isoformat(),
                        "completed_at": getattr(task, 'completed_at', datetime.utcnow()).isoformat(),
                        "result": getattr(task, 'result', None),
                        "error": getattr(task, 'error', None)
                    }
        
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_task(
    task_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create and submit a new task."""
    try:
        # Create task
        task = Task(
            id=str(uuid4()),
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            priority=TaskPriority(task_data.get("priority", "medium")),
            agent_type=task_data.get("agent_type", "auto"),
            parameters=task_data.get("parameters", {}),
            created_at=datetime.utcnow(),
            status=AgentStatus.PENDING
        )
        
        # Submit task to orchestrator
        task_id = await agent_orchestrator.submit_task(task)
        
        await db_manager.log_system_event(
            "info", "tasks", "Task created",
            {"user": current_user.get("username"), "task_id": task_id, "name": task.name}
        )
        
        return {
            "task_id": task_id,
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Cancel a pending task."""
    try:
        # Try to remove from pending queue
        if hasattr(agent_orchestrator, 'task_queue'):
            queue_items = list(agent_orchestrator.task_queue.queue)
            for i, task in enumerate(queue_items):
                if isinstance(task, Task) and task.id == task_id:
                    # Remove from queue
                    queue_items.pop(i)
                    agent_orchestrator.task_queue.queue.clear()
                    for remaining_task in queue_items:
                        agent_orchestrator.task_queue.put_nowait(remaining_task)
                    
                    await db_manager.log_system_event(
                        "info", "tasks", "Task cancelled",
                        {"user": current_user.get("username"), "task_id": task_id}
                    )
                    
                    return {
                        "task_id": task_id,
                        "status": "cancelled",
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be cancelled")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/summary")
async def get_task_stats(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get task statistics summary."""
    try:
        stats = {
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "total": 0
        }
        
        # Count pending tasks
        if hasattr(agent_orchestrator, 'task_queue'):
            stats["pending"] = agent_orchestrator.task_queue.qsize()
        
        # Count completed tasks
        if hasattr(agent_orchestrator, 'completed_tasks'):
            completed_tasks = agent_orchestrator.completed_tasks
            stats["completed"] = len([t for t in completed_tasks if getattr(t, 'error', None) is None])
            stats["failed"] = len([t for t in completed_tasks if getattr(t, 'error', None) is not None])
        
        stats["total"] = stats["pending"] + stats["completed"] + stats["failed"]
        
        return {
            "stats": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def create_batch_tasks(
    tasks_data: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Create multiple tasks in batch."""
    try:
        task_ids = []
        
        for task_data in tasks_data:
            task = Task(
                id=str(uuid4()),
                name=task_data.get("name", "Unnamed Task"),
                description=task_data.get("description", ""),
                priority=TaskPriority(task_data.get("priority", "medium")),
                agent_type=task_data.get("agent_type", "auto"),
                parameters=task_data.get("parameters", {}),
                created_at=datetime.utcnow(),
                status=AgentStatus.PENDING
            )
            
            task_id = await agent_orchestrator.submit_task(task)
            task_ids.append(task_id)
        
        await db_manager.log_system_event(
            "info", "tasks", "Batch tasks created",
            {"user": current_user.get("username"), "count": len(task_ids)}
        )
        
        return {
            "task_ids": task_ids,
            "count": len(task_ids),
            "status": "created",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating batch tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
