"""
High-Performance Background Task Queue
Handles long-running operations without blocking the main event loop
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Callable
import pickle

from app.core.connection_pool import get_redis

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class Task:
    """Task data structure"""
    id: str
    type: str
    payload: Dict[str, Any]
    status: TaskStatus
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


class BackgroundTaskQueue:
    """Advanced task queue with priority, retries, and persistence"""
    
    def __init__(self, num_workers: int = 5):
        self.num_workers = num_workers
        self.workers = []
        self.tasks = {}  # In-memory task storage
        self.handlers = {}  # Task type handlers
        self.running = False
        
        # Priority queues (higher priority = processed first)
        self.high_priority_queue = asyncio.Queue(maxsize=50)
        self.normal_priority_queue = asyncio.Queue(maxsize=100)
        self.low_priority_queue = asyncio.Queue(maxsize=200)
        
        self._stats = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_retried': 0,
            'avg_processing_time': 0
        }
        
    async def start(self):
        """Start the task queue workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
            
        # Start task persistence
        asyncio.create_task(self._persist_tasks())
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_completed_tasks())
        
        logger.info(f"Task queue started with {self.num_workers} workers")
        
    async def stop(self):
        """Stop the task queue"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Task queue stopped")
        
    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a task type"""
        self.handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
        
    async def create_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """Create a new task and add to queue"""
        
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            payload=payload,
            status=TaskStatus.PENDING,
            priority=priority,
            max_retries=max_retries
        )
        
        # Store task
        self.tasks[task_id] = task
        self._stats['tasks_created'] += 1
        
        # Add to appropriate queue based on priority
        if priority >= 10:
            await self.high_priority_queue.put(task_id)
        elif priority >= 5:
            await self.normal_priority_queue.put(task_id)
        else:
            await self.low_priority_queue.put(task_id)
            
        # Persist to Redis
        await self._persist_task(task)
        
        logger.debug(f"Created task {task_id} with type {task_type} and priority {priority}")
        
        return task_id
        
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result"""
        
        # Check in-memory first
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
            
        # Check Redis
        try:
            redis_client = await get_redis()
            task_data = await redis_client.get(f"task:{task_id}")
            
            if task_data:
                return pickle.loads(task_data)
                
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            
        return None
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        
        if task_id in self.tasks:
            task = self.tasks[task_id]
            
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                await self._persist_task(task)
                return True
                
        return False
        
    async def _worker(self, worker_id: int):
        """Background worker to process tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Try to get task from queues (priority order)
                task_id = None
                
                # Check high priority first
                try:
                    task_id = await asyncio.wait_for(
                        self.high_priority_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    pass
                    
                # Check normal priority
                if not task_id:
                    try:
                        task_id = await asyncio.wait_for(
                            self.normal_priority_queue.get(),
                            timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        pass
                        
                # Check low priority
                if not task_id:
                    try:
                        task_id = await asyncio.wait_for(
                            self.low_priority_queue.get(),
                            timeout=1.0
                        )
                    except asyncio.TimeoutError:
                        continue
                        
                # Process the task
                if task_id and task_id in self.tasks:
                    await self._process_task(self.tasks[task_id], worker_id)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Worker {worker_id} stopped")
        
    async def _process_task(self, task: Task, worker_id: int):
        """Process a single task"""
        
        logger.debug(f"Worker {worker_id} processing task {task.id}")
        
        # Update task status
        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        await self._persist_task(task)
        
        try:
            # Get handler for task type
            handler = self.handlers.get(task.type)
            
            if not handler:
                raise ValueError(f"No handler registered for task type: {task.type}")
                
            # Execute handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task.payload)
            else:
                # Run sync handler in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, handler, task.payload)
                
            # Update task with result
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            # Update stats
            self._stats['tasks_completed'] += 1
            processing_time = (task.completed_at - task.started_at).total_seconds()
            self._update_avg_processing_time(processing_time)
            
            logger.info(f"Task {task.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            
            task.error = str(e)
            task.retries += 1
            
            # Check if we should retry
            if task.retries < task.max_retries:
                task.status = TaskStatus.RETRYING
                self._stats['tasks_retried'] += 1
                
                # Re-queue with exponential backoff
                await asyncio.sleep(2 ** task.retries)
                
                # Re-add to queue based on priority
                if task.priority >= 10:
                    await self.high_priority_queue.put(task.id)
                elif task.priority >= 5:
                    await self.normal_priority_queue.put(task.id)
                else:
                    await self.low_priority_queue.put(task.id)
                    
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                self._stats['tasks_failed'] += 1
                
        finally:
            # Persist final task state
            await self._persist_task(task)
            
    async def _persist_task(self, task: Task):
        """Persist task to Redis"""
        try:
            redis_client = await get_redis()
            
            # Serialize task
            task_data = pickle.dumps(task.to_dict())
            
            # Store with TTL (7 days for completed tasks)
            ttl = 604800 if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] else 86400
            
            await redis_client.setex(
                f"task:{task.id}",
                ttl,
                task_data
            )
            
        except Exception as e:
            logger.error(f"Error persisting task: {e}")
            
    async def _persist_tasks(self):
        """Periodically persist all tasks to Redis"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                for task in self.tasks.values():
                    await self._persist_task(task)
                    
            except Exception as e:
                logger.error(f"Error in task persistence: {e}")
                
    async def _cleanup_completed_tasks(self):
        """Remove old completed tasks from memory"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                now = datetime.now()
                tasks_to_remove = []
                
                for task_id, task in self.tasks.items():
                    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        if task.completed_at and (now - task.completed_at) > timedelta(hours=1):
                            tasks_to_remove.append(task_id)
                            
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                    
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
                    
            except Exception as e:
                logger.error(f"Error in task cleanup: {e}")
                
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time"""
        current_avg = self._stats['avg_processing_time']
        completed = self._stats['tasks_completed']
        
        self._stats['avg_processing_time'] = (
            (current_avg * (completed - 1) + processing_time) / completed
        )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self._stats,
            'pending_high': self.high_priority_queue.qsize(),
            'pending_normal': self.normal_priority_queue.qsize(),
            'pending_low': self.low_priority_queue.qsize(),
            'in_memory_tasks': len(self.tasks),
            'num_workers': self.num_workers
        }


# Global task queue instance
_task_queue: Optional[BackgroundTaskQueue] = None


async def get_task_queue() -> BackgroundTaskQueue:
    """Get or create the global task queue"""
    global _task_queue
    
    if _task_queue is None:
        _task_queue = BackgroundTaskQueue(num_workers=5)
        await _task_queue.start()
        
    return _task_queue


async def create_background_task(
    task_type: str,
    payload: Dict[str, Any],
    priority: int = 0
) -> str:
    """Convenience function to create a background task"""
    queue = await get_task_queue()
    return await queue.create_task(task_type, payload, priority)