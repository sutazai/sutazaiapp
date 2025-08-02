#!/usr/bin/env python3
"""
SutazAI Task Scheduler - Autonomous Task Management and Orchestration
"""

import os
import asyncio
import logging
import json
import schedule
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import threading

import asyncpg
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Task Scheduler", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict = None
    result: Dict = None
    error_message: str = None
    retry_count: int = 0
    max_retries: int = 3

class TaskRequest(BaseModel):
    name: str
    description: str
    task_type: str
    priority: str = "normal"
    scheduled_at: Optional[str] = None
    parameters: Optional[Dict] = None
    max_retries: int = 3

class TaskScheduler:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://sutazai:sutazai_password@postgres:5432/sutazai")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.backend_url = os.getenv("BACKEND_URL", "http://backend:8000")
        self.jarvis_url = os.getenv("JARVIS_URL", "http://jarvis-ai:8120")
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
        
        self.db_pool = None
        self.redis = None
        self.running_tasks = set()
        self.scheduler_running = False
        
    async def init_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                self.db_url,
                min_size=1,
                max_size=10
            )
            
            # Initialize Redis connection
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            
            # Create tables if they don't exist
            await self.create_tables()
            
            logger.info("Database and Redis connections initialized")
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    task_type VARCHAR(100) NOT NULL,
                    priority INTEGER DEFAULT 2,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT NOW(),
                    scheduled_at TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    parameters JSONB,
                    result JSONB,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                );
                
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_scheduled_at ON tasks(scheduled_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
            """)
    
    async def create_task(self, task_request: TaskRequest) -> str:
        """Create a new task"""
        task_id = None
        
        # Parse scheduled time if provided
        scheduled_at = None
        if task_request.scheduled_at:
            try:
                scheduled_at = datetime.fromisoformat(task_request.scheduled_at)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid scheduled_at format")
        
        # Map priority string to enum
        priority_map = {
            "low": TaskPriority.LOW.value,
            "normal": TaskPriority.NORMAL.value,
            "high": TaskPriority.HIGH.value,
            "critical": TaskPriority.CRITICAL.value
        }
        priority = priority_map.get(task_request.priority.lower(), TaskPriority.NORMAL.value)
        
        async with self.db_pool.acquire() as conn:
            task_id = await conn.fetchval("""
                INSERT INTO tasks (name, description, task_type, priority, scheduled_at, parameters, max_retries)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """, task_request.name, task_request.description, task_request.task_type,
                priority, scheduled_at, json.dumps(task_request.parameters or {}),
                task_request.max_retries)
        
        # Add to Redis task queue if immediate execution
        if not scheduled_at:
            await self.redis.lpush("task_queue", str(task_id))
        
        logger.info(f"Created task {task_id}: {task_request.name}")
        return str(task_id)
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task by ID"""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
            if row:
                return dict(row)
        return None
    
    async def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """List tasks with optional status filter"""
        query = "SELECT * FROM tasks"
        params = []
        
        if status:
            query += " WHERE status = $1"
            params.append(status)
        
        query += " ORDER BY priority DESC, created_at ASC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def update_task_status(self, task_id: str, status: TaskStatus, 
                               result: Dict = None, error_message: str = None):
        """Update task status"""
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            if status == TaskStatus.RUNNING:
                await conn.execute("""
                    UPDATE tasks SET status = $1, started_at = $2 WHERE id = $3
                """, status.value, now, task_id)
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                await conn.execute("""
                    UPDATE tasks SET status = $1, completed_at = $2, result = $3, error_message = $4
                    WHERE id = $5
                """, status.value, now, json.dumps(result or {}), error_message, task_id)
            else:
                await conn.execute("""
                    UPDATE tasks SET status = $1 WHERE id = $2
                """, status.value, task_id)
        
        # Remove from running tasks set
        if task_id in self.running_tasks:
            self.running_tasks.remove(task_id)
    
    async def execute_task(self, task_id: str):
        """Execute a specific task"""
        if task_id in self.running_tasks:
            logger.warning(f"Task {task_id} is already running")
            return
        
        if len(self.running_tasks) >= self.max_concurrent_tasks:
            logger.warning(f"Maximum concurrent tasks reached ({self.max_concurrent_tasks})")
            return
        
        task = await self.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        if task["status"] != TaskStatus.PENDING.value:
            logger.warning(f"Task {task_id} is not in pending status")
            return
        
        self.running_tasks.add(task_id)
        await self.update_task_status(task_id, TaskStatus.RUNNING)
        
        logger.info(f"Executing task {task_id}: {task['name']}")
        
        try:
            result = await self.execute_task_by_type(task)
            await self.update_task_status(task_id, TaskStatus.COMPLETED, result)
            logger.info(f"Task {task_id} completed successfully")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task_id} failed: {error_msg}")
            
            # Increment retry count
            retry_count = task["retry_count"] + 1
            if retry_count <= task["max_retries"]:
                # Retry the task
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE tasks SET retry_count = $1, status = 'pending'
                        WHERE id = $2
                    """, retry_count, task_id)
                
                # Add back to queue for retry
                await self.redis.lpush("task_queue", task_id)
                logger.info(f"Task {task_id} queued for retry ({retry_count}/{task['max_retries']})")
            else:
                await self.update_task_status(task_id, TaskStatus.FAILED, error_message=error_msg)
    
    async def execute_task_by_type(self, task: Dict) -> Dict:
        """Execute task based on its type"""
        task_type = task["task_type"]
        parameters = json.loads(task["parameters"]) if task["parameters"] else {}
        
        if task_type == "jarvis_command":
            return await self.execute_jarvis_command(task, parameters)
        elif task_type == "backend_operation":
            return await self.execute_backend_operation(task, parameters)
        elif task_type == "model_training":
            return await self.execute_model_training(task, parameters)
        elif task_type == "data_processing":
            return await self.execute_data_processing(task, parameters)
        elif task_type == "system_maintenance":
            return await self.execute_system_maintenance(task, parameters)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def execute_jarvis_command(self, task: Dict, parameters: Dict) -> Dict:
        """Execute JARVIS command"""
        try:
            response = requests.post(
                f"{self.jarvis_url}/command",
                json={
                    "command": parameters.get("command", ""),
                    "context": parameters.get("context", {}),
                    "voice_enabled": parameters.get("voice_enabled", False)
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"JARVIS command failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to execute JARVIS command: {e}")
    
    async def execute_backend_operation(self, task: Dict, parameters: Dict) -> Dict:
        """Execute backend operation"""
        try:
            endpoint = parameters.get("endpoint", "/")
            method = parameters.get("method", "GET")
            data = parameters.get("data", {})
            
            response = requests.request(
                method,
                f"{self.backend_url}{endpoint}",
                json=data,
                timeout=30
            )
            
            if response.status_code < 400:
                return {"status": "success", "response": response.json()}
            else:
                raise Exception(f"Backend operation failed: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to execute backend operation: {e}")
    
    async def execute_model_training(self, task: Dict, parameters: Dict) -> Dict:
        """Execute model training task"""
        # Placeholder for model training logic
        await asyncio.sleep(5)  # Simulate training time
        return {
            "status": "success",
            "model": parameters.get("model_name", "unknown"),
            "epochs": parameters.get("epochs", 1),
            "training_time": 5
        }
    
    async def execute_data_processing(self, task: Dict, parameters: Dict) -> Dict:
        """Execute data processing task"""
        # Placeholder for data processing logic
        await asyncio.sleep(2)  # Simulate processing time
        return {
            "status": "success",
            "processed_items": parameters.get("item_count", 0),
            "processing_time": 2
        }
    
    async def execute_system_maintenance(self, task: Dict, parameters: Dict) -> Dict:
        """Execute system maintenance task"""
        # Placeholder for maintenance logic
        await asyncio.sleep(1)  # Simulate maintenance time
        return {
            "status": "success",
            "maintenance_type": parameters.get("type", "general"),
            "maintenance_time": 1
        }
    
    async def process_task_queue(self):
        """Process tasks from the queue"""
        while self.scheduler_running:
            try:
                # Get task from queue (blocking with timeout)
                task_id = await self.redis.brpop("task_queue", timeout=5)
                
                if task_id:
                    task_id = task_id[1].decode("utf-8")
                    await self.execute_task(task_id)
                
                # Process scheduled tasks
                await self.process_scheduled_tasks()
                
            except Exception as e:
                logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(5)
    
    async def process_scheduled_tasks(self):
        """Process scheduled tasks that are due"""
        now = datetime.now()
        
        async with self.db_pool.acquire() as conn:
            due_tasks = await conn.fetch("""
                SELECT id FROM tasks 
                WHERE status = 'pending' 
                AND scheduled_at IS NOT NULL 
                AND scheduled_at <= $1
                ORDER BY priority DESC, scheduled_at ASC
                LIMIT 10
            """, now)
            
            for task in due_tasks:
                task_id = str(task["id"])
                await self.redis.lpush("task_queue", task_id)
                
                # Clear scheduled_at to prevent reprocessing
                await conn.execute("""
                    UPDATE tasks SET scheduled_at = NULL WHERE id = $1
                """, task["id"])

# Initialize scheduler
scheduler = TaskScheduler()

@app.on_event("startup")
async def startup_event():
    """Initialize scheduler on startup"""
    await scheduler.init_connections()
    
    # Start task processing in background
    scheduler.scheduler_running = True
    asyncio.create_task(scheduler.process_task_queue())
    
    logger.info("Task Scheduler started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.scheduler_running = False
    
    if scheduler.db_pool:
        await scheduler.db_pool.close()
    
    if scheduler.redis:
        await scheduler.redis.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        async with scheduler.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        # Check Redis connection
        await scheduler.redis.ping()
        
        return {
            "status": "healthy",
            "service": "Task Scheduler",
            "timestamp": datetime.now().isoformat(),
            "running_tasks": len(scheduler.running_tasks),
            "max_concurrent": scheduler.max_concurrent_tasks,
            "queue_processor": scheduler.scheduler_running
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.post("/tasks")
async def create_task(task_request: TaskRequest):
    """Create a new task"""
    task_id = await scheduler.create_task(task_request)
    return {"task_id": task_id, "status": "created"}

@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, limit: int = 100):
    """List tasks"""
    tasks = await scheduler.list_tasks(status, limit)
    return {"tasks": tasks, "count": len(tasks)}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task"""
    task = await scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a task"""
    task = await scheduler.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(status_code=400, detail="Task cannot be cancelled")
    
    await scheduler.update_task_status(task_id, TaskStatus.CANCELLED)
    return {"message": "Task cancelled"}

@app.get("/stats")
async def get_stats():
    """Get scheduler statistics"""
    stats = {
        "running_tasks": len(scheduler.running_tasks),
        "max_concurrent_tasks": scheduler.max_concurrent_tasks,
        "scheduler_running": scheduler.scheduler_running,
        "timestamp": datetime.now().isoformat()
    }
    
    # Get task counts by status
    async with scheduler.db_pool.acquire() as conn:
        status_counts = await conn.fetch("""
            SELECT status, COUNT(*) as count 
            FROM tasks 
            GROUP BY status
        """)
        
        stats["task_counts"] = {row["status"]: row["count"] for row in status_counts}
    
    return stats

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("Starting SutazAI Task Scheduler...")
    logger.info(f"Scheduler URL: http://{host}:{port}")
    logger.info(f"Max concurrent tasks: {scheduler.max_concurrent_tasks}")
    
    uvicorn.run(app, host=host, port=port) 