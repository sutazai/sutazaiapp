#!/usr/bin/env python3
"""
Task Assignment Coordinator - Real Implementation with RabbitMQ
Manages task queuing, priority scheduling, and assignment strategies
"""
import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import asynccontextmanager
from collections import deque
from dataclasses import dataclass, field
import heapq
import uuid

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append('/app')
sys.path.append('/opt/sutazaiapp/agents')

from core.messaging import (
    RabbitMQClient, MessageProcessor,
    TaskMessage, StatusMessage, 
    MessageType, Priority
)

# Import metrics module
try:
    from agents.core.metrics import AgentMetrics, setup_metrics_endpoint
except ImportError:
    # Fallback if running in container
    sys.path.append('/app/agents')
    from core.metrics import AgentMetrics, setup_metrics_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "task-assignment-coordinator"
REDIS_QUEUE_KEY = "task:queue"
REDIS_PROCESSING_KEY = "task:processing"
TASK_TTL_SECONDS = 3600
MAX_QUEUE_SIZE = 10000

# Data Models
@dataclass(order=True)
class PrioritizedTask:
    """Task with priority for heap queue"""
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    task: TaskMessage = field(compare=False)

class TaskMetrics(BaseModel):
    """Task processing metrics"""
    total_received: int = 0
    total_assigned: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_timeout: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0
    queue_depth: int = 0

class AssignmentStrategy(BaseModel):
    """Task assignment strategy configuration"""
    strategy_type: str = "round_robin"  # round_robin, least_loaded, capability_match, priority_based
    load_threshold: float = 0.8
    timeout_seconds: int = 300
    max_retries: int = 3
    batch_size: int = 10

class QueuedTask(BaseModel):
    """Task in queue with metadata"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: Priority
    source_agent: str
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    previous_assignments: List[str] = []
    timeout_at: Optional[datetime] = None


class TaskAssignmentMessageProcessor(MessageProcessor):
    """Message processor for task assignment coordinator"""
    
    def __init__(self, coordinator):
        super().__init__(AGENT_ID)
        self.coordinator = coordinator
        
    async def handle_task_request(self, message: Dict[str, Any]):
        """Handle incoming task requests"""
        try:
            task_msg = TaskMessage(**message)
            
            # Add to priority queue
            await self.coordinator.enqueue_task(task_msg)
            
            # Trigger assignment processing
            asyncio.create_task(self.coordinator.process_assignments())
            
            logger.info(f"Queued task {task_msg.task_id} with priority {task_msg.priority}")
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
            await self.rabbitmq_client.publish_error(
                error_code="TASK_QUEUE_ERROR",
                error_message=str(e),
                original_message_id=message.get("message_id")
            )
    
    async def handle_status_update(self, message: Dict[str, Any]):
        """Handle task status updates"""
        try:
            status_msg = StatusMessage(**message)
            
            # Update task metrics
            await self.coordinator.update_metrics(status_msg)
            
            # Handle failed tasks for retry
            if status_msg.status == "failed":
                await self.coordinator.handle_failed_task(status_msg.task_id)
            
        except Exception as e:
            logger.error(f"Error handling status update: {e}")
    
    async def handle_agent_registration(self, message: Dict[str, Any]):
        """Handle agent registration messages"""
        try:
            agent_id = message.get("source_agent")
            capabilities = message.get("payload", {}).get("capabilities", [])
            
            await self.coordinator.register_agent_capabilities(agent_id, capabilities)
            
        except Exception as e:
            logger.error(f"Error handling agent registration: {e}")


class TaskAssignmentCoordinator:
    """Main task assignment coordinator implementation"""
    
    def __init__(self):
        self.redis_client = None
        self.message_processor = None
        self.task_queue: List[PrioritizedTask] = []  # Min heap
        self.processing_tasks: Dict[str, QueuedTask] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_load: Dict[str, float] = {}
        self.metrics = TaskMetrics()
        self.strategy = AssignmentStrategy()
        self.running = False
        self.assignment_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the coordinator"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Initialize message processor
            self.message_processor = TaskAssignmentMessageProcessor(self)
            
            # Connect RabbitMQ client first
            await self.message_processor.rabbitmq_client.connect()
            
            # Register ALL handlers before starting consumption
            self.message_processor.rabbitmq_client.register_handler(
                MessageType.TASK_REQUEST,
                self.message_processor.handle_task_request
            )
            self.message_processor.rabbitmq_client.register_handler(
                MessageType.TASK_STATUS,
                self.message_processor.handle_status_update
            )
            self.message_processor.rabbitmq_client.register_handler(
                MessageType.AGENT_REGISTRATION,
                self.message_processor.handle_agent_registration
            )
            
            # Start consuming messages in background (NON-BLOCKING)
            self.message_processor.consumer_task = await self.message_processor.rabbitmq_client.start_consuming(
                self.message_processor.handle_message
            )
            
            logger.info("Message processor started in background")
            
            # Load queued tasks from Redis
            await self.load_queued_tasks()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self.assignment_scheduler())
            asyncio.create_task(self.timeout_monitor())
            asyncio.create_task(self.metrics_reporter())
            
            logger.info("Task Assignment Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize coordinator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the coordinator"""
        self.running = False
        
        # Save queue state to Redis
        await self.save_queue_state()
        
        if self.message_processor:
            await self.message_processor.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Task Assignment Coordinator shutdown complete")
    
    async def enqueue_task(self, task_msg: TaskMessage):
        """Add task to priority queue"""
        async with self.assignment_lock:
            if len(self.task_queue) >= MAX_QUEUE_SIZE:
                raise Exception(f"Queue full (max {MAX_QUEUE_SIZE} tasks)")
            
            # Create prioritized task (negative priority for min heap)
            prioritized_task = PrioritizedTask(
                priority=-task_msg.priority,  # Negative for max priority
                timestamp=datetime.utcnow().timestamp(),
                task=task_msg
            )
            
            # Add to heap queue
            heapq.heappush(self.task_queue, prioritized_task)
            
            # Update metrics
            self.metrics.total_received += 1
            self.metrics.queue_depth = len(self.task_queue)
            
            # Store in Redis
            await self.redis_client.zadd(
                REDIS_QUEUE_KEY,
                {task_msg.task_id: task_msg.priority}
            )
            await self.redis_client.hset(
                f"{REDIS_QUEUE_KEY}:tasks",
                task_msg.task_id,
                task_msg.json()
            )
    
    async def dequeue_task(self) -> Optional[TaskMessage]:
        """Get next task from priority queue"""
        async with self.assignment_lock:
            if not self.task_queue:
                return None
            
            prioritized_task = heapq.heappop(self.task_queue)
            task = prioritized_task.task
            
            # Remove from Redis queue
            await self.redis_client.zrem(REDIS_QUEUE_KEY, task.task_id)
            await self.redis_client.hdel(f"{REDIS_QUEUE_KEY}:tasks", task.task_id)
            
            # Update metrics
            self.metrics.queue_depth = len(self.task_queue)
            
            return task
    
    async def process_assignments(self):
        """Process task assignments"""
        try:
            batch_count = 0
            
            while batch_count < self.strategy.batch_size and self.task_queue:
                task = await self.dequeue_task()
                if not task:
                    break
                
                # Select agent based on strategy
                agent_id = await self.select_agent(task)
                
                if not agent_id:
                    # No suitable agent, requeue with lower priority
                    task.priority = max(0, task.priority - 1)
                    await self.enqueue_task(task)
                    continue
                
                # Create queued task
                queued_task = QueuedTask(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    payload=task.payload,
                    priority=Priority(task.priority),
                    source_agent=task.source_agent,
                    timeout_at=datetime.utcnow() + timedelta(seconds=self.strategy.timeout_seconds)
                )
                
                # Track processing task
                self.processing_tasks[task.task_id] = queued_task
                
                # Forward to orchestrator for actual assignment
                await self.message_processor.rabbitmq_client.publish_message(
                    task,
                    routing_key=f"orchestrator.assign.{agent_id}"
                )
                
                # Update metrics
                self.metrics.total_assigned += 1
                
                # Update agent load
                if agent_id in self.agent_load:
                    self.agent_load[agent_id] = min(1.0, self.agent_load[agent_id] + 0.1)
                
                batch_count += 1
                
                logger.info(f"Assigned task {task.task_id} to agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error processing assignments: {e}")
    
    async def select_agent(self, task: TaskMessage) -> Optional[str]:
        """Select agent based on assignment strategy"""
        try:
            available_agents = [
                agent_id for agent_id, load in self.agent_load.items()
                if load < self.strategy.load_threshold
            ]
            
            if not available_agents:
                return None
            
            if self.strategy.strategy_type == "round_robin":
                # Simple round-robin
                return available_agents[0]
                
            elif self.strategy.strategy_type == "least_loaded":
                # Select least loaded agent
                return min(available_agents, key=lambda a: self.agent_load.get(a, 0))
                
            elif self.strategy.strategy_type == "capability_match":
                # Match task type to agent capabilities
                best_match = None
                best_score = 0
                
                for agent_id in available_agents:
                    capabilities = self.agent_capabilities.get(agent_id, [])
                    if task.task_type in capabilities:
                        score = len(capabilities)  # More capabilities = more versatile
                        if score > best_score:
                            best_match = agent_id
                            best_score = score
                
                return best_match or available_agents[0]
                
            elif self.strategy.strategy_type == "priority_based":
                # High priority tasks go to best agents
                if task.priority >= Priority.HIGH:
                    # Select agent with best success rate (simplified)
                else:
                    return available_agents[-1]
            
            return available_agents[0]
            
        except Exception as e:
            logger.error(f"Error selecting agent: {e}")
            return None
    
    async def handle_failed_task(self, task_id: str):
        """Handle failed task for potential retry"""
        try:
            if task_id in self.processing_tasks:
                queued_task = self.processing_tasks[task_id]
                
                if queued_task.retry_count < self.strategy.max_retries:
                    # Requeue for retry
                    queued_task.retry_count += 1
                    
                    # Create new task message
                    task_msg = TaskMessage(
                        message_id=f"{task_id}_retry_{queued_task.retry_count}",
                        message_type=MessageType.TASK_REQUEST,
                        source_agent=queued_task.source_agent,
                        task_id=task_id,
                        task_type=queued_task.task_type,
                        payload=queued_task.payload,
                        priority=queued_task.priority,
                        retry_count=queued_task.retry_count
                    )
                    
                    await self.enqueue_task(task_msg)
                    logger.info(f"Requeued failed task {task_id} for retry {queued_task.retry_count}")
                else:
                    # Max retries exceeded
                    self.metrics.total_failed += 1
                    logger.error(f"Task {task_id} failed after {queued_task.retry_count} retries")
                
                # Remove from processing
                del self.processing_tasks[task_id]
            
        except Exception as e:
            logger.error(f"Error handling failed task: {e}")
    
    async def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """Register agent capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        self.agent_load[agent_id] = 0.0
        
        # Store in Redis
        await self.redis_client.hset(
            "agent:capabilities",
            agent_id,
            json.dumps(capabilities)
        )
        
        logger.info(f"Registered capabilities for agent {agent_id}: {capabilities}")
    
    async def update_metrics(self, status_msg: StatusMessage):
        """Update task processing metrics"""
        try:
            if status_msg.status == "completed":
                self.metrics.total_completed += 1
                
                # Update agent load
                if status_msg.source_agent in self.agent_load:
                    self.agent_load[status_msg.source_agent] = max(
                        0, 
                        self.agent_load[status_msg.source_agent] - 0.1
                    )
                
                # Calculate processing time if task was tracked
                if status_msg.task_id in self.processing_tasks:
                    queued_task = self.processing_tasks[status_msg.task_id]
                    processing_time = (datetime.utcnow() - queued_task.queued_at).total_seconds()
                    
                    # Update average (simple moving average)
                    self.metrics.average_processing_time = (
                        (self.metrics.average_processing_time * (self.metrics.total_completed - 1) +
                         processing_time) / self.metrics.total_completed
                    )
                    
                    del self.processing_tasks[status_msg.task_id]
            
            elif status_msg.status == "failed":
                # Will be handled by handle_failed_task
                pass
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def assignment_scheduler(self):
        """Background task for continuous assignment processing"""
        while self.running:
            try:
                if self.task_queue:
                    await self.process_assignments()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in assignment scheduler: {e}")
                await asyncio.sleep(5)
    
    async def timeout_monitor(self):
        """Monitor tasks for timeouts"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                timed_out_tasks = []
                
                for task_id, queued_task in self.processing_tasks.items():
                    if queued_task.timeout_at and current_time > queued_task.timeout_at:
                        timed_out_tasks.append(task_id)
                        self.metrics.total_timeout += 1
                        logger.warning(f"Task {task_id} timed out")
                
                # Handle timed out tasks
                for task_id in timed_out_tasks:
                    await self.handle_failed_task(task_id)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(30)
    
    async def metrics_reporter(self):
        """Report metrics periodically"""
        while self.running:
            try:
                # Store metrics in Redis
                await self.redis_client.hset(
                    "coordinator:metrics",
                    AGENT_ID,
                    json.dumps(self.metrics.dict())
                )
                
                logger.info(f"Metrics: Queue={self.metrics.queue_depth}, "
                          f"Assigned={self.metrics.total_assigned}, "
                          f"Completed={self.metrics.total_completed}, "
                          f"Failed={self.metrics.total_failed}")
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")
                await asyncio.sleep(120)
    
    async def load_queued_tasks(self):
        """Load queued tasks from Redis on startup"""
        try:
            # Get task IDs from sorted set
            task_ids = await self.redis_client.zrange(REDIS_QUEUE_KEY, 0, -1)
            
            for task_id in task_ids:
                # Get task data
                task_data = await self.redis_client.hget(f"{REDIS_QUEUE_KEY}:tasks", task_id)
                if task_data:
                    task_msg = TaskMessage(**json.loads(task_data))
                    
                    # Re-add to in-memory queue
                    prioritized_task = PrioritizedTask(
                        priority=-task_msg.priority,
                        timestamp=datetime.utcnow().timestamp(),
                        task=task_msg
                    )
                    heapq.heappush(self.task_queue, prioritized_task)
            
            self.metrics.queue_depth = len(self.task_queue)
            logger.info(f"Loaded {len(self.task_queue)} queued tasks from Redis")
            
        except Exception as e:
            logger.error(f"Error loading queued tasks: {e}")
    
    async def save_queue_state(self):
        """Save queue state to Redis on shutdown"""
        try:
            # Queue is already persisted in Redis during normal operation
            logger.info(f"Queue state saved: {len(self.task_queue)} tasks")
        except Exception as e:
            logger.error(f"Error saving queue state: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get coordinator status"""
        return {
            "status": "healthy",
            "queue_depth": self.metrics.queue_depth,
            "processing_tasks": len(self.processing_tasks),
            "total_assigned": self.metrics.total_assigned,
            "total_completed": self.metrics.total_completed,
            "total_failed": self.metrics.total_failed,
            "total_timeout": self.metrics.total_timeout,
            "average_processing_time": round(self.metrics.average_processing_time, 2),
            "registered_agents": len(self.agent_capabilities),
            "strategy": self.strategy.strategy_type
        }


# Global coordinator instance
coordinator = TaskAssignmentCoordinator()
metrics: Optional[AgentMetrics] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global metrics
    # Startup
    try:
        logger.info("Starting Task Assignment Coordinator lifespan")
        metrics = AgentMetrics("task_assignment_coordinator")
        setup_metrics_endpoint(app, metrics)
        await coordinator.initialize()
        logger.info("Task Assignment Coordinator startup complete")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Task Assignment Coordinator")
        await coordinator.shutdown()

# Create FastAPI app
app = FastAPI(
    title="Task Assignment Coordinator",
    description="Real implementation with priority queuing and RabbitMQ",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = await coordinator.get_status()
    return {
        "status": "healthy",
        "agent": AGENT_ID,
        "timestamp": datetime.utcnow().isoformat(),
        **status
    }

@app.get("/queue")
async def get_queue_status():
    """Get queue status"""
    return {
        "queue_depth": coordinator.metrics.queue_depth,
        "processing_count": len(coordinator.processing_tasks),
        "max_queue_size": MAX_QUEUE_SIZE,
        "tasks": [
            {
                "task_id": pt.task.task_id,
                "priority": -pt.priority,  # Convert back to positive
                "task_type": pt.task.task_type
            }
            for pt in coordinator.task_queue[:10]  # First 10 tasks
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get detailed statistics"""
    return coordinator.metrics.dict()

@app.post("/strategy")
async def update_strategy(strategy: AssignmentStrategy):
    """Update assignment strategy"""
    coordinator.strategy = strategy
    return {
        "status": "updated",
        "strategy": strategy.dict()
    }

@app.get("/agents")
async def list_agents():
    """List registered agents with capabilities"""
    agents = []
    for agent_id, capabilities in coordinator.agent_capabilities.items():
        agents.append({
            "agent_id": agent_id,
            "capabilities": capabilities,
            "current_load": coordinator.agent_load.get(agent_id, 0.0)
        })
    return {"agents": agents}

@app.get("/status")
async def get_coordinator_status():
    """Get detailed coordinator status"""
    return await coordinator.get_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8551"))
    uvicorn.run(app, host="0.0.0.0", port=port)