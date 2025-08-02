#!/usr/bin/env python3
"""
Dify Workflow Manager for SutazAI automation System
Manages workflow execution, monitoring, and optimization
"""

import os
import sys
import json
import time
import redis
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class WorkflowTask:
    id: str
    workflow_id: str
    agent_id: str
    action: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    timeout: int
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Dict[str, Any] = None
    error_message: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class WorkflowExecution:
    id: str
    template_id: str
    name: str
    input_data: Dict[str, Any]
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: List[WorkflowTask] = None
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    execution_time_ms: int = None
    result: Dict[str, Any] = None
    error_message: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.tasks is None:
            self.tasks = []

class DifyWorkflowManager:
    """Manages Dify workflows for SutazAI agent system"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/workflows"):
        self.config_path = config_path
        self.redis_client = redis.Redis(
            host='redis', 
            port=6379, 
            password='redis_password',
            decode_responses=True
        )
        
        # API endpoints
        self.agent_registry_url = "http://agent-registry:8300/api/v1"
        self.backend_api_url = "http://backend:8000/api/v1"
        self.message_bus_url = "http://agent-message-bus:8299/api/v1"
        
        # Runtime state
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.running = False
        
        # Load configurations
        self.load_configurations()
        
    def load_configurations(self):
        """Load workflow configurations"""
        try:
            with open(f"{self.config_path}/dify_config.yaml", 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
                
            with open(f"{self.config_path}/templates/agent_coordination_patterns.json", 'r') as f:
                self.workflow_templates = json.load(f)
                
            with open(f"{self.config_path}/automation/task_distribution_router.json", 'r') as f:
                self.routing_config = json.load(f)
                
            logger.info("Configurations loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise
            
    async def start(self):
        """Start the workflow manager"""
        logger.info("Starting Dify Workflow Manager...")
        self.running = True
        
        # Initialize components
        await self.initialize_agent_registry()
        await self.load_agent_capabilities()
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self.workflow_executor()),
            asyncio.create_task(self.health_monitor()),
            asyncio.create_task(self.performance_collector()),
            asyncio.create_task(self.cleanup_completed_workflows())
        ]
        
        logger.info("Workflow Manager started successfully")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down Workflow Manager...")
            self.running = False
            for task in tasks:
                task.cancel()
                
    async def initialize_agent_registry(self):
        """Initialize connection to agent registry"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.agent_registry_url}/health") as response:
                    if response.status == 200:
                        logger.info("Connected to agent registry")
                    else:
                        logger.warning(f"Agent registry health check failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to connect to agent registry: {e}")
            
    async def load_agent_capabilities(self):
        """Load agent capabilities from registry"""
        try:
            agent_profiles = self.routing_config["agent_profiles"]
            for agent_id, profile in agent_profiles.items():
                self.agent_capabilities[agent_id] = profile["capabilities"]
                
            logger.info(f"Loaded capabilities for {len(self.agent_capabilities)} agents")
            
        except Exception as e:
            logger.error(f"Failed to load agent capabilities: {e}")
            
    async def create_workflow_execution(self, template_id: str, input_data: Dict[str, Any]) -> str:
        """Create a new workflow execution"""
        try:
            # Find workflow template
            template = None
            for tmpl in self.workflow_templates["workflow_templates"]:
                if tmpl["id"] == template_id:
                    template = tmpl
                    break
                    
            if not template:
                raise ValueError(f"Workflow template {template_id} not found")
                
            # Create execution
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                id=execution_id,
                template_id=template_id,
                name=template["name"],
                input_data=input_data
            )
            
            # Generate tasks from template
            await self.generate_tasks_from_template(execution, template)
            
            # Store execution
            self.active_executions[execution_id] = execution
            
            # Queue for execution
            await self.queue_workflow(execution_id)
            
            logger.info(f"Created workflow execution: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow execution: {e}")
            raise
            
    async def generate_tasks_from_template(self, execution: WorkflowExecution, template: Dict):
        """Generate tasks from workflow template"""
        try:
            for node in template["nodes"]:
                if node["type"] == "agent":
                    task = WorkflowTask(
                        id=str(uuid.uuid4()),
                        workflow_id=execution.id,
                        agent_id=node["agent_id"],
                        action=node["config"].get("action", "execute"),
                        input_data=execution.input_data,
                        priority=TaskPriority(node.get("priority", "medium")),
                        timeout=node.get("timeout", 300),
                        max_retries=node.get("max_retries", 3)
                    )
                    execution.tasks.append(task)
                    
            logger.info(f"Generated {len(execution.tasks)} tasks for workflow {execution.id}")
            
        except Exception as e:
            logger.error(f"Failed to generate tasks: {e}")
            raise
            
    async def queue_workflow(self, execution_id: str):
        """Queue workflow for execution"""
        try:
            workflow_data = {
                "execution_id": execution_id,
                "priority": "medium",
                "queued_at": datetime.now().isoformat()
            }
            
            self.redis_client.lpush("workflow_queue", json.dumps(workflow_data))
            logger.info(f"Queued workflow {execution_id} for execution")
            
        except Exception as e:
            logger.error(f"Failed to queue workflow: {e}")
            raise
            
    async def workflow_executor(self):
        """Main workflow execution loop"""
        logger.info("Starting workflow executor")
        
        while self.running:
            try:
                # Get next workflow from queue
                workflow_data = self.redis_client.brpop("workflow_queue", timeout=5)
                
                if workflow_data:
                    _, data_json = workflow_data
                    data = json.loads(data_json)
                    execution_id = data["execution_id"]
                    
                    # Execute workflow
                    await self.execute_workflow(execution_id)
                    
            except Exception as e:
                logger.error(f"Workflow executor error: {e}")
                await asyncio.sleep(1)
                
    async def execute_workflow(self, execution_id: str):
        """Execute a specific workflow"""
        try:
            execution = self.active_executions.get(execution_id)
            if not execution:
                logger.error(f"Workflow execution {execution_id} not found")
                return
                
            logger.info(f"Starting execution of workflow {execution_id}")
            execution.status = WorkflowStatus.RUNNING
            execution.started_at = datetime.now()
            
            # Execute tasks based on workflow pattern
            template = self.get_template_by_id(execution.template_id)
            pattern = template["pattern"]
            
            if pattern == "sequential":
                await self.execute_sequential_workflow(execution)
            elif pattern == "parallel":
                await self.execute_parallel_workflow(execution)
            elif pattern == "hierarchical":
                await self.execute_hierarchical_workflow(execution)
            elif pattern == "collaborative":
                await self.execute_collaborative_workflow(execution)
            else:
                raise ValueError(f"Unknown workflow pattern: {pattern}")
                
            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.execution_time_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            
            logger.info(f"Workflow {execution_id} completed in {execution.execution_time_ms}ms")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
    async def execute_sequential_workflow(self, execution: WorkflowExecution):
        """Execute workflow tasks sequentially"""
        for task in execution.tasks:
            result = await self.execute_task(task)
            if not result:
                raise Exception(f"Task {task.id} failed")
                
    async def execute_parallel_workflow(self, execution: WorkflowExecution):
        """Execute workflow tasks in parallel"""
        tasks = [self.execute_task(task) for task in execution.tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise Exception(f"Task {execution.tasks[i].id} failed: {result}")
                
    async def execute_hierarchical_workflow(self, execution: WorkflowExecution):
        """Execute workflow with hierarchical task dependencies"""
        # Implementation would handle task dependencies
        await self.execute_sequential_workflow(execution)  # Simplified
        
    async def execute_collaborative_workflow(self, execution: WorkflowExecution):
        """Execute workflow with collaborative agent interaction"""
        # Implementation would handle agent collaboration
        await self.execute_parallel_workflow(execution)  # Simplified
        
    async def execute_task(self, task: WorkflowTask) -> bool:
        """Execute a single task"""
        try:
            logger.info(f"Executing task {task.id} on agent {task.agent_id}")
            
            task.status = WorkflowStatus.RUNNING
            task.started_at = datetime.now()
            
            # Find best agent for task
            selected_agent = await self.select_agent_for_task(task)
            if not selected_agent:
                raise Exception(f"No suitable agent found for task {task.id}")
                
            # Execute task on agent
            result = await self.call_agent(selected_agent, task)
            
            task.status = WorkflowStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            logger.info(f"Task {task.id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                logger.info(f"Retrying task {task.id} (attempt {task.retry_count + 1})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self.execute_task(task)
            else:
                task.status = WorkflowStatus.FAILED
                task.error_message = str(e)
                task.completed_at = datetime.now()
                return False
                
    async def select_agent_for_task(self, task: WorkflowTask) -> Optional[str]:
        """Select best agent for a task using routing logic"""
        try:
            # Get available agents with required capabilities
            suitable_agents = []
            
            for agent_id, capabilities in self.agent_capabilities.items():
                if task.agent_id == agent_id or any(cap in capabilities for cap in ["general", task.action]):
                    # Check agent availability and load
                    agent_load = await self.get_agent_load(agent_id)
                    if agent_load < 0.8:  # 80% load threshold
                        suitable_agents.append((agent_id, agent_load))
                        
            if not suitable_agents:
                return None
                
            # Select agent with lowest load
            suitable_agents.sort(key=lambda x: x[1])
            return suitable_agents[0][0]
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            return task.agent_id  # Fallback to specified agent
            
    async def get_agent_load(self, agent_id: str) -> float:
        """Get current load for an agent"""
        try:
            # This would query the agent registry for current load
            # For now, return a simulated load
            return 0.5  # 50% load
            
        except Exception as e:
            logger.error(f"Failed to get agent load: {e}")
            return 1.0  # Assume full load on error
            
    async def call_agent(self, agent_id: str, task: WorkflowTask) -> Dict[str, Any]:
        """Call an agent to execute a task"""
        try:
            agent_endpoint = f"{self.backend_api_url}/agents/{agent_id}/execute"
            
            payload = {
                "task_id": task.id,
                "action": task.action,
                "input_data": task.input_data,
                "timeout": task.timeout
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    agent_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=task.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Agent call failed: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Agent call failed: {e}")
            raise
            
    def get_template_by_id(self, template_id: str) -> Optional[Dict]:
        """Get template by ID"""
        for template in self.workflow_templates["workflow_templates"]:
            if template["id"] == template_id:
                return template
        return None
        
    async def health_monitor(self):
        """Monitor workflow system health"""
        while self.running:
            try:
                # Check system health metrics
                active_count = len([e for e in self.active_executions.values() 
                                 if e.status == WorkflowStatus.RUNNING])
                
                queue_size = self.redis_client.llen("workflow_queue")
                
                logger.info(f"Health: {active_count} active workflows, {queue_size} queued")
                
                # Store metrics
                self.redis_client.set("workflow_active_count", active_count, ex=300)
                self.redis_client.set("workflow_queue_size", queue_size, ex=300)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)
                
    async def performance_collector(self):
        """Collect performance metrics"""
        while self.running:
            try:
                # Collect execution metrics
                completed_executions = [
                    e for e in self.active_executions.values()
                    if e.status == WorkflowStatus.COMPLETED
                ]
                
                if completed_executions:
                    avg_execution_time = sum(e.execution_time_ms for e in completed_executions) / len(completed_executions)
                    
                    # Store performance metrics
                    self.redis_client.set("workflow_avg_execution_time", avg_execution_time, ex=3600)
                    
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Performance collector error: {e}")
                await asyncio.sleep(30)
                
    async def cleanup_completed_workflows(self):
        """Clean up completed workflow executions"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=1)
                
                to_remove = []
                for execution_id, execution in self.active_executions.items():
                    if (execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and
                        execution.completed_at and execution.completed_at < cutoff_time):
                        to_remove.append(execution_id)
                        
                for execution_id in to_remove:
                    del self.active_executions[execution_id]
                    
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} completed workflows")
                    
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
                
    def get_workflow_status(self, execution_id: str) -> Optional[Dict]:
        """Get status of a workflow execution"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None
            
        return {
            "id": execution.id,
            "name": execution.name,
            "status": execution.status.value,
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "execution_time_ms": execution.execution_time_ms,
            "tasks": [
                {
                    "id": task.id,
                    "agent_id": task.agent_id,
                    "status": task.status.value,
                    "retry_count": task.retry_count
                }
                for task in execution.tasks
            ]
        }

async def main():
    """Main entry point"""
    try:
        manager = DifyWorkflowManager()
        await manager.start()
        
    except KeyboardInterrupt:
        logger.info("Workflow Manager stopped by user")
    except Exception as e:
        logger.error(f"Workflow Manager failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())