#!/usr/bin/env python3
"""
Unified AI Agent Orchestrator for SutazAI
This orchestrator ensures all AI agents work together seamlessly
"""

import asyncio
import json
import logging
import redis.asyncio as redis
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import aiohttp
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    type: str
    description: str
    priority: int
    status: TaskStatus
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: datetime = None

class UnifiedOrchestrator:
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/agent_orchestration.yaml"):
        self.config = self._load_config(config_path)
        self.redis_client = None
        self.agent_registry = {}
        self.active_tasks = {}
        self.agent_capabilities = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load orchestration configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    async def initialize(self):
        """Initialize the orchestrator"""
        # Connect to Redis
        self.redis_client = await redis.Redis(
            host=self.config['communication']['redis']['host'],
            port=self.config['communication']['redis']['port'],
            db=self.config['communication']['redis']['db'],
            decode_responses=True
        )
        
        # Initialize agent registry
        await self._initialize_agent_registry()
        
        # Start background tasks
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._task_processor())
        
        logger.info("Unified Orchestrator initialized successfully")
    
    async def _initialize_agent_registry(self):
        """Initialize and verify agent registry"""
        for agent_name, agent_config in self.config['agents'].items():
            self.agent_registry[agent_name] = agent_config
            self.agent_capabilities[agent_name] = agent_config.get('capabilities', [])
            
            # Register agent in Redis
            await self.redis_client.hset(
                f"agent:registry:{agent_name}",
                mapping={
                    "type": agent_config.get('type', 'general'),
                    "priority": agent_config.get('priority', 'normal'),
                    "status": "initializing",
                    "capabilities": json.dumps(agent_config.get('capabilities', [])),
                    "last_seen": datetime.now().isoformat()
                }
            )
    
    async def deploy_all_agents(self):
        """Deploy all configured agents"""
        logger.info("Starting deployment of all AI agents...")
        
        # Group agents by priority
        critical_agents = []
        high_priority_agents = []
        normal_agents = []
        
        for agent_name, config in self.agent_registry.items():
            priority = config.get('priority', 'normal')
            if priority == 'critical':
                critical_agents.append(agent_name)
            elif priority == 'high':
                high_priority_agents.append(agent_name)
            else:
                normal_agents.append(agent_name)
        
        # Deploy in priority order
        await self._deploy_agent_group(critical_agents, "Critical")
        await self._deploy_agent_group(high_priority_agents, "High Priority")
        await self._deploy_agent_group(normal_agents, "Normal Priority")
        
        logger.info("All agents deployed successfully!")
    
    async def _deploy_agent_group(self, agents: List[str], group_name: str):
        """Deploy a group of agents"""
        logger.info(f"Deploying {group_name} agents: {agents}")
        
        for agent_name in agents:
            try:
                # Use MCP server to deploy agent
                await self._deploy_agent_via_mcp(agent_name)
                
                # Update status
                await self.redis_client.hset(
                    f"agent:registry:{agent_name}",
                    "status", "active"
                )
                
                logger.info(f"✓ Deployed {agent_name}")
                
            except Exception as e:
                logger.error(f"✗ Failed to deploy {agent_name}: {e}")
    
    async def _deploy_agent_via_mcp(self, agent_name: str):
        """Deploy an agent using MCP server"""
        # Map agent names to MCP agent types
        agent_type_mapping = {
            "infrastructure-devops-manager": "infra-manager",
            "ollama-integration-specialist": "ollama-specialist",
            "hardware-resource-optimizer": "hardware-optimizer",
            "senior-ai-engineer": "ai-engineer",
            "senior-backend-developer": "backend-dev",
            "senior-frontend-developer": "frontend-dev",
            "ai-agent-orchestrator": "orchestrator",
            "deployment-automation-master": "deploy-master",
            "testing-qa-validator": "qa-tester",
            "security-pentesting-specialist": "security-tester",
            "agi-system-architect": "agi-architect",
            "context-optimization-engineer": "context-optimizer"
        }
        
        agent_type = agent_type_mapping.get(agent_name, agent_name)
        
        # Call MCP server to deploy agent
        async with aiohttp.ClientSession() as session:
            url = "http://mcp-server:8100/deploy_agent"
            payload = {
                "agent_type": agent_type,
                "name": agent_name,
                "capabilities": self.agent_capabilities.get(agent_name, [])
            }
            
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"MCP deployment failed: {await response.text()}")
    
    async def create_task(self, task_type: str, description: str, priority: int = 5) -> Task:
        """Create a new task for agents to handle"""
        task = Task(
            id=f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_tasks)}",
            type=task_type,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Store task
        self.active_tasks[task.id] = task
        
        # Queue task for processing
        await self.redis_client.lpush("agent:tasks", json.dumps({
            "id": task.id,
            "type": task.type,
            "description": task.description,
            "priority": task.priority
        }))
        
        logger.info(f"Created task {task.id}: {description}")
        return task
    
    async def assign_task_to_best_agent(self, task: Task) -> Optional[str]:
        """Find and assign task to the best available agent"""
        # Find agents with matching capabilities
        capable_agents = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            # Check if agent can handle this task type
            if any(task.type in cap for cap in capabilities):
                # Check if agent is healthy
                status = await self.redis_client.hget(f"agent:registry:{agent_name}", "status")
                if status == "active":
                    capable_agents.append(agent_name)
        
        if not capable_agents:
            logger.warning(f"No capable agents found for task {task.id} of type {task.type}")
            return None
        
        # Select best agent based on workload
        best_agent = await self._select_least_loaded_agent(capable_agents)
        
        # Assign task
        task.assigned_to = best_agent
        task.status = TaskStatus.ASSIGNED
        
        # Notify agent
        await self.redis_client.publish(f"agent:tasks:{best_agent}", json.dumps({
            "task_id": task.id,
            "type": task.type,
            "description": task.description
        }))
        
        logger.info(f"Assigned task {task.id} to {best_agent}")
        return best_agent
    
    async def _select_least_loaded_agent(self, agents: List[str]) -> str:
        """Select the agent with the least workload"""
        agent_loads = {}
        
        for agent in agents:
            # Count active tasks for this agent
            active_count = sum(1 for t in self.active_tasks.values() 
                             if t.assigned_to == agent and t.status == TaskStatus.IN_PROGRESS)
            agent_loads[agent] = active_count
        
        # Return agent with minimum load
        return min(agent_loads, key=agent_loads.get)
    
    async def _task_processor(self):
        """Process queued tasks"""
        while True:
            try:
                # Get task from queue
                task_data = await self.redis_client.brpop("agent:tasks", timeout=1)
                if task_data:
                    task_json = json.loads(task_data[1])
                    task = self.active_tasks.get(task_json['id'])
                    
                    if task and task.status == TaskStatus.PENDING:
                        await self.assign_task_to_best_agent(task)
                
            except Exception as e:
                logger.error(f"Error processing tasks: {e}")
            
            await asyncio.sleep(1)
    
    async def _health_monitor(self):
        """Monitor agent health"""
        while True:
            try:
                for agent_name in self.agent_registry:
                    # Check agent health endpoint
                    agent_config = self.agent_registry[agent_name]
                    if 'endpoints' in agent_config and 'health' in agent_config['endpoints']:
                        health_url = agent_config['endpoints']['health']
                        
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(health_url, timeout=5) as response:
                                    if response.status == 200:
                                        await self.redis_client.hset(
                                            f"agent:registry:{agent_name}",
                                            mapping={
                                                "status": "active",
                                                "last_seen": datetime.now().isoformat()
                                            }
                                        )
                                    else:
                                        await self.redis_client.hset(
                                            f"agent:registry:{agent_name}",
                                            "status", "unhealthy"
                                        )
                        except (IOError, OSError, FileNotFoundError) as e:
                            # TODO: Review this exception handling
                            logger.error(f"Unexpected exception: {e}", exc_info=True)
                            # Agent not responding
                            await self.redis_client.hset(
                                f"agent:registry:{agent_name}",
                                "status", "unreachable"
                            )
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def coordinate_agents_for_complex_task(self, task_description: str):
        """Coordinate multiple agents to handle a complex task"""
        logger.info(f"Coordinating agents for complex task: {task_description}")
        
        # Break down the task into subtasks
        subtasks = await self._analyze_and_decompose_task(task_description)
        
        # Create workflow
        workflow_results = {}
        
        for subtask in subtasks:
            # Create task
            task = await self.create_task(
                task_type=subtask['type'],
                description=subtask['description'],
                priority=subtask.get('priority', 5)
            )
            
            # Assign to best agent
            agent = await self.assign_task_to_best_agent(task)
            
            if agent:
                # Wait for completion (with timeout)
                result = await self._wait_for_task_completion(task.id, timeout=300)
                workflow_results[subtask['name']] = result
            else:
                workflow_results[subtask['name']] = {"error": "No agent available"}
        
        return workflow_results
    
    async def _analyze_and_decompose_task(self, task_description: str) -> List[Dict]:
        """Use AI to analyze and decompose complex tasks"""
        # This would use an AI agent to analyze the task
        # For now, using simple pattern matching
        
        subtasks = []
        
        if "deploy" in task_description.lower():
            subtasks.extend([
                {"name": "pre_check", "type": "deployment", "description": "Pre-deployment validation"},
                {"name": "test", "type": "testing", "description": "Run automated tests"},
                {"name": "security", "type": "security", "description": "Security scan"},
                {"name": "deploy", "type": "deployment", "description": "Execute deployment"},
                {"name": "verify", "type": "testing", "description": "Post-deployment verification"}
            ])
        
        elif "optimize" in task_description.lower():
            subtasks.extend([
                {"name": "analyze", "type": "optimization", "description": "Analyze current performance"},
                {"name": "plan", "type": "optimization", "description": "Create optimization plan"},
                {"name": "implement", "type": "optimization", "description": "Implement optimizations"},
                {"name": "measure", "type": "testing", "description": "Measure improvements"}
            ])
        
        else:
            # Default subtasks
            subtasks.append({
                "name": "main", 
                "type": "general", 
                "description": task_description
            })
        
        return subtasks
    
    async def _wait_for_task_completion(self, task_id: str, timeout: int = 300):
        """Wait for a task to complete"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            task = self.active_tasks.get(task_id)
            if task and task.status == TaskStatus.COMPLETED:
                return task.result
            elif task and task.status == TaskStatus.FAILED:
                return {"error": task.error}
            
            await asyncio.sleep(1)
        
        return {"error": "Task timeout"}
    
    async def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            "orchestrator": "active",
            "agents": {},
            "active_tasks": len([t for t in self.active_tasks.values() 
                               if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]]),
            "completed_tasks": len([t for t in self.active_tasks.values() 
                                  if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.active_tasks.values() 
                               if t.status == TaskStatus.FAILED])
        }
        
        # Get agent statuses
        for agent_name in self.agent_registry:
            agent_data = await self.redis_client.hgetall(f"agent:registry:{agent_name}")
            status["agents"][agent_name] = {
                "status": agent_data.get("status", "unknown"),
                "type": agent_data.get("type", "unknown"),
                "last_seen": agent_data.get("last_seen", "never")
            }
        
        return status

async def main():
    """Main function to run the orchestrator"""
    orchestrator = UnifiedOrchestrator()
    
    # Initialize
    await orchestrator.initialize()
    
    # Deploy all agents
    await orchestrator.deploy_all_agents()
    
    # Example: Coordinate agents for a complex task
    result = await orchestrator.coordinate_agents_for_complex_task(
        "Deploy the new AI model with full testing and security validation"
    )
    
    logger.info(f"Complex task result: {result}")
    
    # Get system status
    status = await orchestrator.get_system_status()
    logger.info(f"System status: {json.dumps(status, indent=2)}")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")

if __name__ == "__main__":
    asyncio.run(main())