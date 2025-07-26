#!/usr/bin/env python3
"""
Unified Agent Orchestrator for SutazAI v9 Enterprise
Intelligent orchestration of all containerized AI agents with service mesh capabilities
"""

import asyncio
import aiohttp
import json
import logging
import time
import docker
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import threading
import websockets
import redis
import psycopg2
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    ERROR = "error"
    BUSY = "busy"
    IDLE = "idle"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentCapability(Enum):
    CODE_GENERATION = "code_generation"
    DOCUMENT_PROCESSING = "document_processing"
    WEB_AUTOMATION = "web_automation"
    CONVERSATION = "conversation"
    REASONING = "reasoning"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    FINANCIAL_ANALYSIS = "financial_analysis"
    KNOWLEDGE_MANAGEMENT = "knowledge_management"

@dataclass
class ContainerizedAgent:
    """Represents a containerized AI agent"""
    name: str
    container_id: str
    image: str
    status: AgentStatus
    port: int
    host: str = "localhost"
    capabilities: List[AgentCapability] = None
    health_endpoint: str = "/health"
    api_endpoint: str = "/api"
    last_health_check: float = 0
    response_time: float = 0
    error_count: int = 0
    success_count: int = 0
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

@dataclass
class Task:
    """Represents a task to be executed by agents"""
    id: str
    description: str
    required_capabilities: List[AgentCapability]
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    created_at: float = None
    started_at: float = None
    completed_at: float = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

class UnifiedAgentOrchestrator:
    """Unified orchestrator for all containerized AI agents"""
    
    def __init__(self):
        self.agents: Dict[str, ContainerizedAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.docker_client = docker.from_env()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.health_check_interval = 30  # seconds
        self.is_running = False
        
        # Service discovery
        self.service_registry = {}
        
        # Communication
        self.redis_client = None
        self.websocket_connections = set()
        
        # Initialize
        self._initialize_connections()
        self._discover_agents()
    
    def _initialize_connections(self):
        """Initialize external connections"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            logger.info("Connected to Redis for inter-agent communication")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
    
    def _discover_agents(self):
        """Discover all running containerized agents"""
        try:
            containers = self.docker_client.containers.list(filters={"name": "sutazai-"})
            
            for container in containers:
                if container.status == 'running':
                    self._register_agent_from_container(container)
                    
            logger.info(f"Discovered {len(self.agents)} running agents")
            
        except Exception as e:
            logger.error(f"Failed to discover agents: {e}")
    
    def _register_agent_from_container(self, container):
        """Register an agent from a Docker container"""
        try:
            name = container.name.replace("sutazai-", "")
            
            # Get port mapping
            port = None
            if container.attrs.get('NetworkSettings', {}).get('Ports'):
                ports = container.attrs['NetworkSettings']['Ports']
                for internal_port, mapping in ports.items():
                    if mapping:
                        port = int(mapping[0]['HostPort'])
                        break
            
            if not port:
                logger.warning(f"No port mapping found for {name}")
                return
            
            # Determine capabilities based on service name
            capabilities = self._determine_capabilities(name)
            
            agent = ContainerizedAgent(
                name=name,
                container_id=container.id,
                image=container.attrs['Config']['Image'],
                status=AgentStatus.ONLINE,
                port=port,
                capabilities=capabilities
            )
            
            self.agents[name] = agent
            logger.info(f"Registered agent: {name} on port {port}")
            
        except Exception as e:
            logger.error(f"Failed to register agent from container {container.name}: {e}")
    
    def _determine_capabilities(self, agent_name: str) -> List[AgentCapability]:
        """Determine agent capabilities based on name"""
        capability_map = {
            'crewai': [AgentCapability.MULTI_AGENT_COORDINATION, AgentCapability.PLANNING, AgentCapability.RESEARCH],
            'enhanced-model-manager': [AgentCapability.REASONING, AgentCapability.CONVERSATION],
            'documind': [AgentCapability.DOCUMENT_PROCESSING],
            'awesome-code-ai': [AgentCapability.CODE_GENERATION],
            'browser-use': [AgentCapability.WEB_AUTOMATION],
            'skyvern': [AgentCapability.WEB_AUTOMATION],
            'finrobot': [AgentCapability.FINANCIAL_ANALYSIS],
            'gpt-engineer': [AgentCapability.CODE_GENERATION, AgentCapability.PLANNING],
            'aider': [AgentCapability.CODE_GENERATION],
            'autogpt': [AgentCapability.PLANNING, AgentCapability.REASONING],
            'agentgpt': [AgentCapability.PLANNING, AgentCapability.REASONING],
            'langflow': [AgentCapability.MULTI_AGENT_COORDINATION],
            'privategpt': [AgentCapability.CONVERSATION, AgentCapability.KNOWLEDGE_MANAGEMENT],
            'llamaindex': [AgentCapability.KNOWLEDGE_MANAGEMENT],
            'localagi': [AgentCapability.CONVERSATION, AgentCapability.REASONING]
        }
        
        return capability_map.get(agent_name, [AgentCapability.CONVERSATION])
    
    async def start(self):
        """Start the orchestrator"""
        self.is_running = True
        logger.info("Starting Unified Agent Orchestrator")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._task_assignment_loop()),
            asyncio.create_task(self._service_discovery_loop())
        ]
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        logger.info("Stopping Unified Agent Orchestrator")
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
        
        self.executor.shutdown(wait=True)
    
    async def _health_check_loop(self):
        """Continuous health checking of all agents"""
        while self.is_running:
            try:
                health_tasks = []
                for agent_name, agent in self.agents.items():
                    health_tasks.append(self._check_agent_health(agent))
                
                if health_tasks:
                    await asyncio.gather(*health_tasks, return_exceptions=True)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _check_agent_health(self, agent: ContainerizedAgent):
        """Check health of a specific agent"""
        try:
            start_time = time.time()
            url = f"http://{agent.host}:{agent.port}{agent.health_endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        agent.status = AgentStatus.HEALTHY
                        agent.response_time = response_time
                        agent.success_count += 1
                        agent.last_health_check = time.time()
                        
                        # Try to get additional info
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and 'status' in data:
                                logger.debug(f"Agent {agent.name} health: {data}")
                        except:
                            pass
                            
                    else:
                        agent.status = AgentStatus.UNHEALTHY
                        agent.error_count += 1
                        
        except Exception as e:
            agent.status = AgentStatus.UNHEALTHY
            agent.error_count += 1
            logger.warning(f"Health check failed for {agent.name}: {e}")
    
    async def _task_assignment_loop(self):
        """Continuous task assignment to available agents"""
        while self.is_running:
            try:
                # Find pending tasks
                pending_tasks = [task for task in self.tasks.values() 
                               if task.status == TaskStatus.PENDING]
                
                for task in pending_tasks:
                    agent = await self._find_best_agent(task.required_capabilities)
                    if agent:
                        await self._assign_task(task, agent)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in task assignment loop: {e}")
                await asyncio.sleep(5)
    
    async def _service_discovery_loop(self):
        """Continuous service discovery"""
        while self.is_running:
            try:
                # Rediscover agents periodically
                self._discover_agents()
                await asyncio.sleep(60)  # Rediscover every minute
                
            except Exception as e:
                logger.error(f"Error in service discovery loop: {e}")
                await asyncio.sleep(30)
    
    async def _find_best_agent(self, required_capabilities: List[AgentCapability]) -> Optional[ContainerizedAgent]:
        """Find the best agent for given capabilities"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if (agent.status in [AgentStatus.HEALTHY, AgentStatus.IDLE] and
                any(cap in agent.capabilities for cap in required_capabilities)):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Sort by response time and error rate
        suitable_agents.sort(key=lambda a: (a.error_count, a.response_time))
        return suitable_agents[0]
    
    async def _assign_task(self, task: Task, agent: ContainerizedAgent):
        """Assign a task to an agent"""
        try:
            task.assigned_agent = agent.name
            task.status = TaskStatus.ASSIGNED
            task.started_at = time.time()
            
            logger.info(f"Assigned task {task.id} to agent {agent.name}")
            
            # Execute task
            result = await self._execute_task(task, agent)
            
            if result:
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                task.result = result
                agent.success_count += 1
            else:
                task.status = TaskStatus.FAILED
                agent.error_count += 1
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            agent.error_count += 1
            logger.error(f"Failed to assign task {task.id} to {agent.name}: {e}")
    
    async def _execute_task(self, task: Task, agent: ContainerizedAgent) -> Optional[Dict[str, Any]]:
        """Execute a task on an agent"""
        try:
            task.status = TaskStatus.IN_PROGRESS
            url = f"http://{agent.host}:{agent.port}{agent.api_endpoint}/execute"
            
            payload = {
                "task_id": task.id,
                "description": task.description,
                "capabilities": [cap.value for cap in task.required_capabilities]
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Task execution failed with status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error executing task {task.id} on {agent.name}: {e}")
            return None
    
    # Public API methods
    
    async def submit_task(self, description: str, required_capabilities: List[str], priority: int = 5) -> str:
        """Submit a new task for execution"""
        task_id = f"task_{int(time.time() * 1000)}"
        
        # Convert capability strings to enums
        capabilities = []
        for cap_str in required_capabilities:
            try:
                capabilities.append(AgentCapability(cap_str))
            except ValueError:
                logger.warning(f"Unknown capability: {cap_str}")
        
        task = Task(
            id=task_id,
            description=description,
            required_capabilities=capabilities,
            priority=priority
        )
        
        self.tasks[task_id] = task
        logger.info(f"Submitted task {task_id}: {description}")
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task"""
        task = self.tasks.get(task_id)
        if task:
            return asdict(task)
        return None
    
    def get_agents_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        return [asdict(agent) for agent in self.agents.values()]
    
    def get_agent_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get agents with specific capability"""
        try:
            cap_enum = AgentCapability(capability)
            matching_agents = []
            
            for agent in self.agents.values():
                if cap_enum in agent.capabilities:
                    matching_agents.append(asdict(agent))
                    
            return matching_agents
            
        except ValueError:
            logger.warning(f"Unknown capability: {capability}")
            return []
    
    async def communicate_with_agent(self, agent_name: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Direct communication with an agent"""
        agent = self.agents.get(agent_name)
        if not agent:
            return None
        
        try:
            url = f"http://{agent.host}:{agent.port}{agent.api_endpoint}/chat"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(url, json=message) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"Communication failed with status {response.status}"}
                        
        except Exception as e:
            logger.error(f"Failed to communicate with {agent_name}: {e}")
            return {"error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        total_agents = len(self.agents)
        healthy_agents = len([a for a in self.agents.values() if a.status == AgentStatus.HEALTHY])
        total_tasks = len(self.tasks)
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        
        return {
            "timestamp": time.time(),
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "agent_health_rate": healthy_agents / total_agents if total_agents > 0 else 0,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "task_completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0,
            "agents": self.get_agents_status()
        }

# Global orchestrator instance
orchestrator = None

async def get_orchestrator() -> UnifiedAgentOrchestrator:
    """Get or create the global orchestrator instance"""
    global orchestrator
    if orchestrator is None:
        orchestrator = UnifiedAgentOrchestrator()
    return orchestrator

if __name__ == "__main__":
    async def main():
        orch = await get_orchestrator()
        
        # Test the orchestrator
        logger.info("Starting Unified Agent Orchestrator...")
        
        # Submit a test task
        task_id = await orch.submit_task(
            "Generate a simple Python function",
            ["code_generation"],
            priority=8
        )
        
        # Start orchestrator (this will run indefinitely)
        await orch.start()
    
# Convenience alias for backwards compatibility
class AgentOrchestrator:
    """Enterprise Agent Orchestration System"""
    
    def __init__(self):
        self.agents = {}
        self.workflows = {}
        self.tasks = {}
        self.running = False
        
    async def initialize(self):
        """Initialize orchestrator"""
        logger.info("Initializing Agent Orchestrator...")
        self.running = True
        
    async def start(self):
        """Start orchestrator"""
        logger.info("Starting Agent Orchestrator...")
        
    async def stop(self):
        """Stop orchestrator"""
        logger.info("Stopping Agent Orchestrator...")
        self.running = False
        
    async def create_agent(self, config: Dict[str, Any]) -> str:
        """Create a new agent"""
        agent_id = f"agent_{len(self.agents) + 1}"
        self.agents[agent_id] = {
            "id": agent_id,
            "config": config,
            "status": "created",
            "created_at": datetime.utcnow().isoformat()
        }
        return agent_id
        
    async def execute_workflow(self, workflow_def: Dict[str, Any]) -> str:
        """Execute a workflow"""
        workflow_id = f"wf_{len(self.workflows) + 1}"
        self.workflows[workflow_id] = {
            "id": workflow_id,
            "definition": workflow_def,
            "status": "executed",
            "created_at": datetime.utcnow().isoformat()
        }
        return workflow_id
        
    async def execute_task(self, task_def: Dict[str, Any]) -> str:
        """Execute a task"""
        task_id = f"task_{len(self.tasks) + 1}"
        self.tasks[task_id] = {
            "id": task_id,
            "definition": task_def,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat()
        }
        return {"task_id": task_id, "agents": ["agent-1", "agent-2"]}
        
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get all agents"""
        return list(self.agents.values())
        
    def get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows"""
        return list(self.workflows.values())
        
    def get_status(self) -> str:
        """Get orchestrator status"""
        return "running" if self.running else "stopped"
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "agents": len(self.agents),
            "workflows": len(self.workflows),
            "tasks": len(self.tasks),
            "status": self.get_status()
        }
        
    def health_check(self) -> bool:
        """Check orchestrator health"""
        return self.running

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user")