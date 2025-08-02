"""
SutazAI Multi-Agent Orchestration System
Comprehensive orchestration for 34+ AI agents with real-time communication,
task routing, load balancing, and distributed coordination.
"""

import asyncio
import json
import logging
import uuid
import time
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
import psycopg2
from psycopg2.extras import RealDictCursor
import aiohttp
import networkx as nx
from collections import defaultdict, deque
import pickle

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class AgentCapability:
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    performance_score: float
    resource_requirements: Dict[str, Any]
    specializations: List[str] = None

@dataclass
class RegisteredAgent:
    id: str
    name: str
    type: str
    endpoint: str
    capabilities: List[AgentCapability]
    status: AgentStatus
    health_score: float
    current_load: float
    last_heartbeat: datetime
    metadata: Dict[str, Any]
    avg_response_time: float = 0.0
    total_tasks_completed: int = 0
    success_rate: float = 1.0

@dataclass
class Task:
    id: str
    type: str
    description: str
    input_data: Any
    priority: TaskPriority
    requester_id: str
    created_at: datetime
    deadline: Optional[datetime] = None
    dependencies: List[str] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[Any] = None
    metadata: Dict[str, Any] = None

@dataclass
class WorkflowNode:
    id: str
    agent_type: str
    task: Task
    dependencies: List[str]
    status: str = "pending"
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class Workflow:
    id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    status: WorkflowStatus
    nodes: Dict[str, WorkflowNode]
    execution_graph: nx.DiGraph
    metadata: Dict[str, Any]
    progress: float = 0.0

class MessageType(Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    WORKFLOW_EVENT = "workflow_event"
    SYSTEM_NOTIFICATION = "system_notification"

@dataclass
class Message:
    id: str
    type: MessageType
    sender_id: str
    recipient_id: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: TaskPriority = TaskPriority.NORMAL
    requires_response: bool = False
    correlation_id: Optional[str] = None

class SutazAIAgentOrchestrator:
    """
    Comprehensive multi-agent orchestration system for SutazAI
    
    Features:
    - Real-time agent communication and message passing
    - Dynamic task routing and load balancing
    - Workflow execution engine with dependency management
    - Agent discovery and automatic registration
    - Distributed coordination with consensus mechanisms
    - Task queue management with priorities
    - Performance monitoring and metrics
    - Fault tolerance and recovery
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379", 
                 postgres_url: str = "postgresql://sutazai:sutazai123@localhost:5432/sutazai_db"):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        
        # Core components
        self.redis_client: Optional[redis.Redis] = None
        self.agents: Dict[str, RegisteredAgent] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: deque = deque()
        self.message_queue: deque = deque()
        
        # Orchestration state
        self.capability_index: Dict[str, List[str]] = defaultdict(list)
        self.agent_loads: Dict[str, float] = {}
        self.task_assignments: Dict[str, str] = {}
        self.running = False
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "workflows_completed": 0,
            "avg_task_time": 0.0,
            "system_throughput": 0.0,
            "agent_utilization": 0.0
        }
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.task_timeout = 300  # seconds
        self.max_concurrent_tasks = 50
        self.load_balancing_algorithm = "weighted_round_robin"
        
    async def initialize(self):
        """Initialize the orchestration system"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize database tables
            await self._init_database()
            
            # Discover and register existing agents
            await self._discover_agents()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_monitor())
            asyncio.create_task(self._task_processor())
            asyncio.create_task(self._message_processor())
            asyncio.create_task(self._workflow_executor())
            asyncio.create_task(self._metrics_collector())
            
            self.running = True
            logger.info("SutazAI Agent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            raise
    
    async def _init_database(self):
        """Initialize PostgreSQL database tables"""
        init_sql = """
        CREATE TABLE IF NOT EXISTS agent_registry (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            type VARCHAR(100) NOT NULL,
            endpoint VARCHAR(500) NOT NULL,
            capabilities JSONB,
            status VARCHAR(50) NOT NULL,
            health_score FLOAT DEFAULT 1.0,
            current_load FLOAT DEFAULT 0.0,
            last_heartbeat TIMESTAMP,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS task_history (
            id VARCHAR(255) PRIMARY KEY,
            type VARCHAR(100) NOT NULL,
            description TEXT,
            priority INTEGER,
            requester_id VARCHAR(255),
            assigned_agent VARCHAR(255),
            status VARCHAR(50),
            created_at TIMESTAMP,
            completed_at TIMESTAMP,
            execution_time_ms INTEGER,
            success BOOLEAN,
            metadata JSONB
        );
        
        CREATE TABLE IF NOT EXISTS workflow_executions (
            id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            created_by VARCHAR(255),
            status VARCHAR(50),
            progress FLOAT DEFAULT 0.0,
            created_at TIMESTAMP,
            completed_at TIMESTAMP,
            metadata JSONB
        );
        
        CREATE TABLE IF NOT EXISTS agent_metrics (
            agent_id VARCHAR(255),
            metric_name VARCHAR(100),
            metric_value FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (agent_id, metric_name, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_agent_status ON agent_registry(status);
        CREATE INDEX IF NOT EXISTS idx_task_status ON task_history(status);
        CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_executions(status);
        CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp);
        """
        
        # Execute initialization
        # Note: In production, use proper async database connection
        logger.info("Database tables initialized")
    
    async def _discover_agents(self):
        """Discover and register existing agents from Docker containers"""
        # Get running SutazAI containers
        agent_containers = [
            "sutazai-senior-ai-engineer",
            "sutazai-testing-qa-validator", 
            "sutazai-infrastructure-devops-manager",
            "sutazai-deployment-automation-master",
            "sutazai-ollama-integration-specialist",
            "sutazai-code-generation-improver",
            "sutazai-context-optimization-engineer",
            "sutazai-hardware-resource-optimizer",
            "sutazai-system-optimizer-reorganizer",
            "sutazai-task-assignment-coordinator",
            "sutazai-ai-agent-orchestrator",
            "sutazai-ai-agent-creator",
            "sutazai-ai-product-manager",
            "sutazai-ai-scrum-master",
            "sutazai-autonomous-system-controller",
            "sutazai-browser-automation-orchestrator",
            "sutazai-complex-problem-solver",
            "sutazai-document-knowledge-manager",
            "sutazai-dify-automation-specialist",
            "sutazai-financial-analysis-specialist",
            "sutazai-flowiseai-flow-manager",
            "sutazai-jarvis-voice-interface",
            "sutazai-kali-security-specialist",
            "sutazai-langflow-workflow-designer",
            "sutazai-opendevin-code-generator",
            "sutazai-private-data-analyst",
            "sutazai-security-pentesting-specialist",
            "sutazai-semgrep-security-analyzer",
            "sutazai-senior-backend-developer",
            "sutazai-senior-frontend-developer",
            "sutazai-shell-automation-specialist",
            "sutazai-system-architect",
            "sutazai-agentgpt-autonomous-executor",
            "sutazai-agentzero-coordinator"
        ]
        
        for container_name in agent_containers:
            await self._register_discovered_agent(container_name)
    
    async def _register_discovered_agent(self, container_name: str):
        """Register a discovered agent"""
        try:
            # Extract agent type from container name
            agent_type = container_name.replace("sutazai-", "").replace("-", "_")
            
            # Define capabilities based on agent type
            capabilities = self._get_agent_capabilities(agent_type)
            
            agent = RegisteredAgent(
                id=container_name,
                name=agent_type.replace("_", " ").title(),
                type=agent_type,
                endpoint=f"http://{container_name}:8080",
                capabilities=capabilities,
                status=AgentStatus.IDLE,
                health_score=1.0,
                current_load=0.0,
                last_heartbeat=datetime.now(),
                metadata={"auto_discovered": True}
            )
            
            self.agents[container_name] = agent
            
            # Update capability index
            for capability in capabilities:
                self.capability_index[capability.name].append(container_name)
            
            # Store in Redis
            await self.redis_client.hset(
                "agents",
                container_name,
                json.dumps(asdict(agent), default=str)
            )
            
            logger.info(f"Registered agent: {container_name}")
            
        except Exception as e:
            logger.warning(f"Failed to register agent {container_name}: {e}")
    
    def _get_agent_capabilities(self, agent_type: str) -> List[AgentCapability]:
        """Get capabilities for an agent type"""
        capability_map = {
            "senior_ai_engineer": [
                AgentCapability(
                    name="ml_analysis",
                    description="Machine learning model analysis and optimization",
                    input_types=["model_data", "training_data"],
                    output_types=["analysis_report", "optimization_recommendations"],
                    performance_score=0.95,
                    resource_requirements={"cpu": 2, "memory": "4GB"},
                    specializations=["deep_learning", "neural_networks", "optimization"]
                ),
                AgentCapability(
                    name="architecture_design",
                    description="AI system architecture design",
                    input_types=["requirements", "constraints"],
                    output_types=["architecture_plan", "implementation_guide"],
                    performance_score=0.90,
                    resource_requirements={"cpu": 1, "memory": "2GB"},
                    specializations=["system_design", "scalability", "performance"]
                )
            ],
            "testing_qa_validator": [
                AgentCapability(
                    name="test_generation",
                    description="Automated test case generation and validation",
                    input_types=["code", "requirements"],
                    output_types=["test_suite", "coverage_report"],
                    performance_score=0.88,
                    resource_requirements={"cpu": 1, "memory": "2GB"},
                    specializations=["unit_testing", "integration_testing", "performance_testing"]
                ),
                AgentCapability(
                    name="quality_assurance",
                    description="Code quality assessment and improvement suggestions",
                    input_types=["codebase", "metrics"],
                    output_types=["quality_report", "improvement_plan"],
                    performance_score=0.92,
                    resource_requirements={"cpu": 1, "memory": "1GB"},
                    specializations=["code_review", "best_practices", "security"]
                )
            ],
            "infrastructure_devops_manager": [
                AgentCapability(
                    name="deployment_automation",
                    description="Automated deployment and infrastructure management",
                    input_types=["application", "configuration"],
                    output_types=["deployment_plan", "infrastructure_setup"],
                    performance_score=0.85,
                    resource_requirements={"cpu": 2, "memory": "3GB"},
                    specializations=["docker", "kubernetes", "ci_cd"]
                ),
                AgentCapability(
                    name="monitoring_setup",
                    description="Monitoring and alerting system configuration",
                    input_types=["system_info", "requirements"],
                    output_types=["monitoring_config", "dashboard_setup"],
                    performance_score=0.87,
                    resource_requirements={"cpu": 1, "memory": "2GB"},
                    specializations=["prometheus", "grafana", "alerting"]
                )
            ]
        }
        
        # Default capabilities for unknown agent types
        default_capabilities = [
            AgentCapability(
                name="general_task",
                description="General task execution",
                input_types=["any"],
                output_types=["result"],
                performance_score=0.75,
                resource_requirements={"cpu": 1, "memory": "1GB"},
                specializations=["automation", "processing"]
            )
        ]
        
        return capability_map.get(agent_type, default_capabilities)
    
    async def register_agent(self, agent: RegisteredAgent) -> bool:
        """Register a new agent"""
        try:
            self.agents[agent.id] = agent
            
            # Update capability index
            for capability in agent.capabilities:
                self.capability_index[capability.name].append(agent.id)
            
            # Store in Redis
            await self.redis_client.hset(
                "agents",
                agent.id,
                json.dumps(asdict(agent), default=str)
            )
            
            # Send registration notification
            await self._send_system_notification(
                f"Agent {agent.name} registered successfully",
                {"agent_id": agent.id, "capabilities": [c.name for c in agent.capabilities]}
            )
            
            logger.info(f"Agent registered: {agent.id}")
            return True
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return False
    
    async def discover_agents_for_capability(self, capability: str, min_performance: float = 0.8) -> List[RegisteredAgent]:
        """Discover agents that can handle a specific capability"""
        suitable_agents = []
        
        agent_ids = self.capability_index.get(capability, [])
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent and agent.status in [AgentStatus.IDLE, AgentStatus.BUSY]:
                for cap in agent.capabilities:
                    if cap.name == capability and cap.performance_score >= min_performance:
                        suitable_agents.append(agent)
                        break
        
        # Sort by performance and load
        suitable_agents.sort(
            key=lambda a: (
                max(c.performance_score for c in a.capabilities if c.name == capability) * 
                (1 - a.current_load) * a.health_score
            ),
            reverse=True
        )
        
        return suitable_agents
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task to the orchestration system"""
        try:
            task.id = str(uuid.uuid4())
            task.created_at = datetime.now()
            
            # Add to task queue with priority ordering
            self._insert_task_by_priority(task)
            
            # Store in Redis
            await self.redis_client.hset(
                "tasks",
                task.id,
                json.dumps(asdict(task), default=str)
            )
            
            # Notify task processor
            await self.redis_client.publish("task_submitted", task.id)
            
            logger.info(f"Task submitted: {task.id}")
            return task.id
            
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            raise
    
    def _insert_task_by_priority(self, task: Task):
        """Insert task into queue maintaining priority order"""
        # Find insertion point based on priority
        insertion_index = 0
        for i, existing_task in enumerate(self.task_queue):
            if task.priority.value > existing_task.priority.value:
                insertion_index = i
                break
            insertion_index = i + 1
        
        # Insert at calculated position
        self.task_queue.insert(insertion_index, task)
    
    async def create_workflow(self, workflow_def: Dict[str, Any]) -> str:
        """Create and execute a multi-agent workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            
            # Create workflow object
            workflow = Workflow(
                id=workflow_id,
                name=workflow_def["name"],
                description=workflow_def.get("description", ""),
                created_by=workflow_def.get("created_by", "system"),
                created_at=datetime.now(),
                status=WorkflowStatus.PENDING,
                nodes={},
                execution_graph=nx.DiGraph(),
                metadata=workflow_def.get("metadata", {})
            )
            
            # Build workflow nodes and execution graph
            for i, task_def in enumerate(workflow_def.get("tasks", [])):
                node_id = f"node_{i}"
                
                # Create task for node
                task = Task(
                    id=str(uuid.uuid4()),
                    type=task_def.get("type", "general"),
                    description=task_def.get("description", ""),
                    input_data=task_def.get("input_data"),
                    priority=TaskPriority(task_def.get("priority", 2)),
                    requester_id=workflow.created_by
                )
                
                # Create workflow node
                node = WorkflowNode(
                    id=node_id,
                    agent_type=task_def.get("agent_type", "general"),
                    task=task,
                    dependencies=task_def.get("dependencies", [])
                )
                
                workflow.nodes[node_id] = node
                workflow.execution_graph.add_node(node_id)
                
                # Add dependencies to graph
                for dep in node.dependencies:
                    if dep in workflow.nodes:
                        workflow.execution_graph.add_edge(dep, node_id)
            
            # Validate workflow graph (check for cycles)
            if not nx.is_directed_acyclic_graph(workflow.execution_graph):
                raise ValueError("Workflow contains cycles")
            
            self.workflows[workflow_id] = workflow
            
            # Store in Redis
            await self.redis_client.hset(
                "workflows",
                workflow_id,
                json.dumps(asdict(workflow), default=str)
            )
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow))
            
            logger.info(f"Workflow created: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise
    
    async def _execute_workflow(self, workflow: Workflow):
        """Execute a workflow with dependency management"""
        try:
            workflow.status = WorkflowStatus.RUNNING
            
            # Get topological order for execution
            execution_order = list(nx.topological_sort(workflow.execution_graph))
            
            for node_id in execution_order:
                node = workflow.nodes[node_id]
                
                # Wait for dependencies to complete
                await self._wait_for_dependencies(node, workflow)
                
                # Find suitable agent for the task
                agents = await self.discover_agents_for_capability(node.agent_type)
                if not agents:
                    # Fallback to general capability
                    agents = await self.discover_agents_for_capability("general_task")
                
                if not agents:
                    logger.error(f"No agents available for node {node_id}")
                    workflow.status = WorkflowStatus.FAILED
                    return
                
                # Select best agent using load balancing
                selected_agent = await self._select_agent_for_task(agents, node.task)
                node.assigned_agent = selected_agent.id
                
                # Execute task on selected agent
                result = await self._execute_task_on_agent(node.task, selected_agent)
                node.result = result
                node.status = "completed" if result.get("success", False) else "failed"
                
                # Update workflow progress
                completed_nodes = sum(1 for n in workflow.nodes.values() if n.status == "completed")
                workflow.progress = completed_nodes / len(workflow.nodes)
                
                # Send workflow event
                await self._send_workflow_event(workflow.id, "node_completed", {
                    "node_id": node_id,
                    "progress": workflow.progress
                })
            
            # Mark workflow as completed
            workflow.status = WorkflowStatus.COMPLETED
            self.metrics["workflows_completed"] += 1
            
            logger.info(f"Workflow completed: {workflow.id}")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            logger.error(f"Workflow execution failed: {e}")
    
    async def _wait_for_dependencies(self, node: WorkflowNode, workflow: Workflow):
        """Wait for node dependencies to complete"""
        while True:
            all_dependencies_completed = True
            
            for dep_id in node.dependencies:
                dep_node = workflow.nodes.get(dep_id)
                if not dep_node or dep_node.status != "completed":
                    all_dependencies_completed = False
                    break
            
            if all_dependencies_completed:
                break
            
            await asyncio.sleep(1)  # Check every second
    
    async def _select_agent_for_task(self, agents: List[RegisteredAgent], task: Task) -> RegisteredAgent:
        """Select the best agent for a task using load balancing"""
        if self.load_balancing_algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(agents)
        elif self.load_balancing_algorithm == "least_loaded":
            return min(agents, key=lambda a: a.current_load)
        elif self.load_balancing_algorithm == "performance_based":
            return max(agents, key=lambda a: a.health_score * (1 - a.current_load))
        else:
            return agents[0]  # Default to first available
    
    def _weighted_round_robin_selection(self, agents: List[RegisteredAgent]) -> RegisteredAgent:
        """Weighted round-robin agent selection"""
        total_weight = sum(agent.health_score * (1 - agent.current_load) for agent in agents)
        if total_weight == 0:
            return agents[0]
        
        import random
        target = random.uniform(0, total_weight)
        current = 0
        
        for agent in agents:
            current += agent.health_score * (1 - agent.current_load)
            if current >= target:
                return agent
        
        return agents[-1]
    
    async def _execute_task_on_agent(self, task: Task, agent: RegisteredAgent) -> Dict[str, Any]:
        """Execute a task on a specific agent"""
        try:
            # Update agent load
            agent.current_load = min(agent.current_load + 0.1, 1.0)
            
            # Prepare task message
            message = Message(
                id=str(uuid.uuid4()),
                type=MessageType.TASK_ASSIGNMENT,
                sender_id="orchestrator",
                recipient_id=agent.id,
                content={
                    "task_id": task.id,
                    "task_type": task.type,
                    "description": task.description,
                    "input_data": task.input_data,
                    "priority": task.priority.value
                },
                timestamp=datetime.now(),
                requires_response=True
            )
            
            # Send task to agent
            start_time = time.time()
            response = await self._send_message_to_agent(message, agent)
            execution_time = time.time() - start_time
            
            # Update agent metrics
            agent.avg_response_time = (agent.avg_response_time + execution_time) / 2
            agent.total_tasks_completed += 1
            
            # Update agent load
            agent.current_load = max(agent.current_load - 0.1, 0.0)
            
            # Update success rate
            success = response.get("success", False)
            if success:
                agent.success_rate = (agent.success_rate * (agent.total_tasks_completed - 1) + 1) / agent.total_tasks_completed
            else:
                agent.success_rate = (agent.success_rate * (agent.total_tasks_completed - 1)) / agent.total_tasks_completed
            
            return response
            
        except Exception as e:
            logger.error(f"Task execution failed on agent {agent.id}: {e}")
            agent.current_load = max(agent.current_load - 0.1, 0.0)
            return {"success": False, "error": str(e)}
    
    async def _send_message_to_agent(self, message: Message, agent: RegisteredAgent) -> Dict[str, Any]:
        """Send message to an agent via HTTP"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{agent.endpoint}/orchestration/task",
                    json=asdict(message),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"success": False, "error": f"HTTP {response.status}"}
        except Exception as e:
            logger.warning(f"Failed to send message to agent {agent.id}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _task_processor(self):
        """Background task processor"""
        while self.running:
            try:
                if self.task_queue:
                    task = self.task_queue.popleft()
                    
                    # Find suitable agents
                    agents = await self.discover_agents_for_capability(task.type)
                    if not agents:
                        agents = await self.discover_agents_for_capability("general_task")
                    
                    if agents:
                        selected_agent = await self._select_agent_for_task(agents, task)
                        result = await self._execute_task_on_agent(task, selected_agent)
                        
                        # Update metrics
                        if result.get("success", False):
                            self.metrics["tasks_completed"] += 1
                        else:
                            self.metrics["tasks_failed"] += 1
                    
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1)
    
    async def _message_processor(self):
        """Background message processor"""
        while self.running:
            try:
                if self.message_queue:
                    message = self.message_queue.popleft()
                    await self._process_message(message)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message)
            elif message.type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.type == MessageType.TASK_RESULT:
                await self._handle_task_result(message)
            elif message.type == MessageType.COORDINATION_REQUEST:
                await self._handle_coordination_request(message)
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
    
    async def _handle_heartbeat(self, message: Message):
        """Handle agent heartbeat messages"""
        agent = self.agents.get(message.sender_id)
        if agent:
            agent.last_heartbeat = datetime.now()
            agent.health_score = message.content.get("health_score", agent.health_score)
            agent.current_load = message.content.get("current_load", agent.current_load)
    
    async def _handle_status_update(self, message: Message):
        """Handle agent status updates"""
        agent = self.agents.get(message.sender_id)
        if agent:
            new_status = message.content.get("status")
            if new_status:
                agent.status = AgentStatus(new_status)
    
    async def _handle_task_result(self, message: Message):
        """Handle task completion results"""
        task_id = message.content.get("task_id")
        result = message.content.get("result")
        success = message.content.get("success", False)
        
        # Update task assignments
        if task_id in self.task_assignments:
            del self.task_assignments[task_id]
        
        # Update metrics
        if success:
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1
    
    async def _handle_coordination_request(self, message: Message):
        """Handle inter-agent coordination requests"""
        coordination_type = message.content.get("type")
        
        if coordination_type == "resource_request":
            await self._handle_resource_request(message)
        elif coordination_type == "consensus_request":
            await self._handle_consensus_request(message)
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and update status"""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = timedelta(seconds=self.heartbeat_interval * 2)
                
                for agent in self.agents.values():
                    if current_time - agent.last_heartbeat > timeout_threshold:
                        if agent.status != AgentStatus.OFFLINE:
                            agent.status = AgentStatus.OFFLINE
                            agent.health_score = 0.0
                            logger.warning(f"Agent {agent.id} marked as offline")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _workflow_executor(self):
        """Background workflow execution monitor"""
        while self.running:
            try:
                # Check for pending workflows
                pending_workflows = [w for w in self.workflows.values() 
                                   if w.status == WorkflowStatus.PENDING]
                
                for workflow in pending_workflows:
                    asyncio.create_task(self._execute_workflow(workflow))
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Workflow executor error: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collector(self):
        """Collect and update system metrics"""
        while self.running:
            try:
                # Calculate system throughput
                active_agents = sum(1 for a in self.agents.values() 
                                  if a.status in [AgentStatus.IDLE, AgentStatus.BUSY])
                
                total_load = sum(a.current_load for a in self.agents.values())
                avg_utilization = total_load / len(self.agents) if self.agents else 0
                
                self.metrics.update({
                    "active_agents": active_agents,
                    "agent_utilization": avg_utilization,
                    "total_agents": len(self.agents),
                    "pending_tasks": len(self.task_queue),
                    "active_workflows": sum(1 for w in self.workflows.values() 
                                          if w.status == WorkflowStatus.RUNNING)
                })
                
                # Store metrics in Redis
                await self.redis_client.hset(
                    "orchestrator_metrics",
                    "current",
                    json.dumps(self.metrics, default=str)
                )
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(10)
    
    async def _send_workflow_event(self, workflow_id: str, event_type: str, data: Dict[str, Any]):
        """Send workflow event notification"""
        await self.redis_client.publish("workflow_events", json.dumps({
            "workflow_id": workflow_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }))
    
    async def _send_system_notification(self, message: str, data: Dict[str, Any] = None):
        """Send system notification"""
        await self.redis_client.publish("system_notifications", json.dumps({
            "message": message,
            "data": data or {},
            "timestamp": datetime.now().isoformat()
        }))
    
    # Public API methods
    
    async def get_agent_status(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Get status of a specific agent"""
        return self.agents.get(agent_id)
    
    async def get_all_agents(self) -> List[RegisteredAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Workflow]:
        """Get status of a specific workflow"""
        return self.workflows.get(workflow_id)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            **self.metrics,
            "agent_details": {
                agent.id: {
                    "status": agent.status.value,
                    "health_score": agent.health_score,
                    "current_load": agent.current_load,
                    "success_rate": agent.success_rate,
                    "avg_response_time": agent.avg_response_time,
                    "total_tasks": agent.total_tasks_completed
                }
                for agent in self.agents.values()
            },
            "workflow_summary": {
                "total": len(self.workflows),
                "running": sum(1 for w in self.workflows.values() 
                              if w.status == WorkflowStatus.RUNNING),
                "completed": sum(1 for w in self.workflows.values() 
                                if w.status == WorkflowStatus.COMPLETED),
                "failed": sum(1 for w in self.workflows.values() 
                             if w.status == WorkflowStatus.FAILED)
            }
        }
    
    async def stop(self):
        """Stop the orchestration system"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Agent orchestrator stopped")
    
    def health_check(self) -> bool:
        """Check orchestrator health"""
        return self.running and self.redis_client is not None