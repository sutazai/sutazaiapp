"""
Universal Agent Client Library
Provides unified interface to interact with all 38 AI agents in the SutazAI automation system.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure structured logging (Rule 8 compliance)
from backend.app.core.logging_config import get_logger
logger = get_logger(__name__)


class AgentType(Enum):
    """Enumeration of all available agent types in the SutazAI system."""
    # Core System Agents
    SYSTEM_ARCHITECT = "system-architect"
    AUTONOMOUS_SYSTEM_CONTROLLER = "autonomous-system-controller"
    AI_AGENT_ORCHESTRATOR = "agent-orchestrator"
    
    # Infrastructure & DevOps
    INFRASTRUCTURE_DEVOPS_MANAGER = "infrastructure-devops-manager"
    DEPLOYMENT_AUTOMATION_MASTER = "deployment-automation-master"
    HARDWARE_RESOURCE_OPTIMIZER = "hardware-resource-optimizer"
    
    # AI & ML Specialists
    OLLAMA_INTEGRATION_SPECIALIST = "ollama-integration-specialist"
    SENIOR_AI_ENGINEER = "senior-engineer"
    DEEP_LEARNING_BRAIN_MANAGER = "deep-learning-coordinator-manager"
    
    # Development Specialists
    CODE_GENERATION_IMPROVER = "code-generation-improver"
    OPENDEVIN_CODE_GENERATOR = "opendevin-code-generator"
    SENIOR_FRONTEND_DEVELOPER = "senior-frontend-developer"
    SENIOR_BACKEND_DEVELOPER = "senior-backend-developer"
    
    # Quality & Testing
    TESTING_QA_VALIDATOR = "testing-qa-validator"
    SECURITY_PENTESTING_SPECIALIST = "security-pentesting-specialist"
    KALI_SECURITY_SPECIALIST = "kali-security-specialist"
    SEMGREP_SECURITY_ANALYZER = "semgrep-security-analyzer"
    
    # Workflow & Process Management
    AI_PRODUCT_MANAGER = "ai-product-manager"
    AI_SCRUM_MASTER = "ai-scrum-master"
    TASK_ASSIGNMENT_COORDINATOR = "task-assignment-coordinator"
    
    # Automation & Integration
    SHELL_AUTOMATION_SPECIALIST = "shell-automation-specialist"
    BROWSER_AUTOMATION_ORCHESTRATOR = "browser-automation-orchestrator"
    LANGFLOW_WORKFLOW_DESIGNER = "langflow-workflow-designer"
    DIFY_AUTOMATION_SPECIALIST = "dify-automation-specialist"
    FLOWISEAI_FLOW_MANAGER = "flowiseai-flow-manager"
    
    # Specialized Services
    DOCUMENT_KNOWLEDGE_MANAGER = "document-knowledge-manager"
    PRIVATE_DATA_ANALYST = "private-data-analyst"
    FINANCIAL_ANALYSIS_SPECIALIST = "financial-analysis-specialist"
    JARVIS_VOICE_INTERFACE = "jarvis-voice-interface"
    
    # System Management
    SYSTEM_OPTIMIZER_REORGANIZER = "system-optimizer-reorganizer"
    CONTEXT_OPTIMIZATION_ENGINEER = "context-optimization-engineer"
    COMPLEX_PROBLEM_SOLVER = "complex-problem-solver"
    AI_AGENT_CREATOR = "ai-agent-creator"
    
    # External Platform Agents
    AUTOGPT_AUTONOMOUS_EXECUTOR = "autogpt-autonomous-executor"
    AGENTGPT_AUTONOMOUS_EXECUTOR = "agentgpt-autonomous-executor"
    AGENTZERO_COORDINATOR = "agentzero-coordinator"
    BIGAGI_SYSTEM_MANAGER = "bigagi-system-manager"
    LOCALAGI_ORCHESTRATION_MANAGER = "localagi-orchestration-manager"


class Priority(Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    URGENT = "urgent"


class AgentStatus(Enum):
    """Agent operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class AgentCapability:
    """Represents a specific capability of an agent."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration: Optional[int] = None  # in seconds


@dataclass
class AgentInfo:
    """Complete information about an agent."""
    id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    status: AgentStatus
    priority: Priority
    endpoint: str
    health_check_url: str
    version: str = "1.0"
    last_heartbeat: Optional[datetime] = None
    load_percentage: float = 0.0
    response_time_ms: float = 0.0


@dataclass
class TaskRequest:
    """Represents a task request to an agent."""
    task_id: str
    agent_type: AgentType
    task_description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    timeout: int = 300  # seconds
    context: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskResponse:
    """Response from an agent after task execution."""
    task_id: str
    agent_id: str
    status: str
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class AgentClientInterface(ABC):
    """Abstract interface for agent clients."""
    
    @abstractmethod
    async def execute_task(self, request: TaskRequest) -> TaskResponse:
        """Execute a task on the agent."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the agent is healthy."""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities."""
        pass


class HTTPAgentClient(AgentClientInterface):
    """HTTP-based agent client implementation."""
    
    def __init__(self, agent_info: AgentInfo, session: Optional[aiohttp.ClientSession] = None):
        self.agent_info = agent_info
        self.session = session
        self._own_session = session is None
    
    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()
    
    async def execute_task(self, request: TaskRequest) -> TaskResponse:
        """Execute a task via HTTP API."""
        start_time = time.time()
        
        try:
            payload = {
                "task_id": request.task_id,
                "task_description": request.task_description,
                "parameters": request.parameters,
                "priority": request.priority.value,
                "context": request.context
            }
            
            timeout = aiohttp.ClientTimeout(total=request.timeout)
            
            async with self.session.post(
                f"{self.agent_info.endpoint}/execute",
                json=payload,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                result_data = await response.json()
                
                execution_time = time.time() - start_time
                
                return TaskResponse(
                    task_id=request.task_id,
                    agent_id=self.agent_info.id,
                    status=result_data.get("status", "completed"),
                    result=result_data.get("result"),
                    error=result_data.get("error"),
                    execution_time=execution_time,
                    metadata=result_data.get("metadata", {})
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed for {self.agent_info.id}: {str(e)}")
            
            return TaskResponse(
                task_id=request.task_id,
                agent_id=self.agent_info.id,
                status="error",
                error=str(e),
                execution_time=execution_time
            )
    
    async def health_check(self) -> bool:
        """Perform health check via HTTP."""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with self.session.get(
                self.agent_info.health_check_url,
                timeout=timeout
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed for {self.agent_info.id}: {str(e)}")
            return False
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities via HTTP."""
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with self.session.get(
                f"{self.agent_info.endpoint}/capabilities",
                timeout=timeout
            ) as response:
                response.raise_for_status()
                caps_data = await response.json()
                
                return [
                    AgentCapability(
                        name=cap["name"],
                        description=cap["description"],
                        input_types=cap.get("input_types", []),
                        output_types=cap.get("output_types", []),
                        prerequisites=cap.get("prerequisites", []),
                        estimated_duration=cap.get("estimated_duration")
                    )
                    for cap in caps_data.get("capabilities", [])
                ]
        except Exception as e:
            logger.error(f"Failed to get capabilities for {self.agent_info.id}: {str(e)}")
            # Return empty list on capability retrieval failure
            return []  # Valid empty list: Failed to get agent capabilities


class UniversalAgentClient:
    """Universal client for interacting with all agents in the SutazAI system."""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        base_url: str = "http://localhost",
        default_timeout: int = 300
    ):
        self.base_url = base_url
        self.default_timeout = default_timeout
        self.agents: Dict[str, AgentInfo] = {}
        self.clients: Dict[str, AgentClientInterface] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        if config_file:
            self.load_config(config_file)
        else:
            self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent configurations."""
        # Define all 38 agents with their default configurations
        agent_configs = [
            # Core System Agents
            {
                "id": "system-architect",
                "name": "automation System Architect",
                "description": "Master architect for automation system design and optimization",
                "port": 8001,
                "capabilities": ["system_design", "architecture_optimization", "integration_planning"]
            },
            {
                "id": "autonomous-system-controller",
                "name": "Autonomous System Controller",
                "description": "Master orchestrator for autonomous operations",
                "port": 8002,
                "capabilities": ["autonomous_control", "decision_making", "resource_allocation"]
            },
            {
                "id": "ai-agent-orchestrator",
                "name": "AI Agent Orchestrator",
                "description": "Coordinates multi-agent workflows",
                "port": 8003,
                "capabilities": ["workflow_orchestration", "agent_coordination", "task_distribution"]
            },
            
            # Infrastructure & DevOps
            {
                "id": "infrastructure-devops-manager",
                "name": "Infrastructure DevOps Manager",
                "description": "Manages Docker, Kubernetes, and CI/CD",
                "port": 8004,
                "capabilities": ["container_management", "deployment_automation", "monitoring"]
            },
            {
                "id": "deployment-automation-master",
                "name": "Deployment Automation Master",
                "description": "Ensures reliable system deployment",
                "port": 8005,
                "capabilities": ["deployment_automation", "error_recovery", "health_validation"]
            },
            {
                "id": "hardware-resource-optimizer",
                "name": "Hardware Resource Optimizer",
                "description": "Optimizes performance within hardware constraints",
                "port": 8006,
                "capabilities": ["resource_monitoring", "performance_optimization", "memory_management"]
            },
            
            # AI & ML Specialists
            {
                "id": "ollama-integration-specialist",
                "name": "Ollama Integration Specialist",
                "description": "Expert in Ollama configuration and optimization",
                "port": 8007,
                "capabilities": ["model_management", "api_configuration", "performance_tuning"]
            },
            {
                "port": 8008,
                "capabilities": ["proxy_management", "api_translation", "model_mapping"]
            },
            {
                "id": "senior-ai-engineer",
                "name": "Senior AI Engineer",
                "description": "AI/ML expert for model integration and RAG systems",
                "port": 8009,
                "capabilities": ["ml_architecture", "model_optimization", "rag_development"]
            },
            {
                "id": "deep-learning-coordinator-manager",
                "name": "Deep Learning Coordinator Manager",
                "description": "Manages processing intelligence core",
                "port": 8010,
                "capabilities": ["processing_architecture", "continuous_learning", "meta_learning"]
            },
            
            # Development Specialists
            {
                "id": "code-generation-improver",
                "name": "Code Generation Improver",
                "description": "Analyzes and improves code quality",
                "port": 8011,
                "capabilities": ["code_analysis", "refactoring", "optimization"]
            },
            {
                "id": "opendevin-code-generator",
                "name": "OpenDevin Code Generator",
                "description": "Manages OpenDevin platform for autonomous software engineering",
                "port": 8012,
                "capabilities": ["autonomous_coding", "debugging", "refactoring"]
            },
            {
                "id": "senior-frontend-developer",
                "name": "Senior Frontend Developer",
                "description": "Frontend specialist for modern web interfaces",
                "port": 8013,
                "capabilities": ["frontend_development", "ui_design", "real_time_features"]
            },
            {
                "id": "senior-backend-developer",
                "name": "Senior Backend Developer",
                "description": "Backend specialist for APIs and databases",
                "port": 8014,
                "capabilities": ["backend_development", "api_design", "database_management"]
            },
            
            # Quality & Testing
            {
                "id": "testing-qa-validator",
                "name": "Testing QA Validator",
                "description": "Comprehensive testing and quality assurance",
                "port": 8015,
                "capabilities": ["test_automation", "quality_assurance", "security_testing"]
            },
            {
                "id": "security-pentesting-specialist",
                "name": "Security Pentesting Specialist",
                "description": "Security expert for vulnerability assessment",
                "port": 8016,
                "capabilities": ["security_auditing", "penetration_testing", "vulnerability_scanning"]
            },
            {
                "id": "kali-security-specialist",
                "name": "Kali Security Specialist",
                "description": "Advanced security testing with Kali Linux tools",
                "port": 8017,
                "capabilities": ["advanced_pentesting", "exploit_development", "forensics"]
            },
            {
                "id": "semgrep-security-analyzer",
                "name": "Semgrep Security Analyzer",
                "description": "Static analysis security testing with Semgrep",
                "port": 8018,
                "capabilities": ["static_analysis", "code_security", "vulnerability_detection"]
            },
            
            # Workflow & Process Management
            {
                "id": "ai-product-manager",
                "name": "AI Product Manager",
                "description": "Central coordinator with web search capabilities",
                "port": 8019,
                "capabilities": ["requirement_analysis", "web_search", "project_coordination"]
            },
            {
                "id": "ai-scrum-master",
                "name": "AI Scrum Master",
                "description": "Facilitates agile processes and manages sprints",
                "port": 8020,
                "capabilities": ["sprint_management", "impediment_removal", "process_optimization"]
            },
            {
                "id": "task-assignment-coordinator",
                "name": "Task Assignment Coordinator",
                "description": "Automatically assigns tasks to suitable agents",
                "port": 8021,
                "capabilities": ["task_analysis", "agent_matching", "workload_balancing"]
            },
            
            # Automation & Integration
            {
                "id": "shell-automation-specialist",
                "name": "Shell Automation Specialist",
                "description": "Expert in shell scripting and command automation",
                "port": 8022,
                "capabilities": ["shell_scripting", "command_automation", "system_tasks"]
            },
            {
                "id": "browser-automation-orchestrator",
                "name": "Browser Automation Orchestrator",
                "description": "Manages browser automation and web scraping",
                "port": 8023,
                "capabilities": ["web_automation", "scraping", "browser_testing"]
            },
            {
                "id": "langflow-workflow-designer",
                "name": "Langflow Workflow Designer",
                "description": "Creates visual AI workflows using Langflow",
                "port": 8024,
                "capabilities": ["visual_workflows", "drag_drop_design", "flow_management"]
            },
            {
                "id": "dify-automation-specialist",
                "name": "Dify Automation Specialist",
                "description": "Manages Dify platform for AI application development",
                "port": 8025,
                "capabilities": ["app_development", "workflow_automation", "api_management"]
            },
            {
                "id": "flowiseai-flow-manager",
                "name": "FlowiseAI Flow Manager",
                "description": "Manages FlowiseAI for LLM application building",
                "port": 8026,
                "capabilities": ["llm_apps", "chatflow_design", "integration_management"]
            },
            
            # Specialized Services
            {
                "id": "document-knowledge-manager",
                "name": "Document Knowledge Manager",
                "description": "Manages documentation and knowledge bases",
                "port": 8027,
                "capabilities": ["documentation", "knowledge_management", "rag_systems"]
            },
            {
                "id": "private-data-analyst",
                "name": "Private Data Analyst",
                "description": "Secure document analysis using PrivateGPT",
                "port": 8028,
                "capabilities": ["private_processing", "data_anonymization", "secure_analysis"]
            },
            {
                "id": "financial-analysis-specialist",
                "name": "Financial Analysis Specialist",
                "description": "Financial analysis and trading strategies using FinRobot",
                "port": 8029,
                "capabilities": ["market_analysis", "trading_strategies", "risk_assessment"]
            },
            {
                "id": "jarvis-voice-interface",
                "name": "Jarvis Voice Interface",
                "description": "Voice-controlled AI assistant",
                "port": 8030,
                "capabilities": ["voice_recognition", "natural_language", "text_to_speech"]
            },
            
            # System Management
            {
                "id": "system-optimizer-reorganizer",
                "name": "System Optimizer Reorganizer",
                "description": "Maintains system organization and optimization",
                "port": 8031,
                "capabilities": ["system_cleanup", "organization", "dependency_management"]
            },
            {
                "id": "context-optimization-engineer",
                "name": "Context Optimization Engineer",
                "description": "Optimizes context windows and prompt engineering",
                "port": 8032,
                "capabilities": ["context_management", "prompt_optimization", "token_reduction"]
            },
            {
                "id": "complex-problem-solver",
                "name": "Complex Problem Solver",
                "description": "Meta-cognitive agent for complex problem solving",
                "port": 8033,
                "capabilities": ["problem_analysis", "research", "solution_synthesis"]
            },
            {
                "id": "ai-agent-creator",
                "name": "AI Agent Creator",
                "description": "Meta-agent that creates new AI agents",
                "port": 8034,
                "capabilities": ["gap_analysis", "agent_design", "system_evolution"]
            },
            
            # External Platform Agents
            {
                "id": "autogpt-autonomous-executor",
                "name": "AutoGPT Autonomous Executor",
                "description": "Autonomous task execution using AutoGPT",
                "port": 8035,
                "capabilities": ["autonomous_execution", "goal_planning", "resource_management"]
            },
            {
                "id": "agentgpt-autonomous-executor",
                "name": "AgentGPT Autonomous Executor",
                "description": "Web-based autonomous AI agent using AgentGPT",
                "port": 8036,
                "capabilities": ["web_automation", "goal_achievement", "task_breakdown"]
            },
            {
                "id": "agentzero-coordinator",
                "name": "AgentZero Coordinator",
                "description": "Multi-agent coordination using AgentZero framework",
                "port": 8037,
                "capabilities": ["multi_agent_coordination", "dynamic_planning", "tool_usage"]
            },
            {
                "id": "bigagi-system-manager",
                "name": "BigAGI System Manager",
                "description": "Advanced AI system management using BigAGI",
                "port": 8038,
                "capabilities": ["system_management", "multi_model_support", "advanced_ui"]
            },
            {
                "id": "localagi-orchestration-manager",
                "name": "LocalAGI Orchestration Manager",
                "description": "Local automation system orchestration",
                "port": 8039,
                "capabilities": ["local_orchestration", "privacy_focused", "offline_capability"]
            }
        ]
        
        # Initialize agent info objects
        for config in agent_configs:
            capabilities = [
                AgentCapability(
                    name=cap,
                    description=f"{cap.replace('_', ' ').title()} capability",
                    input_types=["text", "json"],
                    output_types=["text", "json"]
                )
                for cap in config["capabilities"]
            ]
            
            agent_info = AgentInfo(
                id=config["id"],
                name=config["name"],
                description=config["description"],
                capabilities=capabilities,
                status=AgentStatus.OFFLINE,
                priority=Priority.HIGH,
                endpoint=f"{self.base_url}:{config['port']}",
                health_check_url=f"{self.base_url}:{config['port']}/health"
            )
            
            self.agents[config["id"]] = agent_info
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        
        # Initialize HTTP clients for all agents
        for agent_id, agent_info in self.agents.items():
            self.clients[agent_id] = HTTPAgentClient(agent_info, self.session)
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def load_config(self, config_file: str):
        """Load agent configurations from file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            for agent_config in config.get("agents", []):
                # Convert config to AgentInfo object
                capabilities = [
                    AgentCapability(
                        name=cap,
                        description=f"{cap.replace('_', ' ').title()} capability",
                        input_types=["text", "json"],
                        output_types=["text", "json"]
                    )
                    for cap in agent_config.get("capabilities", [])
                ]
                
                agent_info = AgentInfo(
                    id=agent_config["id"],
                    name=agent_config["name"],
                    description=agent_config["description"],
                    capabilities=capabilities,
                    status=AgentStatus.OFFLINE,
                    priority=Priority(agent_config.get("priority", "medium")),
                    endpoint=agent_config.get("endpoint", f"{self.base_url}:8000"),
                    health_check_url=agent_config.get("health_check_url", f"{self.base_url}:8000/health")
                )
                
                self.agents[agent_config["id"]] = agent_info
                
        except Exception as e:
            logger.error(f"Failed to load config file {config_file}: {str(e)}")
            self._initialize_default_agents()
    
    async def execute_task(
        self,
        agent_type: Union[AgentType, str],
        task_description: str,
        parameters: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.MEDIUM,
        timeout: int = None
    ) -> TaskResponse:
        """Execute a task on a specific agent."""
        if isinstance(agent_type, AgentType):
            agent_id = agent_type.value
        else:
            agent_id = agent_type
        
        if agent_id not in self.clients:
            raise ValueError(f"Agent {agent_id} not found")
        
        task_request = TaskRequest(
            task_id=f"{agent_id}_{int(time.time())}",
            agent_type=AgentType(agent_id),
            task_description=task_description,
            parameters=parameters or {},
            priority=priority,
            timeout=timeout or self.default_timeout
        )
        
        client = self.clients[agent_id]
        return await client.execute_task(task_request)
    
    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[TaskResponse]:
        """Execute multiple tasks in parallel."""
        task_coroutines = []
        
        for task in tasks:
            coro = self.execute_task(
                agent_type=task["agent_type"],
                task_description=task["task_description"],
                parameters=task.get("parameters"),
                priority=Priority(task.get("priority", "medium")),
                timeout=task.get("timeout")
            )
            task_coroutines.append(coro)
        
        return await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all agents."""
        health_results = {}
        
        tasks = []
        for agent_id, client in self.clients.items():
            tasks.append((agent_id, client.health_check()))
        
        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        
        for i, (agent_id, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                health_results[agent_id] = False
                logger.error(f"Health check failed for {agent_id}: {str(result)}")
            else:
                health_results[agent_id] = result
                # Update agent status
                if agent_id in self.agents:
                    self.agents[agent_id].status = AgentStatus.ONLINE if result else AgentStatus.OFFLINE
        
        return health_results
    
    def get_agent_info(self, agent_id: str) -> Optional[AgentInfo]:
        """Get information about a specific agent."""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[AgentInfo]:
        """List all available agents."""
        return list(self.agents.values())
    
    def find_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Find agents that have a specific capability."""
        matching_agents = []
        
        for agent in self.agents.values():
            for cap in agent.capabilities:
                if capability.lower() in cap.name.lower():
                    matching_agents.append(agent)
                    break
        
        return matching_agents
    
    def get_agents_by_priority(self, priority: Priority) -> List[AgentInfo]:
        """Get agents filtered by priority level."""
        return [agent for agent in self.agents.values() if agent.priority == priority]
    
    def get_online_agents(self) -> List[AgentInfo]:
        """Get all currently online agents."""
        return [agent for agent in self.agents.values() if agent.status == AgentStatus.ONLINE]


# Convenience functions for common operations
async def quick_execute(
    agent_type: Union[AgentType, str],
    task_description: str,
    parameters: Optional[Dict[str, Any]] = None,
    base_url: str = "http://localhost"
) -> TaskResponse:
    """Quick execute a single task without managing client lifecycle."""
    async with UniversalAgentClient(base_url=base_url) as client:
        return await client.execute_task(agent_type, task_description, parameters)


async def batch_execute(
    tasks: List[Dict[str, Any]],
    base_url: str = "http://localhost"
) -> List[TaskResponse]:
    """Execute multiple tasks in batch."""
    async with UniversalAgentClient(base_url=base_url) as client:
        return await client.execute_parallel_tasks(tasks)


async def system_health_check(base_url: str = "http://localhost") -> Dict[str, bool]:
    """Quick system-wide health check."""
    async with UniversalAgentClient(base_url=base_url) as client:
        return await client.health_check_all()


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the Universal Agent Client."""
        
        # Initialize client
        async with UniversalAgentClient() as client:
            
            # List all agents
            agents = client.list_agents()
            logger.info(f"AGENT_DEMO - Found {len(agents)} agents")
            
            # Find agents with specific capabilities
            code_agents = client.find_agents_by_capability("code")
            logger.info(f"AGENT_DEMO - Found {len(code_agents)} agents with code capabilities")
            
            # Execute a simple task
            try:
                response = await client.execute_task(
                    AgentType.CODE_GENERATION_IMPROVER,
                    "Analyze the quality of a Python function",
                    parameters={"code": "def hello(): logger.info('Hello, World!')"}
                )
                logger.info(f"AGENT_DEMO - Task result: {response.status}")
                
            except Exception as e:
                logger.error(f"AGENT_DEMO - Task execution failed: {str(e)}")
            
            # Perform system health check
            health_status = await client.health_check_all()
            online_count = sum(1 for status in health_status.values() if status)
            logger.info(f"AGENT_DEMO - Health check: {online_count}/{len(health_status)} agents online")
    
    # Run the example
    asyncio.run(main())