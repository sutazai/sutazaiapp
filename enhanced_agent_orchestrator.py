#!/usr/bin/env python3
"""
Enhanced Agent Orchestrator for SutazAI AGI/ASI System
Complete integration and management of all AI agents
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import subprocess
import docker
import redis
import psycopg2
from sqlalchemy import create_engine, text
import threading
from concurrent.futures import ThreadPoolExecutor

# Enhanced logging setup
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

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10

@dataclass
class AgentConfig:
    name: str
    type: str
    url: str
    port: int
    capabilities: List[str]
    health_endpoint: str
    status: AgentStatus = AgentStatus.OFFLINE
    last_heartbeat: Optional[datetime] = None
    task_count: int = 0
    error_count: int = 0

@dataclass
class Task:
    id: str
    type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EnhancedAgentOrchestrator:
    """
    Complete orchestration system for all SutazAI AI agents
    Manages lifecycle, health monitoring, task distribution, and coordination
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.tasks: Dict[str, Task] = {}
        self.docker_client = docker.from_env()
        self.redis_client = None
        self.db_engine = None
        self.running = False
        self.health_check_interval = 30
        self.task_executor = ThreadPoolExecutor(max_workers=20)
        
        # Initialize connections
        self._initialize_connections()
        
        # Register all agents
        self._register_all_agents()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Redis connection
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Database connection
            database_url = "postgresql://sutazai:sutazai_password@localhost:5432/sutazai"
            self.db_engine = create_engine(database_url)
            
            # Test database connection
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
    
    def _register_all_agents(self):
        """Register all available AI agents"""
        
        # Internal orchestrator agents
        internal_agents = [
            AgentConfig(
                name="CodeMaster",
                type="code_generator", 
                url="http://localhost:8000",
                port=8000,
                capabilities=["python", "javascript", "java", "cpp", "rust", "go", "code_review", "debugging"],
                health_endpoint="/api/agents/CodeMaster/health"
            ),
            AgentConfig(
                name="SecurityGuard",
                type="security_analyst",
                url="http://localhost:8000", 
                port=8000,
                capabilities=["vulnerability_scan", "security_audit", "penetration_testing", "compliance_check"],
                health_endpoint="/api/agents/SecurityGuard/health"
            ),
            AgentConfig(
                name="DocProcessor", 
                type="document_processor",
                url="http://localhost:8000",
                port=8000,
                capabilities=["pdf_processing", "text_extraction", "summarization", "translation"],
                health_endpoint="/api/agents/DocProcessor/health"
            ),
            AgentConfig(
                name="FinAnalyst",
                type="financial_analyst", 
                url="http://localhost:8000",
                port=8000,
                capabilities=["market_analysis", "risk_assessment", "portfolio_optimization", "financial_modeling"],
                health_endpoint="/api/agents/FinAnalyst/health"
            ),
            AgentConfig(
                name="WebAutomator",
                type="web_automator",
                url="http://localhost:8000",
                port=8000, 
                capabilities=["web_scraping", "browser_automation", "form_filling", "data_extraction"],
                health_endpoint="/api/agents/WebAutomator/health"
            ),
            AgentConfig(
                name="TaskCoordinator",
                type="task_coordinator",
                url="http://localhost:8000",
                port=8000,
                capabilities=["task_planning", "workflow_management", "resource_allocation", "coordination"],
                health_endpoint="/api/agents/TaskCoordinator/health"
            ),
            AgentConfig(
                name="SystemMonitor", 
                type="system_monitor",
                url="http://localhost:8000",
                port=8000,
                capabilities=["health_monitoring", "performance_tracking", "alerting", "diagnostics"],
                health_endpoint="/api/agents/SystemMonitor/health"
            ),
            AgentConfig(
                name="DataScientist",
                type="data_scientist",
                url="http://localhost:8000",
                port=8000,
                capabilities=["data_analysis", "machine_learning", "statistical_modeling", "visualization"],
                health_endpoint="/api/agents/DataScientist/health"
            ),
            AgentConfig(
                name="DevOpsEngineer",
                type="devops_engineer", 
                url="http://localhost:8000",
                port=8000,
                capabilities=["container_management", "ci_cd", "infrastructure_automation", "deployment"],
                health_endpoint="/api/agents/DevOpsEngineer/health"
            ),
            AgentConfig(
                name="GeneralAssistant",
                type="general_assistant",
                url="http://localhost:8000",
                port=8000,
                capabilities=["conversation", "general_knowledge", "problem_solving", "assistance"],
                health_endpoint="/api/agents/GeneralAssistant/health"
            )
        ]
        
        # External specialized agents
        external_agents = [
            AgentConfig(
                name="AutoGPT",
                type="task_automation",
                url="http://localhost:8080", 
                port=8080,
                capabilities=["task_automation", "web_browsing", "file_operations", "code_execution"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="LocalAGI",
                type="agi_orchestration",
                url="http://localhost:8082",
                port=8082,
                capabilities=["agi_orchestration", "multi_agent_coordination", "task_planning"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="TabbyML", 
                type="code_completion",
                url="http://localhost:8081",
                port=8081,
                capabilities=["code_completion", "code_suggestions", "ide_integration"],
                health_endpoint="/v1/health"
            ),
            AgentConfig(
                name="BrowserUse",
                type="web_automation",
                url="http://localhost:8083",
                port=8083,
                capabilities=["web_automation", "form_filling", "data_extraction"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="Skyvern",
                type="web_automation", 
                url="http://localhost:8084",
                port=8084,
                capabilities=["web_scraping", "browser_automation", "data_mining"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="Documind",
                type="document_processing",
                url="http://localhost:8085",
                port=8085,
                capabilities=["document_processing", "text_extraction", "pdf_parsing"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="FinRobot",
                type="financial_analysis",
                url="http://localhost:8086", 
                port=8086,
                capabilities=["financial_analysis", "market_data", "portfolio_optimization"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="GPT-Engineer",
                type="code_generation",
                url="http://localhost:8087",
                port=8087,
                capabilities=["code_generation", "project_scaffolding", "architecture_design"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="Aider",
                type="code_editing",
                url="http://localhost:8088",
                port=8088,
                capabilities=["code_editing", "refactoring", "bug_fixing"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="LangFlow",
                type="workflow_orchestration",
                url="http://localhost:7860",
                port=7860,
                capabilities=["workflow_design", "flow_orchestration", "visual_programming"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="Dify",
                type="app_development",
                url="http://localhost:5001",
                port=5001,
                capabilities=["app_development", "workflow_automation", "llm_orchestration"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="AutoGen",
                type="multi_agent_conversation",
                url="http://localhost:8092", 
                port=8092,
                capabilities=["multi_agent_conversation", "collaborative_problem_solving"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="OpenWebUI",
                type="chat_interface",
                url="http://localhost:8089",
                port=8089,
                capabilities=["chat_interface", "model_interaction", "conversation_management"],
                health_endpoint="/health"
            ),
            AgentConfig(
                name="BigAGI",
                type="advanced_ai_interface",
                url="http://localhost:8090",
                port=8090,
                capabilities=["advanced_ai_interface", "multi_model_support", "conversation_branching"],
                health_endpoint="/api/health"
            ),
            AgentConfig(
                name="AgentZero",
                type="specialized_agent",
                url="http://localhost:8091",
                port=8091,
                capabilities=["specialized_tasks", "custom_workflows", "agent_framework"],
                health_endpoint="/health"
            )
        ]
        
        # Register all agents
        for agent in internal_agents + external_agents:
            self.agents[agent.name] = agent
            logger.info(f"Registered agent: {agent.name} ({agent.type})")
    
    def _start_background_tasks(self):
        """Start background monitoring and management tasks"""
        self.running = True
        
        # Health monitoring thread
        health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        health_thread.start()
        
        # Task processor thread  
        task_thread = threading.Thread(target=self._task_processor_loop, daemon=True)
        task_thread.start()
        
        # Agent startup thread
        startup_thread = threading.Thread(target=self._start_all_agents, daemon=True)
        startup_thread.start()
        
        logger.info("Background tasks started")
    
    def _health_monitor_loop(self):
        """Continuous health monitoring for all agents"""
        while self.running:
            try:
                for agent_name, agent in self.agents.items():
                    self._check_agent_health(agent)
                time.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_agent_health(self, agent: AgentConfig):
        """Check health of individual agent"""
        try:
            # First check if Docker container is running
            container_running = self._is_container_running(agent.name)
            
            if not container_running:
                agent.status = AgentStatus.OFFLINE
                return
            
            # Check HTTP health endpoint
            health_url = f"{agent.url}{agent.health_endpoint}"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                agent.status = AgentStatus.ONLINE
                agent.last_heartbeat = datetime.now()
                
                # Reset error count on successful health check
                if agent.error_count > 0:
                    agent.error_count = 0
                    logger.info(f"Agent {agent.name} recovered from errors")
            else:
                agent.status = AgentStatus.ERROR
                agent.error_count += 1
                
        except requests.exceptions.ConnectionError:
            agent.status = AgentStatus.OFFLINE
        except requests.exceptions.Timeout:
            agent.status = AgentStatus.ERROR
        except Exception as e:
            logger.error(f"Health check failed for {agent.name}: {e}")
            agent.status = AgentStatus.ERROR
            agent.error_count += 1
    
    def _is_container_running(self, agent_name: str) -> bool:
        """Check if Docker container for agent is running"""
        try:
            container_name = f"sutazai-{agent_name.lower()}"
            container = self.docker_client.containers.get(container_name)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking container for {agent_name}: {e}")
            return False
    
    def _start_all_agents(self):
        """Start all agent services"""
        logger.info("Starting all agent services...")
        
        # Start Docker containers for external agents
        external_services = [
            "autogpt", "localagi", "tabby", "browser-use", "skyvern",
            "documind", "finrobot", "gpt-engineer", "aider", "langflow",
            "dify", "autogen", "open-webui", "bigagi", "agentzero"
        ]
        
        for service in external_services:
            try:
                container_name = f"sutazai-{service}"
                container = self.docker_client.containers.get(container_name)
                
                if container.status != 'running':
                    logger.info(f"Starting container: {container_name}")
                    container.start()
                    time.sleep(5)  # Wait for startup
                else:
                    logger.info(f"Container already running: {container_name}")
                    
            except docker.errors.NotFound:
                logger.warning(f"Container not found: {container_name}")
            except Exception as e:
                logger.error(f"Failed to start {service}: {e}")
        
        # Initialize internal agents
        self._initialize_internal_agents()
        
        logger.info("Agent startup process completed")
    
    def _initialize_internal_agents(self):
        """Initialize internal orchestrator agents"""
        logger.info("Initializing internal agents...")
        
        for agent_name, agent in self.agents.items():
            if agent.port == 8000:  # Internal agents
                try:
                    # Send initialization request to backend
                    init_url = f"{agent.url}/api/agents/{agent_name}/initialize"
                    response = requests.post(init_url, timeout=10)
                    
                    if response.status_code == 200:
                        logger.info(f"Internal agent {agent_name} initialized successfully")
                        agent.status = AgentStatus.ONLINE
                    else:
                        logger.warning(f"Failed to initialize {agent_name}: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Error initializing {agent_name}: {e}")
    
    def _task_processor_loop(self):
        """Continuous task processing loop"""
        while self.running:
            try:
                # Get pending tasks
                pending_tasks = [task for task in self.tasks.values() if task.status == "pending"]
                
                # Sort by priority
                pending_tasks.sort(key=lambda x: x.priority.value, reverse=True)
                
                # Process high priority tasks
                for task in pending_tasks[:10]:  # Process top 10 tasks
                    self._assign_and_execute_task(task)
                
                time.sleep(5)  # Process tasks every 5 seconds
                
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                time.sleep(10)
    
    def _assign_and_execute_task(self, task: Task):
        """Assign task to appropriate agent and execute"""
        try:
            # Find best agent for this task type
            suitable_agents = []
            
            for agent_name, agent in self.agents.items():
                if agent.status == AgentStatus.ONLINE:
                    # Check if agent has required capabilities
                    task_capabilities = task.payload.get('required_capabilities', [])
                    if not task_capabilities or any(cap in agent.capabilities for cap in task_capabilities):
                        suitable_agents.append(agent)
            
            if not suitable_agents:
                logger.warning(f"No suitable agent found for task {task.id}")
                task.status = "no_agent_available"
                return
            
            # Select agent with lowest task count (load balancing)
            selected_agent = min(suitable_agents, key=lambda x: x.task_count)
            
            # Assign task
            task.assigned_agent = selected_agent.name
            task.status = "assigned"
            task.started_at = datetime.now()
            selected_agent.task_count += 1
            
            logger.info(f"Assigned task {task.id} to agent {selected_agent.name}")
            
            # Execute task asynchronously
            self.task_executor.submit(self._execute_task, task, selected_agent)
            
        except Exception as e:
            logger.error(f"Task assignment error for {task.id}: {e}")
            task.status = "assignment_failed"
            task.error = str(e)
    
    def _execute_task(self, task: Task, agent: AgentConfig):
        """Execute task on assigned agent"""
        try:
            task.status = "executing"
            
            # Build execution URL
            if agent.port == 8000:  # Internal agent
                execute_url = f"{agent.url}/api/agents/{agent.name}/execute"
            else:  # External agent
                execute_url = f"{agent.url}/execute"
            
            # Prepare request payload
            request_payload = {
                "task_id": task.id,
                "task_type": task.type,
                "priority": task.priority.value,
                "payload": task.payload
            }
            
            # Execute task
            response = requests.post(
                execute_url, 
                json=request_payload,
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                task.result = response.json()
                task.status = "completed"
                task.completed_at = datetime.now()
                logger.info(f"Task {task.id} completed successfully")
            else:
                task.status = "failed"
                task.error = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Task {task.id} failed: {task.error}")
            
        except requests.exceptions.Timeout:
            task.status = "timeout"
            task.error = "Task execution timeout"
            logger.error(f"Task {task.id} timed out")
            
        except Exception as e:
            task.status = "error"
            task.error = str(e)
            logger.error(f"Task {task.id} error: {e}")
            
        finally:
            # Decrement agent task count
            agent.task_count = max(0, agent.task_count - 1)
    
    async def submit_task(self, task_type: str, payload: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Submit a new task to the orchestrator"""
        task_id = f"task_{int(time.time() * 1000)}_{len(self.tasks)}"
        
        task = Task(
            id=task_id,
            type=task_type,
            priority=priority,
            payload=payload,
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        logger.info(f"Task submitted: {task_id} (type: {task_type}, priority: {priority.name})")
        
        # Store task in Redis for persistence
        if self.redis_client:
            try:
                task_data = {
                    "id": task.id,
                    "type": task.type,
                    "priority": task.priority.value,
                    "payload": task.payload,
                    "status": task.status,
                    "created_at": task.created_at.isoformat()
                }
                self.redis_client.hset(f"task:{task_id}", mapping=task_data)
            except Exception as e:
                logger.error(f"Failed to store task in Redis: {e}")
        
        return task_id
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        online_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ONLINE)
        total_agents = len(self.agents)
        
        agent_details = {}
        for name, agent in self.agents.items():
            agent_details[name] = {
                "name": agent.name,
                "type": agent.type,
                "status": agent.status.value,
                "url": agent.url,
                "capabilities": agent.capabilities,
                "task_count": agent.task_count,
                "error_count": agent.error_count,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            }
        
        return {
            "total_agents": total_agents,
            "online_agents": online_agents,
            "offline_agents": total_agents - online_agents,
            "agents": agent_details,
            "system_health": "healthy" if online_agents > total_agents * 0.7 else "degraded"
        }
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get comprehensive task status"""
        task_counts = {}
        for status in ["pending", "assigned", "executing", "completed", "failed", "timeout", "error"]:
            task_counts[status] = sum(1 for task in self.tasks.values() if task.status == status)
        
        recent_tasks = []
        for task in sorted(self.tasks.values(), key=lambda x: x.created_at, reverse=True)[:10]:
            recent_tasks.append({
                "id": task.id,
                "type": task.type,
                "status": task.status,
                "assigned_agent": task.assigned_agent,
                "created_at": task.created_at.isoformat(),
                "priority": task.priority.value
            })
        
        return {
            "total_tasks": len(self.tasks),
            "task_counts": task_counts,
            "recent_tasks": recent_tasks
        }
    
    async def start_orchestration(self):
        """Start the complete orchestration system"""
        logger.info("Starting SutazAI Agent Orchestration System")
        
        # Ensure all services are ready
        await self._wait_for_core_services()
        
        # Start all agent services
        self._start_all_agents()
        
        # Wait for agents to come online
        await self._wait_for_agents()
        
        logger.info("SutazAI Agent Orchestration System started successfully")
    
    async def _wait_for_core_services(self):
        """Wait for core services to be ready"""
        services = [
            ("PostgreSQL", "localhost", 5432),
            ("Redis", "localhost", 6379), 
            ("Qdrant", "localhost", 6333),
            ("ChromaDB", "localhost", 8001),
            ("Ollama", "localhost", 11434)
        ]
        
        for service_name, host, port in services:
            logger.info(f"Waiting for {service_name} to be ready...")
            
            for attempt in range(30):  # 5 minutes total
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    sock.close()
                    
                    if result == 0:
                        logger.info(f"{service_name} is ready")
                        break
                except Exception:
                    pass
                
                if attempt == 29:
                    logger.warning(f"{service_name} not ready after 5 minutes")
                
                await asyncio.sleep(10)
    
    async def _wait_for_agents(self):
        """Wait for agents to come online"""
        logger.info("Waiting for agents to come online...")
        
        for attempt in range(60):  # 10 minutes total
            online_count = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.ONLINE)
            total_count = len(self.agents)
            
            logger.info(f"Agents online: {online_count}/{total_count}")
            
            if online_count >= total_count * 0.7:  # 70% of agents online
                logger.info("Sufficient agents are online")
                break
            
            await asyncio.sleep(10)
    
    async def stop_orchestration(self):
        """Stop the orchestration system"""
        logger.info("Stopping SutazAI Agent Orchestration System")
        self.running = False
        
        # Wait for running tasks to complete
        running_tasks = [task for task in self.tasks.values() if task.status in ["assigned", "executing"]]
        if running_tasks:
            logger.info(f"Waiting for {len(running_tasks)} running tasks to complete...")
            
            for attempt in range(30):  # 5 minutes
                running_tasks = [task for task in self.tasks.values() if task.status in ["assigned", "executing"]]
                if not running_tasks:
                    break
                await asyncio.sleep(10)
        
        logger.info("Agent Orchestration System stopped")
    
    def force_restart_agent(self, agent_name: str) -> bool:
        """Force restart an agent container"""
        try:
            container_name = f"sutazai-{agent_name.lower()}"
            container = self.docker_client.containers.get(container_name)
            
            logger.info(f"Restarting container: {container_name}")
            container.restart()
            
            # Update agent status
            if agent_name in self.agents:
                self.agents[agent_name].status = AgentStatus.STARTING
                self.agents[agent_name].error_count = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")
            return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        agent_status = self.get_agent_status()
        task_status = self.get_task_status()
        
        # Calculate performance metrics
        completed_tasks = sum(1 for task in self.tasks.values() if task.status == "completed")
        failed_tasks = sum(1 for task in self.tasks.values() if task.status in ["failed", "timeout", "error"])
        
        success_rate = completed_tasks / len(self.tasks) if self.tasks else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": agent_status,
            "tasks": task_status,
            "performance": {
                "total_tasks_processed": len(self.tasks),
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": success_rate,
                "average_task_completion_time": self._calculate_avg_completion_time()
            },
            "system_status": "operational" if agent_status["online_agents"] > 0 else "degraded"
        }
    
    def _calculate_avg_completion_time(self) -> float:
        """Calculate average task completion time"""
        completed_tasks = [
            task for task in self.tasks.values() 
            if task.status == "completed" and task.started_at and task.completed_at
        ]
        
        if not completed_tasks:
            return 0.0
        
        total_time = sum(
            (task.completed_at - task.started_at).total_seconds() 
            for task in completed_tasks
        )
        
        return total_time / len(completed_tasks)

# Global orchestrator instance
orchestrator_instance = None

def get_orchestrator() -> EnhancedAgentOrchestrator:
    """Get global orchestrator instance"""
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = EnhancedAgentOrchestrator()
    return orchestrator_instance

async def main():
    """Main function for testing"""
    orchestrator = get_orchestrator()
    
    # Start orchestration
    await orchestrator.start_orchestration()
    
    # Submit a test task
    task_id = await orchestrator.submit_task(
        task_type="code_generation",
        payload={
            "description": "Create a Python function to calculate fibonacci numbers",
            "language": "python",
            "required_capabilities": ["python", "code_generation"]
        },
        priority=TaskPriority.HIGH
    )
    
    print(f"Submitted test task: {task_id}")
    
    # Monitor system for a while
    for i in range(12):  # 2 minutes
        await asyncio.sleep(10)
        
        agent_status = orchestrator.get_agent_status()
        task_status = orchestrator.get_task_status()
        
        print(f"\nStatus update {i+1}:")
        print(f"Agents online: {agent_status['online_agents']}/{agent_status['total_agents']}")
        print(f"Tasks: {task_status['task_counts']}")
    
    # Stop orchestration
    await orchestrator.stop_orchestration()

if __name__ == "__main__":
    asyncio.run(main())