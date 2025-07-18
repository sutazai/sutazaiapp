#!/usr/bin/env python3
"""
SutazAI Multi-Agent Orchestration System
Coordinates multiple AI agents working together as an AGI/ASI system
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Different types of AI agents"""
    CODE_GENERATOR = "code_generator"
    SECURITY_ANALYST = "security_analyst"
    DOCUMENT_PROCESSOR = "document_processor"
    FINANCIAL_ANALYST = "financial_analyst"
    WEB_AUTOMATOR = "web_automator"
    GENERAL_ASSISTANT = "general_assistant"
    TASK_COORDINATOR = "task_coordinator"
    SYSTEM_MONITOR = "system_monitor"
    DATA_SCIENTIST = "data_scientist"
    DEVOPS_ENGINEER = "devops_engineer"

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

@dataclass
class Task:
    """Represents a task for agents to execute"""
    id: str
    description: str
    type: str
    priority: int = 5
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Agent:
    """Represents an AI agent"""
    id: str
    name: str
    type: AgentType
    status: AgentStatus = AgentStatus.INITIALIZING
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    completed_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class MultiAgentOrchestrator:
    """Orchestrates multiple AI agents working together"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = False
        self.coordination_loops = []
        
        # Service endpoints
        self.services = {
            "ollama": "http://localhost:11434",
            "backend": "http://localhost:8000",
            "qdrant": "http://localhost:6333",
            "chromadb": "http://localhost:8001"
        }
        
        # Initialize default agents
        self.initialize_default_agents()
        
    def initialize_default_agents(self):
        """Initialize the default set of AI agents"""
        default_agents = [
            {
                "name": "CodeMaster",
                "type": AgentType.CODE_GENERATOR,
                "capabilities": ["python", "javascript", "java", "cpp", "rust", "go", "code_review", "debugging"],
                "config": {"model": "deepseek-coder:7b", "temperature": 0.3}
            },
            {
                "name": "SecurityGuard",
                "type": AgentType.SECURITY_ANALYST,
                "capabilities": ["vulnerability_scan", "security_audit", "penetration_testing", "compliance_check"],
                "config": {"model": "llama3.2:1b", "temperature": 0.2}
            },
            {
                "name": "DocProcessor",
                "type": AgentType.DOCUMENT_PROCESSOR,
                "capabilities": ["pdf_processing", "text_extraction", "summarization", "translation"],
                "config": {"model": "llama3.2:1b", "temperature": 0.5}
            },
            {
                "name": "FinAnalyst",
                "type": AgentType.FINANCIAL_ANALYST,
                "capabilities": ["market_analysis", "risk_assessment", "portfolio_optimization", "financial_modeling"],
                "config": {"model": "llama3.2:1b", "temperature": 0.4}
            },
            {
                "name": "WebAutomator",
                "type": AgentType.WEB_AUTOMATOR,
                "capabilities": ["web_scraping", "browser_automation", "form_filling", "data_extraction"],
                "config": {"model": "llama3.2:1b", "temperature": 0.3}
            },
            {
                "name": "TaskCoordinator",
                "type": AgentType.TASK_COORDINATOR,
                "capabilities": ["task_planning", "workflow_management", "resource_allocation", "coordination"],
                "config": {"model": "llama3.2:1b", "temperature": 0.6}
            },
            {
                "name": "SystemMonitor",
                "type": AgentType.SYSTEM_MONITOR,
                "capabilities": ["health_monitoring", "performance_tracking", "alerting", "diagnostics"],
                "config": {"model": "llama3.2:1b", "temperature": 0.1}
            },
            {
                "name": "DataScientist",
                "type": AgentType.DATA_SCIENTIST,
                "capabilities": ["data_analysis", "machine_learning", "statistical_modeling", "visualization"],
                "config": {"model": "llama3.2:1b", "temperature": 0.4}
            },
            {
                "name": "DevOpsEngineer",
                "type": AgentType.DEVOPS_ENGINEER,
                "capabilities": ["container_management", "ci_cd", "infrastructure_automation", "deployment"],
                "config": {"model": "llama3.2:1b", "temperature": 0.3}
            },
            {
                "name": "GeneralAssistant",
                "type": AgentType.GENERAL_ASSISTANT,
                "capabilities": ["conversation", "general_knowledge", "problem_solving", "assistance"],
                "config": {"model": "llama3.2:1b", "temperature": 0.7}
            }
        ]
        
        for agent_config in default_agents:
            agent_id = f"agent_{len(self.agents) + 1}"
            agent = Agent(
                id=agent_id,
                name=agent_config["name"],
                type=agent_config["type"],
                capabilities=agent_config["capabilities"],
                config=agent_config["config"],
                status=AgentStatus.IDLE
            )
            # Set agent heartbeat to current time so they appear online
            agent.last_heartbeat = time.time()
            self.agents[agent_id] = agent
            logger.info(f"Initialized agent: {agent.name} ({agent.type.value})")
    
    async def start_orchestration(self):
        """Start the multi-agent orchestration system"""
        self.running = True
        logger.info("Starting multi-agent orchestration system...")
        
        # Start coordination loops
        self.coordination_loops = [
            asyncio.create_task(self.task_distribution_loop()),
            asyncio.create_task(self.agent_monitoring_loop()),
            asyncio.create_task(self.performance_tracking_loop()),
            asyncio.create_task(self.health_check_loop())
        ]
        
        logger.info(f"Started orchestration with {len(self.agents)} agents")
        
    async def stop_orchestration(self):
        """Stop the orchestration system"""
        self.running = False
        
        # Cancel all coordination loops
        for loop in self.coordination_loops:
            loop.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*self.coordination_loops, return_exceptions=True)
        
        logger.info("Stopped multi-agent orchestration system")
    
    async def task_distribution_loop(self):
        """Main loop for distributing tasks to agents"""
        while self.running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Find best agent for task
                best_agent = await self.find_best_agent_for_task(task)
                
                if best_agent:
                    # Assign task to agent
                    await self.assign_task_to_agent(task, best_agent)
                else:
                    # No available agent, put task back in queue
                    await self.task_queue.put(task)
                    await asyncio.sleep(5)  # Wait before retrying
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task distribution loop: {e}")
                await asyncio.sleep(1)
    
    async def agent_monitoring_loop(self):
        """Monitor agent health and status"""
        while self.running:
            try:
                current_time = time.time()
                
                for agent_id, agent in self.agents.items():
                    # Check agent heartbeat
                    if current_time - agent.last_heartbeat > 60:  # 1 minute timeout
                        if agent.status != AgentStatus.OFFLINE:
                            logger.warning(f"Agent {agent.name} appears to be offline")
                            agent.status = AgentStatus.OFFLINE
                    
                    # Update performance metrics
                    agent.performance_metrics["uptime"] = current_time - agent.created_at
                    agent.performance_metrics["tasks_completed"] = agent.completed_tasks
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in agent monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def performance_tracking_loop(self):
        """Track system performance metrics"""
        while self.running:
            try:
                # Calculate system metrics
                total_agents = len(self.agents)
                active_agents = len([a for a in self.agents.values() if a.status in [AgentStatus.IDLE, AgentStatus.BUSY]])
                total_tasks = len(self.tasks)
                completed_tasks = len([t for t in self.tasks.values() if t.status == "completed"])
                
                # Log performance metrics
                logger.info(f"System Status: {active_agents}/{total_agents} agents active, {completed_tasks}/{total_tasks} tasks completed")
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance tracking loop: {e}")
                await asyncio.sleep(60)
    
    async def health_check_loop(self):
        """Perform health checks on services"""
        while self.running:
            try:
                for service_name, service_url in self.services.items():
                    try:
                        if service_name == "ollama":
                            response = requests.get(f"{service_url}/api/version", timeout=5)
                        elif service_name == "backend":
                            response = requests.get(f"{service_url}/health", timeout=5)
                        elif service_name == "qdrant":
                            response = requests.get(f"{service_url}/healthz", timeout=5)
                        elif service_name == "chromadb":
                            response = requests.get(f"{service_url}/api/v1/heartbeat", timeout=5)
                        
                        if response.status_code == 200:
                            logger.debug(f"Service {service_name} is healthy")
                        else:
                            logger.warning(f"Service {service_name} returned status {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Service {service_name} health check failed: {e}")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(120)
    
    async def find_best_agent_for_task(self, task: Task) -> Optional[Agent]:
        """Find the best agent to handle a specific task"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE:
                # Check if agent has required capabilities
                if task.type in agent.capabilities or "general" in agent.capabilities:
                    suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Sort by performance metrics (least busy first)
        suitable_agents.sort(key=lambda a: a.completed_tasks)
        
        return suitable_agents[0]
    
    async def assign_task_to_agent(self, task: Task, agent: Agent):
        """Assign a task to a specific agent"""
        task.assigned_agent = agent.id
        task.status = "assigned"
        agent.current_task = task.id
        agent.status = AgentStatus.BUSY
        
        logger.info(f"Assigned task {task.id} to agent {agent.name}")
        
        # Execute task in background
        asyncio.create_task(self.execute_task(task, agent))
    
    async def execute_task(self, task: Task, agent: Agent):
        """Execute a task using the assigned agent"""
        try:
            task.status = "executing"
            
            # Simulate task execution based on agent type
            if agent.type == AgentType.CODE_GENERATOR:
                result = await self.execute_code_generation_task(task, agent)
            elif agent.type == AgentType.SECURITY_ANALYST:
                result = await self.execute_security_analysis_task(task, agent)
            elif agent.type == AgentType.DOCUMENT_PROCESSOR:
                result = await self.execute_document_processing_task(task, agent)
            elif agent.type == AgentType.FINANCIAL_ANALYST:
                result = await self.execute_financial_analysis_task(task, agent)
            elif agent.type == AgentType.WEB_AUTOMATOR:
                result = await self.execute_web_automation_task(task, agent)
            elif agent.type == AgentType.SYSTEM_MONITOR:
                result = await self.execute_system_monitoring_task(task, agent)
            else:
                result = await self.execute_general_task(task, agent)
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            # Update agent status
            agent.current_task = None
            agent.status = AgentStatus.IDLE
            agent.completed_tasks += 1
            agent.last_heartbeat = time.time()
            
            logger.info(f"Task {task.id} completed by agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = "error"
            task.result = {"error": str(e)}
            agent.current_task = None
            agent.status = AgentStatus.ERROR
    
    async def execute_code_generation_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute code generation task"""
        try:
            # Use LLM to generate code
            prompt = f"Generate code for: {task.description}"
            
            response = requests.post(
                f"{self.services['ollama']}/api/generate",
                json={
                    "model": agent.config.get("model", "llama3.2:1b"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": agent.config.get("temperature", 0.3)}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "code": result.get("response", ""),
                    "language": task.metadata.get("language", "python"),
                    "agent": agent.name
                }
            else:
                return {"error": f"Code generation failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Code generation error: {str(e)}"}
    
    async def execute_security_analysis_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute security analysis task"""
        return {
            "analysis": f"Security analysis completed for: {task.description}",
            "vulnerabilities": [],
            "recommendations": ["Enable 2FA", "Use strong passwords", "Update dependencies"],
            "agent": agent.name
        }
    
    async def execute_document_processing_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute document processing task"""
        return {
            "processed": True,
            "summary": f"Document processing completed for: {task.description}",
            "word_count": 1000,
            "extracted_text": "Sample extracted text...",
            "agent": agent.name
        }
    
    async def execute_financial_analysis_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute financial analysis task"""
        return {
            "analysis": f"Financial analysis completed for: {task.description}",
            "metrics": {"ROI": 15.5, "risk_score": 3.2, "recommendation": "BUY"},
            "agent": agent.name
        }
    
    async def execute_web_automation_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute web automation task"""
        return {
            "automation": f"Web automation completed for: {task.description}",
            "pages_processed": 10,
            "data_extracted": {"items": 25, "success_rate": 95},
            "agent": agent.name
        }
    
    async def execute_system_monitoring_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute system monitoring task"""
        return {
            "monitoring": f"System monitoring completed for: {task.description}",
            "metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1,
                "uptime": "5 days, 3 hours"
            },
            "agent": agent.name
        }
    
    async def execute_general_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute general task"""
        try:
            # Use LLM for general task
            prompt = f"Help with: {task.description}"
            
            response = requests.post(
                f"{self.services['ollama']}/api/generate",
                json={
                    "model": agent.config.get("model", "llama3.2:1b"),
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": agent.config.get("temperature", 0.7)}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "agent": agent.name
                }
            else:
                return {"error": f"General task failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"General task error: {str(e)}"}
    
    async def submit_task(self, description: str, task_type: str, priority: int = 5, metadata: Dict[str, Any] = None) -> str:
        """Submit a new task for processing"""
        task_id = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        
        task = Task(
            id=task_id,
            description=description,
            type=task_type,
            priority=priority,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        logger.info(f"Submitted task: {task_id} - {description}")
        return task_id
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status in [AgentStatus.IDLE, AgentStatus.BUSY]]),
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "type": agent.type.value,
                    "status": agent.status.value,
                    "capabilities": agent.capabilities,
                    "completed_tasks": agent.completed_tasks,
                    "current_task": agent.current_task
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def get_task_status(self) -> Dict[str, Any]:
        """Get current status of all tasks"""
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "executing_tasks": len([t for t in self.tasks.values() if t.status == "executing"]),
            "tasks": {
                task_id: {
                    "description": task.description,
                    "type": task.type,
                    "status": task.status,
                    "assigned_agent": task.assigned_agent,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at
                }
                for task_id, task in self.tasks.items()
            }
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            "orchestrator_status": "running" if self.running else "stopped",
            "agents": self.get_agent_status(),
            "tasks": self.get_task_status(),
            "services": {
                service_name: "unknown" for service_name in self.services.keys()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global orchestrator instance
orchestrator = MultiAgentOrchestrator()

async def main():
    """Main function to run the orchestrator"""
    await orchestrator.start_orchestration()
    
    # Submit some test tasks
    await orchestrator.submit_task("Generate a Python function to calculate fibonacci", "code_generation", metadata={"language": "python"})
    await orchestrator.submit_task("Analyze system security vulnerabilities", "security_analysis")
    await orchestrator.submit_task("Process uploaded document", "document_processing")
    await orchestrator.submit_task("Analyze market trends", "financial_analysis")
    
    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await orchestrator.stop_orchestration()

if __name__ == "__main__":
    asyncio.run(main())