"""
Agent Orchestrator - Manages and coordinates all AI agents
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of AI agents"""
    AUTOGPT = "autogpt"
    LOCALAGI = "localagi"
    TABBYML = "tabbyml"
    SEMGREP = "semgrep"
    BROWSER_USE = "browser_use"
    SKYVERN = "skyvern"
    DOCUMIND = "documind"
    FINROBOT = "finrobot"
    GPT_ENGINEER = "gpt_engineer"
    AIDER = "aider"
    BIGAGI = "bigagi"
    AGENTZERO = "agentzero"
    LANGFLOW = "langflow"
    DIFY = "dify"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    AGENTGPT = "agentgpt"
    PRIVATEGPT = "privategpt"
    LLAMAINDEX = "llamaindex"
    FLOWISE = "flowise"
    SHELLGPT = "shellgpt"
    PENTESTGPT = "pentestgpt"

@dataclass
class Agent:
    """Agent configuration"""
    name: str
    type: AgentType
    url: str
    port: int
    capabilities: List[str]
    status: str = "unknown"
    last_health_check: Optional[datetime] = None

class AgentOrchestrator:
    """Orchestrates all AI agents in the system"""
    
    def __init__(self):
        self.agents: Dict[AgentType, Agent] = {}
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize all agents"""
        logger.info("Initializing Agent Orchestrator...")
        
        # Register all agents
        self._register_agents()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitor())
        
        # Start task processor
        asyncio.create_task(self._task_processor())
        
        self.initialized = True
        logger.info("Agent Orchestrator initialized")
        
    def _register_agents(self):
        """Register all available agents"""
        self.agents = {
            AgentType.AUTOGPT: Agent(
                name="AutoGPT",
                type=AgentType.AUTOGPT,
                url="http://autogpt",
                port=8080,
                capabilities=["task_automation", "planning", "execution"]
            ),
            AgentType.LOCALAGI: Agent(
                name="LocalAGI",
                type=AgentType.LOCALAGI,
                url="http://localagi",
                port=8080,
                capabilities=["agi_orchestration", "workflow_management"]
            ),
            AgentType.TABBYML: Agent(
                name="TabbyML",
                type=AgentType.TABBYML,
                url="http://tabbyml",
                port=8080,
                capabilities=["code_completion", "code_suggestions"]
            ),
            AgentType.SEMGREP: Agent(
                name="Semgrep",
                type=AgentType.SEMGREP,
                url="http://semgrep",
                port=8080,
                capabilities=["security_scanning", "vulnerability_detection"]
            ),
            AgentType.CREWAI: Agent(
                name="CrewAI",
                type=AgentType.CREWAI,
                url="http://crewai",
                port=8080,
                capabilities=["multi_agent_collaboration", "team_coordination"]
            ),
            AgentType.GPT_ENGINEER: Agent(
                name="GPT-Engineer",
                type=AgentType.GPT_ENGINEER,
                url="http://gpt-engineer",
                port=8080,
                capabilities=["code_generation", "project_scaffolding"]
            ),
            AgentType.AIDER: Agent(
                name="Aider",
                type=AgentType.AIDER,
                url="http://aider",
                port=8080,
                capabilities=["code_editing", "pair_programming"]
            ),
            AgentType.LANGFLOW: Agent(
                name="LangFlow",
                type=AgentType.LANGFLOW,
                url="http://langflow",
                port=7860,
                capabilities=["visual_workflows", "flow_orchestration"]
            ),
            AgentType.DIFY: Agent(
                name="Dify",
                type=AgentType.DIFY,
                url="http://dify",
                port=5001,
                capabilities=["app_building", "ai_app_development"]
            ),
            AgentType.BIGAGI: Agent(
                name="BigAGI",
                type=AgentType.BIGAGI,
                url="http://bigagi",
                port=3000,
                capabilities=["advanced_conversation", "multi_model_chat"]
            ),
            AgentType.AGENTZERO: Agent(
                name="AgentZero",
                type=AgentType.AGENTZERO,
                url="http://agentzero",
                port=8080,
                capabilities=["autonomous_agent", "self_directed"]
            ),
            AgentType.BROWSER_USE: Agent(
                name="BrowserUse",
                type=AgentType.BROWSER_USE,
                url="http://browser-use",
                port=8080,
                capabilities=["web_automation", "browser_control"]
            ),
            AgentType.SKYVERN: Agent(
                name="Skyvern",
                type=AgentType.SKYVERN,
                url="http://skyvern",
                port=8080,
                capabilities=["web_scraping", "data_extraction"]
            ),
            AgentType.DOCUMIND: Agent(
                name="Documind",
                type=AgentType.DOCUMIND,
                url="http://documind",
                port=8080,
                capabilities=["document_processing", "pdf_analysis"]
            ),
            AgentType.FINROBOT: Agent(
                name="FinRobot",
                type=AgentType.FINROBOT,
                url="http://finrobot",
                port=8080,
                capabilities=["financial_analysis", "market_data"]
            ),
            AgentType.AUTOGEN: Agent(
                name="AutoGen",
                type=AgentType.AUTOGEN,
                url="http://autogen",
                port=8080,
                capabilities=["multi_agent_chat", "collaborative_problem_solving"]
            ),
            AgentType.AGENTGPT: Agent(
                name="AgentGPT",
                type=AgentType.AGENTGPT,
                url="http://agentgpt",
                port=3000,
                capabilities=["goal_oriented_ai", "task_decomposition"]
            ),
            AgentType.PRIVATEGPT: Agent(
                name="PrivateGPT",
                type=AgentType.PRIVATEGPT,
                url="http://privategpt",
                port=8001,
                capabilities=["private_llm", "document_qa"]
            ),
            AgentType.LLAMAINDEX: Agent(
                name="LlamaIndex",
                type=AgentType.LLAMAINDEX,
                url="http://llamaindex",
                port=8080,
                capabilities=["data_indexing", "rag", "retrieval"]
            ),
            AgentType.FLOWISE: Agent(
                name="FlowiseAI",
                type=AgentType.FLOWISE,
                url="http://flowise",
                port=3000,
                capabilities=["chatflow_builder", "visual_ai_flows"]
            ),
            AgentType.SHELLGPT: Agent(
                name="ShellGPT",
                type=AgentType.SHELLGPT,
                url="http://shellgpt",
                port=8080,
                capabilities=["cli_assistance", "command_generation"]
            ),
            AgentType.PENTESTGPT: Agent(
                name="PentestGPT",
                type=AgentType.PENTESTGPT,
                url="http://pentestgpt",
                port=8080,
                capabilities=["security_testing", "penetration_testing"]
            )
        }
        
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the appropriate agent(s)"""
        task_id = f"task_{datetime.now().timestamp()}"
        task["id"] = task_id
        
        # Determine best agent for the task
        agent_type = self._select_agent(task)
        
        if agent_type:
            agent = self.agents[agent_type]
            
            # Check if agent is healthy
            if agent.status != "healthy":
                await self._check_agent_health(agent)
                
            if agent.status == "healthy":
                result = await self._execute_on_agent(agent, task)
                return {
                    "task_id": task_id,
                    "agent": agent.name,
                    "result": result,
                    "status": "completed"
                }
            else:
                return {
                    "task_id": task_id,
                    "error": f"Agent {agent.name} is not available",
                    "status": "failed"
                }
        else:
            # Use multi-agent collaboration
            return await self._multi_agent_execution(task)
            
    def _select_agent(self, task: Dict[str, Any]) -> Optional[AgentType]:
        """Select the best agent for a task"""
        task_type = task.get("type", "").lower()
        task_desc = task.get("description", "").lower()
        
        # Match task to agent capabilities
        if "code" in task_type or "code" in task_desc:
            if "generate" in task_desc:
                return AgentType.GPT_ENGINEER
            elif "complete" in task_desc:
                return AgentType.TABBYML
            elif "edit" in task_desc or "fix" in task_desc:
                return AgentType.AIDER
        elif "security" in task_type:
            if "scan" in task_desc:
                return AgentType.SEMGREP
            elif "pentest" in task_desc:
                return AgentType.PENTESTGPT
        elif "document" in task_type:
            return AgentType.DOCUMIND
        elif "financial" in task_type:
            return AgentType.FINROBOT
        elif "web" in task_type:
            if "automate" in task_desc:
                return AgentType.BROWSER_USE
            elif "scrape" in task_desc:
                return AgentType.SKYVERN
        elif "workflow" in task_type:
            return AgentType.LANGFLOW
        elif "chat" in task_type:
            return AgentType.BIGAGI
            
        # Default to AutoGPT for general tasks
        return AgentType.AUTOGPT
        
    async def _execute_on_agent(self, agent: Agent, task: Dict[str, Any]) -> Any:
        """Execute a task on a specific agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent.url}:{agent.port}/execute",
                    json=task,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Agent returned status {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error executing on {agent.name}: {e}")
            return {"error": str(e)}
            
    async def _multi_agent_execution(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using multiple agents collaboratively"""
        # Use CrewAI for multi-agent coordination
        crew_agent = self.agents[AgentType.CREWAI]
        
        if crew_agent.status == "healthy":
            return await self._execute_on_agent(crew_agent, task)
        else:
            # Fallback to sequential execution
            results = []
            for agent_type, agent in self.agents.items():
                if agent.status == "healthy":
                    result = await self._execute_on_agent(agent, task)
                    results.append({
                        "agent": agent.name,
                        "result": result
                    })
                    
            return {
                "task_id": task["id"],
                "results": results,
                "status": "completed"
            }
            
    async def _health_monitor(self):
        """Monitor health of all agents"""
        while True:
            try:
                for agent in self.agents.values():
                    await self._check_agent_health(agent)
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                
    async def _check_agent_health(self, agent: Agent):
        """Check health of a specific agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{agent.url}:{agent.port}/health",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    agent.status = "healthy"
                else:
                    agent.status = "unhealthy"
                    
        except Exception:
            agent.status = "unreachable"
            
        agent.last_health_check = datetime.now()
        
    async def _task_processor(self):
        """Process tasks from the queue"""
        while True:
            try:
                task = await self.task_queue.get()
                result = await self.execute_task(task)
                
                # Store result
                self.active_tasks[task["id"]] = result
                
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents and their status"""
        return [
            {
                "name": agent.name,
                "type": agent.type.value,
                "url": f"{agent.url}:{agent.port}",
                "capabilities": agent.capabilities,
                "status": agent.status,
                "last_health_check": agent.last_health_check.isoformat() if agent.last_health_check else None
            }
            for agent in self.agents.values()
        ]
        
    async def health_check(self) -> Dict[str, Any]:
        """Check orchestrator health"""
        healthy_agents = sum(1 for a in self.agents.values() if a.status == "healthy")
        total_agents = len(self.agents)
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "task_queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks)
        }
        
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Agent Orchestrator...")
        # Cancel running tasks
        # Clean up resources
        self.initialized = False 