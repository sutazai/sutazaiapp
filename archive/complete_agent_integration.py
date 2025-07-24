#!/usr/bin/env python3
"""
Complete Agent Integration Module for SutazAI
Manages all AI agents and services with full integration
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of AI agents in the system"""
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

@dataclass
class AgentConfig:
    """Configuration for an AI agent"""
    name: str
    type: AgentType
    url: str
    port: int
    capabilities: List[str]
    health_endpoint: str
    api_key: Optional[str] = None

class CompleteAgentIntegration:
    """Manages all AI agents and their interactions"""
    
    def __init__(self):
        self.agents = self._initialize_agents()
        self.session = None
        self.agent_status = {}
        
    def _initialize_agents(self) -> Dict[AgentType, AgentConfig]:
        """Initialize all agent configurations"""
        return {
            AgentType.AUTOGPT: AgentConfig(
                name="AutoGPT",
                type=AgentType.AUTOGPT,
                url=os.getenv("AUTOGPT_URL", "http://autogpt:8080"),
                port=8080,
                capabilities=["task_automation", "planning", "execution"],
                health_endpoint="/health"
            ),
            AgentType.LOCALAGI: AgentConfig(
                name="LocalAGI",
                type=AgentType.LOCALAGI,
                url=os.getenv("LOCALAGI_URL", "http://localagi:8080"),
                port=8082,
                capabilities=["orchestration", "local_ai", "workflow"],
                health_endpoint="/health"
            ),
            AgentType.TABBYML: AgentConfig(
                name="TabbyML",
                type=AgentType.TABBYML,
                url=os.getenv("TABBY_URL", "http://tabby:8080"),
                port=8081,
                capabilities=["code_completion", "code_suggestion"],
                health_endpoint="/health"
            ),
            AgentType.SEMGREP: AgentConfig(
                name="Semgrep",
                type=AgentType.SEMGREP,
                url=os.getenv("SEMGREP_URL", "http://semgrep:8080"),
                port=8080,
                capabilities=["code_security", "vulnerability_scan"],
                health_endpoint="/health"
            ),
            AgentType.BROWSER_USE: AgentConfig(
                name="BrowserUse",
                type=AgentType.BROWSER_USE,
                url=os.getenv("BROWSER_USE_URL", "http://browser-use:8080"),
                port=8083,
                capabilities=["web_automation", "browser_control"],
                health_endpoint="/health"
            ),
            AgentType.SKYVERN: AgentConfig(
                name="Skyvern",
                type=AgentType.SKYVERN,
                url=os.getenv("SKYVERN_URL", "http://skyvern:8080"),
                port=8084,
                capabilities=["web_scraping", "automation"],
                health_endpoint="/health"
            ),
            AgentType.DOCUMIND: AgentConfig(
                name="Documind",
                type=AgentType.DOCUMIND,
                url=os.getenv("DOCUMIND_URL", "http://documind:8080"),
                port=8085,
                capabilities=["document_processing", "pdf_analysis", "text_extraction"],
                health_endpoint="/health"
            ),
            AgentType.FINROBOT: AgentConfig(
                name="FinRobot",
                type=AgentType.FINROBOT,
                url=os.getenv("FINROBOT_URL", "http://finrobot:8080"),
                port=8086,
                capabilities=["financial_analysis", "market_data", "trading"],
                health_endpoint="/health"
            ),
            AgentType.GPT_ENGINEER: AgentConfig(
                name="GPT Engineer",
                type=AgentType.GPT_ENGINEER,
                url=os.getenv("GPT_ENGINEER_URL", "http://gpt-engineer:8080"),
                port=8087,
                capabilities=["code_generation", "project_creation"],
                health_endpoint="/health"
            ),
            AgentType.AIDER: AgentConfig(
                name="Aider",
                type=AgentType.AIDER,
                url=os.getenv("AIDER_URL", "http://aider:8080"),
                port=8088,
                capabilities=["code_editing", "ai_pair_programming"],
                health_endpoint="/health"
            ),
            AgentType.BIGAGI: AgentConfig(
                name="BigAGI",
                type=AgentType.BIGAGI,
                url=os.getenv("BIGAGI_URL", "http://bigagi:3000"),
                port=8090,
                capabilities=["advanced_ai", "multi_model", "conversation"],
                health_endpoint="/api/health"
            ),
            AgentType.AGENTZERO: AgentConfig(
                name="AgentZero",
                type=AgentType.AGENTZERO,
                url=os.getenv("AGENTZERO_URL", "http://agentzero:8080"),
                port=8091,
                capabilities=["autonomous_agent", "task_execution"],
                health_endpoint="/health"
            ),
            AgentType.LANGFLOW: AgentConfig(
                name="LangFlow",
                type=AgentType.LANGFLOW,
                url=os.getenv("LANGFLOW_URL", "http://langflow:7860"),
                port=7860,
                capabilities=["workflow_builder", "visual_programming"],
                health_endpoint="/health"
            ),
            AgentType.DIFY: AgentConfig(
                name="Dify",
                type=AgentType.DIFY,
                url=os.getenv("DIFY_URL", "http://dify:5001"),
                port=5001,
                capabilities=["app_builder", "workflow_automation"],
                health_endpoint="/health"
            ),
            AgentType.AUTOGEN: AgentConfig(
                name="AutoGen",
                type=AgentType.AUTOGEN,
                url=os.getenv("AUTOGEN_URL", "http://autogen:8080"),
                port=8092,
                capabilities=["multi_agent", "collaboration"],
                health_endpoint="/health"
            ),
            AgentType.CREWAI: AgentConfig(
                name="CrewAI",
                type=AgentType.CREWAI,
                url=os.getenv("CREWAI_URL", "http://crewai:8080"),
                port=8102,
                capabilities=["team_collaboration", "task_distribution"],
                health_endpoint="/health"
            ),
            AgentType.AGENTGPT: AgentConfig(
                name="AgentGPT",
                type=AgentType.AGENTGPT,
                url=os.getenv("AGENTGPT_URL", "http://agentgpt:3000"),
                port=8103,
                capabilities=["web_agent", "autonomous_tasks"],
                health_endpoint="/api/health"
            ),
            AgentType.PRIVATEGPT: AgentConfig(
                name="PrivateGPT",
                type=AgentType.PRIVATEGPT,
                url=os.getenv("PRIVATEGPT_URL", "http://privategpt:8001"),
                port=8104,
                capabilities=["private_llm", "document_qa"],
                health_endpoint="/health"
            ),
            AgentType.LLAMAINDEX: AgentConfig(
                name="LlamaIndex",
                type=AgentType.LLAMAINDEX,
                url=os.getenv("LLAMAINDEX_URL", "http://llamaindex:8080"),
                port=8105,
                capabilities=["data_indexing", "retrieval", "rag"],
                health_endpoint="/health"
            ),
            AgentType.FLOWISE: AgentConfig(
                name="FlowiseAI",
                type=AgentType.FLOWISE,
                url=os.getenv("FLOWISE_URL", "http://flowise:3000"),
                port=8106,
                capabilities=["chatflow", "visual_builder"],
                health_endpoint="/api/v1/ping"
            )
        }
    
    async def initialize(self):
        """Initialize the agent integration system"""
        self.session = aiohttp.ClientSession()
        await self.check_all_agents()
        
    async def close(self):
        """Close the agent integration system"""
        if self.session:
            await self.session.close()
            
    async def check_agent_health(self, agent_type: AgentType) -> Dict[str, Any]:
        """Check health of a specific agent"""
        agent = self.agents.get(agent_type)
        if not agent:
            return {"status": "unknown", "error": "Agent not configured"}
            
        try:
            url = f"{agent.url}{agent.health_endpoint}"
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "agent": agent.name,
                        "type": agent.type.value,
                        "capabilities": agent.capabilities
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "agent": agent.name,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "status": "error",
                "agent": agent.name,
                "error": str(e)
            }
    
    async def check_all_agents(self) -> Dict[str, Any]:
        """Check health of all agents"""
        tasks = []
        for agent_type in self.agents:
            tasks.append(self.check_agent_health(agent_type))
            
        results = await asyncio.gather(*tasks)
        
        # Update agent status
        for i, agent_type in enumerate(self.agents):
            self.agent_status[agent_type.value] = results[i]
            
        healthy_count = sum(1 for r in results if r.get("status") == "healthy")
        total_count = len(results)
        
        return {
            "total_agents": total_count,
            "healthy_agents": healthy_count,
            "unhealthy_agents": total_count - healthy_count,
            "agents": self.agent_status,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_task(self, task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the appropriate agent"""
        # Map task types to agents
        task_agent_mapping = {
            "code_generation": [AgentType.GPT_ENGINEER, AgentType.AIDER],
            "code_completion": [AgentType.TABBYML],
            "code_security": [AgentType.SEMGREP],
            "web_automation": [AgentType.BROWSER_USE, AgentType.SKYVERN],
            "document_processing": [AgentType.DOCUMIND],
            "financial_analysis": [AgentType.FINROBOT],
            "workflow_automation": [AgentType.LANGFLOW, AgentType.DIFY, AgentType.FLOWISE],
            "task_automation": [AgentType.AUTOGPT, AgentType.AGENTZERO, AgentType.AGENTGPT],
            "multi_agent_collaboration": [AgentType.AUTOGEN, AgentType.CREWAI],
            "conversation": [AgentType.BIGAGI, AgentType.PRIVATEGPT],
            "data_retrieval": [AgentType.LLAMAINDEX]
        }
        
        # Get suitable agents for the task
        suitable_agents = task_agent_mapping.get(task_type, [])
        if not suitable_agents:
            return {
                "status": "error",
                "message": f"No suitable agent found for task type: {task_type}"
            }
        
        # Try each suitable agent until one succeeds
        for agent_type in suitable_agents:
            agent = self.agents.get(agent_type)
            if not agent:
                continue
                
            # Check if agent is healthy
            health = await self.check_agent_health(agent_type)
            if health.get("status") != "healthy":
                continue
                
            try:
                # Execute task on agent
                result = await self._execute_on_agent(agent, task_data)
                if result.get("status") == "success":
                    return {
                        "status": "success",
                        "agent": agent.name,
                        "result": result.get("result"),
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error executing on {agent.name}: {e}")
                continue
        
        return {
            "status": "error",
            "message": "All suitable agents failed to execute the task"
        }
    
    async def _execute_on_agent(self, agent: AgentConfig, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task on a specific agent"""
        # This is a generic implementation - each agent would have its specific API
        url = f"{agent.url}/api/execute"
        
        try:
            async with self.session.post(
                url,
                json=task_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"status": "success", "result": result}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def coordinate_agents(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents to complete a workflow"""
        results = []
        
        for step in workflow.get("steps", []):
            task_type = step.get("type")
            task_data = step.get("data", {})
            
            # Add results from previous steps if needed
            if step.get("use_previous_results"):
                task_data["previous_results"] = results
            
            result = await self.execute_task(task_type, task_data)
            results.append({
                "step": step.get("name"),
                "result": result
            })
            
            # Stop on error if required
            if result.get("status") == "error" and not step.get("continue_on_error"):
                break
        
        return {
            "workflow": workflow.get("name"),
            "status": "completed",
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get all agent capabilities"""
        capabilities = {}
        for agent_type, agent in self.agents.items():
            capabilities[agent.name] = agent.capabilities
        return capabilities
    
    def get_agents_by_capability(self, capability: str) -> List[str]:
        """Get agents that have a specific capability"""
        agents = []
        for agent_type, agent in self.agents.items():
            if capability in agent.capabilities:
                agents.append(agent.name)
        return agents

# Singleton instance
_agent_integration = None

def get_agent_integration() -> CompleteAgentIntegration:
    """Get the singleton agent integration instance"""
    global _agent_integration
    if _agent_integration is None:
        _agent_integration = CompleteAgentIntegration()
    return _agent_integration