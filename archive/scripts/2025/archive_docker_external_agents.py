#!/usr/bin/env python3
"""
Docker External AI Agents Integration for SutazAI
Integrates external AI agents running in Docker containers
"""

import asyncio
import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DockerAgent:
    """Docker-based external AI agent"""
    name: str
    container_name: str
    port: int
    api_url: str
    capabilities: List[str]
    status: str = "inactive"
    health_endpoint: str = "/health"

class DockerExternalAgentManager:
    """Manages Docker-based external AI agents"""
    
    def __init__(self):
        self.agents: Dict[str, DockerAgent] = {}
        self.setup_docker_agents()
    
    def setup_docker_agents(self):
        """Setup Docker-based external AI agents"""
        
        # LangChain Agents
        self.agents["langchain"] = DockerAgent(
            name="LangChain Agents",
            container_name="sutazai-langchain-agents",
            port=8084,
            api_url="http://localhost:8084",
            capabilities=["general_chat", "research", "code_help", "task_planning"]
        )
        
        # AutoGen Agents
        self.agents["autogen"] = DockerAgent(
            name="AutoGen",
            container_name="sutazai-autogen",
            port=8085,
            api_url="http://localhost:8085",
            capabilities=["multi_agent_chat", "code_execution", "task_automation"]
        )
        
        # Browser Use
        self.agents["browser_use"] = DockerAgent(
            name="Browser Use",
            container_name="sutazai-browser-use",
            port=8088,
            api_url="http://localhost:8088",
            capabilities=["web_automation", "content_extraction", "web_navigation"]
        )
        
        # OpenWebUI
        self.agents["open_webui"] = DockerAgent(
            name="OpenWebUI",
            container_name="sutazai-open-webui",
            port=8090,
            api_url="http://localhost:8090",
            capabilities=["web_interface", "chat_interface", "model_management"]
        )
        
        # Semgrep
        self.agents["semgrep"] = DockerAgent(
            name="Semgrep",
            container_name="sutazai-semgrep",
            port=8083,
            api_url="http://localhost:8083",
            capabilities=["code_security", "vulnerability_scanning", "code_analysis"]
        )
        
        # TabbyML
        self.agents["tabbyml"] = DockerAgent(
            name="TabbyML",
            container_name="sutazai-tabbyml",
            port=8082,
            api_url="http://localhost:8082",
            capabilities=["code_completion", "code_generation", "ai_assistance"]
        )
        
        logger.info(f"Configured {len(self.agents)} Docker external agents")
    
    async def check_agent_health(self, agent_name: str) -> bool:
        """Check if an agent is healthy"""
        if agent_name not in self.agents:
            return False
        
        agent = self.agents[agent_name]
        try:
            response = requests.get(
                f"{agent.api_url}{agent.health_endpoint}",
                timeout=5
            )
            if response.status_code == 200:
                agent.status = "active"
                return True
            else:
                agent.status = "inactive"
                return False
        except Exception as e:
            logger.debug(f"Health check failed for {agent_name}: {e}")
            agent.status = "inactive"
            return False
    
    async def check_all_agents_health(self) -> Dict[str, bool]:
        """Check health of all agents"""
        health_status = {}
        
        for agent_name in self.agents:
            is_healthy = await self.check_agent_health(agent_name)
            health_status[agent_name] = is_healthy
        
        return health_status
    
    async def call_agent(self, agent_name: str, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an API call to an external agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        
        try:
            # Check if agent is healthy first
            if not await self.check_agent_health(agent_name):
                return {"error": f"Agent {agent_name} is not healthy"}
            
            response = requests.post(
                f"{agent.api_url}{endpoint}",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API call failed with status {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Failed to call agent {agent_name}: {e}")
            return {"error": f"Failed to call agent: {str(e)}"}
    
    async def execute_task(self, agent_name: str, task: str, task_type: str = "general") -> Dict[str, Any]:
        """Execute a task using an external agent"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        
        # Prepare task data based on agent type
        if agent_name == "langchain":
            return await self.call_agent(agent_name, "/execute", {
                "task": task,
                "agent_type": task_type
            })
        
        elif agent_name == "autogen":
            return await self.call_agent(agent_name, "/execute", {
                "task": task,
                "agent_type": task_type
            })
        
        elif agent_name == "browser_use":
            return await self.call_agent(agent_name, "/execute", {
                "task": task,
                "action": "navigate" if "navigate" in task.lower() else "extract"
            })
        
        else:
            # Generic task execution
            return await self.call_agent(agent_name, "/execute", {
                "task": task,
                "type": task_type
            })
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == "active"]),
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            status["agents"][name] = {
                "name": agent.name,
                "status": agent.status,
                "port": agent.port,
                "capabilities": agent.capabilities
            }
        
        return status
    
    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all agents"""
        capabilities = {}
        for name, agent in self.agents.items():
            capabilities[name] = agent.capabilities
        return capabilities
    
    async def start_monitoring(self):
        """Start monitoring all agents"""
        while True:
            await self.check_all_agents_health()
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def get_available_agents(self) -> List[str]:
        """Get list of available (active) agents"""
        await self.check_all_agents_health()
        return [name for name, agent in self.agents.items() if agent.status == "active"]
    
    async def distribute_task(self, task: str, preferred_agent: str = None) -> Dict[str, Any]:
        """Distribute a task to the most appropriate agent"""
        if preferred_agent and preferred_agent in self.agents:
            return await self.execute_task(preferred_agent, task)
        
        # Simple task routing based on keywords
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["code", "programming", "python", "javascript"]):
            if "langchain" in self.agents:
                return await self.execute_task("langchain", task, "code_helper")
        
        elif any(keyword in task_lower for keyword in ["web", "browser", "navigate", "scrape"]):
            if "browser_use" in self.agents:
                return await self.execute_task("browser_use", task)
        
        elif any(keyword in task_lower for keyword in ["security", "vulnerability", "scan"]):
            if "semgrep" in self.agents:
                return await self.execute_task("semgrep", task)
        
        elif any(keyword in task_lower for keyword in ["chat", "conversation", "multi-agent"]):
            if "autogen" in self.agents:
                return await self.execute_task("autogen", task)
        
        else:
            # Default to LangChain for general tasks
            if "langchain" in self.agents:
                return await self.execute_task("langchain", task, "general")
        
        return {"error": "No suitable agent found for task"}

# Global manager instance
docker_agent_manager = DockerExternalAgentManager()

async def start_agent_monitoring():
    """Start agent monitoring in background"""
    await docker_agent_manager.start_monitoring()

if __name__ == "__main__":
    # Test the agent manager
    async def test_agents():
        print("Testing Docker External Agents...")
        status = await docker_agent_manager.check_all_agents_health()
        print(f"Agent health status: {status}")
        
        agent_status = docker_agent_manager.get_agent_status()
        print(f"Agent status: {json.dumps(agent_status, indent=2)}")
        
        # Test a simple task
        result = await docker_agent_manager.distribute_task("Hello, can you help me with a Python function?")
        print(f"Task result: {result}")
    
    asyncio.run(test_agents())