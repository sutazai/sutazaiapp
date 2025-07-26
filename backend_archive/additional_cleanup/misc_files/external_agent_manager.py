#!/usr/bin/env python3
"""
External Agent Manager for SutazAI System
Provides integration layer for external AI agents
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import requests
import docker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalAgentManager:
    """
    External agent management system that works with the enhanced orchestrator
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.external_agents = {
            "AutoGPT": {
                "name": "AutoGPT",
                "type": "task_automation", 
                "url": "http://localhost:8080",
                "container_name": "sutazai-autogpt",
                "capabilities": ["task_automation", "web_browsing", "file_operations", "code_execution"],
                "status": "offline"
            },
            "LocalAGI": {
                "name": "LocalAGI",
                "type": "agi_orchestration",
                "url": "http://localhost:8082",
                "container_name": "sutazai-localagi", 
                "capabilities": ["agi_orchestration", "multi_agent_coordination", "task_planning"],
                "status": "offline"
            },
            "TabbyML": {
                "name": "TabbyML",
                "type": "code_completion",
                "url": "http://localhost:8081",
                "container_name": "sutazai-tabby",
                "capabilities": ["code_completion", "code_suggestions", "ide_integration"],
                "status": "offline"
            },
            "BrowserUse": {
                "name": "BrowserUse", 
                "type": "web_automation",
                "url": "http://localhost:8083",
                "container_name": "sutazai-browser-use",
                "capabilities": ["web_automation", "form_filling", "data_extraction"],
                "status": "offline"
            },
            "Skyvern": {
                "name": "Skyvern",
                "type": "web_automation",
                "url": "http://localhost:8084", 
                "container_name": "sutazai-skyvern",
                "capabilities": ["web_scraping", "browser_automation", "data_mining"],
                "status": "offline"
            },
            "Documind": {
                "name": "Documind",
                "type": "document_processing",
                "url": "http://localhost:8085",
                "container_name": "sutazai-documind",
                "capabilities": ["document_processing", "text_extraction", "pdf_parsing"],
                "status": "offline"
            },
            "FinRobot": {
                "name": "FinRobot",
                "type": "financial_analysis", 
                "url": "http://localhost:8086",
                "container_name": "sutazai-finrobot",
                "capabilities": ["financial_analysis", "market_data", "portfolio_optimization"],
                "status": "offline"
            },
            "GPT-Engineer": {
                "name": "GPT-Engineer",
                "type": "code_generation",
                "url": "http://localhost:8087",
                "container_name": "sutazai-gpt-engineer",
                "capabilities": ["code_generation", "project_scaffolding", "architecture_design"],
                "status": "offline"
            },
            "Aider": {
                "name": "Aider",
                "type": "code_editing",
                "url": "http://localhost:8088",
                "container_name": "sutazai-aider", 
                "capabilities": ["code_editing", "refactoring", "bug_fixing"],
                "status": "offline"
            },
            "LangFlow": {
                "name": "LangFlow",
                "type": "workflow_orchestration",
                "url": "http://localhost:7860",
                "container_name": "sutazai-langflow",
                "capabilities": ["workflow_design", "flow_orchestration", "visual_programming"],
                "status": "offline"
            },
            "Dify": {
                "name": "Dify",
                "type": "app_development",
                "url": "http://localhost:5001",
                "container_name": "sutazai-dify",
                "capabilities": ["app_development", "workflow_automation", "llm_orchestration"],
                "status": "offline"
            },
            "AutoGen": {
                "name": "AutoGen",
                "type": "multi_agent_conversation",
                "url": "http://localhost:8092",
                "container_name": "sutazai-autogen",
                "capabilities": ["multi_agent_conversation", "collaborative_problem_solving"],
                "status": "offline"
            },
            "BigAGI": {
                "name": "BigAGI",
                "type": "advanced_ai_interface",
                "url": "http://localhost:8090",
                "container_name": "sutazai-bigagi",
                "capabilities": ["advanced_ai_interface", "multi_model_support", "conversation_branching"],
                "status": "offline"
            },
            "AgentZero": {
                "name": "AgentZero",
                "type": "specialized_agent",
                "url": "http://localhost:8091",
                "container_name": "sutazai-agentzero",
                "capabilities": ["specialized_tasks", "custom_workflows", "agent_framework"],
                "status": "offline"
            },
            "CrewAI": {
                "name": "CrewAI",
                "type": "multi_agent_collaboration",
                "url": "http://localhost:8089",
                "container_name": "sutazai-crewai",
                "capabilities": ["multi_agent_collaboration", "team_based_problem_solving", "role_based_agents"],
                "status": "offline"
            },
            "AgentGPT": {
                "name": "AgentGPT",
                "type": "autonomous_agent",
                "url": "http://localhost:8094",
                "container_name": "sutazai-agentgpt",
                "capabilities": ["autonomous_task_execution", "goal_achievement", "task_planning"],
                "status": "offline"
            },
            "PrivateGPT": {
                "name": "PrivateGPT",
                "type": "private_document_qa",
                "url": "http://localhost:8095",
                "container_name": "sutazai-privategpt",
                "capabilities": ["private_document_qa", "local_knowledge_base", "secure_data_processing"],
                "status": "offline"
            },
            "LlamaIndex": {
                "name": "LlamaIndex",
                "type": "document_indexing",
                "url": "http://localhost:8096",
                "container_name": "sutazai-llamaindex",
                "capabilities": ["document_indexing", "semantic_search", "knowledge_retrieval"],
                "status": "offline"
            },
            "FlowiseAI": {
                "name": "FlowiseAI",
                "type": "visual_flow_orchestration",
                "url": "http://localhost:8097",
                "container_name": "sutazai-flowise",
                "capabilities": ["visual_flow_design", "no_code_ai_workflows", "drag_drop_orchestration"],
                "status": "offline"
            }
        }
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        import threading
        health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        health_thread.start()
    
    def _health_monitor_loop(self):
        """Monitor agent health continuously"""
        while True:
            try:
                for agent_name, agent_info in self.external_agents.items():
                    self._check_agent_health(agent_info)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(30)
    
    def _check_agent_health(self, agent_info: Dict[str, Any]):
        """Check health of individual agent"""
        try:
            # Check if container is running
            container_running = self._is_container_running(agent_info["container_name"])
            
            if not container_running:
                agent_info["status"] = "offline"
                return
            
            # Check HTTP endpoint with various possible health endpoints
            health_endpoints = ["/health", "/", "/api/health", "/healthz"]
            
            for endpoint in health_endpoints:
                try:
                    response = requests.get(f"{agent_info['url']}{endpoint}", timeout=5)
                    if response.status_code == 200:
                        agent_info["status"] = "online"
                        return
                except requests.exceptions.RequestException:
                    continue
            
            # If no endpoint responds, but container is running, mark as starting
            agent_info["status"] = "starting"
                
        except Exception as e:
            logger.error(f"Health check failed for {agent_info['name']}: {e}")
            agent_info["status"] = "error"
    
    def _is_container_running(self, container_name: str) -> bool:
        """Check if Docker container is running"""
        try:
            container = self.docker_client.containers.get(container_name)
            return container.status == 'running'
        except docker.errors.NotFound:
            return False
        except Exception as e:
            logger.error(f"Error checking container {container_name}: {e}")
            return False
    
    async def start_all_agents(self) -> Dict[str, bool]:
        """Start all external agent containers"""
        results = {}
        
        for agent_name, agent_info in self.external_agents.items():
            try:
                container_name = agent_info["container_name"]
                container = self.docker_client.containers.get(container_name)
                
                if container.status != 'running':
                    logger.info(f"Starting {agent_name} container...")
                    container.start()
                    await asyncio.sleep(5)  # Wait for startup
                    results[agent_name] = True
                else:
                    logger.info(f"{agent_name} container already running")
                    results[agent_name] = True
                    
            except docker.errors.NotFound:
                logger.warning(f"Container not found for {agent_name}")
                results[agent_name] = False
            except Exception as e:
                logger.error(f"Failed to start {agent_name}: {e}")
                results[agent_name] = False
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all external agents"""
        online_agents = sum(1 for agent in self.external_agents.values() if agent["status"] == "online")
        total_agents = len(self.external_agents)
        
        return {
            "total_agents": total_agents,
            "active_agents": online_agents,
            "offline_agents": total_agents - online_agents,
            "agents": self.external_agents,
            "system_health": "healthy" if online_agents > total_agents * 0.7 else "degraded"
        }
    
    def get_available_capabilities(self) -> List[str]:
        """Get all available capabilities from external agents"""
        capabilities = set()
        for agent_info in self.external_agents.values():
            if agent_info["status"] == "online":
                capabilities.update(agent_info["capabilities"])
        return list(capabilities)
    
    async def call_agent(self, agent_name: str, task: str, **kwargs) -> Dict[str, Any]:
        """Call an external agent with a task"""
        if agent_name not in self.external_agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent_info = self.external_agents[agent_name]
        
        if agent_info["status"] != "online":
            return {"error": f"Agent {agent_name} is not online"}
        
        try:
            payload = {
                "task": task,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"{agent_info['url']}/execute",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "agent": agent_name,
                            "result": result,
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        return {
                            "error": f"Agent {agent_name} returned status {response.status}",
                            "timestamp": datetime.now().isoformat()
                        }
                        
        except Exception as e:
            logger.error(f"Error calling agent {agent_name}: {e}")
            return {
                "error": f"Failed to call agent {agent_name}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Global instance
external_agent_manager = ExternalAgentManager()

def get_external_agent_manager() -> ExternalAgentManager:
    """Get global external agent manager instance"""
    return external_agent_manager