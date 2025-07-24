#!/usr/bin/env python3
"""
Docker Agent Manager for SutazAI System
Provides Docker-based agent management and task distribution
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

class DockerAgentManager:
    """
    Docker-based agent management system for specialized AI agents
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.docker_agents = {
            "pytorch": {
                "name": "PyTorch",
                "type": "ml_framework",
                "url": "http://localhost:8093",
                "container_name": "sutazai-pytorch",
                "capabilities": ["deep_learning", "neural_networks", "model_training", "tensor_operations"],
                "status": "offline"
            },
            "tensorflow": {
                "name": "TensorFlow",
                "type": "ml_framework", 
                "url": "http://localhost:8094",
                "container_name": "sutazai-tensorflow",
                "capabilities": ["machine_learning", "neural_networks", "model_serving", "distributed_training"],
                "status": "offline"
            },
            "jax": {
                "name": "JAX",
                "type": "ml_framework",
                "url": "http://localhost:8095",
                "container_name": "sutazai-jax",
                "capabilities": ["scientific_computing", "automatic_differentiation", "high_performance_computing"],
                "status": "offline"
            },
            "faiss": {
                "name": "FAISS",
                "type": "vector_search",
                "url": "http://localhost:8096",
                "container_name": "sutazai-faiss",
                "capabilities": ["vector_similarity", "nearest_neighbor_search", "indexing", "clustering"],
                "status": "offline"
            },
            "awesome-code-ai": {
                "name": "Awesome Code AI",
                "type": "code_analysis",
                "url": "http://localhost:8097",
                "container_name": "sutazai-awesome-code-ai",
                "capabilities": ["code_analysis", "pattern_recognition", "code_quality", "refactoring_suggestions"],
                "status": "offline"
            },
            "enhanced-model-manager": {
                "name": "Enhanced Model Manager",
                "type": "model_management",
                "url": "http://localhost:8098",
                "container_name": "sutazai-enhanced-model-manager",
                "capabilities": ["model_loading", "model_switching", "resource_optimization", "model_caching"],
                "status": "offline"
            },
            "context-engineering": {
                "name": "Context Engineering",
                "type": "context_optimization",
                "url": "http://localhost:8099",
                "container_name": "sutazai-context-engineering",
                "capabilities": ["context_optimization", "prompt_engineering", "token_management"],
                "status": "offline"
            },
            "fms-fsdp": {
                "name": "FMS FSDP",
                "type": "distributed_training",
                "url": "http://localhost:8100",
                "container_name": "sutazai-fms-fsdp",
                "capabilities": ["distributed_training", "model_parallelism", "memory_optimization"],
                "status": "offline"
            },
            "realtimestt": {
                "name": "RealtimeSTT",
                "type": "speech_processing",
                "url": "http://localhost:8101",
                "container_name": "sutazai-realtimestt",
                "capabilities": ["speech_to_text", "real_time_transcription", "audio_processing"],
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
                for agent_name, agent_info in self.docker_agents.items():
                    self._check_agent_health(agent_info)
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Docker agent health monitor error: {e}")
                time.sleep(30)
    
    def _check_agent_health(self, agent_info: Dict[str, Any]):
        """Check health of individual Docker agent"""
        try:
            # Check if container is running
            container_running = self._is_container_running(agent_info["container_name"])
            
            if not container_running:
                agent_info["status"] = "offline"
                return
            
            # Check HTTP endpoint
            try:
                response = requests.get(f"{agent_info['url']}/health", timeout=5)
                if response.status_code == 200:
                    agent_info["status"] = "online"
                else:
                    agent_info["status"] = "error"
            except requests.exceptions.ConnectionError:
                agent_info["status"] = "offline"
            except requests.exceptions.Timeout:
                agent_info["status"] = "timeout"
                
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
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all Docker agents"""
        online_agents = sum(1 for agent in self.docker_agents.values() if agent["status"] == "online")
        total_agents = len(self.docker_agents)
        
        return {
            "total_agents": total_agents,
            "active_agents": online_agents,
            "offline_agents": total_agents - online_agents,
            "agents": self.docker_agents,
            "system_health": "healthy" if online_agents > total_agents * 0.5 else "degraded"
        }
    
    def get_agent_capabilities(self) -> List[str]:
        """Get all available capabilities from Docker agents"""
        capabilities = set()
        for agent_info in self.docker_agents.values():
            if agent_info["status"] == "online":
                capabilities.update(agent_info["capabilities"])
        return list(capabilities)
    
    async def execute_task(self, agent_name: str, task: str, task_type: str = "general") -> Dict[str, Any]:
        """Execute a task on a specific Docker agent"""
        if agent_name not in self.docker_agents:
            return {"error": f"Docker agent {agent_name} not found"}
        
        agent_info = self.docker_agents[agent_name]
        
        if agent_info["status"] != "online":
            return {"error": f"Docker agent {agent_name} is not online"}
        
        try:
            payload = {
                "task": task,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
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
                            "error": f"Docker agent {agent_name} returned status {response.status}",
                            "timestamp": datetime.now().isoformat()
                        }
                        
        except Exception as e:
            logger.error(f"Error executing task on Docker agent {agent_name}: {e}")
            return {
                "error": f"Failed to execute task on Docker agent {agent_name}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def distribute_task(self, task: str, preferred_agent: Optional[str] = None) -> Dict[str, Any]:
        """Distribute a task to the most appropriate Docker agent"""
        # If preferred agent is specified and available, use it
        if preferred_agent and preferred_agent in self.docker_agents:
            if self.docker_agents[preferred_agent]["status"] == "online":
                return await self.execute_task(preferred_agent, task)
        
        # Find the best agent based on capabilities and availability
        online_agents = [
            (name, info) for name, info in self.docker_agents.items() 
            if info["status"] == "online"
        ]
        
        if not online_agents:
            return {"error": "No Docker agents are currently online"}
        
        # Simple load balancing - use the first available agent
        # In a more sophisticated system, you could analyze the task and match capabilities
        selected_agent_name, _ = online_agents[0]
        
        return await self.execute_task(selected_agent_name, task)
    
    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available Docker agents with their capabilities"""
        available_agents = []
        
        for name, info in self.docker_agents.items():
            if info["status"] == "online":
                available_agents.append({
                    "name": name,
                    "type": info["type"],
                    "capabilities": info["capabilities"],
                    "url": info["url"],
                    "status": info["status"]
                })
        
        return available_agents
    
    async def start_agent(self, agent_name: str) -> bool:
        """Start a specific Docker agent"""
        if agent_name not in self.docker_agents:
            logger.error(f"Docker agent {agent_name} not found")
            return False
        
        try:
            agent_info = self.docker_agents[agent_name]
            container_name = agent_info["container_name"]
            container = self.docker_client.containers.get(container_name)
            
            if container.status != 'running':
                logger.info(f"Starting Docker agent {agent_name}...")
                container.start()
                await asyncio.sleep(10)  # Wait for startup
                return True
            else:
                logger.info(f"Docker agent {agent_name} already running")
                return True
                
        except docker.errors.NotFound:
            logger.error(f"Container not found for Docker agent {agent_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to start Docker agent {agent_name}: {e}")
            return False
    
    async def stop_agent(self, agent_name: str) -> bool:
        """Stop a specific Docker agent"""
        if agent_name not in self.docker_agents:
            logger.error(f"Docker agent {agent_name} not found")
            return False
        
        try:
            agent_info = self.docker_agents[agent_name]
            container_name = agent_info["container_name"]
            container = self.docker_client.containers.get(container_name)
            
            if container.status == 'running':
                logger.info(f"Stopping Docker agent {agent_name}...")
                container.stop()
                agent_info["status"] = "offline"
                return True
            else:
                logger.info(f"Docker agent {agent_name} already stopped")
                return True
                
        except docker.errors.NotFound:
            logger.error(f"Container not found for Docker agent {agent_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to stop Docker agent {agent_name}: {e}")
            return False

# Global instance
docker_agent_manager = DockerAgentManager()

def get_docker_agent_manager() -> DockerAgentManager:
    """Get global Docker agent manager instance"""
    return docker_agent_manager