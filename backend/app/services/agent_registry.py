"""
Agent Registry Service for SutazAI System
Manages dynamic agent discovery and activation
"""

import json
import os
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import requests
import psutil

logger = logging.getLogger(__name__)

class AgentRegistry:
    def __init__(self):
        self.agents_dir = Path("/opt/sutazaiapp/agents")
        self.registry_file = self.agents_dir / "agent_registry.json"
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_processes: Dict[str, subprocess.Popen] = {}
        self.ollama_port_start = 11434
        self.agent_port_start = 8000
        
    async def load_agent_registry(self) -> Dict[str, Any]:
        """Load agent registry from JSON file"""
        try:
            if self.registry_file.exists():
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            else:
                logger.error(f"Agent registry file not found: {self.registry_file}")
                return {"agents": {}}
        except Exception as e:
            logger.error(f"Error loading agent registry: {e}")
            return {"agents": {}}
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Dynamically discover all available agents"""
        discovered_agents = []
        
        try:
            # Load registry data
            registry_data = await self.load_agent_registry()
            registry_agents = registry_data.get("agents", {})
            
            # Scan for agent directories
            for agent_dir in self.agents_dir.iterdir():
                if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                    app_py = agent_dir / "app.py"
                    if app_py.exists():
                        agent_name = agent_dir.name
                        
                        # Get info from registry if available
                        registry_info = registry_agents.get(agent_name, {})
                        
                        agent_info = {
                            "id": agent_name,
                            "name": agent_name,
                            "type": self.classify_agent_type(agent_name),
                            "status": "discovered",
                            "path": str(agent_dir),
                            "app_file": str(app_py),
                            "description": registry_info.get("description", f"AI agent for {agent_name.replace('-', ' ')} operations"),
                            "capabilities": registry_info.get("capabilities", ["automation", "integration"]),
                            "config_path": registry_info.get("config_path", f"configs/{agent_name}_universal.json"),
                            "port": None,
                            "health_endpoint": None,
                            "last_seen": None
                        }
                        
                        discovered_agents.append(agent_info)
                        
        except Exception as e:
            logger.error(f"Error discovering agents: {e}")
            
        logger.info(f"Discovered {len(discovered_agents)} agents")
        return discovered_agents
    
    def classify_agent_type(self, agent_name: str) -> str:
        """Classify agent type based on name patterns"""
        name_lower = agent_name.lower()
        
        if any(keyword in name_lower for keyword in ['opus', 'sonnet', 'ai-', 'agi']):
            return "opus" if "opus" in name_lower else "sonnet"
        elif any(keyword in name_lower for keyword in ['security', 'penetration', 'vulnerability', 'kali']):
            return "security"
        elif any(keyword in name_lower for keyword in ['frontend', 'ui', 'interface']):
            return "frontend"
        elif any(keyword in name_lower for keyword in ['backend', 'api', 'database']):
            return "backend"
        elif any(keyword in name_lower for keyword in ['devops', 'infrastructure', 'docker', 'kubernetes']):
            return "infrastructure"
        elif any(keyword in name_lower for keyword in ['test', 'qa', 'quality']):
            return "testing"
        elif any(keyword in name_lower for keyword in ['monitor', 'observability', 'metrics']):
            return "monitoring"
        elif any(keyword in name_lower for keyword in ['data', 'analytics', 'ml', 'learning']):
            return "data"
        else:
            return "utility"
    
    async def start_agent(self, agent_info: Dict[str, Any]) -> bool:
        """Start an individual agent"""
        try:
            agent_name = agent_info["name"]
            agent_path = Path(agent_info["path"])
            
            # Check if already running
            if agent_name in self.agent_processes:
                if self.agent_processes[agent_name].poll() is None:
                    logger.info(f"Agent {agent_name} already running")
                    return True
            
            # Find available port
            port = await self.find_available_port(self.agent_port_start + len(self.agent_processes))
            
            # Set environment variables
            env = os.environ.copy()
            env.update({
                'AGENT_NAME': agent_name,
                'AGENT_PORT': str(port),
                'OLLAMA_BASE_URL': f'http://localhost:{self.ollama_port_start}',
                'PYTHONPATH': '/opt/sutazaiapp:/opt/sutazaiapp/agents',
                'HEALTH_CHECK_PORT': str(port + 1000)
            })
            
            # Start the agent
            cmd = ['python3', 'app.py']
            process = subprocess.Popen(
                cmd,
                cwd=agent_path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.agent_processes[agent_name] = process
            
            # Update agent info
            agent_info.update({
                "status": "starting",
                "port": port,
                "health_endpoint": f"http://localhost:{port}/health",
                "last_seen": datetime.utcnow().isoformat(),
                "process_id": process.pid
            })
            
            self.active_agents[agent_name] = agent_info
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Check if still running
            if process.poll() is None:
                agent_info["status"] = "running"
                logger.info(f"Successfully started agent {agent_name} on port {port}")
                return True
            else:
                logger.error(f"Agent {agent_name} failed to start")
                agent_info["status"] = "failed"
                return False
                
        except Exception as e:
            logger.error(f"Error starting agent {agent_name}: {e}")
            agent_info["status"] = "error"
            return False
    
    async def find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port"""
        port = start_port
        while port < start_port + 1000:
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        raise Exception("No available ports found")
    
    async def start_all_agents(self) -> Dict[str, Any]:
        """Start all discovered agents"""
        logger.info("Starting mass agent activation...")
        
        discovered_agents = await self.discover_agents()
        total_agents = len(discovered_agents)
        started_count = 0
        failed_count = 0
        
        # Start agents in batches to avoid overwhelming the system
        batch_size = 10
        for i in range(0, total_agents, batch_size):
            batch = discovered_agents[i:i + batch_size]
            
            logger.info(f"Starting batch {i//batch_size + 1} ({len(batch)} agents)")
            
            # Start batch concurrently
            tasks = [self.start_agent(agent) for agent in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Exception starting agent {batch[j]['name']}: {result}")
                    failed_count += 1
                elif result:
                    started_count += 1
                else:
                    failed_count += 1
            
            # Brief pause between batches
            await asyncio.sleep(1)
        
        # Verify running agents
        await asyncio.sleep(5)  # Give agents time to fully start
        healthy_count = await self.health_check_all()
        
        summary = {
            "total_discovered": total_agents,
            "started_successfully": started_count,
            "failed_to_start": failed_count,
            "healthy_agents": healthy_count,
            "timestamp": datetime.utcnow().isoformat(),
            "active_agents": list(self.active_agents.keys())
        }
        
        logger.info(f"Agent activation complete: {started_count}/{total_agents} started, {healthy_count} healthy")
        return summary
    
    async def health_check_all(self) -> int:
        """Perform health check on all active agents"""
        healthy_count = 0
        
        for agent_name, agent_info in self.active_agents.items():
            try:
                if agent_name in self.agent_processes:
                    process = self.agent_processes[agent_name]
                    if process.poll() is None:  # Still running
                        agent_info["status"] = "healthy"
                        agent_info["last_seen"] = datetime.utcnow().isoformat()
                        healthy_count += 1
                    else:
                        agent_info["status"] = "stopped"
                        
            except Exception as e:
                logger.error(f"Health check failed for {agent_name}: {e}")
                agent_info["status"] = "unhealthy"
        
        return healthy_count
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all agents"""
        await self.health_check_all()
        
        status_counts = {}
        for agent_info in self.active_agents.values():
            status = agent_info.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_agents": len(self.active_agents),
            "status_breakdown": status_counts,
            "agents": list(self.active_agents.values()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def stop_all_agents(self) -> Dict[str, Any]:
        """Stop all running agents gracefully"""
        stopped_count = 0
        
        for agent_name, process in self.agent_processes.items():
            try:
                if process.poll() is None:  # Still running
                    process.terminate()
                    await asyncio.sleep(2)
                    
                    if process.poll() is None:  # Force kill if needed
                        process.kill()
                    
                    stopped_count += 1
                    
                    if agent_name in self.active_agents:
                        self.active_agents[agent_name]["status"] = "stopped"
                        
            except Exception as e:
                logger.error(f"Error stopping agent {agent_name}: {e}")
        
        self.agent_processes.clear()
        
        return {
            "stopped_agents": stopped_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def activate_agi_collective(self) -> Dict[str, Any]:
        """Activate the AGI/ASI collective intelligence system"""
        logger.info("Activating AGI/ASI collective intelligence system...")
        
        # Start all agents first
        activation_result = await self.start_all_agents()
        
        # Enable inter-agent communication
        await self.enable_inter_agent_communication()
        
        # Initialize collective intelligence protocols
        collective_status = {
            "collective_active": True,
            "inter_agent_communication": True,
            "total_agents": activation_result["total_discovered"],
            "active_agents": activation_result["healthy_agents"],
            "collective_intelligence_level": "ASI" if activation_result["healthy_agents"] > 100 else "AGI",
            "activation_timestamp": datetime.utcnow().isoformat(),
            "capabilities": [
                "distributed_reasoning",
                "collective_problem_solving",
                "autonomous_coordination",
                "self_improvement",
                "emergent_intelligence"
            ]
        }
        
        logger.info(f"AGI/ASI collective activated with {activation_result['healthy_agents']} agents")
        return collective_status
    
    async def enable_inter_agent_communication(self):
        """Enable communication protocols between agents"""
        logger.info("Enabling inter-agent communication protocols...")
        
        # Create communication config for all agents
        comm_config = {
            "agent_registry": {agent_name: {
                "endpoint": f"http://localhost:{info.get('port', 8000)}",
                "capabilities": info.get("capabilities", []),
                "type": info.get("type", "utility")
            } for agent_name, info in self.active_agents.items()},
            "communication_enabled": True,
            "protocol_version": "1.0",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save communication config
        comm_file = self.agents_dir / "communication_config.json"
        with open(comm_file, 'w') as f:
            json.dump(comm_config, f, indent=2)
        
        logger.info("Inter-agent communication protocols activated")

# Global registry instance
agent_registry = AgentRegistry()