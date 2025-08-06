#!/usr/bin/env python3
"""
Generic SutazAI Agent Implementation
This serves as a base template for all AI agents in the system
"""

import os
import json
import time
import asyncio
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    id: str
    timestamp: str
    agent_id: str
    agent_type: str
    message_type: str
    content: str
    metadata: Dict[str, Any]

class GenericAgent:
    """
    Generic agent that can be customized for different agent types
    """
    
    def __init__(self, config_path: str = "/app/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.agent_name = os.getenv("AGENT_NAME", "generic-agent")
        self.agent_type = os.getenv("AGENT_TYPE", "generic")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://sutazai-ollama:11434")
        self.backend_url = os.getenv("BACKEND_URL", "http://sutazai-backend:8000")
        
        # Setup logging
        self._setup_logging()
        
        # Agent state
        self.is_running = False
        self.last_heartbeat = None
        
        self.logger.info(f"Initialized {self.agent_name} ({self.agent_type})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for generic agent"""
        return {
            "name": self.agent_name,
            "type": self.agent_type,
            "description": f"Generic {self.agent_type} agent",
            "capabilities": [
                "text_processing",
                "task_execution",
                "communication"
            ],
            "settings": {
                "heartbeat_interval": 30,
                "max_retries": 3,
                "timeout": 300
            }
        }
    
    def _setup_logging(self):
        """Setup logging for the agent"""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=f'%(asctime)s - {self.agent_name} - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.agent_name)
    
    async def start(self):
        """Start the agent and begin processing"""
        self.logger.info(f"Starting {self.agent_name}...")
        self.is_running = True
        
        # Register with backend
        await self._register_with_backend()
        
        # Start main processing loop
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the agent"""
        self.logger.info(f"Stopping {self.agent_name}...")
        self.is_running = False
        await self._unregister_from_backend()
    
    async def _main_loop(self):
        """Main processing loop for the agent"""
        heartbeat_interval = self.config.get("settings", {}).get("heartbeat_interval", 30)
        
        while self.is_running:
            try:
                # Send heartbeat
                await self._send_heartbeat()
                
                # Check for tasks
                tasks = await self._get_tasks()
                
                # Process tasks
                for task in tasks:
                    await self._process_task(task)
                
                # Wait before next iteration
                await asyncio.sleep(heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in main loop iteration: {e}")
                await asyncio.sleep(5)
    
    async def _register_with_backend(self):
        """Register this agent with the backend"""
        try:
            registration_data = {
                "agent_id": self.agent_name,
                "agent_type": self.agent_type,
                "config": self.config,
                "status": "online",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/agents/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Successfully registered with backend")
            else:
                self.logger.warning(f"Failed to register with backend: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error registering with backend: {e}")
    
    async def _unregister_from_backend(self):
        """Unregister this agent from the backend"""
        try:
            response = requests.delete(
                f"{self.backend_url}/api/v1/agents/{self.agent_name}",
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Successfully unregistered from backend")
            else:
                self.logger.warning(f"Failed to unregister from backend: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error unregistering from backend: {e}")
    
    async def _send_heartbeat(self):
        """Send heartbeat to backend"""
        try:
            heartbeat_data = {
                "agent_id": self.agent_name,
                "agent_type": self.agent_type,
                "status": "online",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": await self._get_agent_metrics()
            }
            
            response = requests.post(
                f"{self.backend_url}/api/v1/agents/heartbeat",
                json=heartbeat_data,
                timeout=5
            )
            
            if response.status_code == 200:
                self.last_heartbeat = datetime.utcnow()
            else:
                self.logger.warning(f"Heartbeat failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
    
    async def _get_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks from backend for this agent"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/v1/agents/{self.agent_name}/tasks",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json().get("tasks", [])
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting tasks: {e}")
            return []
    
    async def _process_task(self, task: Dict[str, Any]):
        """Process a single task"""
        task_id = task.get("id", "unknown")
        task_type = task.get("type", "unknown")
        
        self.logger.info(f"Processing task {task_id} ({task_type})")
        
        try:
            # Update task status to processing
            await self._update_task_status(task_id, "processing")
            
            # Execute the task based on agent type
            result = await self._execute_task(task)
            
            # Update task status to completed
            await self._update_task_status(task_id, "completed", result)
            
            self.logger.info(f"Completed task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            await self._update_task_status(task_id, "failed", {"error": str(e)})
    
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task - override this method in specific agent implementations"""
        task_type = task.get("type", "unknown")
        task_content = task.get("content", "")
        
        # Default generic task execution
        if task_type == "text_analysis":
            return await self._analyze_text(task_content)
        elif task_type == "code_review":
            return await self._review_code(task_content)
        elif task_type == "system_check":
            return await self._system_check()
        else:
            # Use LLM for general task processing
            return await self._process_with_llm(task)
    
    async def _process_with_llm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task using Ollama LLM"""
        try:
            prompt = self._build_prompt(task)
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "gpt-oss:1.1b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result.get("response", ""),
                    "agent": self.agent_name
                }
            else:
                return {
                    "success": False,
                    "error": f"LLM request failed: {response.status_code}",
                    "agent": self.agent_name
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM processing error: {str(e)}",
                "agent": self.agent_name
            }
    
    def _build_prompt(self, task: Dict[str, Any]) -> str:
        """Build prompt for LLM based on agent type and task"""
        agent_context = self.config.get("description", f"You are a {self.agent_type} agent")
        task_content = task.get("content", "")
        task_type = task.get("type", "general")
        
        return f"""
{agent_context}

Task Type: {task_type}
Task Content: {task_content}

Please provide a helpful response based on your role as a {self.agent_type} agent.
"""
    
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Generic text analysis"""
        return {
            "analysis": f"Analyzed text of length {len(text)}",
            "word_count": len(text.split()),
            "character_count": len(text),
            "agent": self.agent_name
        }
    
    async def _review_code(self, code: str) -> Dict[str, Any]:
        """Generic code review"""
        return {
            "review": f"Code review completed for {len(code.split())} lines",
            "suggestions": ["Consider adding comments", "Check for error handling"],
            "agent": self.agent_name
        }
    
    async def _system_check(self) -> Dict[str, Any]:
        """Generic system check"""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.agent_name,
            "uptime": time.time() - (self.last_heartbeat.timestamp() if self.last_heartbeat else time.time())
        }
    
    async def _update_task_status(self, task_id: str, status: str, result: Dict[str, Any] = None):
        """Update task status in backend"""
        try:
            data = {
                "task_id": task_id,
                "status": status,
                "agent_id": self.agent_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if result:
                data["result"] = result
            
            response = requests.put(
                f"{self.backend_url}/api/v1/tasks/{task_id}/status",
                json=data,
                timeout=10
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Failed to update task status: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error updating task status: {e}")
    
    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "status": "online" if self.is_running else "offline",
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "agent_type": self.agent_type,
            "uptime": time.time()
        }

def main():
    """Main entry point for generic agent"""
    agent = GenericAgent()
    asyncio.run(agent.start())

if __name__ == "__main__":
    main()