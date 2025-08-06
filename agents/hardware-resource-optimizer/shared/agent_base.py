#!/usr/bin/env python3
"""
Base Agent for SutazAI System
Provides core functionality for all agent types
"""

import os
import sys
import json
import time
import logging
import asyncio
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class BaseAgent:
    """Base class for all SutazAI agents"""
    
    def __init__(self, config_path: str = '/app/config.json'):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        self.agent_name = os.getenv('AGENT_NAME', 'base-agent')
        self.agent_type = os.getenv('AGENT_TYPE', 'base')
        self.backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://ollama:10104')
        self.is_running = False
        self.tasks_processed = 0
        self.last_heartbeat = time.time()
        
        self.logger.info(f"Initializing {self.agent_name} agent")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        return {}
    
    def register_with_coordinator(self) -> bool:
        """Register this agent with the coordinator"""
        try:
            registration_data = {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "capabilities": self.config.get("capabilities", []),
                "status": "active",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.backend_url}/api/agents/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully registered {self.agent_name}")
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
    def send_heartbeat(self):
        """Send periodic heartbeat to coordinator"""
        while self.is_running:
            try:
                heartbeat_data = {
                    "agent_name": self.agent_name,
                    "status": "active",
                    "tasks_processed": self.tasks_processed,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = requests.post(
                    f"{self.backend_url}/api/agents/heartbeat",
                    json=heartbeat_data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.logger.debug(f"Heartbeat sent successfully")
                else:
                    self.logger.warning(f"Heartbeat failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
            
            time.sleep(30)  # Send heartbeat every 30 seconds
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Fetch next task from the coordinator"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/tasks/next/{self.agent_type}",
                timeout=10
            )
            
            if response.status_code == 200:
                task = response.json()
                if task:
                    self.logger.info(f"Received task: {task.get('id', 'unknown')}")
                    return task
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task: {e}")
            return None
    
    def report_task_complete(self, task_id: str, result: Dict[str, Any]):
        """Report task completion to coordinator"""
        try:
            completion_data = {
                "task_id": task_id,
                "agent_name": self.agent_name,
                "status": "completed",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = requests.post(
                f"{self.backend_url}/api/tasks/complete",
                json=completion_data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Task {task_id} marked as complete")
                self.tasks_processed += 1
            else:
                self.logger.error(f"Failed to report task completion: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error reporting task completion: {e}")
    
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - to be overridden by specific agents"""
        self.logger.info(f"Processing task: {task}")
        # Base implementation - just echo the task
        return {
            "status": "success",
            "message": f"Task processed by {self.agent_name}",
            "task_id": task.get("id", "unknown")
        }
    
    def query_ollama(self, prompt: str, model: str = "tinyllama") -> Optional[str]:
        """Query Ollama for AI assistance"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                self.logger.error(f"Ollama query failed: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Ollama query error: {e}")
            return None
    
    async def run_async(self):
        """Async run loop for the agent"""
        self.is_running = True
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.send_heartbeat)
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        
        # Register with coordinator
        if not self.register_with_coordinator():
            self.logger.warning("Failed to register with coordinator, continuing anyway...")
        
        self.logger.info(f"{self.agent_name} is running and waiting for tasks...")
        
        while self.is_running:
            try:
                # Get next task
                task = self.get_next_task()
                
                if task:
                    # Process the task
                    result = self.process_task(task)
                    
                    # Report completion
                    self.report_task_complete(task.get("id", "unknown"), result)
                else:
                    # No task available, wait a bit
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                self.logger.info("Received interrupt signal")
                self.is_running = False
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
        
        self.logger.info(f"{self.agent_name} shutting down...")
    
    def run(self):
        """Run the agent"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")


if __name__ == "__main__":
    agent = BaseAgent()
    agent.run()