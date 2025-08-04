#!/usr/bin/env python3
"""
Simplified Base Agent for SutazAI System
Compatible constructor signature for existing agent implementations
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgentV2:
    """
    Simplified base class for SutazAI agents
    Compatible with existing agent constructor patterns
    """
    
    def __init__(self, 
                 agent_id: str,
                 name: str,
                 port: int = 8080,
                 description: str = "SutazAI Agent"):
        
        self.agent_id = agent_id
        self.name = name
        self.port = port
        self.description = description
        
        # Basic configuration
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.status = "initializing"
        
        # Environment variables
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        
        # Simple metrics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.startup_time = datetime.utcnow()
        
        self.logger.info(f"Initialized {name} ({agent_id}) on port {port}")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task - basic implementation"""
        try:
            task_type = task.get("type", "unknown")
            
            if task_type == "health":
                return {"status": "healthy", "agent": self.agent_id}
            
            # Simple echo response for base implementation
            self.tasks_processed += 1
            return {
                "status": "success",
                "message": f"Task processed by {self.name}",
                "task_id": task.get("id", "unknown"),
                "agent": self.agent_id,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.tasks_failed += 1
            self.logger.error(f"Error processing task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent": self.agent_id
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Basic health check"""
        uptime = (datetime.utcnow() - self.startup_time).total_seconds()
        
        return {
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "status": self.status,
            "healthy": True,
            "uptime_seconds": uptime,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "description": self.description,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def start(self):
        """Start the agent in standalone mode"""
        self.status = "active"
        self.logger.info(f"Agent {self.name} started in standalone mode")
        
        # Simple event loop for standalone operation
        try:
            asyncio.run(self._standalone_loop())
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent error: {e}")
    
    async def _standalone_loop(self):
        """Simple standalone event loop"""
        self.logger.info(f"Agent {self.name} running on port {self.port}")
        
        # In standalone mode, just keep the agent alive
        while True:
            await asyncio.sleep(10)
            self.logger.debug(f"Agent {self.name} heartbeat - processed {self.tasks_processed} tasks")

# Backward compatibility
BaseAgent = BaseAgentV2