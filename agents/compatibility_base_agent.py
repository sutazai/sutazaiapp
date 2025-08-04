#!/usr/bin/env python3
"""
Backward compatibility wrapper for BaseAgentV2
Handles import fallbacks without duplication
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Setup paths for imports
def setup_import_paths():
    """Setup import paths for agent compatibility"""
    current_dir = Path(__file__).parent.absolute()
    paths_to_add = [
        str(current_dir / 'core'),
        str(current_dir),
        str(current_dir.parent)
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

setup_import_paths()

# Try to import the full BaseAgentV2
try:
    from core.base_agent_v2 import BaseAgentV2
except ImportError:
    try:
        from agents.core.base_agent_v2 import BaseAgentV2
    except ImportError:
        # Create minimal fallback if all imports fail
        class BaseAgentV2:
            """Minimal fallback BaseAgentV2 for container compatibility"""
            
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
                self.logger.info(f"Agent {self.name} started in compatibility mode")
                
                # Simple event loop for standalone operation
                try:
                    import asyncio
                    asyncio.run(self._standalone_loop())
                except KeyboardInterrupt:
                    self.logger.info("Agent stopped by user")
                except Exception as e:
                    self.logger.error(f"Agent error: {e}")
            
            async def _standalone_loop(self):
                """Simple standalone event loop"""
                import asyncio
                self.logger.info(f"Agent {self.name} running on port {self.port}")
                
                # In standalone mode, just keep the agent alive
                while True:
                    await asyncio.sleep(10)
                    self.logger.debug(f"Agent {self.name} heartbeat - processed {self.tasks_processed} tasks")

# For backward compatibility
BaseAgent = BaseAgentV2

__all__ = ['BaseAgentV2', 'BaseAgent']