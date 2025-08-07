#!/usr/bin/env python3
"""
Enhanced Generic SutazAI Agent with Health Check Server
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
from threading import Thread
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

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

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for health check endpoints"""
    
    def __init__(self, agent_instance, *args, **kwargs):
        self.agent = agent_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for health checks"""
        path = self.path.split('?')[0]
        
        if path == '/health' or path == '/healthz':
            self._handle_health_check()
        elif path == '/status':
            self._handle_status_check()
        elif path == '/metrics':
            self._handle_metrics()
        else:
            self._send_response(404, {'error': 'Not found'})
    
    def _handle_health_check(self):
        """Basic health check endpoint"""
        if self.agent and self.agent.is_running:
            health_data = {
                'status': 'healthy',
                'agent_name': self.agent.agent_name,
                'agent_type': self.agent.agent_type,
                'timestamp': datetime.utcnow().isoformat(),
                'uptime': time.time() - self.agent.start_time if hasattr(self.agent, 'start_time') else 0
            }
            self._send_response(200, health_data)
        else:
            self._send_response(503, {'status': 'unhealthy', 'reason': 'Agent not running'})
    
    def _handle_status_check(self):
        """Detailed status check"""
        if self.agent:
            status_data = {
                'agent_name': self.agent.agent_name,
                'agent_type': self.agent.agent_type,
                'is_running': self.agent.is_running,
                'last_heartbeat': self.agent.last_heartbeat.isoformat() if self.agent.last_heartbeat else None,
                'config': self.agent.config,
                'timestamp': datetime.utcnow().isoformat()
            }
            self._send_response(200, status_data)
        else:
            self._send_response(503, {'error': 'Agent not initialized'})
    
    def _handle_metrics(self):
        """Agent metrics endpoint"""
        if self.agent and hasattr(self.agent, '_get_agent_metrics'):
            try:
                metrics = asyncio.run(self.agent._get_agent_metrics())
                self._send_response(200, metrics)
            except Exception as e:
                self._send_response(500, {'error': f'Metrics error: {str(e)}'})
        else:
            self._send_response(503, {'error': 'Metrics not available'})
    
    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging"""
        pass

class GenericAgentWithHealth:
    """
    Enhanced generic agent with built-in health check server
    """
    
    def __init__(self, config_path: str = "/app/config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.agent_name = os.getenv("AGENT_NAME", "generic-agent")
        self.agent_type = os.getenv("AGENT_TYPE", "generic")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://sutazai-ollama:10104")
        self.backend_url = os.getenv("BACKEND_URL", "http://sutazai-backend:8000")
        self.health_port = int(os.getenv("HEALTH_PORT", "8080"))
        
        # Setup logging
        self._setup_logging()
        
        # Agent state
        self.is_running = False
        self.last_heartbeat = None
        self.start_time = time.time()
        self.health_server = None
        self.health_thread = None
        
        self.logger.info(f"Initialized {self.agent_name} ({self.agent_type}) with health check on port {self.health_port}")
    
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
    
    def _start_health_server(self):
        """Start the health check HTTP server"""
        try:
            handler = lambda *args, **kwargs: HealthCheckHandler(self, *args, **kwargs)
            self.health_server = socketserver.TCPServer(("", self.health_port), handler)
            self.health_server.allow_reuse_address = True
            
            def serve_health():
                self.logger.info(f"Health check server starting on port {self.health_port}")
                self.health_server.serve_forever()
            
            self.health_thread = Thread(target=serve_health, daemon=True)
            self.health_thread.start()
            self.logger.info(f"Health check server started on port {self.health_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to start health server: {e}")
    
    def _stop_health_server(self):
        """Stop the health check HTTP server"""
        if self.health_server:
            self.health_server.shutdown()
            self.health_server.server_close()
            self.logger.info("Health check server stopped")
    
    async def start(self):
        """Start the agent and begin processing"""
        self.logger.info(f"Starting {self.agent_name}...")
        self.is_running = True
        
        # Start health check server
        self._start_health_server()
        
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
        self._stop_health_server()
    
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
                "health_endpoint": f"http://{self.agent_name}:{self.health_port}/health",
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
    
    # ... (rest of the methods remain the same as the original generic_agent.py)
    
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
        """Execute a task"""
        return {
            "success": True,
            "result": f"Task processed by {self.agent_name}",
            "agent": self.agent_name,
            "timestamp": datetime.utcnow().isoformat()
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
            "uptime": time.time() - self.start_time,
            "health_port": self.health_port
        }

def main():
    """Main entry point for generic agent with health checks"""
    agent = GenericAgentWithHealth()
    asyncio.run(agent.start())

if __name__ == "__main__":
    main()