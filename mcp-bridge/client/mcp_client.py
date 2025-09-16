#!/usr/bin/env python3
"""
MCP Client Library for AI Agents
Provides easy interface for agents to communicate via MCP Bridge
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable
import httpx
import websockets
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting AI Agents to the MCP Bridge"""
    
    def __init__(self, 
                 agent_id: str,
                 agent_name: str,
                 capabilities: list,
                 port: int,
                 bridge_url: str = "http://localhost:11100"):
        """
        Initialize MCP Client
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable name
            capabilities: List of agent capabilities
            port: Port the agent is running on
            bridge_url: URL of the MCP Bridge
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.port = port
        self.bridge_url = bridge_url
        self.ws_url = bridge_url.replace("http", "ws") + f"/ws/{agent_id}"
        self.websocket = None
        self.message_handlers = {}
        self.is_connected = False
        
    async def register(self) -> bool:
        """Register agent with MCP Bridge"""
        try:
            async with httpx.AsyncClient() as client:
                # Update agent status to online
                response = await client.post(
                    f"{self.bridge_url}/agents/{self.agent_id}/status",
                    params={"status": "online"}
                )
                if response.status_code == 200:
                    logger.info(f"Agent {self.agent_id} registered successfully")
                    return True
                else:
                    logger.error(f"Registration failed: {response.text}")
                    return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
    
    async def connect_websocket(self):
        """Establish WebSocket connection for real-time communication"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            logger.info(f"WebSocket connected for {self.agent_id}")
            
            # Start listening for messages
            asyncio.create_task(self._listen_messages())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
    
    async def _listen_messages(self):
        """Listen for incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for {self.agent_id}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming messages"""
        msg_type = data.get("type", "unknown")
        
        if msg_type in self.message_handlers:
            handler = self.message_handlers[msg_type]
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error handling message type {msg_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {msg_type}")
    
    def on_message(self, msg_type: str):
        """Decorator for registering message handlers"""
        def decorator(func: Callable):
            self.message_handlers[msg_type] = func
            return func
        return decorator
    
    async def send_message(self, 
                          target: str, 
                          msg_type: str, 
                          payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send message to another agent or service
        
        Args:
            target: Target agent or service ID
            msg_type: Type of message
            payload: Message payload
            
        Returns:
            Response from the bridge
        """
        message = {
            "id": str(uuid.uuid4()),
            "source": self.agent_id,
            "target": target,
            "type": msg_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.bridge_url}/route",
                json=message
            )
            return response.json()
    
    async def submit_task(self, 
                          task_type: str,
                          description: str,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Submit a task for execution
        
        Args:
            task_type: Type of task
            description: Task description
            params: Optional parameters
            
        Returns:
            Task submission result
        """
        task = {
            "task_id": str(uuid.uuid4()),
            "task_type": task_type,
            "description": description,
            "agent": self.agent_id,
            "params": params or {}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.bridge_url}/tasks/submit",
                json=task
            )
            return response.json()
    
    async def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Get information about a specific service"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.bridge_url}/services/{service_name}"
            )
            return response.json()
    
    async def get_agent_info(self, agent_id: str) -> Dict[str, Any]:
        """Get information about another agent"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.bridge_url}/agents/{agent_id}"
            )
            return response.json()
    
    async def broadcast(self, msg_type: str, payload: Dict[str, Any]):
        """Broadcast message to all connected clients via WebSocket"""
        if self.websocket and self.is_connected:
            message = {
                "type": "broadcast",
                "payload": {
                    "msg_type": msg_type,
                    "data": payload,
                    "from": self.agent_id
                }
            }
            await self.websocket.send(json.dumps(message))
    
    async def disconnect(self):
        """Disconnect from MCP Bridge"""
        # Update status to offline
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.bridge_url}/agents/{self.agent_id}/status",
                params={"status": "offline"}
            )
        
        # Close WebSocket
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
        
        logger.info(f"Agent {self.agent_id} disconnected")


# Example usage for agents
async def example_agent():
    """Example of how an agent would use the MCP Client"""
    
    # Initialize client
    client = MCPClient(
        agent_id="example-agent",
        agent_name="Example Agent",
        capabilities=["test", "demo"],
        port=11199
    )
    
    # Register with bridge
    await client.register()
    
    # Connect WebSocket
    await client.connect_websocket()
    
    # Register message handlers
    @client.on_message("task.request")
    async def handle_task(data):
        print(f"Received task: {data}")
        # Process task here
        return {"status": "completed", "result": "Task done"}
    
    @client.on_message("query")
    async def handle_query(data):
        print(f"Received query: {data}")
        return {"response": "Query answered"}
    
    # Send a message to another agent
    response = await client.send_message(
        target="letta",
        msg_type="collaboration.request",
        payload={"task": "Help with memory management"}
    )
    print(f"Response from Letta: {response}")
    
    # Submit a task
    task_result = await client.submit_task(
        task_type="code.generation",
        description="Generate Python function for data processing",
        params={"language": "python", "framework": "pandas"}
    )
    print(f"Task result: {task_result}")
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await client.disconnect()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_agent())