#!/usr/bin/env python3
"""
SutazAI Agent Communication System
Handles inter-agent communication and message routing
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class AgentCommunication:
    """Manages communication between agents"""
    
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.active_channels = {}
        self.message_history = []
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize communication system"""
        self.initialized = True
        logger.info("Agent communication system initialized")
    
    async def send_message(self, from_agent: str, to_agent: str, message: Dict[str, Any]) -> bool:
        """Send message between agents"""
        try:
            message_envelope = {
                "id": f"msg_{datetime.now().timestamp()}",
                "from": from_agent,
                "to": to_agent,
                "content": message,
                "timestamp": datetime.now().isoformat(),
                "status": "sent"
            }
            
            await self.message_queue.put(message_envelope)
            self.message_history.append(message_envelope)
            
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Receive message for specific agent"""
        try:
            # Check queue for messages addressed to this agent
            for _ in range(self.message_queue.qsize()):
                message = await self.message_queue.get()
                if message["to"] == agent_id:
                    return message
                else:
                    # Put back if not for this agent
                    await self.message_queue.put(message)
            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None
    
    def create_channel(self, name: str, participants: List[str]) -> bool:
        """Create communication channel"""
        try:
            self.active_channels[name] = {
                "name": name,
                "participants": participants,
                "created_at": datetime.now().isoformat(),
                "messages": []
            }
            return True
        except Exception as e:
            logger.error(f"Failed to create channel: {e}")
            return False
    
    def get_message_history(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get message history for agent or all"""
        if agent_id:
            return [msg for msg in self.message_history 
                   if msg["from"] == agent_id or msg["to"] == agent_id]
        return self.message_history
    
    def cleanup(self) -> None:
        """Cleanup communication system"""
        self.active_channels.clear()
        self.message_history.clear()
        self.initialized = False


