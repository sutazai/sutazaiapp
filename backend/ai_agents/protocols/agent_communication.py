"""
Agent Communication Protocol
Handles inter-agent communication and message routing
"""

import logging
import threading
import asyncio
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentCommunication:
    """Communication system for inter-agent messaging"""
    
    def __init__(self):
        self.agents = {}
        self.subscriptions = defaultdict(list)  # message_type -> list of callbacks
        self.message_queue = asyncio.Queue()
        self.running = False
        self.processor_task = None
        
    def start(self):
        """Start the communication system"""
        if self.running:
            return
            
        self.running = True
        logger.info("Agent communication system started")
        
    def stop(self):
        """Stop the communication system"""
        self.running = False
        if self.processor_task:
            self.processor_task.cancel()
        logger.info("Agent communication system stopped")
        
    def register_agent(self, agent_id: str):
        """Register an agent for communication"""
        self.agents[agent_id] = {
            "registered_at": datetime.now(),
            "message_count": 0
        }
        logger.info(f"Agent {agent_id} registered for communication")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from communication"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered from communication")
            
    def subscribe(self, agent_id: str, message_type: "MessageType", callback: Callable):
        """Subscribe an agent to specific message types"""
        subscription = {
            "agent_id": agent_id,
            "callback": callback,
            "subscribed_at": datetime.now()
        }
        self.subscriptions[message_type].append(subscription)
        logger.info(f"Agent {agent_id} subscribed to {message_type}")
        
    def send_message(self, message: "Message"):
        """Send a message through the communication system"""
        try:
            # Route message to appropriate subscribers
            message_type = getattr(message, 'message_type', None)
            if message_type and message_type in self.subscriptions:
                for subscription in self.subscriptions[message_type]:
                    try:
                        subscription["callback"](message)
                    except Exception as e:
                        logger.error(f"Error delivering message to {subscription['agent_id']}: {e}")
                        
            # Update sender's message count
            sender_id = getattr(message, 'sender_id', None)
            if sender_id in self.agents:
                self.agents[sender_id]["message_count"] += 1
                
            logger.debug(f"Message sent from {sender_id} to {getattr(message, 'recipient_id', 'broadcast')}")
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        return {
            "registered_agents": len(self.agents),
            "active_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "total_messages": sum(agent["message_count"] for agent in self.agents.values()),
            "agents": dict(self.agents),
            "running": self.running
