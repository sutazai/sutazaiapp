"""
Agent Communication

This module provides functionality for agent-to-agent communication,
including message passing, routing, and processing.
"""

import logging
import threading
import queue
from typing import Dict, List, Callable, Any, Optional, Set
from collections import defaultdict

from .message_protocol import Message, MessageType, MessageProtocol

# Configure logging
logger = logging.getLogger(__name__)


class AgentCommunication:
    """
    Facilitates communication between agents in the system.

    This class implements a message bus that allows agents to send messages
    to each other, register for specific message types, and process messages
    asynchronously.
    """

    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize the agent communication system.

        Args:
            max_queue_size: Maximum size of the message queue
        """
        self.message_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.subscribers: Dict[
            MessageType, Dict[str, List[Callable[[Message], None]]]
        ] = defaultdict(lambda: defaultdict(list))
        self.broadcast_subscribers: Dict[
            MessageType, List[Callable[[Message], None]]
        ] = defaultdict(list)
        self.running = False
        self.processing_thread = None
        self.agent_ids: Set[str] = set()

    def start(self) -> None:
        """Start the message processing thread."""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_messages, daemon=True
        )
        self.processing_thread.start()
        logger.info("Agent communication system started")

    def stop(self) -> None:
        """Stop the message processing thread."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Agent communication system stopped")

    def register_agent(self, agent_id: str) -> None:
        """
        Register an agent with the communication system.

        Args:
            agent_id: Agent ID to register
        """
        self.agent_ids.add(agent_id)
        logger.debug(f"Agent {agent_id} registered with communication system")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the communication system.

        Args:
            agent_id: Agent ID to unregister
        """
        self.agent_ids.discard(agent_id)

        # Remove agent's subscriptions
        for msg_type in self.subscribers:
            if agent_id in self.subscribers[msg_type]:
                del self.subscribers[msg_type][agent_id]

        logger.debug(f"Agent {agent_id} unregistered from communication system")

    def send_message(self, message: Message) -> bool:
        """
        Send a message to the specified recipient.

        Args:
            message: Message to send

        Returns:
            bool: True if message was successfully queued, False otherwise
        """
        if not MessageProtocol.validate_message(message):
            logger.warning(f"Invalid message rejected: {message}")
            return False

        try:
            # Add to queue with priority (lower number = higher priority)
            self.message_queue.put((message.priority, message))
            return True
        except queue.Full:
            logger.error("Message queue full, message rejected")
            return False

    def broadcast(
        self, sender_id: str, message_type: MessageType, content: Dict[str, Any]
    ) -> bool:
        """
        Broadcast a message to all registered agents.

        Args:
            sender_id: ID of the sending agent
            message_type: Type of message to broadcast
            content: Message content

        Returns:
            bool: True if message was successfully queued, False otherwise
        """
        message = MessageProtocol.create_message(
            message_type=message_type,
            sender_id=sender_id,
            content=content,
            recipient_id=None,  # No specific recipient for broadcast
        )
        message.metadata["broadcast"] = True

        return self.send_message(message)

    def subscribe(
        self,
        agent_id: str,
        message_type: MessageType,
        callback: Callable[[Message], None],
    ) -> None:
        """
        Subscribe an agent to receive messages of a specific type.

        Args:
            agent_id: ID of the subscribing agent
            message_type: Type of message to subscribe to
            callback: Function to call when a message is received
        """
        self.subscribers[message_type][agent_id].append(callback)
        logger.debug(f"Agent {agent_id} subscribed to {message_type.value} messages")

    def subscribe_broadcast(
        self, message_type: MessageType, callback: Callable[[Message], None]
    ) -> None:
        """
        Subscribe to broadcast messages of a specific type.

        Args:
            message_type: Type of message to subscribe to
            callback: Function to call when a message is received
        """
        self.broadcast_subscribers[message_type].append(callback)

    def unsubscribe(
        self, agent_id: str, message_type: Optional[MessageType] = None
    ) -> None:
        """
        Unsubscribe an agent from receiving messages.

        Args:
            agent_id: ID of the agent to unsubscribe
            message_type: Type of message to unsubscribe from (None for all)
        """
        if message_type:
            if agent_id in self.subscribers[message_type]:
                del self.subscribers[message_type][agent_id]
        else:
            for msg_type in self.subscribers:
                if agent_id in self.subscribers[msg_type]:
                    del self.subscribers[msg_type][agent_id]

    def _process_messages(self) -> None:
        """Process messages from the queue and deliver to subscribers."""
        while self.running:
            try:
                # Get message with priority
                _, message = self.message_queue.get(block=True, timeout=1.0)

                # Skip expired messages
                if message.is_expired():
                    logger.debug(f"Skipping expired message: {message.message_id}")
                    self.message_queue.task_done()
                    continue

                # Handle broadcast messages
                is_broadcast = (
                    message.metadata.get("broadcast", False)
                    or message.recipient_id is None
                )
                if is_broadcast:
                    self._deliver_broadcast(message)
                # Handle directed messages
                elif message.recipient_id:
                    self._deliver_message(message)

                self.message_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def _deliver_message(self, message: Message) -> None:
        """
        Deliver a message to its intended recipient.

        Args:
            message: Message to deliver
        """
        recipient_id = message.recipient_id
        message_type = message.message_type

        # Check if recipient exists
        if recipient_id not in self.agent_ids:
            logger.warning(
                f"Recipient {recipient_id} not found for message {message.message_id}"
            )
            return

        # Deliver to all subscribers of this agent for this message type
        for callback in self.subscribers[message_type][recipient_id]:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in message handler for agent {recipient_id}: {e}")

    def _deliver_broadcast(self, message: Message) -> None:
        """
        Deliver a broadcast message to all subscribers.

        Args:
            message: Message to broadcast
        """
        message_type = message.message_type

        # Deliver to specific broadcast subscribers
        for callback in self.broadcast_subscribers[message_type]:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in broadcast message handler: {e}")

        # Deliver to all agents that subscribe to this message type
        for agent_id, callbacks in self.subscribers[message_type].items():
            if agent_id != message.sender_id:  # Don't send back to sender
                for callback in callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(
                            f"Error in message handler for agent {agent_id}: {e}"
                        )
