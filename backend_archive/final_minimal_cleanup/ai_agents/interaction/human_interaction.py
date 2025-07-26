"""
Human Interaction Module

This module provides functionality for human-in-the-loop interaction with agents,
enabling approval workflows, decision points, and task delegation between agents and humans.
"""

import uuid
import logging
import threading
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue
from collections import defaultdict

from ..protocols.message_protocol import Message, MessageType, MessageProtocol
from ..protocols.agent_communication import AgentCommunication

# Configure logging
logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """Type of human interaction."""

    APPROVAL = "approval"
    DECISION = "decision"
    INFORMATION = "information"
    INPUT = "input"
    ESCALATION = "escalation"


class InteractionStatus(Enum):
    """Status of a human interaction request."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class InteractionRequest:
    """
    Request for human interaction.
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interaction_type: InteractionType = InteractionType.INFORMATION
    title: str = "Interaction Request"
    description: str = ""
    agent_id: str = ""
    user_id: Optional[str] = None
    task_id: Optional[str] = None
    workflow_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    options: List[Dict[str, Any]] = field(default_factory=list)
    status: InteractionStatus = InteractionStatus.PENDING
    priority: int = 3  # 1 (highest) to 5 (lowest)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction request to dictionary."""
        return {
            "request_id": self.request_id,
            "interaction_type": self.interaction_type.value,
            "title": self.title,
            "description": self.description,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "data": self.data,
            "options": self.options,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "response": self.response,
        }

    def is_expired(self) -> bool:
        """Check if the interaction request has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class InteractionResponse:
    """
    Response to a human interaction request.
    """

    request_id: str
    user_id: str
    response: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert interaction response to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "response": self.response,
            "timestamp": self.timestamp.isoformat(),
        }


class HumanInteractionPoint:
    """
    A human interaction point within an agent workflow.

    This class represents a point in the agent workflow where human
    interaction is required, such as for approvals, decisions, or
    information input.
    """

    def __init__(
        self,
        interaction_type: InteractionType,
        title: str,
        description: str,
        agent_id: str,
        options: Optional[List[Dict[str, Any]]] = None,
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        priority: int = 3,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        """
        Initialize a human interaction point.

        Args:
            interaction_type: Type of interaction
            title: Interaction title
            description: Interaction description
            agent_id: ID of the agent requesting interaction
            options: List of options for decision interactions
            data: Additional data for the interaction
            timeout: Timeout in seconds
            priority: Interaction priority (1-5, 1 is highest)
            user_id: ID of the user to interact with (if specific)
            task_id: ID of the related task (if any)
            workflow_id: ID of the related workflow (if any)
        """
        self.interaction_type = interaction_type
        self.title = title
        self.description = description
        self.agent_id = agent_id
        self.options = options or []
        self.data = data or {}
        self.timeout = timeout
        self.priority = priority
        self.user_id = user_id
        self.task_id = task_id
        self.workflow_id = workflow_id

    def create_request(self) -> InteractionRequest:
        """
        Create an interaction request from this interaction point.

        Returns:
            InteractionRequest: Created interaction request
        """
        expires_at = None
        if self.timeout:
            expires_at = datetime.utcnow() + timedelta(seconds=self.timeout)

        return InteractionRequest(
            interaction_type=self.interaction_type,
            title=self.title,
            description=self.description,
            agent_id=self.agent_id,
            user_id=self.user_id,
            task_id=self.task_id,
            workflow_id=self.workflow_id,
            data=self.data,
            options=self.options,
            priority=self.priority,
            expires_at=expires_at,
        )


class Interaction:
    """
    Represents a human interaction.
    """

    def __init__(
        self,
        interaction_id: str,
        interaction_type: InteractionType,
        agent_id: str,
        user_id: Optional[str] = None,
        callback: Optional[Callable[[InteractionResponse], None]] = None,
        request_data: Dict[str, Any] = {},
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.interaction_id = interaction_id
        self.interaction_type = interaction_type
        self.agent_id = agent_id
        self.user_id = user_id
        self.callback = callback
        self.timestamp = datetime.utcnow()
        self.status = InteractionStatus.PENDING
        self.request_data = request_data
        self.timeout = timeout
        self.metadata = metadata or {}
        self.responses: List[InteractionResponse] = []

    def add_response(self, response: "InteractionResponse") -> None:
        """Add a response to this interaction."""
        self.responses.append(response)


class InteractionManager:
    """
    Manager for human interaction requests.

    This class provides functionality for creating, tracking, and responding
    to human interaction requests, as well as handling timeouts and notifications.
    """

    def __init__(self, agent_communication: AgentCommunication):
        """
        Initialize the interaction manager.

        Args:
            agent_communication: Agent communication interface
        """
        self.agent_communication = agent_communication
        self.requests: Dict[str, InteractionRequest] = {}
        self.callbacks: Dict[str, Callable[[InteractionResponse], None]] = {}
        self.default_callbacks: Dict[
            InteractionType, List[Callable[[InteractionResponse], None]]
        ] = defaultdict(list)

        self.running = False
        self.cleanup_thread: Optional[threading.Thread] = None
        self.notification_thread: Optional[threading.Thread] = None
        self._input_thread: Optional[threading.Thread] = None
        self._output_thread: Optional[threading.Thread] = None

        # Queues for input and output processing
        self._input_queue: Queue[InteractionRequest] = Queue()
        self._output_queue: Queue[InteractionResponse] = Queue()

        # Configuration
        self.default_timeout = timedelta(minutes=30)
        self.cleanup_interval = timedelta(minutes=5)

        logger.info("Interaction manager initialized")

    def start(self) -> None:
        """Start the interaction manager."""
        if self.running:
            return

        self.running = True

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired_requests, daemon=True
        )
        self.cleanup_thread.start()

        # Start notification thread
        self.notification_thread = threading.Thread(
            target=self._process_notifications, daemon=True
        )
        self.notification_thread.start()

        # Subscribe to interaction responses
        self.agent_communication.subscribe_broadcast(
            MessageType.HUMAN_INPUT_RESPONSE, self._handle_interaction_response
        )

        # Start input thread
        if not self._input_thread or not self._input_thread.is_alive():
            self._input_thread = threading.Thread(
                target=self._process_input, daemon=True
            )
            if self._input_thread is not None:
                assert isinstance(self._input_thread, threading.Thread)
                self._input_thread.start()

        # Start output thread
        if not self._output_thread or not self._output_thread.is_alive():
            self._output_thread = threading.Thread(
                target=self._process_output, daemon=True
            )
            if self._output_thread is not None:
                assert isinstance(self._output_thread, threading.Thread)
                self._output_thread.start()

        logger.info("Interaction manager started")

    def stop(self) -> None:
        """Stop the interaction manager."""
        self.running = False

        # Wait for threads to finish
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)

        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=5.0)

        # Unsubscribe from interaction responses
        # (No direct method to unsubscribe from broadcasts, would need to add one)

        logger.info("Interaction manager stopped")

    def create_interaction(
        self,
        interaction_point: HumanInteractionPoint,
        callback: Optional[Callable[[InteractionResponse], None]] = None,
    ) -> str:
        """
        Create a new interaction request.

        Args:
            interaction_point: Human interaction point
            callback: Function to call when interaction is completed

        Returns:
            str: Request ID
        """
        # Create request
        request = interaction_point.create_request()

        # Store request
        self.requests[request.request_id] = request

        # Store callback if provided
        if callback:
            self.callbacks[request.request_id] = callback

        # Send interaction request message
        self._send_interaction_request(request)

        logger.info(f"Created interaction request: {request.request_id}")
        return request.request_id

    def register_default_callback(
        self,
        interaction_type: InteractionType,
        callback: Callable[[InteractionResponse], None],
    ) -> None:
        """
        Register a default callback for an interaction type.

        Args:
            interaction_type: Interaction type to register for
            callback: Function to call when interaction is completed
        """
        self.default_callbacks[interaction_type].append(callback)

    def respond_to_interaction(
        self, request_id: str, user_id: str, response: Dict[str, Any]
    ) -> bool:
        """
        Respond to an interaction request.

        Args:
            request_id: Request ID to respond to
            user_id: ID of the user responding
            response: Response data

        Returns:
            bool: True if response was successfully processed, False otherwise

        Raises:
            ValueError: If request ID is invalid or request is no longer active
        """
        if request_id not in self.requests:
            raise ValueError(f"Interaction request {request_id} not found")

        request = self.requests[request_id]

        # Check if request is still active
        if request.status != InteractionStatus.PENDING:
            raise ValueError(f"Interaction request {request_id} is no longer active")

        # Update request
        request.status = InteractionStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        request.response = response

        # Create response object
        interaction_response = InteractionResponse(
            request_id=request_id, user_id=user_id, response=response
        )

        # Process response
        self._process_interaction_response(interaction_response)

        return True

    def cancel_interaction(self, request_id: str) -> bool:
        """
        Cancel an interaction request.

        Args:
            request_id: Request ID to cancel

        Returns:
            bool: True if request was cancelled, False otherwise

        Raises:
            ValueError: If request ID is invalid
        """
        if request_id not in self.requests:
            raise ValueError(f"Interaction request {request_id} not found")

        request = self.requests[request_id]

        # Check if request is already completed
        if request.status in (
            InteractionStatus.COMPLETED,
            InteractionStatus.EXPIRED,
            InteractionStatus.CANCELLED,
        ):
            return False

        # Update request
        request.status = InteractionStatus.CANCELLED

        # Send cancellation notification
        self._send_interaction_cancellation(request)

        return True

    def get_interaction_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get the status of an interaction request.

        Args:
            request_id: Request ID to get status for

        Returns:
            Dict[str, Any]: Interaction request status

        Raises:
            ValueError: If request ID is invalid
        """
        if request_id not in self.requests:
            raise ValueError(f"Interaction request {request_id} not found")

        return self.requests[request_id].to_dict()

    def get_pending_interactions(
        self,
        user_id: Optional[str] = None,
        interaction_type: Optional[InteractionType] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get pending interaction requests.

        Args:
            user_id: Filter by user ID
            interaction_type: Filter by interaction type

        Returns:
            List[Dict[str, Any]]: List of pending interaction requests
        """
        pending = []

        for request in self.requests.values():
            # Skip non-pending requests
            if request.status != InteractionStatus.PENDING:
                continue

            # Apply user filter
            if user_id and request.user_id and request.user_id != user_id:
                continue

            # Apply type filter
            if interaction_type and request.interaction_type != interaction_type:
                continue

            pending.append(request.to_dict())

        return pending

    def acknowledge_interaction(self, request_id: str, user_id: str) -> bool:
        """
        Acknowledge an interaction request.

        Args:
            request_id: Request ID to acknowledge
            user_id: ID of the user acknowledging

        Returns:
            bool: True if request was acknowledged, False otherwise

        Raises:
            ValueError: If request ID is invalid
        """
        if request_id not in self.requests:
            raise ValueError(f"Interaction request {request_id} not found")

        request = self.requests[request_id]

        # Check if request is pending
        if request.status != InteractionStatus.PENDING:
            return False

        # Update request
        request.status = InteractionStatus.ACKNOWLEDGED

        # Send acknowledgement notification
        self._send_interaction_acknowledgement(request, user_id)

        return True

    def _send_interaction_request(self, request: InteractionRequest) -> bool:
        """
        Send an interaction request message.

        Args:
            request: Interaction request to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        content = {"request": request.to_dict()}

        message = MessageProtocol.create_message(
            message_type=MessageType.HUMAN_INPUT_REQUEST,
            sender_id=request.agent_id,
            content=content,
            recipient_id=None,  # Broadcast to all
            priority=request.priority,
        )

        return self.agent_communication.send_message(message)

    def _send_interaction_cancellation(self, request: InteractionRequest) -> bool:
        """
        Send an interaction cancellation message.

        Args:
            request: Interaction request to cancel

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        content = {
            "request_id": request.request_id,
            "status": InteractionStatus.CANCELLED.value,
        }

        message = MessageProtocol.create_message(
            message_type=MessageType.NOTIFICATION,
            sender_id="interaction_manager",
            content=content,
            recipient_id=None,  # Broadcast to all
            priority=request.priority,
        )

        return self.agent_communication.send_message(message)

    def _send_interaction_acknowledgement(
        self, request: InteractionRequest, user_id: str
    ) -> bool:
        """
        Send an interaction acknowledgement message.

        Args:
            request: Interaction request that was acknowledged
            user_id: ID of the user acknowledging

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        content = {
            "request_id": request.request_id,
            "user_id": user_id,
            "status": InteractionStatus.ACKNOWLEDGED.value,
        }

        message = MessageProtocol.create_message(
            message_type=MessageType.NOTIFICATION,
            sender_id="interaction_manager",
            content=content,
            recipient_id=request.agent_id,
            priority=request.priority,
        )

        return self.agent_communication.send_message(message)

    def _handle_interaction_response(self, message: Message) -> None:
        """
        Handle an interaction response message.

        Args:
            message: Interaction response message
        """
        try:
            content = message.content
            if "response" not in content:
                logger.warning("Received interaction response without response data")
                return

            response_data = content["response"]
            if (
                "request_id" not in response_data
                or "user_id" not in response_data
                or "response" not in response_data
            ):
                logger.warning("Received interaction response with missing fields")
                return

            # Create response object
            response = InteractionResponse(
                request_id=response_data["request_id"],
                user_id=response_data["user_id"],
                response=response_data["response"],
            )

            # Process response
            self._process_interaction_response(response)

        except Exception as e:
            logger.error(f"Error handling interaction response: {e}")

    def _process_interaction_response(self, response: InteractionResponse) -> None:
        """
        Process an interaction response.

        Args:
            response: Interaction response to process
        """
        request_id = response.request_id

        if request_id not in self.requests:
            logger.warning(f"Received response for unknown request: {request_id}")
            return

        request = self.requests[request_id]

        # Update request
        request.status = InteractionStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        request.response = response.response

        # Call specific callback if registered
        if request_id in self.callbacks:
            try:
                self.callbacks[request_id](response)
            except Exception as e:
                logger.error(
                    f"Error in interaction callback for request {request_id}: {e}"
                )
            finally:
                # Remove callback after it's called
                del self.callbacks[request_id]

        # Call default callbacks for this interaction type
        for callback in self.default_callbacks[request.interaction_type]:
            try:
                callback(response)
            except Exception as e:
                logger.error(
                    f"Error in default callback for interaction type {request.interaction_type.value}: {e}"
                )

    def _cleanup_expired_requests(self) -> None:
        """Clean up expired interaction requests."""
        while self.running:
            try:
                # Find expired requests
                expired_ids = []
                for request_id, request in self.requests.items():
                    if (
                        request.status == InteractionStatus.PENDING
                        and request.expires_at
                        and datetime.utcnow() > request.expires_at
                    ):
                        expired_ids.append(request_id)

                # Update expired requests
                for request_id in expired_ids:
                    request = self.requests[request_id]
                    request.status = InteractionStatus.EXPIRED

                    # Call callback with None response to indicate expiry
                    if request_id in self.callbacks:
                        try:
                            self.callbacks[request_id](
                                InteractionResponse(
                                    request_id=request_id,
                                    user_id="system",
                                    response={"expired": True},
                                )
                            )
                        except Exception as e:
                            logger.error(
                                f"Error in interaction callback for expired request {request_id}: {e}"
                            )
                        finally:
                            # Remove callback after it's called
                            del self.callbacks[request_id]

                # Sleep for a while
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error in interaction request cleanup: {e}")
                time.sleep(5)  # Sleep longer on error

    def _process_notifications(self) -> None:
        """Process pending interaction notifications."""
        while self.running:
            try:
                # TODO: Implement notification processing
                # (e.g., send reminders for pending interactions)
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in notification processing: {e}")
                time.sleep(60)  # Sleep longer on error

    def _process_input(self) -> None:
        """Process incoming interaction requests."""
        while self.running:
            try:
                # TODO: Implement input processing logic
                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in input processing: {e}")
                time.sleep(5)  # Sleep longer on error

    def _process_output(self) -> None:
        """Process outgoing interaction responses."""
        while self.running:
            try:
                # TODO: Implement output processing logic
                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in output processing: {e}")
                time.sleep(5)  # Sleep longer on error
