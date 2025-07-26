"""
Human Interaction Management
Handles requests for human input and decision-making
"""

import logging
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of human interactions"""
    APPROVAL = "approval"
    DECISION = "decision"
    INFORMATION = "information"
    INPUT = "input"
    ESCALATION = "escalation"
    CONFIRMATION = "confirmation"
    REVIEW = "review"

@dataclass
class InteractionResponse:
    """Response from human interaction"""
    interaction_id: str
    response_type: str
    response_data: Dict[str, Any]
    responded_by: Optional[str]
    responded_at: datetime
    
@dataclass
class HumanInteractionPoint:
    """Defines a point where human interaction is needed"""
    interaction_type: InteractionType
    title: str
    description: str
    agent_id: str
    data: Dict[str, Any]
    options: List[Dict[str, Any]]
    timeout: Optional[int]  # seconds
    priority: int = 5  # 1-10, higher is more urgent
    
    def __post_init__(self):
        if isinstance(self.interaction_type, str):
            self.interaction_type = InteractionType(self.interaction_type)

class PendingInteraction:
    """A pending human interaction request"""
    
    def __init__(self, interaction_point: HumanInteractionPoint, callback: Optional[Callable] = None):
        self.id = str(uuid.uuid4())
        self.interaction_point = interaction_point
        self.callback = callback
        self.status = "pending"
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=interaction_point.timeout) if interaction_point.timeout else None
        self.response: Optional[InteractionResponse] = None
        
    def is_expired(self) -> bool:
        """Check if interaction has expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at
        
    def respond(self, response_data: Dict[str, Any], responded_by: Optional[str] = None) -> InteractionResponse:
        """Respond to the interaction"""
        if self.status != "pending":
            raise ValueError(f"Interaction {self.id} is not pending")
            
        if self.is_expired():
            raise ValueError(f"Interaction {self.id} has expired")
            
        self.response = InteractionResponse(
            interaction_id=self.id,
            response_type=self.interaction_point.interaction_type.value,
            response_data=response_data,
            responded_by=responded_by,
            responded_at=datetime.now()
        )
        
        self.status = "completed"
        
        # Call callback if provided
        if self.callback:
            try:
                self.callback(self.response)
            except Exception as e:
                logger.error(f"Error in interaction callback for {self.id}: {e}")
                
        logger.info(f"Interaction {self.id} completed by {responded_by}")
        return self.response

class InteractionManager:
    """Manages human interaction requests and responses"""
    
    def __init__(self):
        self.pending_interactions: Dict[str, PendingInteraction] = {}
        self.completed_interactions: Dict[str, PendingInteraction] = {}
        self.running = False
        
    def start(self):
        """Start the interaction manager"""
        self.running = True
        logger.info("Human interaction manager started")
        
    def stop(self):
        """Stop the interaction manager"""
        self.running = False
        
        # Expire all pending interactions
        for interaction in list(self.pending_interactions.values()):
            self._expire_interaction(interaction)
            
        logger.info("Human interaction manager stopped")
        
    def create_interaction(self, interaction_point: HumanInteractionPoint, 
                         callback: Optional[Callable] = None) -> str:
        """Create a new human interaction request"""
        if not self.running:
            raise RuntimeError("Interaction manager is not running")
            
        interaction = PendingInteraction(interaction_point, callback)
        self.pending_interactions[interaction.id] = interaction
        
        logger.info(f"Created {interaction_point.interaction_type.value} interaction {interaction.id} for agent {interaction_point.agent_id}")
        return interaction.id
        
    def respond_to_interaction(self, interaction_id: str, response_data: Dict[str, Any], 
                             responded_by: Optional[str] = None) -> bool:
        """Respond to a pending interaction"""
        if interaction_id not in self.pending_interactions:
            logger.warning(f"Interaction {interaction_id} not found")
            return False
            
        interaction = self.pending_interactions[interaction_id]
        
        try:
            interaction.respond(response_data, responded_by)
            
            # Move to completed
            self.completed_interactions[interaction_id] = interaction
            del self.pending_interactions[interaction_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Error responding to interaction {interaction_id}: {e}")
            return False
            
    def get_pending_interactions(self, agent_id: Optional[str] = None, 
                               interaction_type: Optional[InteractionType] = None) -> List[PendingInteraction]:
        """Get pending interactions, optionally filtered"""
        interactions = list(self.pending_interactions.values())
        
        # Clean up expired interactions first
        for interaction in interactions[:]:
            if interaction.is_expired():
                self._expire_interaction(interaction)
                interactions.remove(interaction)
                
        # Apply filters
        if agent_id:
            interactions = [i for i in interactions if i.interaction_point.agent_id == agent_id]
            
        if interaction_type:
            interactions = [i for i in interactions if i.interaction_point.interaction_type == interaction_type]
            
        # Sort by priority and creation time
        interactions.sort(key=lambda i: (i.interaction_point.priority, i.created_at), reverse=True)
        
        return interactions
        
    def get_interaction(self, interaction_id: str) -> Optional[PendingInteraction]:
        """Get specific interaction by ID"""
        # Check pending first
        if interaction_id in self.pending_interactions:
            interaction = self.pending_interactions[interaction_id]
            if interaction.is_expired():
                self._expire_interaction(interaction)
                return None
            return interaction
            
        # Check completed
        return self.completed_interactions.get(interaction_id)
        
    def cancel_interaction(self, interaction_id: str, reason: str = "Cancelled") -> bool:
        """Cancel a pending interaction"""
        if interaction_id not in self.pending_interactions:
            return False
            
        interaction = self.pending_interactions[interaction_id]
        interaction.status = "cancelled"
        
        # Move to completed with cancellation response
        interaction.response = InteractionResponse(
            interaction_id=interaction_id,
            response_type="cancelled",
            response_data={"reason": reason},
            responded_by="system",
            responded_at=datetime.now()
        )
        
        self.completed_interactions[interaction_id] = interaction
        del self.pending_interactions[interaction_id]
        
        logger.info(f"Cancelled interaction {interaction_id}: {reason}")
        return True
        
    def get_interaction_stats(self) -> Dict[str, Any]:
        """Get interaction statistics"""
        pending_by_type = {}
        for interaction in self.pending_interactions.values():
            itype = interaction.interaction_point.interaction_type.value
            pending_by_type[itype] = pending_by_type.get(itype, 0) + 1
            
        completed_by_type = {}
        for interaction in self.completed_interactions.values():
            itype = interaction.interaction_point.interaction_type.value
            completed_by_type[itype] = completed_by_type.get(itype, 0) + 1
            
        return {
            "total_pending": len(self.pending_interactions),
            "total_completed": len(self.completed_interactions),
            "pending_by_type": pending_by_type,
            "completed_by_type": completed_by_type,
            "running": self.running
        }
        
    def _expire_interaction(self, interaction: PendingInteraction):
        """Handle expired interaction"""
        interaction.status = "expired"
        interaction.response = InteractionResponse(
            interaction_id=interaction.id,
            response_type="expired",
            response_data={"reason": "Interaction timed out"},
            responded_by="system",
            responded_at=datetime.now()
        )
        
        # Call callback if provided
        if interaction.callback:
            try:
                interaction.callback(interaction.response)
            except Exception as e:
                logger.error(f"Error in expiration callback for {interaction.id}: {e}")
                
        # Move to completed
        self.completed_interactions[interaction.id] = interaction
        if interaction.id in self.pending_interactions:
            del self.pending_interactions[interaction.id]
            
        logger.warning(f"Interaction {interaction.id} expired")