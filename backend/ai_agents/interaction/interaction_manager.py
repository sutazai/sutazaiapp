#!/usr/bin/env python3
"""
SutazAI Interaction Manager
Manages agent interactions and coordination
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class InteractionManager:
    """Manages agent interactions and coordination"""
    
    def __init__(self):
        self.active_interactions = {}
        self.interaction_history = []
        self.interaction_patterns = {}
        self.initialized = False
    
    def initialize(self) -> None:
        """Initialize interaction manager"""
        self.initialized = True
        logger.info("Interaction manager initialized")
    
    def start_interaction(self, agents: List[str], interaction_type: str = "collaboration") -> str:
        """Start new interaction between agents"""
        interaction_id = f"interaction_{datetime.now().timestamp()}"
        
        self.active_interactions[interaction_id] = {
            "id": interaction_id,
            "agents": agents,
            "type": interaction_type,
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "events": []
        }
        
        return interaction_id
    
    def log_interaction_event(self, interaction_id: str, event: Dict[str, Any]) -> bool:
        """Log event within an interaction"""
        try:
            if interaction_id in self.active_interactions:
                event["timestamp"] = datetime.now().isoformat()
                self.active_interactions[interaction_id]["events"].append(event)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to log interaction event: {e}")
            return False
    
    def end_interaction(self, interaction_id: str, result: Dict[str, Any] = None) -> bool:
        """End an active interaction"""
        try:
            if interaction_id in self.active_interactions:
                interaction = self.active_interactions[interaction_id]
                interaction["status"] = "completed"
                interaction["ended_at"] = datetime.now().isoformat()
                interaction["result"] = result or {}
                
                # Move to history
                self.interaction_history.append(interaction)
                del self.active_interactions[interaction_id]
                
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to end interaction: {e}")
            return False
    
    def get_active_interactions(self) -> List[Dict[str, Any]]:
        """Get all active interactions"""
        return list(self.active_interactions.values())
    
    def get_interaction_history(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get interaction history"""
        if agent_id:
            return [interaction for interaction in self.interaction_history
                   if agent_id in interaction.get("agents", [])]
        return self.interaction_history
    
    def cleanup(self) -> None:
        """Cleanup interaction manager"""
        self.active_interactions.clear()
        self.interaction_history.clear()
        self.interaction_patterns.clear()
        self.initialized = False