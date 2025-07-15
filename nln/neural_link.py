"""
Neural Link Implementation
Connections between neural nodes with adaptive weights
"""

import time
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NeuralLink:
    """Neural link connecting two nodes"""
    
    source_id: str
    target_id: str
    weight: float
    link_type: str  # "excitatory", "inhibitory", "modulatory"
    learning_rate: float = 0.01
    decay_rate: float = 0.001
    
    def __post_init__(self):
        self.created_at = datetime.now()
        self.last_update = time.time()
        self.activation_history = []
        self.weight_history = []
    
    def transmit(self, input_signal: float) -> float:
        """Transmit signal through the link"""
        current_time = time.time()
        
        # Calculate output based on link type
        if self.link_type == "excitatory":
            output = input_signal * self.weight
        elif self.link_type == "inhibitory":
            output = -input_signal * self.weight
        elif self.link_type == "modulatory":
            output = input_signal * self.weight * 0.5  # Modulated transmission
        else:
            output = input_signal * self.weight
        
        # Record activation
        self.activation_history.append({
            "time": current_time,
            "input": input_signal,
            "output": output,
            "weight": self.weight
        })
        
        # Apply weight decay
        self.apply_decay(current_time)
        
        return output
    
    def update_weight(self, pre_activity: float, post_activity: float):
        """Update weight based on Hebbian learning"""
        current_time = time.time()
        
        # Hebbian learning: weights increase when both pre and post are active
        if pre_activity > 0 and post_activity > 0:
            weight_change = self.learning_rate * pre_activity * post_activity
        else:
            weight_change = -self.learning_rate * 0.1  # Small decrease for inactive connections
        
        self.weight += weight_change
        
        # Keep weight within bounds
        self.weight = max(-2.0, min(2.0, self.weight))
        
        # Record weight change
        self.weight_history.append({
            "time": current_time,
            "weight": self.weight,
            "change": weight_change
        })
        
        self.last_update = current_time
    
    def apply_decay(self, current_time: float):
        """Apply weight decay over time"""
        time_since_update = current_time - self.last_update
        decay_factor = 1.0 - (self.decay_rate * time_since_update)
        self.weight *= decay_factor
        self.last_update = current_time
    
    def get_state(self) -> Dict[str, Any]:
        """Get current link state"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "link_type": self.link_type,
            "last_update": self.last_update,
            "activation_count": len(self.activation_history)
        }