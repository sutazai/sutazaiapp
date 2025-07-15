"""
Neural Node Implementation
Advanced neural node with realistic biological modeling
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NeuralNode:
    """Neural node with biological characteristics"""
    
    node_id: str
    node_type: str  # "input", "processing", "output", "memory"
    position: tuple
    threshold: float
    activity: float = 0.0
    learning_rate: float = 0.01
    adaptation_rate: float = 0.001
    refractory_period: float = 0.001
    last_firing_time: float = 0.0
    
    def __post_init__(self):
        self.created_at = datetime.now()
        self.firing_history = []
        self.input_connections = []
        self.output_connections = []
    
    def process_input(self, input_value: float) -> float:
        """Process input and determine output"""
        current_time = time.time()
        
        # Check refractory period
        if current_time - self.last_firing_time < self.refractory_period:
            return 0.0
        
        # Update activity
        self.activity = max(0.0, self.activity + input_value)
        
        # Check if threshold is exceeded
        if self.activity >= self.threshold:
            self.last_firing_time = current_time
            self.firing_history.append(current_time)
            
            # Reset activity after firing
            output = self.activity
            self.activity = 0.0
            
            return output
        
        return 0.0
    
    def adapt_threshold(self, target_activity: float):
        """Adapt threshold based on desired activity level"""
        if self.activity > target_activity:
            self.threshold += self.adaptation_rate
        elif self.activity < target_activity:
            self.threshold -= self.adaptation_rate
        
        # Keep threshold within bounds
        self.threshold = max(0.1, min(2.0, self.threshold))
    
    def get_firing_rate(self, time_window: float = 1.0) -> float:
        """Calculate firing rate over time window"""
        current_time = time.time()
        recent_firings = [t for t in self.firing_history 
                         if current_time - t <= time_window]
        return len(recent_firings) / time_window
    
    def get_state(self) -> Dict[str, Any]:
        """Get current node state"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "position": self.position,
            "threshold": self.threshold,
            "activity": self.activity,
            "firing_rate": self.get_firing_rate(),
            "last_firing_time": self.last_firing_time
        }