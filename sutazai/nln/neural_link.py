"""
Neural Link Implementation
Represents connections between neural nodes in the NLN system
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class LinkType(str, Enum):
    EXCITATORY = "excitatory"      # Positive influence
    INHIBITORY = "inhibitory"      # Negative influence
    MODULATORY = "modulatory"      # Conditional influence
    ASSOCIATIVE = "associative"    # Memory association
    CAUSAL = "causal"             # Cause-effect relationship
    TEMPORAL = "temporal"         # Time-based relationship

class LinkStrength(str, Enum):
    WEAK = "weak"         # 0.0 - 0.3
    MODERATE = "moderate" # 0.3 - 0.7
    STRONG = "strong"     # 0.7 - 1.0

@dataclass
class LinkDynamics:
    """Dynamic properties of a neural link"""
    plasticity: float = 0.1      # How much the link can change
    adaptation_rate: float = 0.01 # Speed of adaptation
    saturation_point: float = 1.0 # Maximum strength
    decay_rate: float = 0.001    # Natural decay over time
    last_used: float = 0.0       # Timestamp of last activation

class NeuralLink:
    """
    Neural Link connecting two nodes in the NLN
    Manages weight, direction, and learning dynamics
    """
    
    def __init__(
        self,
        link_id: str = None,
        source_node_id: str = "",
        target_node_id: str = "",
        weight: float = 0.5,
        link_type: LinkType = LinkType.EXCITATORY,
        bidirectional: bool = False,
        learning_enabled: bool = True
    ):
        self.id = link_id or str(uuid.uuid4())
        self.source_node_id = source_node_id
        self.target_node_id = target_node_id
        self.weight = max(-2.0, min(2.0, weight))  # Clip to reasonable range
        self.link_type = link_type
        self.bidirectional = bidirectional
        self.learning_enabled = learning_enabled
        
        # Dynamic properties
        self.dynamics = LinkDynamics()
        
        # Usage tracking
        self.activation_count = 0
        self.signal_history = []  # Recent signal transmissions
        self.weight_history = []  # Weight change history
        
        # Metadata
        self.metadata = {
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
            "description": ""
        }
        
        # Initialize based on link type
        self._initialize_link_properties()
    
    def _initialize_link_properties(self):
        """Initialize properties based on link type"""
        if self.link_type == LinkType.EXCITATORY:
            self.weight = abs(self.weight)  # Ensure positive
            self.dynamics.plasticity = 0.15
            
        elif self.link_type == LinkType.INHIBITORY:
            self.weight = -abs(self.weight)  # Ensure negative
            self.dynamics.plasticity = 0.1
            
        elif self.link_type == LinkType.MODULATORY:
            self.dynamics.plasticity = 0.2
            self.dynamics.adaptation_rate = 0.02
            
        elif self.link_type == LinkType.ASSOCIATIVE:
            self.dynamics.plasticity = 0.25
            self.dynamics.decay_rate = 0.0005  # Slower decay for memory
            
        elif self.link_type == LinkType.CAUSAL:
            self.dynamics.plasticity = 0.05  # More stable
            self.weight = abs(self.weight)
            
        elif self.link_type == LinkType.TEMPORAL:
            self.dynamics.adaptation_rate = 0.05  # Faster adaptation
            self.dynamics.decay_rate = 0.002  # Faster decay
    
    def transmit_signal(
        self, 
        signal: float, 
        source_activation: float = 1.0,
        context: Dict[str, Any] = None
    ) -> float:
        """Transmit signal through the link"""
        try:
            # Apply link weight
            transmitted_signal = signal * self.weight
            
            # Apply link type modulation
            transmitted_signal = self._apply_link_modulation(
                transmitted_signal, 
                source_activation, 
                context or {}
            )
            
            # Update usage tracking
            self.activation_count += 1
            self.dynamics.last_used = time.time()
            
            # Store signal in history
            self.signal_history.append({
                "timestamp": time.time(),
                "input_signal": signal,
                "output_signal": transmitted_signal,
                "source_activation": source_activation,
                "weight": self.weight
            })
            
            # Keep only recent history
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
            
            # Apply learning if enabled
            if self.learning_enabled:
                self._update_weight(signal, transmitted_signal, source_activation)
            
            return transmitted_signal
            
        except Exception as e:
            print(f"Signal transmission failed: {e}")
            return 0.0
    
    def _apply_link_modulation(
        self, 
        signal: float, 
        source_activation: float, 
        context: Dict[str, Any]
    ) -> float:
        """Apply link-type specific modulation"""
        
        if self.link_type == LinkType.EXCITATORY:
            # Amplify positive signals
            return signal * (1 + source_activation * 0.1)
            
        elif self.link_type == LinkType.INHIBITORY:
            # Suppress based on source activation
            return signal * (1 - source_activation * 0.2)
            
        elif self.link_type == LinkType.MODULATORY:
            # Conditional transmission based on context
            modulation_factor = context.get("modulation", 1.0)
            return signal * modulation_factor
            
        elif self.link_type == LinkType.ASSOCIATIVE:
            # Strengthen with repeated activation
            frequency_bonus = min(0.5, self.activation_count * 0.001)
            return signal * (1 + frequency_bonus)
            
        elif self.link_type == LinkType.CAUSAL:
            # Reliable transmission with minimal modulation
            return signal * 0.95  # Slight signal loss
            
        elif self.link_type == LinkType.TEMPORAL:
            # Time-dependent transmission
            time_factor = context.get("time_factor", 1.0)
            return signal * time_factor
        
        return signal
    
    def _update_weight(self, input_signal: float, output_signal: float, source_activation: float):
        """Update link weight based on Hebbian learning and other rules"""
        if not self.learning_enabled:
            return
        
        try:
            old_weight = self.weight
            
            # Hebbian learning: strengthen when both nodes are active
            hebbian_update = (
                self.dynamics.adaptation_rate * 
                input_signal * 
                source_activation * 
                self.dynamics.plasticity
            )
            
            # Apply weight update
            new_weight = old_weight + hebbian_update
            
            # Apply link type constraints
            if self.link_type == LinkType.EXCITATORY:
                new_weight = max(0, new_weight)
            elif self.link_type == LinkType.INHIBITORY:
                new_weight = min(0, new_weight)
            
            # Clip to saturation point
            new_weight = max(-self.dynamics.saturation_point, 
                           min(self.dynamics.saturation_point, new_weight))
            
            # Apply natural decay
            if abs(new_weight) > 0.01:
                decay = self.dynamics.decay_rate * abs(new_weight)
                if new_weight > 0:
                    new_weight -= decay
                else:
                    new_weight += decay
            
            self.weight = new_weight
            
            # Record weight change
            self.weight_history.append({
                "timestamp": time.time(),
                "old_weight": old_weight,
                "new_weight": new_weight,
                "hebbian_update": hebbian_update,
                "input_signal": input_signal,
                "source_activation": source_activation
            })
            
            # Keep only recent history
            if len(self.weight_history) > 50:
                self.weight_history.pop(0)
            
            self.metadata["updated_at"] = time.time()
            
        except Exception as e:
            print(f"Weight update failed: {e}")
    
    def strengthen(self, amount: float = 0.1):
        """Manually strengthen the link"""
        if self.link_type == LinkType.INHIBITORY:
            self.weight = max(-self.dynamics.saturation_point, self.weight - amount)
        else:
            self.weight = min(self.dynamics.saturation_point, self.weight + amount)
        
        self.metadata["updated_at"] = time.time()
    
    def weaken(self, amount: float = 0.1):
        """Manually weaken the link"""
        if self.link_type == LinkType.INHIBITORY:
            self.weight = min(0, self.weight + amount)
        else:
            self.weight = max(0, self.weight - amount)
        
        self.metadata["updated_at"] = time.time()
    
    def get_strength_category(self) -> LinkStrength:
        """Get categorical strength of the link"""
        abs_weight = abs(self.weight)
        if abs_weight < 0.3:
            return LinkStrength.WEAK
        elif abs_weight < 0.7:
            return LinkStrength.MODERATE
        else:
            return LinkStrength.STRONG
    
    def is_active(self, threshold: float = 0.1) -> bool:
        """Check if link is considered active"""
        return abs(self.weight) >= threshold
    
    def get_signal_efficiency(self) -> float:
        """Calculate signal transmission efficiency"""
        if not self.signal_history:
            return 0.0
        
        # Calculate average signal preservation
        efficiencies = []
        for record in self.signal_history[-20:]:  # Last 20 transmissions
            input_sig = abs(record["input_signal"])
            output_sig = abs(record["output_signal"])
            if input_sig > 0:
                efficiency = output_sig / input_sig
                efficiencies.append(min(1.0, efficiency))
        
        return sum(efficiencies) / max(len(efficiencies), 1)
    
    def get_usage_frequency(self, time_window: float = 3600) -> float:
        """Get usage frequency within time window (default 1 hour)"""
        current_time = time.time()
        recent_uses = [
            record for record in self.signal_history
            if current_time - record["timestamp"] <= time_window
        ]
        
        return len(recent_uses) / (time_window / 60)  # Uses per minute
    
    def reset_dynamics(self):
        """Reset dynamic properties to default"""
        self.dynamics = LinkDynamics()
        self._initialize_link_properties()
        self.metadata["updated_at"] = time.time()
    
    def get_link_info(self) -> Dict[str, Any]:
        """Get comprehensive link information"""
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "weight": self.weight,
            "link_type": self.link_type.value,
            "strength_category": self.get_strength_category().value,
            "bidirectional": self.bidirectional,
            "learning_enabled": self.learning_enabled,
            "activation_count": self.activation_count,
            "signal_efficiency": self.get_signal_efficiency(),
            "usage_frequency": self.get_usage_frequency(),
            "dynamics": asdict(self.dynamics),
            "metadata": self.metadata
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get link statistics"""
        recent_signals = [
            record for record in self.signal_history
            if time.time() - record["timestamp"] < 3600  # Last hour
        ]
        
        return {
            "total_activations": self.activation_count,
            "recent_activations": len(recent_signals),
            "average_signal_strength": (
                sum(abs(record["output_signal"]) for record in recent_signals) /
                max(len(recent_signals), 1)
            ),
            "weight_stability": self._calculate_weight_stability(),
            "signal_efficiency": self.get_signal_efficiency(),
            "last_used": self.dynamics.last_used,
            "age": time.time() - self.metadata["created_at"]
        }
    
    def _calculate_weight_stability(self) -> float:
        """Calculate how stable the weight has been"""
        if len(self.weight_history) < 2:
            return 1.0
        
        # Calculate variance in recent weight changes
        recent_weights = [record["new_weight"] for record in self.weight_history[-10:]]
        if len(recent_weights) < 2:
            return 1.0
        
        variance = np.var(recent_weights)
        stability = 1.0 / (1.0 + variance)  # Higher variance = lower stability
        return stability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert link to dictionary for serialization"""
        return {
            "id": self.id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "weight": self.weight,
            "link_type": self.link_type.value,
            "bidirectional": self.bidirectional,
            "learning_enabled": self.learning_enabled,
            "dynamics": asdict(self.dynamics),
            "activation_count": self.activation_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralLink':
        """Create link from dictionary"""
        link = cls(
            link_id=data["id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            weight=data["weight"],
            link_type=LinkType(data["link_type"]),
            bidirectional=data["bidirectional"],
            learning_enabled=data["learning_enabled"]
        )
        
        # Restore state
        link.dynamics = LinkDynamics(**data["dynamics"])
        link.activation_count = data["activation_count"]
        link.metadata = data["metadata"]
        
        return link
    
    def __str__(self) -> str:
        direction = "<->" if self.bidirectional else "->"
        return f"NeuralLink({self.source_node_id} {direction} {self.target_node_id}, weight={self.weight:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()