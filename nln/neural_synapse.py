"""
Neural Synapse Implementation
Detailed synaptic modeling with neurotransmitter simulation
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class NeuralSynapse:
    """Neural synapse with neurotransmitter modeling"""
    
    pre_synaptic_id: str
    post_synaptic_id: str
    neurotransmitter: str  # "glutamate", "GABA", "dopamine", "serotonin"
    strength: float
    plasticity_type: str = "hebbian"  # "hebbian", "spike_timing", "homeostatic"
    
    def __post_init__(self):
        self.created_at = datetime.now()
        self.vesicle_count = 100
        self.release_probability = 0.5
        self.receptor_density = 1.0
        self.last_release_time = 0.0
        self.release_history = []
        
        # Neurotransmitter-specific properties
        self.nt_properties = {
            "glutamate": {"excitatory": True, "decay_rate": 0.01},
            "GABA": {"excitatory": False, "decay_rate": 0.02},
            "dopamine": {"excitatory": True, "decay_rate": 0.005},
            "serotonin": {"excitatory": True, "decay_rate": 0.008}
        }
    
    def release_neurotransmitter(self, pre_activity: float) -> float:
        """Release neurotransmitter based on pre-synaptic activity"""
        current_time = time.time()
        
        # Determine if release occurs
        if pre_activity > 0 and self.vesicle_count > 0:
            release_amount = pre_activity * self.release_probability * self.strength
            
            # Update vesicle count
            self.vesicle_count = max(0, self.vesicle_count - 1)
            
            # Record release
            self.release_history.append({
                "time": current_time,
                "amount": release_amount,
                "vesicles_remaining": self.vesicle_count
            })
            
            self.last_release_time = current_time
            
            return release_amount
        
        return 0.0
    
    def bind_to_receptors(self, nt_amount: float) -> float:
        """Bind neurotransmitter to post-synaptic receptors"""
        nt_props = self.nt_properties.get(self.neurotransmitter, 
                                         {"excitatory": True, "decay_rate": 0.01})
        
        # Calculate binding efficiency
        binding_efficiency = self.receptor_density * 0.8
        bound_amount = nt_amount * binding_efficiency
        
        # Apply excitatory/inhibitory effect
        if nt_props["excitatory"]:
            return bound_amount
        else:
            return -bound_amount
    
    def update_plasticity(self, pre_activity: float, post_activity: float):
        """Update synaptic plasticity"""
        current_time = time.time()
        
        if self.plasticity_type == "hebbian":
            # Hebbian plasticity: strengthen when both pre and post are active
            if pre_activity > 0 and post_activity > 0:
                self.strength += 0.01 * pre_activity * post_activity
            else:
                self.strength -= 0.001  # Gradual weakening
        
        elif self.plasticity_type == "spike_timing":
            # Spike-timing dependent plasticity (simplified)
            time_diff = current_time - self.last_release_time
            if 0 < time_diff < 0.02:  # 20ms window
                self.strength += 0.01  # LTP
            elif 0.02 < time_diff < 0.05:  # 50ms window
                self.strength -= 0.005  # LTD
        
        elif self.plasticity_type == "homeostatic":
            # Homeostatic scaling
            target_strength = 0.5
            if self.strength > target_strength:
                self.strength -= 0.001
            elif self.strength < target_strength:
                self.strength += 0.001
        
        # Keep strength within bounds
        self.strength = max(0.0, min(2.0, self.strength))
    
    def replenish_vesicles(self):
        """Replenish neurotransmitter vesicles"""
        replenishment_rate = 10  # vesicles per second
        current_time = time.time()
        
        if self.vesicle_count < 100:
            time_since_release = current_time - self.last_release_time
            new_vesicles = int(replenishment_rate * time_since_release)
            self.vesicle_count = min(100, self.vesicle_count + new_vesicles)
    
    def process_transmission(self, pre_activity: float, post_activity: float) -> float:
        """Process complete synaptic transmission"""
        # Replenish vesicles
        self.replenish_vesicles()
        
        # Release neurotransmitter
        nt_amount = self.release_neurotransmitter(pre_activity)
        
        # Bind to receptors
        post_synaptic_current = self.bind_to_receptors(nt_amount)
        
        # Update plasticity
        self.update_plasticity(pre_activity, post_activity)
        
        return post_synaptic_current
    
    def get_state(self) -> Dict[str, Any]:
        """Get current synapse state"""
        return {
            "pre_synaptic_id": self.pre_synaptic_id,
            "post_synaptic_id": self.post_synaptic_id,
            "neurotransmitter": self.neurotransmitter,
            "strength": self.strength,
            "vesicle_count": self.vesicle_count,
            "release_probability": self.release_probability,
            "receptor_density": self.receptor_density,
            "release_count": len(self.release_history)
        }