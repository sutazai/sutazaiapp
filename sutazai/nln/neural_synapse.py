"""
Neural Synapse Implementation
Advanced synaptic connections with neurotransmitter simulation and plasticity
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class NeurotransmitterType(str, Enum):
    DOPAMINE = "dopamine"         # Reward/motivation
    SEROTONIN = "serotonin"       # Mood/learning
    ACETYLCHOLINE = "acetylcholine" # Attention/learning
    GABA = "gaba"                 # Inhibitory
    GLUTAMATE = "glutamate"       # Excitatory
    NOREPINEPHRINE = "norepinephrine" # Arousal/attention

class SynapseType(str, Enum):
    CHEMICAL = "chemical"         # Standard synaptic transmission
    ELECTRICAL = "electrical"    # Direct electrical coupling
    MIXED = "mixed"              # Both chemical and electrical

class PlasticityType(str, Enum):
    LTP = "ltp"                  # Long-term potentiation
    LTD = "ltd"                  # Long-term depression
    HOMEOSTATIC = "homeostatic"  # Homeostatic scaling
    METAPLASTIC = "metaplastic"  # Plasticity of plasticity

@dataclass
class NeurotransmitterPool:
    """Simulates neurotransmitter availability and dynamics"""
    neurotransmitter: NeurotransmitterType
    concentration: float = 1.0
    release_probability: float = 0.5
    reuptake_rate: float = 0.1
    synthesis_rate: float = 0.05
    vesicle_count: int = 100

@dataclass
class SynapticPlasticity:
    """Manages synaptic plasticity mechanisms"""
    plasticity_type: PlasticityType
    induction_threshold: float = 0.5
    expression_magnitude: float = 0.1
    decay_time_constant: float = 3600  # seconds
    saturation_level: float = 2.0

class NeuralSynapse:
    """
    Advanced Neural Synapse with neurotransmitter simulation and plasticity
    Provides sophisticated synaptic transmission and learning
    """
    
    def __init__(
        self,
        synapse_id: str = None,
        presynaptic_node_id: str = "",
        postsynaptic_node_id: str = "",
        synapse_type: SynapseType = SynapseType.CHEMICAL,
        base_strength: float = 1.0,
        delay: float = 0.001,  # seconds
        learning_enabled: bool = True
    ):
        self.id = synapse_id or str(uuid.uuid4())
        self.presynaptic_node_id = presynaptic_node_id
        self.postsynaptic_node_id = postsynaptic_node_id
        self.synapse_type = synapse_type
        self.base_strength = base_strength
        self.delay = delay
        self.learning_enabled = learning_enabled
        
        # Current synaptic strength (modifiable by plasticity)
        self.current_strength = base_strength
        
        # Neurotransmitter systems
        self.neurotransmitter_pools = {}
        self._initialize_neurotransmitters()
        
        # Plasticity mechanisms
        self.plasticity_mechanisms = {}
        self._initialize_plasticity()
        
        # Activity tracking
        self.transmission_history = []
        self.plasticity_events = []
        self.spike_timing = []  # For STDP
        
        # State variables
        self.last_presynaptic_spike = 0.0
        self.last_postsynaptic_spike = 0.0
        self.resource_depletion = 1.0  # 1.0 = no depletion
        
        # Metadata
        self.metadata = {
            "created_at": time.time(),
            "updated_at": time.time(),
            "total_transmissions": 0,
            "total_plasticity_events": 0
        }
    
    def _initialize_neurotransmitters(self):
        """Initialize neurotransmitter pools based on synapse type"""
        if self.synapse_type in [SynapseType.CHEMICAL, SynapseType.MIXED]:
            # Default excitatory neurotransmitter
            self.neurotransmitter_pools[NeurotransmitterType.GLUTAMATE] = NeurotransmitterPool(
                neurotransmitter=NeurotransmitterType.GLUTAMATE,
                concentration=1.0,
                release_probability=0.6
            )
            
            # Add dopamine for reward-based learning
            self.neurotransmitter_pools[NeurotransmitterType.DOPAMINE] = NeurotransmitterPool(
                neurotransmitter=NeurotransmitterType.DOPAMINE,
                concentration=0.5,
                release_probability=0.3,
                synthesis_rate=0.02
            )
    
    def _initialize_plasticity(self):
        """Initialize plasticity mechanisms"""
        if self.learning_enabled:
            # Long-term potentiation
            self.plasticity_mechanisms[PlasticityType.LTP] = SynapticPlasticity(
                plasticity_type=PlasticityType.LTP,
                induction_threshold=0.6,
                expression_magnitude=0.15,
                decay_time_constant=7200  # 2 hours
            )
            
            # Long-term depression
            self.plasticity_mechanisms[PlasticityType.LTD] = SynapticPlasticity(
                plasticity_type=PlasticityType.LTD,
                induction_threshold=0.3,
                expression_magnitude=-0.1,
                decay_time_constant=3600  # 1 hour
            )
            
            # Homeostatic plasticity
            self.plasticity_mechanisms[PlasticityType.HOMEOSTATIC] = SynapticPlasticity(
                plasticity_type=PlasticityType.HOMEOSTATIC,
                induction_threshold=0.1,
                expression_magnitude=0.05,
                decay_time_constant=86400  # 24 hours
            )
    
    def transmit(
        self,
        presynaptic_signal: float,
        presynaptic_spike_time: float = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Transmit signal across synapse with realistic synaptic dynamics"""
        try:
            current_time = time.time()
            spike_time = presynaptic_spike_time or current_time
            context = context or {}
            
            # Record spike timing
            self.last_presynaptic_spike = spike_time
            self.spike_timing.append({
                "type": "presynaptic",
                "time": spike_time,
                "amplitude": presynaptic_signal
            })
            
            # Calculate transmission based on synapse type
            if self.synapse_type == SynapseType.ELECTRICAL:
                output_signal = self._electrical_transmission(presynaptic_signal)
            elif self.synapse_type == SynapseType.CHEMICAL:
                output_signal = self._chemical_transmission(presynaptic_signal, context)
            else:  # MIXED
                electrical_component = self._electrical_transmission(presynaptic_signal) * 0.3
                chemical_component = self._chemical_transmission(presynaptic_signal, context) * 0.7
                output_signal = electrical_component + chemical_component
            
            # Apply synaptic delay (simplified for real-time simulation)
            # In a full simulation, this would be handled by the scheduler
            
            # Apply current synaptic strength
            final_signal = output_signal * self.current_strength * self.resource_depletion
            
            # Record transmission
            transmission_record = {
                "timestamp": current_time,
                "presynaptic_signal": presynaptic_signal,
                "output_signal": final_signal,
                "synaptic_strength": self.current_strength,
                "resource_depletion": self.resource_depletion,
                "neurotransmitter_levels": {
                    nt.value: pool.concentration 
                    for nt, pool in self.neurotransmitter_pools.items()
                }
            }
            
            self.transmission_history.append(transmission_record)
            if len(self.transmission_history) > 1000:
                self.transmission_history.pop(0)
            
            # Update neurotransmitter pools
            self._update_neurotransmitter_pools(presynaptic_signal)
            
            # Apply plasticity if enabled
            if self.learning_enabled:
                self._apply_plasticity(presynaptic_signal, context)
            
            # Update metadata
            self.metadata["total_transmissions"] += 1
            self.metadata["updated_at"] = current_time
            
            return {
                "signal": final_signal,
                "delay": self.delay,
                "transmission_success": final_signal > 0.01,
                "resource_depletion": self.resource_depletion,
                "synaptic_strength": self.current_strength
            }
            
        except Exception as e:
            print(f"Synaptic transmission failed: {e}")
            return {"signal": 0.0, "delay": self.delay, "transmission_success": False}
    
    def _electrical_transmission(self, signal: float) -> float:
        """Simulate electrical transmission (gap junction)"""
        # Electrical synapses are bidirectional and fast
        # Signal attenuation based on electrical resistance
        attenuation = 0.8  # 20% signal loss
        return signal * attenuation
    
    def _chemical_transmission(self, signal: float, context: Dict[str, Any]) -> float:
        """Simulate chemical transmission with neurotransmitter dynamics"""
        total_output = 0.0
        
        for nt_type, pool in self.neurotransmitter_pools.items():
            # Calculate neurotransmitter release
            release_amount = self._calculate_neurotransmitter_release(signal, pool)
            
            # Apply neurotransmitter-specific effects
            nt_effect = self._apply_neurotransmitter_effect(release_amount, nt_type, context)
            
            total_output += nt_effect
        
        return total_output
    
    def _calculate_neurotransmitter_release(
        self, 
        signal: float, 
        pool: NeurotransmitterPool
    ) -> float:
        """Calculate amount of neurotransmitter released"""
        # Release probability depends on signal strength and available neurotransmitter
        release_probability = min(1.0, pool.release_probability * signal)
        
        # Stochastic release
        if np.random.random() < release_probability:
            # Amount released depends on vesicle availability and concentration
            vesicle_release = min(pool.vesicle_count, int(signal * 10))
            release_amount = vesicle_release * pool.concentration / 100
            
            # Update pool
            pool.vesicle_count = max(0, pool.vesicle_count - vesicle_release)
            pool.concentration = max(0, pool.concentration - release_amount * 0.1)
            
            return release_amount
        
        return 0.0
    
    def _apply_neurotransmitter_effect(
        self, 
        amount: float, 
        nt_type: NeurotransmitterType, 
        context: Dict[str, Any]
    ) -> float:
        """Apply neurotransmitter-specific effects"""
        if nt_type == NeurotransmitterType.GLUTAMATE:
            # Primary excitatory effect
            return amount * 1.0
            
        elif nt_type == NeurotransmitterType.GABA:
            # Inhibitory effect
            return amount * -0.8
            
        elif nt_type == NeurotransmitterType.DOPAMINE:
            # Modulatory effect based on reward context
            reward_signal = context.get("reward", 0.0)
            return amount * (0.5 + reward_signal * 0.5)
            
        elif nt_type == NeurotransmitterType.SEROTONIN:
            # Mood and learning modulation
            learning_context = context.get("learning_signal", 1.0)
            return amount * 0.7 * learning_context
            
        elif nt_type == NeurotransmitterType.ACETYLCHOLINE:
            # Attention and learning enhancement
            attention_level = context.get("attention", 1.0)
            return amount * 0.6 * attention_level
            
        elif nt_type == NeurotransmitterType.NOREPINEPHRINE:
            # Arousal and attention
            arousal_level = context.get("arousal", 1.0)
            return amount * 0.8 * arousal_level
        
        return amount * 0.5  # Default effect
    
    def _update_neurotransmitter_pools(self, signal_strength: float):
        """Update neurotransmitter pool dynamics"""
        dt = 0.1  # Time step
        
        for pool in self.neurotransmitter_pools.values():
            # Reuptake (clearance)
            pool.concentration *= (1 - pool.reuptake_rate * dt)
            
            # Synthesis (replenishment)
            synthesis = pool.synthesis_rate * dt
            pool.concentration = min(2.0, pool.concentration + synthesis)
            
            # Vesicle replenishment
            if pool.vesicle_count < 100:
                replenishment = int(synthesis * 50)
                pool.vesicle_count = min(100, pool.vesicle_count + replenishment)
    
    def _apply_plasticity(self, presynaptic_signal: float, context: Dict[str, Any]):
        """Apply synaptic plasticity mechanisms"""
        current_time = time.time()
        
        for plasticity_type, mechanism in self.plasticity_mechanisms.items():
            if plasticity_type == PlasticityType.LTP:
                self._apply_ltp(presynaptic_signal, mechanism, context)
            elif plasticity_type == PlasticityType.LTD:
                self._apply_ltd(presynaptic_signal, mechanism, context)
            elif plasticity_type == PlasticityType.HOMEOSTATIC:
                self._apply_homeostatic_plasticity(mechanism)
    
    def _apply_ltp(self, signal: float, mechanism: SynapticPlasticity, context: Dict[str, Any]):
        """Apply long-term potentiation"""
        # LTP requires strong presynaptic activity and postsynaptic depolarization
        postsynaptic_activity = context.get("postsynaptic_activity", 0.0)
        
        if (signal > mechanism.induction_threshold and 
            postsynaptic_activity > mechanism.induction_threshold):
            
            # Strength increase
            strength_increase = mechanism.expression_magnitude * signal * postsynaptic_activity
            self.current_strength = min(
                mechanism.saturation_level,
                self.current_strength + strength_increase
            )
            
            # Record plasticity event
            self.plasticity_events.append({
                "timestamp": time.time(),
                "type": "LTP",
                "magnitude": strength_increase,
                "new_strength": self.current_strength
            })
            
            self.metadata["total_plasticity_events"] += 1
    
    def _apply_ltd(self, signal: float, mechanism: SynapticPlasticity, context: Dict[str, Any]):
        """Apply long-term depression"""
        # LTD can occur with weak correlated activity
        postsynaptic_activity = context.get("postsynaptic_activity", 0.0)
        
        if (signal > 0.1 and signal < mechanism.induction_threshold and
            postsynaptic_activity > 0.1 and postsynaptic_activity < mechanism.induction_threshold):
            
            # Strength decrease
            strength_decrease = abs(mechanism.expression_magnitude) * signal
            self.current_strength = max(
                0.1,  # Minimum strength
                self.current_strength - strength_decrease
            )
            
            # Record plasticity event
            self.plasticity_events.append({
                "timestamp": time.time(),
                "type": "LTD",
                "magnitude": -strength_decrease,
                "new_strength": self.current_strength
            })
            
            self.metadata["total_plasticity_events"] += 1
    
    def _apply_homeostatic_plasticity(self, mechanism: SynapticPlasticity):
        """Apply homeostatic scaling to maintain network stability"""
        # Calculate recent activity level
        recent_transmissions = [
            t for t in self.transmission_history
            if time.time() - t["timestamp"] < 3600  # Last hour
        ]
        
        if len(recent_transmissions) > 10:
            avg_activity = sum(t["output_signal"] for t in recent_transmissions) / len(recent_transmissions)
            target_activity = 0.5  # Target average activity
            
            # Scale synaptic strength to maintain target activity
            if avg_activity > target_activity * 1.2:  # Too active
                scaling = -mechanism.expression_magnitude
            elif avg_activity < target_activity * 0.8:  # Too quiet
                scaling = mechanism.expression_magnitude
            else:
                return  # Activity is within target range
            
            self.current_strength = max(0.1, min(2.0, self.current_strength + scaling))
            
            # Record plasticity event
            self.plasticity_events.append({
                "timestamp": time.time(),
                "type": "HOMEOSTATIC",
                "magnitude": scaling,
                "new_strength": self.current_strength,
                "avg_activity": avg_activity
            })
    
    def set_postsynaptic_spike(self, spike_time: float, amplitude: float = 1.0):
        """Record postsynaptic spike for plasticity calculations"""
        self.last_postsynaptic_spike = spike_time
        self.spike_timing.append({
            "type": "postsynaptic",
            "time": spike_time,
            "amplitude": amplitude
        })
        
        # Keep only recent spikes for STDP
        cutoff_time = spike_time - 0.1  # 100ms window
        self.spike_timing = [
            spike for spike in self.spike_timing
            if spike["time"] > cutoff_time
        ]
    
    def get_synapse_info(self) -> Dict[str, Any]:
        """Get comprehensive synapse information"""
        return {
            "id": self.id,
            "presynaptic_node_id": self.presynaptic_node_id,
            "postsynaptic_node_id": self.postsynaptic_node_id,
            "synapse_type": self.synapse_type.value,
            "base_strength": self.base_strength,
            "current_strength": self.current_strength,
            "delay": self.delay,
            "learning_enabled": self.learning_enabled,
            "resource_depletion": self.resource_depletion,
            "neurotransmitter_pools": {
                nt.value: asdict(pool) 
                for nt, pool in self.neurotransmitter_pools.items()
            },
            "plasticity_mechanisms": {
                pt.value: asdict(mech) 
                for pt, mech in self.plasticity_mechanisms.items()
            },
            "metadata": self.metadata
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synapse statistics"""
        recent_transmissions = [
            t for t in self.transmission_history
            if time.time() - t["timestamp"] < 3600  # Last hour
        ]
        
        return {
            "total_transmissions": self.metadata["total_transmissions"],
            "recent_transmissions": len(recent_transmissions),
            "average_signal_strength": (
                sum(t["output_signal"] for t in recent_transmissions) /
                max(len(recent_transmissions), 1)
            ),
            "plasticity_events": len(self.plasticity_events),
            "strength_change": self.current_strength - self.base_strength,
            "transmission_success_rate": (
                sum(1 for t in recent_transmissions if t["output_signal"] > 0.01) /
                max(len(recent_transmissions), 1)
            ),
            "average_resource_depletion": (
                sum(t["resource_depletion"] for t in recent_transmissions) /
                max(len(recent_transmissions), 1)
            )
        }
    
    def reset_plasticity(self):
        """Reset synaptic strength to base level"""
        self.current_strength = self.base_strength
        self.plasticity_events.clear()
        self.metadata["total_plasticity_events"] = 0
        self.metadata["updated_at"] = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert synapse to dictionary for serialization"""
        return {
            "id": self.id,
            "presynaptic_node_id": self.presynaptic_node_id,
            "postsynaptic_node_id": self.postsynaptic_node_id,
            "synapse_type": self.synapse_type.value,
            "base_strength": self.base_strength,
            "current_strength": self.current_strength,
            "delay": self.delay,
            "learning_enabled": self.learning_enabled,
            "neurotransmitter_pools": {
                nt.value: asdict(pool) 
                for nt, pool in self.neurotransmitter_pools.items()
            },
            "plasticity_mechanisms": {
                pt.value: asdict(mech) 
                for pt, mech in self.plasticity_mechanisms.items()
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralSynapse':
        """Create synapse from dictionary"""
        synapse = cls(
            synapse_id=data["id"],
            presynaptic_node_id=data["presynaptic_node_id"],
            postsynaptic_node_id=data["postsynaptic_node_id"],
            synapse_type=SynapseType(data["synapse_type"]),
            base_strength=data["base_strength"],
            delay=data["delay"],
            learning_enabled=data["learning_enabled"]
        )
        
        # Restore state
        synapse.current_strength = data["current_strength"]
        synapse.metadata = data["metadata"]
        
        # Restore neurotransmitter pools
        for nt_name, pool_data in data["neurotransmitter_pools"].items():
            nt_type = NeurotransmitterType(nt_name)
            synapse.neurotransmitter_pools[nt_type] = NeurotransmitterPool(**pool_data)
        
        # Restore plasticity mechanisms
        for pt_name, mech_data in data["plasticity_mechanisms"].items():
            pt_type = PlasticityType(pt_name)
            synapse.plasticity_mechanisms[pt_type] = SynapticPlasticity(**mech_data)
        
        return synapse
    
    def __str__(self) -> str:
        return f"NeuralSynapse({self.presynaptic_node_id} -> {self.postsynaptic_node_id}, strength={self.current_strength:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()