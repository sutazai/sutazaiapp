#!/usr/bin/env python3
"""
Biological Neural Network Modeling
Advanced biological neuron modeling with realistic neural dynamics
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio
import threading

logger = logging.getLogger(__name__)

@dataclass
class BiologicalConfig:
    """Configuration for biological neural network"""
    # Network architecture
    num_neurons: int = 1000
    num_layers: int = 3
    connection_probability: float = 0.1
    
    # Neuron parameters
    membrane_potential_rest: float = -70.0  # mV
    membrane_potential_threshold: float = -55.0  # mV
    membrane_potential_reset: float = -80.0  # mV
    membrane_time_constant: float = 20.0  # ms
    refractory_period: float = 2.0  # ms
    
    # Synaptic parameters
    synaptic_delay: float = 1.0  # ms
    excitatory_weight: float = 0.5
    inhibitory_weight: float = -0.5
    synaptic_decay: float = 0.1
    
    # Plasticity parameters
    enable_stdp: bool = True  # Spike-timing dependent plasticity
    stdp_learning_rate: float = 0.01
    stdp_time_window: float = 20.0  # ms
    
    # Homeostasis parameters
    enable_homeostasis: bool = True
    target_firing_rate: float = 10.0  # Hz
    homeostasis_time_constant: float = 1000.0  # ms
    
    # Noise parameters
    noise_amplitude: float = 0.1
    enable_noise: bool = True
    
    # Simulation parameters
    dt: float = 0.1  # ms
    simulation_time: float = 1000.0  # ms
    
    # Device settings
    device: str = "auto"
    dtype: str = "float32"

class BiologicalNeuron(nn.Module):
    """
    Biological neuron model with realistic dynamics
    Implements leaky integrate-and-fire model with adaptations
    """
    
    def __init__(self, config: BiologicalConfig):
        super().__init__()
        self.config = config
        
        # Neuron state variables
        self.membrane_potential = nn.Parameter(
            torch.full((config.num_neurons,), config.membrane_potential_rest),
            requires_grad=False
        )
        self.spike_times = []
        self.refractory_timer = torch.zeros(config.num_neurons)
        
        # Synaptic currents
        self.excitatory_current = torch.zeros(config.num_neurons)
        self.inhibitory_current = torch.zeros(config.num_neurons)
        
        # Adaptation variables
        self.adaptation_current = torch.zeros(config.num_neurons)
        self.firing_rate = torch.zeros(config.num_neurons)
        
        # Homeostasis variables
        self.homeostatic_scaling = torch.ones(config.num_neurons)
        
        logger.info(f"Biological neuron initialized with {config.num_neurons} neurons")
    
    def forward(self, input_current: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through biological neuron model
        
        Args:
            input_current: External input current
            
        Returns:
            Dictionary containing neuron states and outputs
        """
        # Update membrane potential
        self._update_membrane_potential(input_current)
        
        # Check for spikes
        spikes = self._check_spikes()
        
        # Update synaptic currents
        self._update_synaptic_currents()
        
        # Update adaptation
        self._update_adaptation(spikes)
        
        # Update homeostasis
        if self.config.enable_homeostasis:
            self._update_homeostasis(spikes)
        
        return {
            "membrane_potential": self.membrane_potential.clone(),
            "spikes": spikes,
            "excitatory_current": self.excitatory_current.clone(),
            "inhibitory_current": self.inhibitory_current.clone(),
            "firing_rate": self.firing_rate.clone()
        }
    
    def _update_membrane_potential(self, input_current: torch.Tensor):
        """Update membrane potential using leaky integrate-and-fire dynamics"""
        # Leak current
        leak_current = -(self.membrane_potential - self.config.membrane_potential_rest) / self.config.membrane_time_constant
        
        # Total current
        total_current = (
            leak_current + 
            input_current + 
            self.excitatory_current + 
            self.inhibitory_current - 
            self.adaptation_current
        )
        
        # Add noise
        if self.config.enable_noise:
            noise = torch.randn_like(self.membrane_potential) * self.config.noise_amplitude
            total_current += noise
        
        # Update membrane potential (only for non-refractory neurons)
        non_refractory = self.refractory_timer <= 0
        self.membrane_potential[non_refractory] += total_current[non_refractory] * self.config.dt
        
        # Update refractory timer
        self.refractory_timer = torch.clamp(self.refractory_timer - self.config.dt, min=0)
    
    def _check_spikes(self) -> torch.Tensor:
        """Check for spikes and handle spike generation"""
        # Find neurons above threshold
        spike_mask = self.membrane_potential >= self.config.membrane_potential_threshold
        
        # Reset spiking neurons
        self.membrane_potential[spike_mask] = self.config.membrane_potential_reset
        
        # Set refractory period
        self.refractory_timer[spike_mask] = self.config.refractory_period
        
        # Record spike times
        current_time = len(self.spike_times) * self.config.dt
        spike_indices = torch.where(spike_mask)[0]
        for idx in spike_indices:
            self.spike_times.append((current_time, idx.item()))
        
        return spike_mask.float()
    
    def _update_synaptic_currents(self):
        """Update excitatory and inhibitory synaptic currents"""
        # Exponential decay
        self.excitatory_current *= torch.exp(-self.config.dt / self.config.synaptic_decay)
        self.inhibitory_current *= torch.exp(-self.config.dt / self.config.synaptic_decay)
    
    def _update_adaptation(self, spikes: torch.Tensor):
        """Update adaptation current based on recent spiking"""
        # Increase adaptation for spiking neurons
        self.adaptation_current[spikes.bool()] += 0.1
        
        # Decay adaptation
        self.adaptation_current *= torch.exp(-self.config.dt / 100.0)  # 100ms time constant
    
    def _update_homeostasis(self, spikes: torch.Tensor):
        """Update homeostatic scaling to maintain target firing rate"""
        # Update firing rate estimate
        self.firing_rate += (spikes - self.firing_rate) * self.config.dt / self.config.homeostasis_time_constant
        
        # Update homeostatic scaling
        firing_rate_error = self.config.target_firing_rate - self.firing_rate
        self.homeostatic_scaling += firing_rate_error * self.config.dt / self.config.homeostasis_time_constant
        self.homeostatic_scaling = torch.clamp(self.homeostatic_scaling, 0.1, 10.0)
    
    def add_synaptic_input(self, synaptic_input: torch.Tensor, is_excitatory: bool = True):
        """Add synaptic input to neurons"""
        if is_excitatory:
            self.excitatory_current += synaptic_input * self.homeostatic_scaling
        else:
            self.inhibitory_current += synaptic_input * self.homeostatic_scaling
    
    def reset_state(self):
        """Reset neuron state"""
        self.membrane_potential.fill_(self.config.membrane_potential_rest)
        self.refractory_timer.fill_(0)
        self.excitatory_current.fill_(0)
        self.inhibitory_current.fill_(0)
        self.adaptation_current.fill_(0)
        self.firing_rate.fill_(0)
        self.homeostatic_scaling.fill_(1.0)
        self.spike_times.clear()

class SynapticConnection(nn.Module):
    """
    Synaptic connection with plasticity
    Implements spike-timing dependent plasticity (STDP)
    """
    
    def __init__(self, config: BiologicalConfig, pre_neurons: int, post_neurons: int):
        super().__init__()
        self.config = config
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        
        # Synaptic weights
        self.weights = nn.Parameter(
            torch.randn(pre_neurons, post_neurons) * 0.1
        )
        
        # Connection mask (random connectivity)
        self.connection_mask = torch.rand(pre_neurons, post_neurons) < config.connection_probability
        self.weights.data[~self.connection_mask] = 0
        
        # STDP traces
        self.pre_trace = torch.zeros(pre_neurons)
        self.post_trace = torch.zeros(post_neurons)
        
        # Synaptic delays
        self.delay_buffer = torch.zeros(int(config.synaptic_delay / config.dt) + 1, pre_neurons)
        self.delay_index = 0
        
        logger.info(f"Synaptic connection initialized: {pre_neurons} -> {post_neurons}")
    
    def forward(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through synaptic connection
        
        Args:
            pre_spikes: Presynaptic spikes
            post_spikes: Postsynaptic spikes
            
        Returns:
            Synaptic output current
        """
        # Add current spikes to delay buffer
        self.delay_buffer[self.delay_index] = pre_spikes
        self.delay_index = (self.delay_index + 1) % self.delay_buffer.size(0)
        
        # Get delayed spikes
        delayed_spikes = self.delay_buffer[self.delay_index]
        
        # Calculate synaptic output
        synaptic_output = torch.matmul(delayed_spikes, self.weights * self.connection_mask.float())
        
        # Update STDP
        if self.config.enable_stdp:
            self._update_stdp(delayed_spikes, post_spikes)
        
        return synaptic_output
    
    def _update_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """Update synaptic weights using STDP"""
        # Update traces
        self.pre_trace += pre_spikes - self.pre_trace * self.config.dt / self.config.stdp_time_window
        self.post_trace += post_spikes - self.post_trace * self.config.dt / self.config.stdp_time_window
        
        # Calculate weight changes
        # LTP: post-before-pre
        ltp = torch.outer(pre_spikes, self.post_trace)
        
        # LTD: pre-before-post
        ltd = torch.outer(self.pre_trace, post_spikes)
        
        # Weight update
        weight_change = self.config.stdp_learning_rate * (ltp - ltd)
        
        # Apply only to connected synapses
        weight_change *= self.connection_mask.float()
        
        # Update weights
        self.weights.data += weight_change
        
        # Clip weights
        self.weights.data = torch.clamp(self.weights.data, -1.0, 1.0)

class BiologicalNeuralNetwork(nn.Module):
    """
    Complete biological neural network with multiple layers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = BiologicalConfig(**config) if config else BiologicalConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Create neuron layers
        self.layers = nn.ModuleList()
        self.synaptic_connections = nn.ModuleList()
        
        # Create layers
        for i in range(self.config.num_layers):
            layer = BiologicalNeuron(self.config)
            self.layers.append(layer)
            
            # Create synaptic connections (except for last layer)
            if i < self.config.num_layers - 1:
                connection = SynapticConnection(
                    self.config, 
                    self.config.num_neurons, 
                    self.config.num_neurons
                )
                self.synaptic_connections.append(connection)
        
        # Network state
        self.is_initialized = False
        self.current_time = 0.0
        self.network_activity = []
        
        self.to(self.device)
        logger.info(f"Biological neural network created with {self.config.num_layers} layers")
    
    async def initialize(self) -> bool:
        """Initialize biological neural network"""
        try:
            # Reset all layers
            for layer in self.layers:
                layer.reset_state()
            
            # Clear network activity
            self.network_activity.clear()
            self.current_time = 0.0
            
            self.is_initialized = True
            logger.info("Biological neural network initialized")
            return True
            
        except Exception as e:
            logger.error(f"Biological neural network initialization failed: {e}")
            return False
    
    async def process(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Process input through biological neural network
        
        Args:
            input_data: Input tensor
            
        Returns:
            Dictionary containing network outputs and states
        """
        try:
            input_data = input_data.to(self.device)
            
            # Convert input to current
            if input_data.dim() == 1:
                input_current = input_data.repeat(self.config.num_neurons, 1).T
            else:
                input_current = input_data
            
            # Process through layers
            layer_outputs = []
            current_input = input_current
            
            for i, layer in enumerate(self.layers):
                # Process through current layer
                layer_output = layer(current_input)
                layer_outputs.append(layer_output)
                
                # Pass spikes to next layer via synaptic connections
                if i < len(self.synaptic_connections):
                    next_layer_input = self.synaptic_connections[i](
                        layer_output["spikes"],
                        torch.zeros_like(layer_output["spikes"]) if i == len(self.layers) - 1 else layer_outputs[i + 1]["spikes"]
                    )
                    current_input = next_layer_input.unsqueeze(0)
                else:
                    current_input = layer_output["spikes"].unsqueeze(0)
            
            # Calculate network activity
            total_spikes = sum(output["spikes"].sum().item() for output in layer_outputs)
            self.network_activity.append(total_spikes)
            
            # Update time
            self.current_time += self.config.dt
            
            # Create output
            final_output = layer_outputs[-1]["spikes"]
            
            return {
                "output": final_output,
                "layer_outputs": layer_outputs,
                "network_activity": total_spikes,
                "current_time": self.current_time,
                "membrane_potentials": [output["membrane_potential"] for output in layer_outputs],
                "firing_rates": [output["firing_rate"] for output in layer_outputs]
            }
            
        except Exception as e:
            logger.error(f"Biological neural network processing failed: {e}")
            raise
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "num_layers": self.config.num_layers,
            "num_neurons": self.config.num_neurons,
            "current_time": self.current_time,
            "network_activity": self.network_activity[-100:],  # Last 100 time steps
            "average_activity": np.mean(self.network_activity) if self.network_activity else 0,
            "total_connections": sum(conn.connection_mask.sum().item() for conn in self.synaptic_connections),
            "is_initialized": self.is_initialized
        }
    
    def get_synaptic_weights(self) -> List[torch.Tensor]:
        """Get synaptic weights from all connections"""
        return [conn.weights.data.clone() for conn in self.synaptic_connections]
    
    def set_synaptic_weights(self, weights: List[torch.Tensor]):
        """Set synaptic weights for all connections"""
        for i, weight_tensor in enumerate(weights):
            if i < len(self.synaptic_connections):
                self.synaptic_connections[i].weights.data = weight_tensor.to(self.device)
    
    async def save(self, path: str) -> bool:
        """Save biological neural network"""
        try:
            save_data = {
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
                "network_activity": self.network_activity,
                "current_time": self.current_time
            }
            
            torch.save(save_data, path)
            logger.info(f"Biological neural network saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save biological neural network: {e}")
            return False
    
    async def load(self, path: str) -> bool:
        """Load biological neural network"""
        try:
            save_data = torch.load(path, map_location=self.device)
            
            # Load state dict
            self.load_state_dict(save_data["state_dict"])
            
            # Restore network state
            self.network_activity = save_data.get("network_activity", [])
            self.current_time = save_data.get("current_time", 0.0)
            
            logger.info(f"Biological neural network loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load biological neural network: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get network status"""
        return {
            "is_initialized": self.is_initialized,
            "current_time": self.current_time,
            "device": str(self.device),
            "config": self.config.__dict__,
            "statistics": self.get_network_statistics()
        }
    
    def health_check(self) -> bool:
        """Check network health"""
        try:
            return (
                self.is_initialized and
                len(self.layers) == self.config.num_layers and
                len(self.synaptic_connections) == self.config.num_layers - 1
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown biological neural network"""
        try:
            # Reset all layers
            for layer in self.layers:
                layer.reset_state()
            
            # Clear network activity
            self.network_activity.clear()
            self.current_time = 0.0
            self.is_initialized = False
            
            logger.info("Biological neural network shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Biological neural network shutdown failed: {e}")
            return False