#!/usr/bin/env python3
"""
Synaptic Plasticity Manager
Implements various forms of synaptic plasticity for neural networks
"""

import logging
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PlasticityConfig:
    """Configuration for synaptic plasticity"""
    # STDP parameters
    enable_stdp: bool = True
    stdp_learning_rate: float = 0.01
    stdp_time_window: float = 20.0  # ms
    stdp_tau_plus: float = 20.0  # ms
    stdp_tau_minus: float = 20.0  # ms
    stdp_a_plus: float = 0.1
    stdp_a_minus: float = 0.12
    
    # Homeostatic plasticity
    enable_homeostatic: bool = True
    target_rate: float = 10.0  # Hz
    homeostatic_alpha: float = 0.1
    homeostatic_beta: float = 0.1
    
    # Metaplasticity
    enable_metaplasticity: bool = True
    metaplasticity_theta: float = 0.5
    metaplasticity_tau: float = 1000.0  # ms
    
    # Structural plasticity
    enable_structural: bool = True
    pruning_threshold: float = 0.01
    growth_rate: float = 0.001
    
    # Device settings
    device: str = "auto"
    dt: float = 0.1  # ms

class STDPRule:
    """Spike-timing dependent plasticity rule"""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # STDP parameters
        self.tau_plus = config.stdp_tau_plus
        self.tau_minus = config.stdp_tau_minus
        self.a_plus = config.stdp_a_plus
        self.a_minus = config.stdp_a_minus
        
        # Trace variables
        self.pre_trace = None
        self.post_trace = None
        
    def initialize(self, num_pre: int, num_post: int):
        """Initialize STDP traces"""
        self.pre_trace = torch.zeros(num_pre, device=self.device)
        self.post_trace = torch.zeros(num_post, device=self.device)
    
    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
               weights: torch.Tensor) -> torch.Tensor:
        """Update weights based on STDP rule"""
        if self.pre_trace is None or self.post_trace is None:
            self.initialize(pre_spikes.size(0), post_spikes.size(0))
        
        # Update traces
        self.pre_trace = self.pre_trace * torch.exp(-self.config.dt / self.tau_plus) + pre_spikes
        self.post_trace = self.post_trace * torch.exp(-self.config.dt / self.tau_minus) + post_spikes
        
        # Calculate weight changes
        # LTP: post spike occurs after pre spike
        ltp = torch.outer(pre_spikes, self.post_trace) * self.a_plus
        
        # LTD: pre spike occurs after post spike
        ltd = torch.outer(self.pre_trace, post_spikes) * self.a_minus
        
        # Total weight change
        weight_change = (ltp - ltd) * self.config.stdp_learning_rate
        
        # Apply weight change
        new_weights = weights + weight_change
        
        return new_weights
    
    def reset(self):
        """Reset STDP traces"""
        if self.pre_trace is not None:
            self.pre_trace.zero_()
        if self.post_trace is not None:
            self.post_trace.zero_()

class HomeostaticPlasticity:
    """Homeostatic plasticity mechanism"""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Homeostatic parameters
        self.target_rate = config.target_rate
        self.alpha = config.homeostatic_alpha
        self.beta = config.homeostatic_beta
        
        # State variables
        self.firing_rates = None
        self.scaling_factors = None
        
    def initialize(self, num_neurons: int):
        """Initialize homeostatic variables"""
        self.firing_rates = torch.zeros(num_neurons, device=self.device)
        self.scaling_factors = torch.ones(num_neurons, device=self.device)
    
    def update(self, spike_activity: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update homeostatic scaling"""
        if self.firing_rates is None or self.scaling_factors is None:
            self.initialize(spike_activity.size(0))
        
        # Update firing rate estimate
        self.firing_rates = (1 - self.alpha) * self.firing_rates + self.alpha * spike_activity
        
        # Calculate scaling factors
        rate_error = self.target_rate - self.firing_rates
        self.scaling_factors = self.scaling_factors + self.beta * rate_error * self.config.dt
        
        # Clamp scaling factors
        self.scaling_factors = torch.clamp(self.scaling_factors, 0.1, 10.0)
        
        # Apply scaling to weights
        scaled_weights = weights * self.scaling_factors.unsqueeze(0)
        
        return scaled_weights, self.scaling_factors
    
    def reset(self):
        """Reset homeostatic variables"""
        if self.firing_rates is not None:
            self.firing_rates.zero_()
        if self.scaling_factors is not None:
            self.scaling_factors.fill_(1.0)

class Metaplasticity:
    """Metaplasticity mechanism (plasticity of plasticity)"""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Metaplasticity parameters
        self.theta = config.metaplasticity_theta
        self.tau = config.metaplasticity_tau
        
        # State variables
        self.omega = None  # Metaplasticity variable
        
    def initialize(self, weight_shape: Tuple[int, int]):
        """Initialize metaplasticity variables"""
        self.omega = torch.zeros(weight_shape, device=self.device)
    
    def update(self, weight_change: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Update metaplasticity and modify weight changes"""
        if self.omega is None:
            self.initialize(weights.shape)
        
        # Update omega (sliding average of weight changes)
        self.omega = self.omega * torch.exp(-self.config.dt / self.tau) + \
                   torch.abs(weight_change)
        
        # Modulate weight change based on omega
        modulation = torch.sigmoid(self.theta - self.omega)
        modulated_change = weight_change * modulation
        
        return modulated_change
    
    def reset(self):
        """Reset metaplasticity variables"""
        if self.omega is not None:
            self.omega.zero_()

class StructuralPlasticity:
    """Structural plasticity for connection pruning and growth"""
    
    def __init__(self, config: PlasticityConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Structural parameters
        self.pruning_threshold = config.pruning_threshold
        self.growth_rate = config.growth_rate
        
        # Connection mask
        self.connection_mask = None
        
    def initialize(self, weight_shape: Tuple[int, int], initial_density: float = 0.1):
        """Initialize connection mask"""
        self.connection_mask = torch.rand(weight_shape, device=self.device) < initial_density
    
    def update(self, weights: torch.Tensor, activity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update structural connectivity"""
        if self.connection_mask is None:
            self.initialize(weights.shape)
        
        # Prune weak connections
        weak_connections = torch.abs(weights) < self.pruning_threshold
        self.connection_mask = self.connection_mask & ~weak_connections
        
        # Grow new connections based on activity
        growth_probability = self.growth_rate * activity.unsqueeze(1)
        growth_mask = torch.rand_like(weights) < growth_probability
        potential_connections = ~self.connection_mask & growth_mask
        
        # Add new connections
        self.connection_mask = self.connection_mask | potential_connections
        
        # Apply connection mask to weights
        masked_weights = weights * self.connection_mask.float()
        
        return masked_weights, self.connection_mask
    
    def reset(self, initial_density: float = 0.1):
        """Reset structural connectivity"""
        if self.connection_mask is not None:
            self.connection_mask = torch.rand_like(self.connection_mask) < initial_density

class SynapticPlasticityManager:
    """
    Manager for all synaptic plasticity mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = PlasticityConfig(**config) if config else PlasticityConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Plasticity mechanisms
        self.stdp = STDPRule(self.config) if self.config.enable_stdp else None
        self.homeostatic = HomeostaticPlasticity(self.config) if self.config.enable_homeostatic else None
        self.metaplasticity = Metaplasticity(self.config) if self.config.enable_metaplasticity else None
        self.structural = StructuralPlasticity(self.config) if self.config.enable_structural else None
        
        # State
        self.is_initialized = False
        self.update_count = 0
        
        # Statistics
        self.plasticity_stats = {
            "weight_changes": [],
            "connection_density": [],
            "firing_rates": [],
            "scaling_factors": []
        }
        
        logger.info("Synaptic plasticity manager created")
    
    async def initialize(self, weight_shape: Tuple[int, int], initial_weights: Optional[torch.Tensor] = None) -> bool:
        """Initialize plasticity mechanisms"""
        try:
            if self.is_initialized:
                return True
            
            # Initialize STDP
            if self.stdp:
                self.stdp.initialize(weight_shape[0], weight_shape[1])
            
            # Initialize homeostatic plasticity
            if self.homeostatic:
                self.homeostatic.initialize(weight_shape[1])
            
            # Initialize metaplasticity
            if self.metaplasticity:
                self.metaplasticity.initialize(weight_shape)
            
            # Initialize structural plasticity
            if self.structural:
                self.structural.initialize(weight_shape)
            
            self.is_initialized = True
            logger.info("Synaptic plasticity manager initialized")
            return True
            
        except Exception as e:
            logger.error(f"Plasticity manager initialization failed: {e}")
            return False
    
    async def update_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, 
                              weights: torch.Tensor) -> Dict[str, Any]:
        """
        Update synaptic plasticity based on neural activity
        
        Args:
            pre_spikes: Presynaptic spike activity
            post_spikes: Postsynaptic spike activity
            weights: Current synaptic weights
            
        Returns:
            Dictionary containing updated weights and plasticity information
        """
        try:
            if not self.is_initialized:
                await self.initialize(weights.shape)
            
            updated_weights = weights.clone()
            plasticity_info = {}
            
            # Apply STDP
            if self.stdp:
                updated_weights = self.stdp.update(pre_spikes, post_spikes, updated_weights)
                plasticity_info["stdp_applied"] = True
            
            # Apply metaplasticity modulation
            if self.metaplasticity:
                weight_change = updated_weights - weights
                modulated_change = self.metaplasticity.update(weight_change, weights)
                updated_weights = weights + modulated_change
                plasticity_info["metaplasticity_applied"] = True
            
            # Apply homeostatic scaling
            if self.homeostatic:
                updated_weights, scaling_factors = self.homeostatic.update(post_spikes, updated_weights)
                plasticity_info["homeostatic_applied"] = True
                plasticity_info["scaling_factors"] = scaling_factors
            
            # Apply structural plasticity
            if self.structural:
                updated_weights, connection_mask = self.structural.update(updated_weights, post_spikes)
                plasticity_info["structural_applied"] = True
                plasticity_info["connection_mask"] = connection_mask
                plasticity_info["connection_density"] = connection_mask.float().mean().item()
            
            # Clamp weights to reasonable range
            updated_weights = torch.clamp(updated_weights, -10.0, 10.0)
            
            # Update statistics
            self._update_statistics(weights, updated_weights, plasticity_info)
            
            self.update_count += 1
            
            return {
                "weights": updated_weights,
                "plasticity_info": plasticity_info,
                "update_count": self.update_count
            }
            
        except Exception as e:
            logger.error(f"Plasticity update failed: {e}")
            raise
    
    def _update_statistics(self, old_weights: torch.Tensor, new_weights: torch.Tensor, 
                          plasticity_info: Dict[str, Any]):
        """Update plasticity statistics"""
        # Weight change magnitude
        weight_change_magnitude = torch.abs(new_weights - old_weights).mean().item()
        self.plasticity_stats["weight_changes"].append(weight_change_magnitude)
        
        # Connection density
        if "connection_density" in plasticity_info:
            self.plasticity_stats["connection_density"].append(plasticity_info["connection_density"])
        
        # Scaling factors
        if "scaling_factors" in plasticity_info:
            avg_scaling = plasticity_info["scaling_factors"].mean().item()
            self.plasticity_stats["scaling_factors"].append(avg_scaling)
        
        # Keep only recent statistics
        max_history = 1000
        for key in self.plasticity_stats:
            if len(self.plasticity_stats[key]) > max_history:
                self.plasticity_stats[key] = self.plasticity_stats[key][-max_history:]
    
    def get_plasticity_statistics(self) -> Dict[str, Any]:
        """Get plasticity statistics"""
        stats = {
            "update_count": self.update_count,
            "average_weight_change": np.mean(self.plasticity_stats["weight_changes"]) if self.plasticity_stats["weight_changes"] else 0,
            "recent_weight_changes": self.plasticity_stats["weight_changes"][-100:],
            "is_initialized": self.is_initialized
        }
        
        if self.plasticity_stats["connection_density"]:
            stats["average_connection_density"] = np.mean(self.plasticity_stats["connection_density"])
            stats["recent_connection_density"] = self.plasticity_stats["connection_density"][-100:]
        
        if self.plasticity_stats["scaling_factors"]:
            stats["average_scaling_factor"] = np.mean(self.plasticity_stats["scaling_factors"])
            stats["recent_scaling_factors"] = self.plasticity_stats["scaling_factors"][-100:]
        
        return stats
    
    def reset_plasticity(self):
        """Reset all plasticity mechanisms"""
        if self.stdp:
            self.stdp.reset()
        
        if self.homeostatic:
            self.homeostatic.reset()
        
        if self.metaplasticity:
            self.metaplasticity.reset()
        
        if self.structural:
            self.structural.reset()
        
        # Clear statistics
        for key in self.plasticity_stats:
            self.plasticity_stats[key].clear()
        
        self.update_count = 0
        logger.info("Plasticity mechanisms reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get plasticity manager status"""
        return {
            "is_initialized": self.is_initialized,
            "config": self.config.__dict__,
            "enabled_mechanisms": {
                "stdp": self.stdp is not None,
                "homeostatic": self.homeostatic is not None,
                "metaplasticity": self.metaplasticity is not None,
                "structural": self.structural is not None
            },
            "statistics": self.get_plasticity_statistics()
        }
    
    def health_check(self) -> bool:
        """Check plasticity manager health"""
        try:
            return self.is_initialized and self.update_count >= 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown plasticity manager"""
        try:
            # Reset all mechanisms
            self.reset_plasticity()
            
            self.is_initialized = False
            logger.info("Synaptic plasticity manager shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Plasticity manager shutdown failed: {e}")
            return False