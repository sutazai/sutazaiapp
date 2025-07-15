"""
Neural Link Network Core
Core neural network implementation with advanced modeling
"""

import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime
from .neural_node import NeuralNode
from .neural_link import NeuralLink
from .neural_synapse import NeuralSynapse

class NeuralLinkNetwork:
    """Advanced neural network with biological modeling"""
    
    def __init__(self):
        self.nodes: Dict[str, NeuralNode] = {}
        self.links: Dict[str, NeuralLink] = {}
        self.synapses: Dict[str, NeuralSynapse] = {}
        self.created_at = datetime.now()
        self.simulation_time = 0.0
        self.global_activity = 0.0
        
    def add_node(self, node: NeuralNode):
        """Add a neural node to the network"""
        self.nodes[node.node_id] = node
    
    def add_link(self, link: NeuralLink):
        """Add a neural link to the network"""
        link_id = f"{link.source_id}_{link.target_id}"
        self.links[link_id] = link
        
        # Update node connections
        if link.source_id in self.nodes:
            self.nodes[link.source_id].output_connections.append(link_id)
        if link.target_id in self.nodes:
            self.nodes[link.target_id].input_connections.append(link_id)
    
    def add_synapse(self, synapse: NeuralSynapse):
        """Add a neural synapse to the network"""
        synapse_id = f"{synapse.pre_synaptic_id}_{synapse.post_synaptic_id}"
        self.synapses[synapse_id] = synapse
    
    def process_input(self, input_data: List[float]) -> List[float]:
        """Process input through the neural network"""
        if not input_data:
            return []
        
        # Get input nodes
        input_nodes = [node for node in self.nodes.values() 
                      if node.node_type == "input"]
        
        if not input_nodes:
            return []
        
        # Apply input to input nodes
        for i, node in enumerate(input_nodes):
            if i < len(input_data):
                node.process_input(input_data[i])
        
        # Propagate through network
        self._propagate_signals()
        
        # Get output from output nodes
        output_nodes = [node for node in self.nodes.values() 
                       if node.node_type == "output"]
        
        output = [node.activity for node in output_nodes]
        
        # Update global activity
        self._update_global_activity()
        
        return output
    
    def _propagate_signals(self):
        """Propagate signals through the network"""
        # Multiple propagation steps for complex networks
        for step in range(3):
            node_inputs = defaultdict(float)
            
            # Process all links
            for link_id, link in self.links.items():
                if link.source_id in self.nodes and link.target_id in self.nodes:
                    source_node = self.nodes[link.source_id]
                    
                    # Transmit signal
                    signal = link.transmit(source_node.activity)
                    node_inputs[link.target_id] += signal
                    
                    # Update link weights
                    target_node = self.nodes[link.target_id]
                    link.update_weight(source_node.activity, target_node.activity)
            
            # Process synapses
            for synapse_id, synapse in self.synapses.items():
                if synapse.pre_synaptic_id in self.nodes and synapse.post_synaptic_id in self.nodes:
                    pre_node = self.nodes[synapse.pre_synaptic_id]
                    post_node = self.nodes[synapse.post_synaptic_id]
                    
                    # Process synaptic transmission
                    current = synapse.process_transmission(pre_node.activity, post_node.activity)
                    node_inputs[synapse.post_synaptic_id] += current
            
            # Update node activities
            for node_id, input_sum in node_inputs.items():
                if node_id in self.nodes:
                    self.nodes[node_id].process_input(input_sum)
    
    def _update_global_activity(self):
        """Update global network activity"""
        if not self.nodes:
            self.global_activity = 0.0
            return
        
        total_activity = sum(node.activity for node in self.nodes.values())
        self.global_activity = total_activity / len(self.nodes)
    
    def get_global_activity(self) -> float:
        """Get global network activity level"""
        return self.global_activity
    
    def apply_learning_rules(self, learning_factor: float):
        """Apply learning rules to the network"""
        # Update all links
        for link in self.links.values():
            link.learning_rate = learning_factor
        
        # Update all synapses
        for synapse in self.synapses.values():
            if hasattr(synapse, 'learning_rate'):
                synapse.learning_rate = learning_factor
        
        # Adapt node thresholds
        for node in self.nodes.values():
            node.adapt_threshold(0.5)  # Target activity level
    
    def get_network_state(self) -> Dict[str, Any]:
        """Get complete network state"""
        return {
            "total_nodes": len(self.nodes),
            "total_links": len(self.links),
            "total_synapses": len(self.synapses),
            "global_activity": self.global_activity,
            "simulation_time": self.simulation_time,
            "created_at": self.created_at.isoformat(),
            "nodes": {node_id: node.get_state() for node_id, node in self.nodes.items()},
            "links": {link_id: link.get_state() for link_id, link in self.links.items()},
            "synapses": {synapse_id: synapse.get_state() for synapse_id, synapse in self.synapses.items()}
        }
    
    def reset_network(self):
        """Reset network to initial state"""
        for node in self.nodes.values():
            node.activity = 0.0
            node.firing_history = []
        
        for link in self.links.values():
            link.activation_history = []
        
        for synapse in self.synapses.values():
            synapse.vesicle_count = 100
            synapse.release_history = []
        
        self.global_activity = 0.0
        self.simulation_time = 0.0