"""
Neural Node Implementation
Represents individual nodes in the Neural Link Network
"""

import time
import uuid
import json
import numpy as np
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

class NodeType(str, Enum):
    CONCEPT = "concept"
    ENTITY = "entity" 
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    PATTERN = "pattern"
    MEMORY = "memory"
    PROCESSING = "processing"

class ActivationFunction(str, Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LINEAR = "linear"

@dataclass
class NodeState:
    """Current state of a neural node"""
    activation_level: float = 0.0
    last_activation: float = 0.0
    total_activations: int = 0
    connection_strength: float = 1.0
    learning_rate: float = 0.01

class NeuralNode:
    """
    Neural Node in the NLN system
    Represents concepts, entities, algorithms, and other cognitive elements
    """
    
    def __init__(
        self,
        node_id: str = None,
        node_type: NodeType = NodeType.CONCEPT,
        name: str = "",
        content: Dict[str, Any] = None,
        activation_function: ActivationFunction = ActivationFunction.RELU,
        threshold: float = 0.5,
        learning_enabled: bool = True
    ):
        self.id = node_id or str(uuid.uuid4())
        self.node_type = node_type
        self.name = name
        self.content = content or {}
        self.activation_function = activation_function
        self.threshold = threshold
        self.learning_enabled = learning_enabled
        
        # Node state
        self.state = NodeState()
        
        # Connections
        self.input_connections = {}  # node_id -> weight
        self.output_connections = {}  # node_id -> weight
        
        # Memory and learning
        self.activation_history = []
        self.weight_updates = []
        self.metadata = {
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1
        }
        
        # Specialized attributes based on node type
        self._initialize_specialized_attributes()
    
    def _initialize_specialized_attributes(self):
        """Initialize attributes specific to node type"""
        if self.node_type == NodeType.CONCEPT:
            self.content.setdefault("semantic_features", {})
            self.content.setdefault("abstraction_level", 0.5)
            
        elif self.node_type == NodeType.ALGORITHM:
            self.content.setdefault("complexity", "O(n)")
            self.content.setdefault("parameters", {})
            self.content.setdefault("implementation", "")
            
        elif self.node_type == NodeType.DATA_STRUCTURE:
            self.content.setdefault("operations", [])
            self.content.setdefault("properties", {})
            self.content.setdefault("capacity", "dynamic")
            
        elif self.node_type == NodeType.MEMORY:
            self.content.setdefault("capacity", 1000)
            self.content.setdefault("retention_time", 3600)
            self.content.setdefault("access_frequency", 0)
    
    def activate(self, input_signal: float = 1.0, source_node: str = None) -> float:
        """Activate the node with given input signal"""
        try:
            # Record input
            if source_node:
                if source_node not in self.input_connections:
                    self.input_connections[source_node] = 0.1  # Default weak connection
                input_signal *= self.input_connections[source_node]
            
            # Apply activation function
            activation = self._apply_activation_function(input_signal)
            
            # Check threshold
            if activation >= self.threshold:
                self.state.activation_level = activation
                self.state.last_activation = time.time()
                self.state.total_activations += 1
                
                # Store in history (keep last 100)
                self.activation_history.append({
                    "timestamp": time.time(),
                    "activation": activation,
                    "input_signal": input_signal,
                    "source": source_node
                })
                if len(self.activation_history) > 100:
                    self.activation_history.pop(0)
                
                # Learning update
                if self.learning_enabled:
                    self._update_weights(activation, source_node)
                
                return activation
            
            return 0.0
            
        except Exception as e:
            print(f"Node activation failed: {e}")
            return 0.0
    
    def _apply_activation_function(self, x: float) -> float:
        """Apply the node's activation function"""
        if self.activation_function == ActivationFunction.RELU:
            return max(0, x)
        elif self.activation_function == ActivationFunction.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == ActivationFunction.TANH:
            return np.tanh(x)
        elif self.activation_function == ActivationFunction.SOFTMAX:
            return np.exp(x) / (1 + np.exp(x))  # Simplified for single value
        else:  # LINEAR
            return x
    
    def _update_weights(self, activation: float, source_node: str = None):
        """Update connection weights based on activation"""
        if not self.learning_enabled:
            return
        
        try:
            # Hebbian learning: strengthen connections that fire together
            learning_rate = self.state.learning_rate
            
            if source_node and source_node in self.input_connections:
                old_weight = self.input_connections[source_node]
                # Weight update based on activation correlation
                weight_delta = learning_rate * activation * self.state.activation_level
                new_weight = old_weight + weight_delta
                
                # Clip weights to reasonable range
                new_weight = max(-2.0, min(2.0, new_weight))
                self.input_connections[source_node] = new_weight
                
                # Record weight update
                self.weight_updates.append({
                    "timestamp": time.time(),
                    "source_node": source_node,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "delta": weight_delta
                })
                
                # Keep only recent updates
                if len(self.weight_updates) > 50:
                    self.weight_updates.pop(0)
            
            # Decay unused connections
            self._decay_unused_connections()
            
        except Exception as e:
            print(f"Weight update failed: {e}")
    
    def _decay_unused_connections(self):
        """Decay weights of unused connections"""
        current_time = time.time()
        decay_rate = 0.001
        
        for node_id, weight in list(self.input_connections.items()):
            # Find last activation from this node
            last_used = 0
            for activation in reversed(self.activation_history):
                if activation.get("source") == node_id:
                    last_used = activation["timestamp"]
                    break
            
            # Decay if not used recently (more than 1 hour)
            if current_time - last_used > 3600:
                new_weight = weight * (1 - decay_rate)
                if abs(new_weight) < 0.01:  # Remove very weak connections
                    del self.input_connections[node_id]
                else:
                    self.input_connections[node_id] = new_weight
    
    def connect_to(self, target_node: 'NeuralNode', weight: float = 0.5):
        """Create connection to another node"""
        self.output_connections[target_node.id] = weight
        target_node.input_connections[self.id] = weight
    
    def disconnect_from(self, target_node: 'NeuralNode'):
        """Remove connection to another node"""
        if target_node.id in self.output_connections:
            del self.output_connections[target_node.id]
        if self.id in target_node.input_connections:
            del target_node.input_connections[self.id]
    
    def get_activation_strength(self) -> float:
        """Get current activation strength"""
        return self.state.activation_level
    
    def get_connection_weight(self, node_id: str) -> float:
        """Get connection weight to specific node"""
        return self.output_connections.get(node_id, 0.0)
    
    def get_input_nodes(self) -> List[str]:
        """Get list of input node IDs"""
        return list(self.input_connections.keys())
    
    def get_output_nodes(self) -> List[str]:
        """Get list of output node IDs"""
        return list(self.output_connections.keys())
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get comprehensive node information"""
        return {
            "id": self.id,
            "type": self.node_type.value,
            "name": self.name,
            "content": self.content,
            "state": asdict(self.state),
            "input_connections": len(self.input_connections),
            "output_connections": len(self.output_connections),
            "activation_history_length": len(self.activation_history),
            "metadata": self.metadata
        }
    
    def update_content(self, new_content: Dict[str, Any]):
        """Update node content"""
        self.content.update(new_content)
        self.metadata["updated_at"] = time.time()
        self.metadata["version"] += 1
    
    def set_learning_rate(self, rate: float):
        """Set learning rate for this node"""
        self.state.learning_rate = max(0.0, min(1.0, rate))
    
    def reset_activation(self):
        """Reset node activation state"""
        self.state.activation_level = 0.0
        self.state.last_activation = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics"""
        recent_activations = [
            act for act in self.activation_history 
            if time.time() - act["timestamp"] < 3600  # Last hour
        ]
        
        return {
            "total_activations": self.state.total_activations,
            "recent_activations": len(recent_activations),
            "average_activation": (
                sum(act["activation"] for act in recent_activations) / 
                max(len(recent_activations), 1)
            ),
            "connection_count": len(self.input_connections) + len(self.output_connections),
            "average_weight": (
                sum(self.input_connections.values()) / max(len(self.input_connections), 1)
            ),
            "last_activation_time": self.state.last_activation,
            "learning_enabled": self.learning_enabled
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization"""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "content": self.content,
            "activation_function": self.activation_function.value,
            "threshold": self.threshold,
            "learning_enabled": self.learning_enabled,
            "state": asdict(self.state),
            "input_connections": self.input_connections,
            "output_connections": self.output_connections,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralNode':
        """Create node from dictionary"""
        node = cls(
            node_id=data["id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            content=data["content"],
            activation_function=ActivationFunction(data["activation_function"]),
            threshold=data["threshold"],
            learning_enabled=data["learning_enabled"]
        )
        
        # Restore state
        node.state = NodeState(**data["state"])
        node.input_connections = data["input_connections"]
        node.output_connections = data["output_connections"]
        node.metadata = data["metadata"]
        
        return node
    
    def __str__(self) -> str:
        return f"NeuralNode(id={self.id}, type={self.node_type.value}, name='{self.name}')"
    
    def __repr__(self) -> str:
        return self.__str__()