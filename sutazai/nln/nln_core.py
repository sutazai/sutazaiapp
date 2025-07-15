"""
Neural Link Network (NLN) Core System
Main neural network system integrating nodes, links, and synapses
"""

import asyncio
import logging
import json
import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import networkx as nx

from .neural_node import NeuralNode, NodeType
from .neural_link import NeuralLink, LinkType
from .neural_synapse import NeuralSynapse, SynapseType

logger = logging.getLogger(__name__)

@dataclass
class NetworkState:
    """Current state of the neural network"""
    total_nodes: int = 0
    total_links: int = 0
    total_synapses: int = 0
    global_activity: float = 0.0
    network_coherence: float = 0.0
    learning_rate: float = 0.01
    simulation_time: float = 0.0

@dataclass
class ActivationPattern:
    """Pattern of network activation"""
    pattern_id: str
    timestamp: float
    active_nodes: List[str]
    activation_strengths: Dict[str, float]
    pattern_signature: str
    coherence_score: float

class NeuralLinkNetwork:
    """
    Advanced Neural Link Network System
    Manages complex neural networks with nodes, links, and synapses
    """
    
    # Hardcoded authorization
    AUTHORIZED_USER = "os.getenv("ADMIN_EMAIL", "admin@localhost")"
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/nln"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Network components
        self.nodes = {}  # node_id -> NeuralNode
        self.links = {}  # link_id -> NeuralLink
        self.synapses = {}  # synapse_id -> NeuralSynapse
        
        # Network structure
        self.graph = nx.MultiDiGraph()
        self.node_layers = defaultdict(set)  # layer -> node_ids
        self.connection_matrix = {}  # For fast lookup
        
        # Network state
        self.state = NetworkState()
        self.activation_patterns = []
        self.learning_history = []
        
        # Simulation
        self.simulation_running = False
        self.simulation_step = 0
        self.time_step = 0.01  # 10ms
        
        # Event handling
        self.event_handlers = {
            "node_activated": [],
            "pattern_detected": [],
            "learning_event": [],
            "network_synchronized": []
        }
        
        # Performance metrics
        self.metrics = {
            "total_activations": 0,
            "patterns_detected": 0,
            "learning_events": 0,
            "network_efficiency": 0.0,
            "synchronization_level": 0.0
        }
        
        # Initialize
        self._load_existing_network()
        self._initialize_default_structures()
        
        logger.info("✅ Neural Link Network initialized")
    
    def _load_existing_network(self):
        """Load existing network from disk"""
        try:
            # Load nodes
            nodes_file = self.data_dir / "nodes.json"
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = NeuralNode.from_dict(node_data)
                        self.nodes[node.id] = node
                        self.graph.add_node(node.id, **node.get_node_info())
            
            # Load links
            links_file = self.data_dir / "links.json"
            if links_file.exists():
                with open(links_file, 'r') as f:
                    data = json.load(f)
                    for link_data in data.get("links", []):
                        link = NeuralLink.from_dict(link_data)
                        self.links[link.id] = link
                        self.graph.add_edge(
                            link.source_node_id,
                            link.target_node_id,
                            key=link.id,
                            **link.get_link_info()
                        )
            
            # Load synapses
            synapses_file = self.data_dir / "synapses.json"
            if synapses_file.exists():
                with open(synapses_file, 'r') as f:
                    data = json.load(f)
                    for synapse_data in data.get("synapses", []):
                        synapse = NeuralSynapse.from_dict(synapse_data)
                        self.synapses[synapse.id] = synapse
            
            self._update_network_state()
            logger.info(f"✅ Loaded network: {len(self.nodes)} nodes, {len(self.links)} links, {len(self.synapses)} synapses")
            
        except Exception as e:
            logger.error(f"Failed to load network: {e}")
    
    def _initialize_default_structures(self):
        """Initialize default network structures if empty"""
        if not self.nodes:
            # Create basic network structure
            self._create_default_network()
    
    def _create_default_network(self):
        """Create a basic default network for demonstration"""
        try:
            # Create concept nodes
            concept_nodes = [
                ("memory", NodeType.MEMORY, "Working Memory"),
                ("attention", NodeType.PROCESSING, "Attention Control"),
                ("learning", NodeType.ALGORITHM, "Learning System"),
                ("pattern", NodeType.PATTERN, "Pattern Recognition"),
                ("reasoning", NodeType.PROCESSING, "Reasoning Engine")
            ]
            
            node_ids = {}
            for node_key, node_type, name in concept_nodes:
                node_id = self.create_node(
                    node_type=node_type,
                    name=name,
                    content={
                        "description": f"Default {name} component",
                        "functionality": node_key,
                        "default_component": True
                    }
                )
                node_ids[node_key] = node_id
            
            # Create connections between nodes
            connections = [
                ("attention", "memory", 0.8, LinkType.EXCITATORY),
                ("memory", "pattern", 0.7, LinkType.EXCITATORY),
                ("pattern", "learning", 0.9, LinkType.EXCITATORY),
                ("learning", "reasoning", 0.6, LinkType.MODULATORY),
                ("reasoning", "attention", 0.5, LinkType.MODULATORY)
            ]
            
            for source, target, weight, link_type in connections:
                if source in node_ids and target in node_ids:
                    self.create_link(
                        source_node_id=node_ids[source],
                        target_node_id=node_ids[target],
                        weight=weight,
                        link_type=link_type
                    )
            
            # Create synapses for enhanced connectivity
            for source, target, _, _ in connections:
                if source in node_ids and target in node_ids:
                    self.create_synapse(
                        presynaptic_node_id=node_ids[source],
                        postsynaptic_node_id=node_ids[target],
                        synapse_type=SynapseType.CHEMICAL,
                        base_strength=0.8
                    )
            
            logger.info("✅ Created default network structure")
            
        except Exception as e:
            logger.error(f"Failed to create default network: {e}")
    
    def create_node(
        self,
        node_type: NodeType = NodeType.CONCEPT,
        name: str = "",
        content: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """Create a new neural node"""
        try:
            node = NeuralNode(
                node_type=node_type,
                name=name or f"{node_type.value}_{len(self.nodes)+1}",
                content=content or {},
                **kwargs
            )
            
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **node.get_node_info())
            
            # Organize into layers based on type
            layer = self._determine_node_layer(node_type)
            self.node_layers[layer].add(node.id)
            
            self._update_network_state()
            
            logger.info(f"✅ Created node: {node.id} ({node_type.value})")
            return node.id
            
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise
    
    def create_link(
        self,
        source_node_id: str,
        target_node_id: str,
        weight: float = 0.5,
        link_type: LinkType = LinkType.EXCITATORY,
        **kwargs
    ) -> str:
        """Create a neural link between nodes"""
        try:
            if source_node_id not in self.nodes or target_node_id not in self.nodes:
                raise ValueError("Source or target node does not exist")
            
            link = NeuralLink(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                weight=weight,
                link_type=link_type,
                **kwargs
            )
            
            self.links[link.id] = link
            self.graph.add_edge(
                source_node_id,
                target_node_id,
                key=link.id,
                **link.get_link_info()
            )
            
            # Update node connections
            source_node = self.nodes[source_node_id]
            target_node = self.nodes[target_node_id]
            source_node.connect_to(target_node, weight)
            
            self._update_connection_matrix()
            self._update_network_state()
            
            logger.info(f"✅ Created link: {source_node_id} -> {target_node_id}")
            return link.id
            
        except Exception as e:
            logger.error(f"Failed to create link: {e}")
            raise
    
    def create_synapse(
        self,
        presynaptic_node_id: str,
        postsynaptic_node_id: str,
        synapse_type: SynapseType = SynapseType.CHEMICAL,
        base_strength: float = 1.0,
        **kwargs
    ) -> str:
        """Create a neural synapse between nodes"""
        try:
            if presynaptic_node_id not in self.nodes or postsynaptic_node_id not in self.nodes:
                raise ValueError("Presynaptic or postsynaptic node does not exist")
            
            synapse = NeuralSynapse(
                presynaptic_node_id=presynaptic_node_id,
                postsynaptic_node_id=postsynaptic_node_id,
                synapse_type=synapse_type,
                base_strength=base_strength,
                **kwargs
            )
            
            self.synapses[synapse.id] = synapse
            self._update_network_state()
            
            logger.info(f"✅ Created synapse: {presynaptic_node_id} -> {postsynaptic_node_id}")
            return synapse.id
            
        except Exception as e:
            logger.error(f"Failed to create synapse: {e}")
            raise
    
    def _determine_node_layer(self, node_type: NodeType) -> int:
        """Determine which layer a node belongs to"""
        layer_mapping = {
            NodeType.MEMORY: 0,      # Input/memory layer
            NodeType.PROCESSING: 1,  # Processing layer
            NodeType.CONCEPT: 2,     # Concept layer
            NodeType.PATTERN: 3,     # Pattern layer
            NodeType.ALGORITHM: 4,   # Algorithm layer
            NodeType.ENTITY: 2,      # Same as concept
            NodeType.DATA_STRUCTURE: 4  # Same as algorithm
        }
        return layer_mapping.get(node_type, 2)
    
    def _update_connection_matrix(self):
        """Update connection matrix for fast lookups"""
        self.connection_matrix = {}
        for node_id in self.nodes:
            self.connection_matrix[node_id] = {
                "incoming": [],
                "outgoing": []
            }
        
        for link in self.links.values():
            source = link.source_node_id
            target = link.target_node_id
            
            self.connection_matrix[source]["outgoing"].append({
                "target": target,
                "link_id": link.id,
                "weight": link.weight
            })
            
            self.connection_matrix[target]["incoming"].append({
                "source": source,
                "link_id": link.id,
                "weight": link.weight
            })
    
    def _update_network_state(self):
        """Update network state metrics"""
        self.state.total_nodes = len(self.nodes)
        self.state.total_links = len(self.links)
        self.state.total_synapses = len(self.synapses)
        
        # Calculate global activity
        if self.nodes:
            total_activation = sum(node.get_activation_strength() for node in self.nodes.values())
            self.state.global_activity = total_activation / len(self.nodes)
        
        # Calculate network coherence
        self.state.network_coherence = self._calculate_network_coherence()
    
    def _calculate_network_coherence(self) -> float:
        """Calculate network coherence based on connectivity"""
        if len(self.nodes) < 2:
            return 0.0
        
        # Use graph connectivity metrics
        try:
            if nx.is_strongly_connected(self.graph):
                coherence = 1.0
            elif nx.is_connected(self.graph.to_undirected()):
                coherence = 0.7
            else:
                # Calculate largest connected component
                largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
                coherence = len(largest_cc) / len(self.nodes)
            
            return coherence
            
        except Exception:
            return 0.0
    
    async def activate_node(
        self,
        node_id: str,
        input_signal: float = 1.0,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Activate a specific node and propagate signals"""
        try:
            if node_id not in self.nodes:
                return {"success": False, "error": "Node not found"}
            
            node = self.nodes[node_id]
            context = context or {}
            
            # Activate the node
            activation = node.activate(input_signal)
            
            if activation > 0:
                # Record activation
                self.metrics["total_activations"] += 1
                
                # Trigger event handlers
                for handler in self.event_handlers.get("node_activated", []):
                    try:
                        await handler(node_id, activation, context)
                    except:
                        pass
                
                # Propagate signal to connected nodes
                propagation_results = await self._propagate_activation(
                    node_id, activation, context
                )
                
                # Detect patterns
                await self._detect_activation_patterns()
                
                return {
                    "success": True,
                    "activation": activation,
                    "propagation": propagation_results,
                    "patterns_detected": len(self.activation_patterns)
                }
            
            return {"success": True, "activation": 0.0, "message": "Below threshold"}
            
        except Exception as e:
            logger.error(f"Node activation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _propagate_activation(
        self,
        source_node_id: str,
        activation: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Propagate activation through network connections"""
        propagation_results = {"activated_nodes": [], "total_signal": 0.0}
        
        try:
            # Get outgoing connections
            connections = self.connection_matrix.get(source_node_id, {}).get("outgoing", [])
            
            for connection in connections:
                target_id = connection["target"]
                link_id = connection["link_id"]
                
                # Get link and transmit signal
                if link_id in self.links:
                    link = self.links[link_id]
                    transmitted_signal = link.transmit_signal(
                        activation,
                        activation,
                        context
                    )
                    
                    # Also use synapse if available
                    synapse_signal = 0.0
                    for synapse in self.synapses.values():
                        if (synapse.presynaptic_node_id == source_node_id and
                            synapse.postsynaptic_node_id == target_id):
                            
                            synapse_result = synapse.transmit(
                                transmitted_signal,
                                context=context
                            )
                            synapse_signal = synapse_result.get("signal", 0.0)
                            break
                    
                    # Use stronger signal
                    final_signal = max(transmitted_signal, synapse_signal)
                    
                    if final_signal > 0.1:  # Threshold for propagation
                        # Activate target node
                        target_node = self.nodes[target_id]
                        target_activation = target_node.activate(final_signal, source_node_id)
                        
                        if target_activation > 0:
                            propagation_results["activated_nodes"].append({
                                "node_id": target_id,
                                "activation": target_activation,
                                "signal": final_signal
                            })
                            propagation_results["total_signal"] += final_signal
            
            return propagation_results
            
        except Exception as e:
            logger.error(f"Activation propagation failed: {e}")
            return propagation_results
    
    async def _detect_activation_patterns(self):
        """Detect and record activation patterns"""
        try:
            current_time = time.time()
            
            # Get currently active nodes
            active_nodes = []
            activation_strengths = {}
            
            for node_id, node in self.nodes.items():
                activation = node.get_activation_strength()
                if activation > 0.1:  # Activity threshold
                    active_nodes.append(node_id)
                    activation_strengths[node_id] = activation
            
            if len(active_nodes) >= 2:  # Need at least 2 nodes for pattern
                # Create pattern signature
                pattern_signature = "_".join(sorted(active_nodes))
                
                # Calculate coherence (how synchronized the activations are)
                if len(activation_strengths) > 1:
                    activations = list(activation_strengths.values())
                    coherence = 1.0 - (np.std(activations) / max(np.mean(activations), 0.1))
                else:
                    coherence = 1.0
                
                # Create pattern record
                pattern = ActivationPattern(
                    pattern_id=str(uuid.uuid4()),
                    timestamp=current_time,
                    active_nodes=active_nodes,
                    activation_strengths=activation_strengths,
                    pattern_signature=pattern_signature,
                    coherence_score=coherence
                )
                
                self.activation_patterns.append(pattern)
                
                # Keep only recent patterns
                if len(self.activation_patterns) > 1000:
                    self.activation_patterns.pop(0)
                
                self.metrics["patterns_detected"] += 1
                
                # Trigger pattern detection event
                for handler in self.event_handlers.get("pattern_detected", []):
                    try:
                        await handler(pattern)
                    except:
                        pass
                
                logger.info(f"🔍 Pattern detected: {len(active_nodes)} nodes, coherence: {coherence:.3f}")
                
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
    
    async def run_network_simulation(self, duration: float = 10.0, external_inputs: Dict[str, float] = None):
        """Run network simulation with optional external inputs"""
        try:
            self.simulation_running = True
            self.simulation_step = 0
            start_time = time.time()
            
            external_inputs = external_inputs or {}
            
            logger.info(f"🔄 Starting network simulation for {duration} seconds")
            
            while self.simulation_running and (time.time() - start_time) < duration:
                # Apply external inputs
                for node_id, input_signal in external_inputs.items():
                    if node_id in self.nodes:
                        await self.activate_node(node_id, input_signal)
                
                # Random background activity
                if np.random.random() < 0.1:  # 10% chance each step
                    random_node = np.random.choice(list(self.nodes.keys()))
                    random_signal = np.random.uniform(0.1, 0.5)
                    await self.activate_node(random_node, random_signal)
                
                # Update simulation state
                self.simulation_step += 1
                self.state.simulation_time = time.time() - start_time
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(self.time_step)
            
            self.simulation_running = False
            
            # Generate simulation report
            report = {
                "duration": self.state.simulation_time,
                "steps": self.simulation_step,
                "total_activations": self.metrics["total_activations"],
                "patterns_detected": len(self.activation_patterns),
                "final_network_state": self.get_network_status()
            }
            
            logger.info(f"✅ Simulation completed: {self.simulation_step} steps, {len(self.activation_patterns)} patterns")
            return report
            
        except Exception as e:
            logger.error(f"Network simulation failed: {e}")
            self.simulation_running = False
            return {"error": str(e)}
    
    def stop_simulation(self):
        """Stop running simulation"""
        self.simulation_running = False
    
    def get_node(self, node_id: str) -> Optional[NeuralNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_link(self, link_id: str) -> Optional[NeuralLink]:
        """Get link by ID"""
        return self.links.get(link_id)
    
    def get_synapse(self, synapse_id: str) -> Optional[NeuralSynapse]:
        """Get synapse by ID"""
        return self.synapses.get(synapse_id)
    
    def find_path(self, source_node_id: str, target_node_id: str) -> List[str]:
        """Find path between nodes"""
        try:
            if source_node_id in self.nodes and target_node_id in self.nodes:
                return nx.shortest_path(self.graph, source_node_id, target_node_id)
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Path finding failed: {e}")
        return []
    
    def get_node_neighbors(self, node_id: str) -> Dict[str, List[str]]:
        """Get neighboring nodes"""
        if node_id not in self.nodes:
            return {"incoming": [], "outgoing": []}
        
        return {
            "incoming": [edge[0] for edge in self.graph.in_edges(node_id)],
            "outgoing": [edge[1] for edge in self.graph.out_edges(node_id)]
        }
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status"""
        try:
            # Calculate additional metrics
            node_types = defaultdict(int)
            link_types = defaultdict(int)
            synapse_types = defaultdict(int)
            
            for node in self.nodes.values():
                node_types[node.node_type.value] += 1
            
            for link in self.links.values():
                link_types[link.link_type.value] += 1
            
            for synapse in self.synapses.values():
                synapse_types[synapse.synapse_type.value] += 1
            
            # Network topology metrics
            topology_metrics = {}
            if self.nodes:
                try:
                    topology_metrics = {
                        "density": nx.density(self.graph),
                        "average_clustering": nx.average_clustering(self.graph.to_undirected()),
                        "connected_components": nx.number_weakly_connected_components(self.graph)
                    }
                except:
                    topology_metrics = {"density": 0.0, "average_clustering": 0.0, "connected_components": 0}
            
            return {
                "network_state": asdict(self.state),
                "node_distribution": dict(node_types),
                "link_distribution": dict(link_types),
                "synapse_distribution": dict(synapse_types),
                "topology_metrics": topology_metrics,
                "performance_metrics": self.metrics.copy(),
                "recent_patterns": len([
                    p for p in self.activation_patterns
                    if time.time() - p.timestamp < 3600  # Last hour
                ]),
                "simulation_running": self.simulation_running,
                "layers": {
                    layer: len(nodes) for layer, nodes in self.node_layers.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get network status: {e}")
            return {"error": str(e)}
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    async def save_network(self):
        """Save network to disk"""
        try:
            # Save nodes
            nodes_data = {
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "nodes.json", 'w') as f:
                json.dump(nodes_data, f, indent=2, default=str)
            
            # Save links
            links_data = {
                "links": [link.to_dict() for link in self.links.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "links.json", 'w') as f:
                json.dump(links_data, f, indent=2, default=str)
            
            # Save synapses
            synapses_data = {
                "synapses": [synapse.to_dict() for synapse in self.synapses.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "synapses.json", 'w') as f:
                json.dump(synapses_data, f, indent=2, default=str)
            
            # Save network state and metrics
            state_data = {
                "state": asdict(self.state),
                "metrics": self.metrics,
                "activation_patterns": [asdict(p) for p in self.activation_patterns[-100:]],  # Last 100
                "saved_at": time.time()
            }
            with open(self.data_dir / "network_state.json", 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info("✅ Neural Link Network saved")
            
        except Exception as e:
            logger.error(f"Failed to save network: {e}")
    
    def cleanup(self):
        """Cleanup network resources"""
        try:
            # Stop simulation
            self.simulation_running = False
            
            # Save network
            asyncio.create_task(self.save_network())
            
            logger.info("✅ Neural Link Network cleaned up")
            
        except Exception as e:
            logger.error(f"Network cleanup failed: {e}")

# Global instance
neural_link_network = NeuralLinkNetwork()

# Convenience functions
def create_neural_node(node_type: NodeType, name: str = "", content: Dict[str, Any] = None, **kwargs) -> str:
    """Create neural node"""
    return neural_link_network.create_node(node_type, name, content, **kwargs)

def create_neural_link(source_node_id: str, target_node_id: str, weight: float = 0.5, link_type: LinkType = LinkType.EXCITATORY, **kwargs) -> str:
    """Create neural link"""
    return neural_link_network.create_link(source_node_id, target_node_id, weight, link_type, **kwargs)

def create_neural_synapse(presynaptic_node_id: str, postsynaptic_node_id: str, synapse_type: SynapseType = SynapseType.CHEMICAL, **kwargs) -> str:
    """Create neural synapse"""
    return neural_link_network.create_synapse(presynaptic_node_id, postsynaptic_node_id, synapse_type, **kwargs)

async def activate_neural_node(node_id: str, input_signal: float = 1.0, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Activate neural node"""
    return await neural_link_network.activate_node(node_id, input_signal, context)

async def run_neural_simulation(duration: float = 10.0, external_inputs: Dict[str, float] = None) -> Dict[str, Any]:
    """Run neural simulation"""
    return await neural_link_network.run_network_simulation(duration, external_inputs)

def get_network_status() -> Dict[str, Any]:
    """Get network status"""
    return neural_link_network.get_network_status()