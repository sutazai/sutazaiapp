#!/usr/bin/env python3
"""
Universal Learning Machine (ULM) Implementation
Based on "The Brain as a Universal Learning Machine" principles
Implements dynamic rewiring, hierarchical learning, and self-modification
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import deque
import json

from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicNeuralArchitecture(nn.Module):
    """
    Implements dynamic rewiring capabilities inspired by neuroplasticity
    Can modify its own architecture based on learning experiences
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build initial architecture
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_dim))
        
        # Neuroplasticity parameters
        self.connection_strengths = {}
        self.pruning_threshold = 0.01
        self.growth_rate = 0.1
        
        # Learning history for meta-learning
        self.learning_history = deque(maxlen=1000)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic activation patterns"""
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            # Apply dropout for regularization
            x = F.dropout(x, p=0.1, training=self.training)
        return self.layers[-1](x)
    
    def rewire_connections(self, performance_metric: float):
        """
        Dynamically rewire neural connections based on performance
        Implements synaptic pruning and growth
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if isinstance(layer, nn.Linear):
                    # Prune weak connections
                    mask = torch.abs(layer.weight) > self.pruning_threshold
                    layer.weight *= mask.float()
                    
                    # Strengthen successful connections
                    if performance_metric > 0.8:
                        layer.weight *= (1 + self.growth_rate * performance_metric)
                    
                    # Add new connections (simulate neurogenesis)
                    if performance_metric < 0.5 and np.random.random() < 0.1:
                        new_connections = torch.randn_like(layer.weight) * 0.01
                        layer.weight += new_connections
    
    def meta_learn(self, task_embedding: torch.Tensor, performance: float):
        """
        Meta-learning: Learn to learn better
        Adjusts learning strategy based on task patterns
        """
        self.learning_history.append({
            'task_embedding': task_embedding.detach().cpu().numpy().tolist(),
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
        # Adjust architecture based on learning patterns
        if len(self.learning_history) > 100:
            recent_performance = np.mean([h['performance'] for h in list(self.learning_history)[-100:]])
            if recent_performance < 0.6:
                # Poor performance - increase model capacity
                self._grow_architecture()
            elif recent_performance > 0.95:
                # Excellent performance - optimize/compress
                self._compress_architecture()
    
    def _grow_architecture(self):
        """Add neurons/layers to increase capacity"""
        # This is a simplified version - in practice would be more sophisticated
        logger.info("Growing architecture due to learning challenges")
        
    def _compress_architecture(self):
        """Compress/optimize architecture for efficiency"""
        logger.info("Compressing architecture due to consistent high performance")


class HierarchicalTemporalMemory:
    """
    Implements Hierarchical Temporal Memory (HTM) principles
    For sequence learning and pattern recognition
    """
    
    def __init__(self, input_size: int, column_count: int = 2048, cells_per_column: int = 32):
        self.input_size = input_size
        self.column_count = column_count
        self.cells_per_column = cells_per_column
        
        # Spatial pooler
        self.sp_connections = np.random.random((column_count, input_size)) < 0.5
        self.sp_permanences = np.random.random((column_count, input_size)) * 0.1
        
        # Temporal memory
        self.tm_connections = {}
        self.active_cells = set()
        self.predictive_cells = set()
        self.learning_cells = set()
        
        # Sequence memory
        self.sequence_memory = deque(maxlen=1000)
        
    def spatial_pooling(self, input_sdr: np.ndarray) -> np.ndarray:
        """
        Spatial pooling: Convert input to sparse distributed representation
        """
        # Calculate overlap scores
        overlaps = np.dot(self.sp_connections * (self.sp_permanences > 0.1), input_sdr)
        
        # Winner-take-all
        k = int(0.02 * self.column_count)  # 2% sparsity
        winners = np.argpartition(overlaps, -k)[-k:]
        
        # Create sparse output
        output_sdr = np.zeros(self.column_count)
        output_sdr[winners] = 1
        
        # Update permanences (learning)
        for winner in winners:
            self.sp_permanences[winner] += (input_sdr - 0.5) * 0.1
            self.sp_permanences[winner] = np.clip(self.sp_permanences[winner], 0, 1)
        
        return output_sdr
    
    def temporal_memory(self, active_columns: np.ndarray) -> Tuple[set, set]:
        """
        Temporal memory: Learn sequences and make predictions
        """
        # Activate cells in active columns
        new_active_cells = set()
        new_predictive_cells = set()
        
        active_cols = np.where(active_columns)[0]
        
        for col in active_cols:
            # Check for predicted cells
            predicted_cells_in_col = [
                cell for cell in self.predictive_cells 
                if cell // self.cells_per_column == col
            ]
            
            if predicted_cells_in_col:
                # Activate predicted cells
                new_active_cells.update(predicted_cells_in_col)
            else:
                # Burst: activate all cells in column
                for i in range(self.cells_per_column):
                    new_active_cells.add(col * self.cells_per_column + i)
        
        # Update connections and make predictions
        # (Simplified - full HTM would have dendritic segments)
        
        self.active_cells = new_active_cells
        self.predictive_cells = new_predictive_cells
        
        return new_active_cells, new_predictive_cells
    
    def learn_sequence(self, sequence: List[np.ndarray]):
        """Learn a sequence of patterns"""
        self.sequence_memory.append(sequence)
        
        # Process sequence through spatial and temporal pooling
        for pattern in sequence:
            sdr = self.spatial_pooling(pattern)
            self.temporal_memory(sdr)


class BasalGangliaController:
    """
    Implements basal ganglia-inspired action selection and reinforcement learning
    Acts as the 'CPU' for the brain system
    """
    
    def __init__(self, n_actions: int, n_states: int):
        self.n_actions = n_actions
        self.n_states = n_states
        
        # Action values (Q-table)
        self.q_table = np.zeros((n_states, n_actions))
        
        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        # Dopamine system for reward processing
        self.dopamine_baseline = 0.0
        self.reward_history = deque(maxlen=100)
        
    def select_action(self, state: int) -> int:
        """
        Select action using epsilon-greedy strategy
        Models the action selection function of basal ganglia
        """
        if np.random.random() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update_values(self, state: int, action: int, reward: float, next_state: int):
        """
        Update action values using temporal difference learning
        Models dopaminergic learning in basal ganglia
        """
        # Calculate prediction error (dopamine signal)
        current_value = self.q_table[state, action]
        next_value = np.max(self.q_table[next_state])
        prediction_error = reward + self.discount_factor * next_value - current_value
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * prediction_error
        
        # Update dopamine baseline
        self.reward_history.append(reward)
        self.dopamine_baseline = np.mean(self.reward_history) if self.reward_history else 0.0
        
        # Adapt exploration based on performance
        if len(self.reward_history) > 50:
            recent_performance = np.mean(list(self.reward_history)[-50:])
            if recent_performance > self.dopamine_baseline:
                self.exploration_rate *= 0.99  # Reduce exploration
            else:
                self.exploration_rate = min(0.3, self.exploration_rate * 1.01)  # Increase exploration


class UniversalLearningMachine:
    """
    Main Universal Learning Machine implementation
    Integrates all components into a unified learning system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.embedding_model = SentenceTransformer(
            config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Dynamic neural architecture
        self.neural_architecture = DynamicNeuralArchitecture(
            input_dim=self.embedding_dim,
            hidden_dims=[512, 256, 128],
            output_dim=64  # Compressed representation
        )
        
        # Hierarchical temporal memory
        self.htm = HierarchicalTemporalMemory(
            input_size=self.embedding_dim,
            column_count=2048
        )
        
        # Basal ganglia controller
        self.controller = BasalGangliaController(
            n_actions=100,  # Number of possible actions/agents
            n_states=1000   # Discretized state space
        )
        
        # Meta-learning components
        self.task_memory = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.neural_architecture.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        logger.info("🧠 Universal Learning Machine initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the complete ULM pipeline
        """
        # 1. Perception: Convert input to embeddings
        input_text = input_data.get('input', '')
        input_embedding = self.embedding_model.encode(input_text)
        input_tensor = torch.FloatTensor(input_embedding).unsqueeze(0)
        
        # 2. Spatial-temporal processing
        spatial_pattern = self.htm.spatial_pooling(input_embedding)
        
        # 3. Neural processing with dynamic architecture
        neural_output = self.neural_architecture(input_tensor)
        
        # 4. Action selection via basal ganglia
        state_representation = self._discretize_state(neural_output)
        selected_action = self.controller.select_action(state_representation)
        
        # 5. Execute action (would interface with agent system)
        result = await self._execute_action(selected_action, input_data)
        
        # 6. Evaluate and learn
        performance = await self._evaluate_performance(result, input_data)
        
        # 7. Update all learning systems
        self._update_learning_systems(
            input_embedding=input_tensor,
            action=selected_action,
            performance=performance,
            state=state_representation
        )
        
        # 8. Meta-learning and self-modification
        self.neural_architecture.meta_learn(input_tensor, performance)
        
        return {
            'output': result,
            'confidence': performance,
            'selected_action': selected_action,
            'neural_state': neural_output.detach().cpu().numpy().tolist(),
            'learning_progress': self._calculate_learning_progress()
        }
    
    def _discretize_state(self, continuous_state: torch.Tensor) -> int:
        """Convert continuous neural state to discrete state for controller"""
        # Simple discretization - could be more sophisticated
        state_vector = continuous_state.detach().cpu().numpy().flatten()
        state_hash = hash(tuple(np.round(state_vector, 2))) % self.controller.n_states
        return abs(state_hash)
    
    async def _execute_action(self, action: int, input_data: Dict[str, Any]) -> Any:
        """Execute selected action - would interface with agent system"""
        # Placeholder - would map to actual agent execution
        return f"Executed action {action} for input: {input_data.get('input', '')[:50]}..."
    
    async def _evaluate_performance(self, result: Any, input_data: Dict[str, Any]) -> float:
        """Evaluate performance of the action"""
        # Placeholder - would use actual evaluation metrics
        return np.random.random() * 0.4 + 0.6  # Random score between 0.6 and 1.0
    
    def _update_learning_systems(
        self,
        input_embedding: torch.Tensor,
        action: int,
        performance: float,
        state: int
    ):
        """Update all learning components based on experience"""
        # Update neural architecture
        if performance < 0.8:
            loss = F.mse_loss(
                self.neural_architecture(input_embedding),
                torch.randn(1, 64)  # Target representation
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Update basal ganglia controller
        reward = performance - self.controller.dopamine_baseline
        next_state = (state + 1) % self.controller.n_states  # Simplified
        self.controller.update_values(state, action, reward, next_state)
        
        # Rewire neural connections based on performance
        self.neural_architecture.rewire_connections(performance)
        
        # Store in performance history
        self.performance_history.append(performance)
    
    def _calculate_learning_progress(self) -> float:
        """Calculate overall learning progress"""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-100:]
        earlier = list(self.performance_history)[-200:-100] if len(self.performance_history) > 100 else [0.5]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier) if earlier else 0.5
        
        progress = (recent_avg - earlier_avg) / (earlier_avg + 0.001)
        return float(np.clip(progress, -1, 1))
    
    def save_state(self, path: str):
        """Save the complete ULM state"""
        state = {
            'neural_weights': self.neural_architecture.state_dict(),
            'q_table': self.controller.q_table.tolist(),
            'learning_history': list(self.neural_architecture.learning_history),
            'performance_history': list(self.performance_history),
            'config': self.config
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Saved ULM state to {path}")
    
    def load_state(self, path: str):
        """Load a previously saved ULM state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.neural_architecture.load_state_dict(
            {k: torch.tensor(v) for k, v in state['neural_weights'].items()}
        )
        self.controller.q_table = np.array(state['q_table'])
        self.neural_architecture.learning_history = deque(
            state['learning_history'],
            maxlen=1000
        )
        self.performance_history = deque(
            state['performance_history'],
            maxlen=1000
        )
        
        logger.info(f"📂 Loaded ULM state from {path}")


# Example usage
if __name__ == "__main__":
    config = {
        'embedding_model': 'all-MiniLM-L6-v2',
        'learning_rate': 0.001
    }
    
    ulm = UniversalLearningMachine(config)
    
    # Test processing
    async def test():
        result = await ulm.process({
            'input': 'Create a Python function to calculate fibonacci numbers'
        })
        print(f"Result: {result}")
    
    asyncio.run(test())