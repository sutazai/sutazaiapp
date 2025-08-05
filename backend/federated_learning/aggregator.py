"""
Federated Learning Aggregator
============================

Implements various federated aggregation algorithms optimized for CPU-only environments.
Supports FedAvg, FedProx, FedOpt with efficient CPU computation for 12-core constraint.

Features:
- FedAvg (Federated Averaging)
- FedProx (Federated Proximal)
- FedOpt (Federated Optimization)
- Asynchronous aggregation
- CPU-optimized implementations
- Compression and quantization
- Byzantine fault tolerance
"""

import asyncio
import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle
import zlib

# CPU-only optimized numerical computing
import numpy as np
from scipy import sparse
from collections import defaultdict


class AggregationAlgorithm(Enum):
    """Supported federated aggregation algorithms"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    FEDOPT = "fedopt"
    FEDAVG_MOMENTUM = "fedavg_momentum"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"


class CompressionType(Enum):
    """Model compression types"""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    GRADIENT_COMPRESSION = "gradient_compression"


@dataclass
class AggregationConfig:
    """Configuration for aggregation algorithm"""
    algorithm: AggregationAlgorithm
    learning_rate: float = 1.0
    momentum: float = 0.9
    prox_mu: float = 0.01  # For FedProx
    compression: CompressionType = CompressionType.NONE
    compression_ratio: float = 0.1
    byzantine_tolerance: bool = False
    max_norm: Optional[float] = None  # Gradient clipping
    adaptive_lr: bool = False
    warmup_rounds: int = 0


@dataclass
class ClientUpdate:
    """Represents a client model update"""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    loss: float
    accuracy: float
    computation_time: float
    communication_time: float
    metadata: Dict[str, Any]
    
    def serialize(self) -> bytes:
        """Serialize client update for storage/transmission"""
        data = {
            'client_id': self.client_id,
            'model_weights': {k: v.tobytes() for k, v in self.model_weights.items()},
            'weight_shapes': {k: v.shape for k, v in self.model_weights.items()},
            'weight_dtypes': {k: str(v.dtype) for k, v in self.model_weights.items()},
            'num_samples': self.num_samples,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'computation_time': self.computation_time,
            'communication_time': self.communication_time,
            'metadata': self.metadata
        }
        return zlib.compress(pickle.dumps(data))
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'ClientUpdate':
        """Deserialize client update from bytes"""
        obj_data = pickle.loads(zlib.decompress(data))
        
        # Reconstruct numpy arrays
        model_weights = {}
        for k in obj_data['model_weights']:
            arr_bytes = obj_data['model_weights'][k]
            shape = obj_data['weight_shapes'][k]
            dtype = np.dtype(obj_data['weight_dtypes'][k])
            model_weights[k] = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
        
        return cls(
            client_id=obj_data['client_id'],
            model_weights=model_weights,
            num_samples=obj_data['num_samples'],
            loss=obj_data['loss'],
            accuracy=obj_data['accuracy'],
            computation_time=obj_data['computation_time'],
            communication_time=obj_data['communication_time'],
            metadata=obj_data['metadata']
        )


@dataclass
class AggregationResult:
    """Result of federated aggregation"""
    aggregated_weights: Dict[str, np.ndarray]
    participating_clients: List[str]
    total_samples: int
    weighted_loss: float
    weighted_accuracy: float
    aggregation_time: float
    compression_ratio: float
    metadata: Dict[str, Any]


class BaseAggregator(ABC):
    """Abstract base class for federated aggregators"""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.logger = logging.getLogger(f"aggregator_{config.algorithm.value}")
        
        # Track aggregation history for adaptive algorithms
        self.aggregation_history: List[AggregationResult] = []
        self.momentum_buffer: Optional[Dict[str, np.ndarray]] = None
    
    @abstractmethod
    async def aggregate(self, client_updates: List[ClientUpdate], 
                       round_number: int) -> AggregationResult:
        """Aggregate client updates"""
        pass
    
    def _apply_compression(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply model compression"""
        if self.config.compression == CompressionType.NONE:
            return weights
        
        compressed_weights = {}
        
        for layer_name, weight_matrix in weights.items():
            if self.config.compression == CompressionType.QUANTIZATION:
                compressed_weights[layer_name] = self._quantize_weights(weight_matrix)
            elif self.config.compression == CompressionType.SPARSIFICATION:
                compressed_weights[layer_name] = self._sparsify_weights(weight_matrix)
            else:
                compressed_weights[layer_name] = weight_matrix
        
        return compressed_weights
    
    def _quantize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Quantize weights to reduce precision"""
        # Simple uniform quantization
        min_val, max_val = weights.min(), weights.max()
        num_levels = int(1.0 / self.config.compression_ratio)
        
        # Quantize to num_levels discrete values
        quantized = np.round((weights - min_val) / (max_val - min_val) * (num_levels - 1))
        quantized = quantized / (num_levels - 1) * (max_val - min_val) + min_val
        
        return quantized.astype(weights.dtype)
    
    def _sparsify_weights(self, weights: np.ndarray) -> np.ndarray:
        """Sparsify weights by keeping only top-k values"""
        flat_weights = weights.flatten()
        k = int(len(flat_weights) * self.config.compression_ratio)
        
        # Get indices of top-k absolute values
        top_k_indices = np.argpartition(np.abs(flat_weights), -k)[-k:]
        
        # Create sparse version
        sparse_weights = np.zeros_like(flat_weights)
        sparse_weights[top_k_indices] = flat_weights[top_k_indices]
        
        return sparse_weights.reshape(weights.shape)
    
    def _clip_gradients(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply gradient clipping"""
        if self.config.max_norm is None:
            return weights
        
        # Calculate global norm
        global_norm = 0.0
        for weight_matrix in weights.values():
            global_norm += np.sum(weight_matrix ** 2)
        global_norm = np.sqrt(global_norm)
        
        # Clip if necessary
        if global_norm > self.config.max_norm:
            clip_factor = self.config.max_norm / global_norm
            clipped_weights = {}
            for layer_name, weight_matrix in weights.items():
                clipped_weights[layer_name] = weight_matrix * clip_factor
            return clipped_weights
        
        return weights


class FedAvgAggregator(BaseAggregator):
    """Federated Averaging (FedAvg) aggregator"""
    
    async def aggregate(self, client_updates: List[ClientUpdate], 
                       round_number: int) -> AggregationResult:
        """Perform FedAvg aggregation"""
        start_time = time.time()
        
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Calculate total samples for weighting
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_update = client_updates[0]
        
        for layer_name in first_update.model_weights:
            aggregated_weights[layer_name] = np.zeros_like(first_update.model_weights[layer_name])
        
        # Weighted aggregation
        for update in client_updates:
            weight = update.num_samples / total_samples
            
            for layer_name, layer_weights in update.model_weights.items():
                aggregated_weights[layer_name] += weight * layer_weights
        
        # Apply gradient clipping
        aggregated_weights = self._clip_gradients(aggregated_weights)
        
        # Apply compression
        aggregated_weights = self._apply_compression(aggregated_weights)
        
        # Calculate metrics
        weighted_loss = sum(update.loss * update.num_samples for update in client_updates) / total_samples
        weighted_accuracy = sum(update.accuracy * update.num_samples for update in client_updates) / total_samples
        
        aggregation_time = time.time() - start_time
        
        result = AggregationResult(
            aggregated_weights=aggregated_weights,
            participating_clients=[update.client_id for update in client_updates],
            total_samples=total_samples,
            weighted_loss=weighted_loss,
            weighted_accuracy=weighted_accuracy,
            aggregation_time=aggregation_time,
            compression_ratio=self.config.compression_ratio if self.config.compression != CompressionType.NONE else 1.0,
            metadata={
                "algorithm": "fedavg",
                "round_number": round_number,
                "num_clients": len(client_updates)
            }
        )
        
        self.aggregation_history.append(result)
        self.logger.info(f"FedAvg round {round_number}: {len(client_updates)} clients, "
                        f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")
        
        return result


class FedProxAggregator(BaseAggregator):
    """Federated Proximal (FedProx) aggregator"""
    
    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.global_model: Optional[Dict[str, np.ndarray]] = None
    
    async def aggregate(self, client_updates: List[ClientUpdate], 
                       round_number: int) -> AggregationResult:
        """Perform FedProx aggregation with proximal term"""
        start_time = time.time()
        
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # Store current global model for proximal term
        if round_number == 1:
            # Initialize global model from first client
            self.global_model = {k: v.copy() for k, v in client_updates[0].model_weights.items()}
        
        # Calculate total samples
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        for layer_name in client_updates[0].model_weights:
            aggregated_weights[layer_name] = np.zeros_like(client_updates[0].model_weights[layer_name])
        
        # Weighted aggregation with proximal term
        for update in client_updates:
            weight = update.num_samples / total_samples
            
            for layer_name, layer_weights in update.model_weights.items():
                # Apply proximal term: w_t+1 = w_t - η(∇f + μ(w_t - w_global))
                if self.global_model is not None:
                    proximal_term = self.config.prox_mu * (layer_weights - self.global_model[layer_name])
                    adjusted_weights = layer_weights - self.config.learning_rate * proximal_term
                else:
                    adjusted_weights = layer_weights
                
                aggregated_weights[layer_name] += weight * adjusted_weights
        
        # Update global model
        self.global_model = {k: v.copy() for k, v in aggregated_weights.items()}
        
        # Apply gradient clipping and compression
        aggregated_weights = self._clip_gradients(aggregated_weights)
        aggregated_weights = self._apply_compression(aggregated_weights)
        
        # Calculate metrics
        weighted_loss = sum(update.loss * update.num_samples for update in client_updates) / total_samples
        weighted_accuracy = sum(update.accuracy * update.num_samples for update in client_updates) / total_samples
        
        aggregation_time = time.time() - start_time
        
        result = AggregationResult(
            aggregated_weights=aggregated_weights,
            participating_clients=[update.client_id for update in client_updates],
            total_samples=total_samples,
            weighted_loss=weighted_loss,
            weighted_accuracy=weighted_accuracy,
            aggregation_time=aggregation_time,
            compression_ratio=self.config.compression_ratio if self.config.compression != CompressionType.NONE else 1.0,
            metadata={
                "algorithm": "fedprox",
                "round_number": round_number,
                "num_clients": len(client_updates),
                "prox_mu": self.config.prox_mu
            }
        )
        
        self.aggregation_history.append(result)
        self.logger.info(f"FedProx round {round_number}: {len(client_updates)} clients, "
                        f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")
        
        return result


class FedOptAggregator(BaseAggregator):
    """Federated Optimization (FedOpt) aggregator with server-side optimization"""
    
    def __init__(self, config: AggregationConfig):
        super().__init__(config)
        self.server_optimizer_state: Dict[str, np.ndarray] = {}
        self.server_momentum: Dict[str, np.ndarray] = {}
        self.server_velocity: Dict[str, np.ndarray] = {}  # For Adam-like optimization
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0  # Time step for bias correction
    
    async def aggregate(self, client_updates: List[ClientUpdate], 
                       round_number: int) -> AggregationResult:
        """Perform FedOpt aggregation with server-side optimization"""
        start_time = time.time()
        self.t += 1
        
        if not client_updates:
            raise ValueError("No client updates provided")
        
        # First, compute FedAvg-style aggregation
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Compute pseudo-gradients (difference from current model)
        if round_number == 1:
            # Initialize server state
            for layer_name in client_updates[0].model_weights:
                layer_shape = client_updates[0].model_weights[layer_name].shape
                self.server_momentum[layer_name] = np.zeros(layer_shape)
                self.server_velocity[layer_name] = np.zeros(layer_shape)
        
        # Aggregate client updates
        aggregated_delta = {}
        for layer_name in client_updates[0].model_weights:
            aggregated_delta[layer_name] = np.zeros_like(client_updates[0].model_weights[layer_name])
        
        for update in client_updates:
            weight = update.num_samples / total_samples
            for layer_name, layer_weights in update.model_weights.items():
                aggregated_delta[layer_name] += weight * layer_weights
        
        # Apply server-side optimization (Adam-like)
        optimized_weights = {}
        
        for layer_name, delta in aggregated_delta.items():
            # Update biased first moment estimate
            self.server_momentum[layer_name] = (self.beta1 * self.server_momentum[layer_name] + 
                                              (1 - self.beta1) * delta)
            
            # Update biased second raw moment estimate  
            self.server_velocity[layer_name] = (self.beta2 * self.server_velocity[layer_name] + 
                                              (1 - self.beta2) * (delta ** 2))
            
            # Compute bias-corrected first moment estimate
            m_hat = self.server_momentum[layer_name] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.server_velocity[layer_name] / (1 - self.beta2 ** self.t)
            
            # Apply server learning rate and optimization
            optimized_weights[layer_name] = delta + self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Apply gradient clipping and compression
        optimized_weights = self._clip_gradients(optimized_weights)
        optimized_weights = self._apply_compression(optimized_weights)
        
        # Calculate metrics
        weighted_loss = sum(update.loss * update.num_samples for update in client_updates) / total_samples
        weighted_accuracy = sum(update.accuracy * update.num_samples for update in client_updates) / total_samples
        
        aggregation_time = time.time() - start_time
        
        result = AggregationResult(
            aggregated_weights=optimized_weights,
            participating_clients=[update.client_id for update in client_updates],
            total_samples=total_samples,
            weighted_loss=weighted_loss,
            weighted_accuracy=weighted_accuracy,
            aggregation_time=aggregation_time,
            compression_ratio=self.config.compression_ratio if self.config.compression != CompressionType.NONE else 1.0,
            metadata={
                "algorithm": "fedopt",
                "round_number": round_number,
                "num_clients": len(client_updates),
                "server_lr": self.config.learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2
            }
        )
        
        self.aggregation_history.append(result)
        self.logger.info(f"FedOpt round {round_number}: {len(client_updates)} clients, "
                        f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")
        
        return result


class ByzantineRobustAggregator(BaseAggregator):
    """Byzantine-robust aggregation using Krum or Trimmed Mean"""
    
    async def aggregate(self, client_updates: List[ClientUpdate], 
                       round_number: int) -> AggregationResult:
        """Perform Byzantine-robust aggregation"""
        start_time = time.time()
        
        if not client_updates:
            raise ValueError("No client updates provided")
        
        if self.config.algorithm == AggregationAlgorithm.KRUM:
            aggregated_weights = await self._krum_aggregation(client_updates)
        elif self.config.algorithm == AggregationAlgorithm.TRIMMED_MEAN:
            aggregated_weights = await self._trimmed_mean_aggregation(client_updates)
        elif self.config.algorithm == AggregationAlgorithm.MEDIAN:
            aggregated_weights = await self._median_aggregation(client_updates)
        else:
            raise ValueError(f"Unsupported Byzantine-robust algorithm: {self.config.algorithm}")
        
        # Apply gradient clipping and compression
        aggregated_weights = self._clip_gradients(aggregated_weights)
        aggregated_weights = self._apply_compression(aggregated_weights)
        
        # Calculate metrics
        total_samples = sum(update.num_samples for update in client_updates)
        weighted_loss = sum(update.loss * update.num_samples for update in client_updates) / total_samples
        weighted_accuracy = sum(update.accuracy * update.num_samples for update in client_updates) / total_samples
        
        aggregation_time = time.time() - start_time
        
        result = AggregationResult(
            aggregated_weights=aggregated_weights,
            participating_clients=[update.client_id for update in client_updates],
            total_samples=total_samples,
            weighted_loss=weighted_loss,
            weighted_accuracy=weighted_accuracy,
            aggregation_time=aggregation_time,
            compression_ratio=self.config.compression_ratio if self.config.compression != CompressionType.NONE else 1.0,
            metadata={
                "algorithm": self.config.algorithm.value,
                "round_number": round_number,
                "num_clients": len(client_updates),
                "byzantine_robust": True
            }
        )
        
        self.aggregation_history.append(result)
        self.logger.info(f"{self.config.algorithm.value} round {round_number}: {len(client_updates)} clients, "
                        f"loss={weighted_loss:.4f}, accuracy={weighted_accuracy:.4f}")
        
        return result
    
    async def _krum_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Krum aggregation - select most representative client"""
        n = len(client_updates)
        f = n // 4  # Assume up to n/4 Byzantine clients
        
        # Flatten all client updates for distance computation
        flattened_updates = []
        for update in client_updates:
            flattened = np.concatenate([weights.flatten() for weights in update.model_weights.values()])
            flattened_updates.append(flattened)
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(flattened_updates[i] - flattened_updates[j])
                distances[i, j] = distances[j, i] = dist
        
        # For each client, compute sum of distances to closest n-f-2 clients
        krum_scores = []
        for i in range(n):
            # Get distances to all other clients
            client_distances = distances[i]
            # Sort and take sum of closest n-f-2 distances (excluding self)
            sorted_distances = np.sort(client_distances[client_distances > 0])
            score = np.sum(sorted_distances[:n-f-2])
            krum_scores.append(score)
        
        # Select client with minimum Krum score
        selected_client_idx = np.argmin(krum_scores)
        selected_update = client_updates[selected_client_idx]
        
        self.logger.info(f"Krum selected client {selected_update.client_id} "
                        f"(score: {krum_scores[selected_client_idx]:.4f})")
        
        return selected_update.model_weights
    
    async def _trimmed_mean_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Trimmed mean aggregation - remove outliers and average"""
        n = len(client_updates)
        trim_ratio = 0.2  # Trim 20% of outliers
        trim_count = int(n * trim_ratio / 2)  # Trim from both ends
        
        aggregated_weights = {}
        
        # For each layer, compute trimmed mean
        for layer_name in client_updates[0].model_weights:
            layer_weights = [update.model_weights[layer_name] for update in client_updates]
            layer_stack = np.stack(layer_weights, axis=0)
            
            # Sort along client axis and trim
            sorted_weights = np.sort(layer_stack, axis=0)
            if trim_count > 0:
                trimmed_weights = sorted_weights[trim_count:-trim_count]
            else:
                trimmed_weights = sorted_weights
            
            # Compute mean of remaining weights
            aggregated_weights[layer_name] = np.mean(trimmed_weights, axis=0)
        
        self.logger.info(f"Trimmed mean aggregation: trimmed {trim_count*2}/{n} outliers")
        return aggregated_weights
    
    async def _median_aggregation(self, client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """Coordinate-wise median aggregation"""
        aggregated_weights = {}
        
        for layer_name in client_updates[0].model_weights:
            layer_weights = [update.model_weights[layer_name] for update in client_updates]
            layer_stack = np.stack(layer_weights, axis=0)
            
            # Compute coordinate-wise median
            aggregated_weights[layer_name] = np.median(layer_stack, axis=0)
        
        return aggregated_weights


class FederatedAggregator:
    """
    Main Federated Aggregator
    
    Factory class that creates appropriate aggregators based on configuration
    and manages CPU-optimized aggregation for the SutazAI system.
    """
    
    def __init__(self, cpu_cores: int = 12):
        self.cpu_cores = cpu_cores
        self.thread_pool = ThreadPoolExecutor(max_workers=cpu_cores)
        
        # Cache aggregators for reuse
        self.aggregators: Dict[str, BaseAggregator] = {}
        
        self.logger = logging.getLogger("federated_aggregator")
    
    async def initialize(self):
        """Initialize the federated aggregator"""
        self.logger.info(f"Initialized Federated Aggregator with {self.cpu_cores} CPU cores")
    
    async def aggregate(self, algorithm: AggregationAlgorithm,
                       client_updates: Dict[str, Dict[str, Any]],
                       round_number: int,
                       config: Optional[AggregationConfig] = None) -> Dict[str, Any]:
        """
        Aggregate client updates using specified algorithm
        
        Args:
            algorithm: Aggregation algorithm to use
            client_updates: Dictionary of client updates from coordinator
            round_number: Current round number
            config: Optional aggregation configuration
        
        Returns:
            Aggregation result dictionary
        """
        try:
            # Convert client updates to ClientUpdate objects
            update_objects = []
            for client_id, update_data in client_updates.items():
                # Convert serialized weights back to numpy arrays
                model_weights = {}
                for layer_name, weight_data in update_data.get("model_weights", {}).items():
                    if isinstance(weight_data, list):
                        model_weights[layer_name] = np.array(weight_data)
                    else:
                        model_weights[layer_name] = weight_data
                
                update_obj = ClientUpdate(
                    client_id=client_id,
                    model_weights=model_weights,
                    num_samples=update_data.get("num_samples", 1),
                    loss=update_data.get("loss", 0.0),
                    accuracy=update_data.get("accuracy", 0.0),
                    computation_time=update_data.get("computation_time", 0.0),
                    communication_time=update_data.get("communication_time", 0.0),
                    metadata=update_data.get("metadata", {})
                )
                update_objects.append(update_obj)
            
            # Get or create aggregator
            aggregator = self._get_aggregator(algorithm, config)
            
            # Perform aggregation
            result = await aggregator.aggregate(update_objects, round_number)
            
            # Convert result to serializable format
            return {
                "aggregated_weights": {
                    k: v.tolist() for k, v in result.aggregated_weights.items()
                },
                "participating_clients": result.participating_clients,
                "total_samples": result.total_samples,
                "weighted_loss": result.weighted_loss,
                "weighted_accuracy": result.weighted_accuracy,
                "aggregation_time": result.aggregation_time,
                "compression_ratio": result.compression_ratio,
                "metadata": result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise
    
    def _get_aggregator(self, algorithm: AggregationAlgorithm, 
                       config: Optional[AggregationConfig] = None) -> BaseAggregator:
        """Get or create aggregator for algorithm"""
        
        if config is None:
            config = AggregationConfig(algorithm=algorithm)
        
        # Create cache key
        cache_key = f"{algorithm.value}_{hash(str(config.__dict__))}"
        
        if cache_key not in self.aggregators:
            if algorithm == AggregationAlgorithm.FEDAVG:
                self.aggregators[cache_key] = FedAvgAggregator(config)
            elif algorithm == AggregationAlgorithm.FEDPROX:
                self.aggregators[cache_key] = FedProxAggregator(config)
            elif algorithm == AggregationAlgorithm.FEDOPT:
                self.aggregators[cache_key] = FedOptAggregator(config)
            elif algorithm in [AggregationAlgorithm.KRUM, AggregationAlgorithm.TRIMMED_MEAN, 
                              AggregationAlgorithm.MEDIAN]:
                self.aggregators[cache_key] = ByzantineRobustAggregator(config)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        return self.aggregators[cache_key]
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported aggregation algorithms"""
        return [alg.value for alg in AggregationAlgorithm]
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        stats = {
            "cpu_cores": self.cpu_cores,
            "active_aggregators": len(self.aggregators),
            "supported_algorithms": self.get_supported_algorithms()
        }
        
        # Add per-algorithm stats
        for cache_key, aggregator in self.aggregators.items():
            algorithm_name = aggregator.config.algorithm.value
            stats[f"{algorithm_name}_rounds"] = len(aggregator.aggregation_history)
            
            if aggregator.aggregation_history:
                recent_results = aggregator.aggregation_history[-10:]  # Last 10 rounds
                stats[f"{algorithm_name}_avg_time"] = np.mean([r.aggregation_time for r in recent_results])
                stats[f"{algorithm_name}_avg_accuracy"] = np.mean([r.weighted_accuracy for r in recent_results])
        
        return stats
    
    async def shutdown(self):
        """Shutdown the aggregator"""
        self.thread_pool.shutdown(wait=True)
        self.logger.info("Federated Aggregator shutdown complete")