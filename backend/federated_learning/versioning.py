"""
Model Versioning and Rollback System
====================================

Manages model versions, checkpoints, and rollback capabilities for federated learning.
Provides version control, model comparison, and automated rollback on performance degradation.

Features:
- Model version management and storage
- Automated checkpointing during training
- Performance-based rollback triggers
- Model diff and comparison tools
- Version history and lineage tracking
- Compressed model storage
- Distributed model registry
"""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
import pickle
import zlib
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import aioredis
import aiofiles
import os
from pathlib import Path


class VersionStatus(Enum):
    """Model version status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ROLLBACK = "rollback"


class CheckpointTrigger(Enum):
    """Triggers for creating checkpoints"""
    ROUND_INTERVAL = "round_interval"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    model_id: str
    version: str
    training_id: str
    round_number: int
    algorithm: str
    performance_metrics: Dict[str, float]
    model_size_bytes: int
    compression_ratio: float
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    description: str = ""
    parent_version: Optional[str] = None
    checkpoint_trigger: CheckpointTrigger = CheckpointTrigger.MANUAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'checkpoint_trigger': self.checkpoint_trigger.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['checkpoint_trigger'] = CheckpointTrigger(data['checkpoint_trigger'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Complete model version with weights and metadata"""
    metadata: ModelMetadata
    model_weights: Dict[str, np.ndarray]
    optimizer_state: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    
    def calculate_size(self) -> int:
        """Calculate total model size in bytes"""
        total_size = 0
        for weights in self.model_weights.values():
            total_size += weights.nbytes
        
        if self.optimizer_state:
            # Estimate optimizer state size
            total_size += len(json.dumps(self.optimizer_state).encode())
        
        return total_size
    
    def calculate_checksum(self) -> str:
        """Calculate model checksum for integrity verification"""
        hasher = hashlib.sha256()
        
        # Hash model weights in deterministic order
        for layer_name in sorted(self.model_weights.keys()):
            hasher.update(layer_name.encode())
            hasher.update(self.model_weights[layer_name].tobytes())
        
        return hasher.hexdigest()


@dataclass
class RollbackConfig:
    """Configuration for automatic rollback"""
    enabled: bool = True
    performance_threshold: float = 0.05  # 5% degradation triggers rollback
    consecutive_degradations: int = 3
    monitoring_rounds: int = 10
    rollback_to_best: bool = True
    notify_on_rollback: bool = True


@dataclass
class VersioningConfig:
    """Configuration for model versioning"""
    storage_backend: str = "filesystem"  # filesystem, redis, database
    storage_path: str = "/tmp/federated_models"
    max_versions_per_training: int = 50
    compression_enabled: bool = True
    auto_cleanup_enabled: bool = True
    cleanup_after_days: int = 30
    checkpoint_interval: int = 5  # Every N rounds
    performance_checkpointing: bool = True


class ModelStorage:
    """Abstract model storage interface"""
    
    async def save_model(self, version: ModelVersion) -> bool:
        raise NotImplementedError
    
    async def load_model(self, model_id: str, version_id: str) -> Optional[ModelVersion]:
        raise NotImplementedError
    
    async def delete_model(self, model_id: str, version_id: str) -> bool:
        raise NotImplementedError
    
    async def list_versions(self, model_id: str) -> List[str]:
        raise NotImplementedError


class FilesystemStorage(ModelStorage):
    """Filesystem-based model storage"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("filesystem_storage")
    
    def _get_model_path(self, model_id: str, version_id: str) -> Path:
        return self.base_path / model_id / f"{version_id}.pkl.gz"
    
    def _get_metadata_path(self, model_id: str, version_id: str) -> Path:
        return self.base_path / model_id / f"{version_id}_metadata.json"
    
    async def save_model(self, version: ModelVersion) -> bool:
        try:
            model_dir = self.base_path / version.metadata.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = self._get_model_path(version.metadata.model_id, version.metadata.version)
            metadata_path = self._get_metadata_path(version.metadata.model_id, version.metadata.version)
            
            # Serialize and compress model
            model_data = {
                'model_weights': version.model_weights,
                'optimizer_state': version.optimizer_state,
                'training_config': version.training_config,
                'validation_results': version.validation_results
            }
            
            serialized = pickle.dumps(model_data)
            compressed = zlib.compress(serialized)
            
            # Save compressed model
            async with aiofiles.open(model_path, 'wb') as f:
                await f.write(compressed)
            
            # Save metadata
            metadata_dict = version.metadata.to_dict()
            metadata_dict['checksum'] = version.calculate_checksum()
            metadata_dict['file_size'] = len(compressed)
            
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata_dict, indent=2))
            
            self.logger.info(f"Saved model {version.metadata.model_id} v{version.metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self, model_id: str, version_id: str) -> Optional[ModelVersion]:
        try:
            model_path = self._get_model_path(model_id, version_id)
            metadata_path = self._get_metadata_path(model_id, version_id)
            
            if not model_path.exists() or not metadata_path.exists():
                return None
            
            # Load metadata
            async with aiofiles.open(metadata_path, 'r') as f:
                metadata_dict = json.loads(await f.read())
            
            metadata = ModelMetadata.from_dict(metadata_dict)
            
            # Load and decompress model
            async with aiofiles.open(model_path, 'rb') as f:
                compressed_data = await f.read()
            
            decompressed = zlib.decompress(compressed_data)
            model_data = pickle.loads(decompressed)
            
            version = ModelVersion(
                metadata=metadata,
                model_weights=model_data['model_weights'],
                optimizer_state=model_data.get('optimizer_state'),
                training_config=model_data.get('training_config'),
                validation_results=model_data.get('validation_results')
            )
            
            # Verify checksum
            if metadata_dict.get('checksum') != version.calculate_checksum():
                self.logger.warning(f"Checksum mismatch for {model_id} v{version_id}")
            
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id} v{version_id}: {e}")
            return None
    
    async def delete_model(self, model_id: str, version_id: str) -> bool:
        try:
            model_path = self._get_model_path(model_id, version_id)
            metadata_path = self._get_metadata_path(model_id, version_id)
            
            if model_path.exists():
                model_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            self.logger.info(f"Deleted model {model_id} v{version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id} v{version_id}: {e}")
            return False
    
    async def list_versions(self, model_id: str) -> List[str]:
        try:
            model_dir = self.base_path / model_id
            if not model_dir.exists():
                return []
            
            versions = []
            for metadata_file in model_dir.glob("*_metadata.json"):
                version_id = metadata_file.stem.replace("_metadata", "")
                versions.append(version_id)
            
            return sorted(versions)
            
        except Exception as e:
            self.logger.error(f"Failed to list versions for {model_id}: {e}")
            return []


class RedisStorage(ModelStorage):
    """Redis-based model storage"""
    
    def __init__(self, redis_url: str, namespace: str = "federated_models"):
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        self.logger = logging.getLogger("redis_storage")
    
    async def initialize(self):
        self.redis = aioredis.from_url(self.redis_url)
        await self.redis.ping()
    
    def _get_model_key(self, model_id: str, version_id: str) -> str:
        return f"{self.namespace}:model:{model_id}:{version_id}"
    
    def _get_metadata_key(self, model_id: str, version_id: str) -> str:
        return f"{self.namespace}:metadata:{model_id}:{version_id}"
    
    def _get_versions_key(self, model_id: str) -> str:
        return f"{self.namespace}:versions:{model_id}"
    
    async def save_model(self, version: ModelVersion) -> bool:
        try:
            if not self.redis:
                await self.initialize()
            
            model_key = self._get_model_key(version.metadata.model_id, version.metadata.version)
            metadata_key = self._get_metadata_key(version.metadata.model_id, version.metadata.version)
            versions_key = self._get_versions_key(version.metadata.model_id)
            
            # Serialize and compress model
            model_data = {
                'model_weights': {k: v.tobytes() for k, v in version.model_weights.items()},
                'weight_shapes': {k: v.shape for k, v in version.model_weights.items()},
                'weight_dtypes': {k: str(v.dtype) for k, v in version.model_weights.items()},
                'optimizer_state': version.optimizer_state,
                'training_config': version.training_config,
                'validation_results': version.validation_results
            }
            
            compressed_data = zlib.compress(pickle.dumps(model_data))
            
            # Store in Redis
            await self.redis.set(model_key, compressed_data)
            await self.redis.set(metadata_key, json.dumps(version.metadata.to_dict()))
            await self.redis.sadd(versions_key, version.metadata.version)
            
            # Set TTL if configured
            # await self.redis.expire(model_key, 30 * 24 * 3600)  # 30 days
            
            self.logger.info(f"Saved model {version.metadata.model_id} v{version.metadata.version} to Redis")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model to Redis: {e}")
            return False
    
    async def load_model(self, model_id: str, version_id: str) -> Optional[ModelVersion]:
        try:
            if not self.redis:
                await self.initialize()
            
            model_key = self._get_model_key(model_id, version_id)
            metadata_key = self._get_metadata_key(model_id, version_id)
            
            # Load metadata
            metadata_data = await self.redis.get(metadata_key)
            if not metadata_data:
                return None
            
            metadata = ModelMetadata.from_dict(json.loads(metadata_data))
            
            # Load model data
            compressed_data = await self.redis.get(model_key)
            if not compressed_data:
                return None
            
            decompressed = zlib.decompress(compressed_data)
            model_data = pickle.loads(decompressed)
            
            # Reconstruct numpy arrays
            model_weights = {}
            for k in model_data['model_weights']:
                arr_bytes = model_data['model_weights'][k]
                shape = model_data['weight_shapes'][k]
                dtype = np.dtype(model_data['weight_dtypes'][k])
                model_weights[k] = np.frombuffer(arr_bytes, dtype=dtype).reshape(shape)
            
            version = ModelVersion(
                metadata=metadata,
                model_weights=model_weights,
                optimizer_state=model_data.get('optimizer_state'),
                training_config=model_data.get('training_config'),
                validation_results=model_data.get('validation_results')
            )
            
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to load model from Redis: {e}")
            return None
    
    async def delete_model(self, model_id: str, version_id: str) -> bool:
        try:
            if not self.redis:
                await self.initialize()
            
            model_key = self._get_model_key(model_id, version_id)
            metadata_key = self._get_metadata_key(model_id, version_id)
            versions_key = self._get_versions_key(model_id)
            
            await self.redis.delete(model_key, metadata_key)
            await self.redis.srem(versions_key, version_id)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model from Redis: {e}")
            return False
    
    async def list_versions(self, model_id: str) -> List[str]:
        try:
            if not self.redis:
                await self.initialize()
            
            versions_key = self._get_versions_key(model_id)
            versions = await self.redis.smembers(versions_key)
            
            return sorted(versions) if versions else []
            
        except Exception as e:
            self.logger.error(f"Failed to list versions from Redis: {e}")
            return []


class ModelComparator:
    """Compare model versions and compute differences"""
    
    @staticmethod
    def compute_model_diff(model1: ModelVersion, model2: ModelVersion) -> Dict[str, Any]:
        """Compute difference between two model versions"""
        diff = {
            "version1": model1.metadata.version,
            "version2": model2.metadata.version,
            "layer_diffs": {},
            "total_param_change": 0.0,
            "max_param_change": 0.0,
            "performance_diff": {},
            "size_diff": 0
        }
        
        # Compare model weights layer by layer
        all_layers = set(model1.model_weights.keys()) | set(model2.model_weights.keys())
        
        for layer_name in all_layers:
            if layer_name in model1.model_weights and layer_name in model2.model_weights:
                w1 = model1.model_weights[layer_name]
                w2 = model2.model_weights[layer_name]
                
                if w1.shape == w2.shape:
                    layer_diff = np.abs(w1 - w2)
                    diff["layer_diffs"][layer_name] = {
                        "mean_diff": float(np.mean(layer_diff)),
                        "max_diff": float(np.max(layer_diff)),
                        "norm_diff": float(np.linalg.norm(layer_diff)),
                        "relative_change": float(np.linalg.norm(layer_diff) / np.linalg.norm(w1))
                    }
                else:
                    diff["layer_diffs"][layer_name] = {"error": "Shape mismatch"}
            else:
                diff["layer_diffs"][layer_name] = {"error": "Layer missing in one model"}
        
        # Overall statistics
        if diff["layer_diffs"]:
            valid_diffs = [d for d in diff["layer_diffs"].values() if "error" not in d]
            if valid_diffs:
                diff["total_param_change"] = sum(d["norm_diff"] for d in valid_diffs)
                diff["max_param_change"] = max(d["max_diff"] for d in valid_diffs)
        
        # Performance difference
        perf1 = model1.metadata.performance_metrics
        perf2 = model2.metadata.performance_metrics
        
        for metric in set(perf1.keys()) | set(perf2.keys()):
            if metric in perf1 and metric in perf2:
                diff["performance_diff"][metric] = perf2[metric] - perf1[metric]
        
        # Size difference
        diff["size_diff"] = model2.metadata.model_size_bytes - model1.metadata.model_size_bytes
        
        return diff
    
    @staticmethod
    def compare_performance(model1: ModelVersion, model2: ModelVersion, 
                          primary_metric: str = "accuracy") -> Dict[str, Any]:
        """Compare performance between two models"""
        perf1 = model1.metadata.performance_metrics.get(primary_metric, 0.0)
        perf2 = model2.metadata.performance_metrics.get(primary_metric, 0.0)
        
        return {
            "model1_performance": perf1,
            "model2_performance": perf2,
            "absolute_diff": perf2 - perf1,
            "relative_diff": (perf2 - perf1) / perf1 if perf1 != 0 else float('inf'),
            "is_improvement": perf2 > perf1,
            "primary_metric": primary_metric
        }


class ModelVersionManager:
    """
    Model Version Manager for Federated Learning
    
    Manages model versions, checkpoints, and rollback functionality
    for federated learning in the SutazAI system.
    """
    
    def __init__(self, config: VersioningConfig = None):
        self.config = config or VersioningConfig()
        
        # Initialize storage backend
        if self.config.storage_backend == "filesystem":
            self.storage = FilesystemStorage(self.config.storage_path)
        elif self.config.storage_backend == "redis":
            self.storage = RedisStorage("redis://localhost:6379")
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend}")
        
        # Version tracking
        self.active_versions: Dict[str, str] = {}  # training_id -> active_version
        self.version_history: Dict[str, List[str]] = defaultdict(list)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_versions_per_training))
        
        # Rollback management
        self.rollback_configs: Dict[str, RollbackConfig] = {}
        self.rollback_candidates: Dict[str, str] = {}  # training_id -> best_version
        
        # Comparator
        self.comparator = ModelComparator()
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("model_version_manager")
    
    async def initialize(self):
        """Initialize the model version manager"""
        try:
            if hasattr(self.storage, 'initialize'):
                await self.storage.initialize()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("Model Version Manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize version manager: {e}")
            raise
    
    def _start_background_tasks(self):
        """Start background tasks"""
        tasks = [
            self._cleanup_old_versions(),
            self._monitor_performance_degradation()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def create_initial_model(self, training_id: str, model_type: str) -> str:
        """Create initial model version for a training session"""
        try:
            # Generate initial model weights (simplified)
            initial_weights = self._generate_initial_weights(model_type)
            
            version_id = f"v0.0.0_{int(time.time())}"
            
            metadata = ModelMetadata(
                model_id=training_id,
                version=version_id,
                training_id=training_id,
                round_number=0,
                algorithm="initialization",
                performance_metrics={"accuracy": 0.0, "loss": float('inf')},
                model_size_bytes=0,  # Will be calculated
                compression_ratio=1.0,
                created_at=datetime.utcnow(),
                created_by="federated_coordinator",
                description="Initial model for federated training",
                checkpoint_trigger=CheckpointTrigger.MANUAL
            )
            
            version = ModelVersion(
                metadata=metadata,
                model_weights=initial_weights,
                training_config={"model_type": model_type}
            )
            
            # Calculate actual size
            version.metadata.model_size_bytes = version.calculate_size()
            
            # Save version
            success = await self.storage.save_model(version)
            if not success:
                raise ValueError("Failed to save initial model")
            
            # Track version
            self.active_versions[training_id] = version_id
            self.version_history[training_id].append(version_id)
            
            self.logger.info(f"Created initial model {training_id} v{version_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create initial model: {e}")
            raise
    
    def _generate_initial_weights(self, model_type: str) -> Dict[str, np.ndarray]:
        """Generate initial model weights"""
        if model_type == "neural_network":
            # Simple neural network initialization
            return {
                "layer1_weights": np.random.normal(0, 0.1, (784, 128)),
                "layer1_bias": np.zeros(128),
                "layer2_weights": np.random.normal(0, 0.1, (128, 64)),
                "layer2_bias": np.zeros(64),
                "output_weights": np.random.normal(0, 0.1, (64, 10)),
                "output_bias": np.zeros(10)
            }
        else:
            # Default initialization
            return {
                "weights": np.random.normal(0, 0.1, (100, 10)),
                "bias": np.zeros(10)
            }
    
    async def create_model_version(self, training_id: str, 
                                 aggregation_result: Dict[str, Any],
                                 round_number: int = None,
                                 trigger: CheckpointTrigger = CheckpointTrigger.ROUND_INTERVAL) -> str:
        """Create a new model version from aggregation result"""
        try:
            # Generate version ID
            major, minor, patch = self._get_next_version_number(training_id, trigger)
            version_id = f"v{major}.{minor}.{patch}_{int(time.time())}"
            
            # Extract model weights
            model_weights = {}
            for layer_name, weights in aggregation_result.get("aggregated_weights", {}).items():
                if isinstance(weights, list):
                    model_weights[layer_name] = np.array(weights)
                else:
                    model_weights[layer_name] = weights
            
            # Get performance metrics
            performance_metrics = {
                "accuracy": aggregation_result.get("weighted_accuracy", 0.0),
                "loss": aggregation_result.get("weighted_loss", float('inf')),
                "num_clients": aggregation_result.get("num_clients", 0),
                "total_samples": aggregation_result.get("total_samples", 0)
            }
            
            # Get parent version
            parent_version = self.active_versions.get(training_id)
            
            metadata = ModelMetadata(
                model_id=training_id,
                version=version_id,
                training_id=training_id,
                round_number=round_number or 0,
                algorithm=aggregation_result.get("algorithm", "unknown"),
                performance_metrics=performance_metrics,
                model_size_bytes=0,  # Will be calculated
                compression_ratio=aggregation_result.get("compression_ratio", 1.0),
                created_at=datetime.utcnow(),
                created_by="federated_coordinator",
                parent_version=parent_version,
                checkpoint_trigger=trigger,
                description=f"Model after round {round_number}"
            )
            
            version = ModelVersion(
                metadata=metadata,
                model_weights=model_weights,
                validation_results=aggregation_result
            )
            
            # Calculate actual size
            version.metadata.model_size_bytes = version.calculate_size()
            
            # Check if this should be a checkpoint
            should_checkpoint = await self._should_create_checkpoint(training_id, version, trigger)
            
            if should_checkpoint:
                # Save version
                success = await self.storage.save_model(version)
                if not success:
                    raise ValueError("Failed to save model version")
                
                # Update tracking
                self.active_versions[training_id] = version_id
                self.version_history[training_id].append(version_id)
                self.performance_history[training_id].append(performance_metrics)
                
                # Check if this is the best version so far
                await self._update_best_version(training_id, version)
                
                self.logger.info(f"Created model version {training_id} v{version_id} "
                               f"(accuracy: {performance_metrics['accuracy']:.4f})")
            else:
                # Just update active version without saving
                self.active_versions[training_id] = version_id
                self.performance_history[training_id].append(performance_metrics)
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create model version: {e}")
            raise
    
    def _get_next_version_number(self, training_id: str, 
                               trigger: CheckpointTrigger) -> Tuple[int, int, int]:
        """Generate next version number based on trigger"""
        versions = self.version_history.get(training_id, [])
        
        if not versions:
            return 1, 0, 0
        
        # Parse latest version
        latest_version = versions[-1]
        try:
            version_part = latest_version.split('_')[0][1:]  # Remove 'v' prefix
            major, minor, patch = map(int, version_part.split('.'))
        except:
            major, minor, patch = 1, 0, 0
        
        # Increment based on trigger
        if trigger in [CheckpointTrigger.PERFORMANCE_IMPROVEMENT, CheckpointTrigger.MANUAL]:
            major += 1
            minor, patch = 0, 0
        elif trigger == CheckpointTrigger.ROUND_INTERVAL:
            minor += 1
            patch = 0
        else:
            patch += 1
        
        return major, minor, patch
    
    async def _should_create_checkpoint(self, training_id: str, version: ModelVersion, 
                                      trigger: CheckpointTrigger) -> bool:
        """Determine if a checkpoint should be created"""
        if trigger == CheckpointTrigger.MANUAL:
            return True
        
        if trigger == CheckpointTrigger.ROUND_INTERVAL:
            return version.metadata.round_number % self.config.checkpoint_interval == 0
        
        if trigger == CheckpointTrigger.PERFORMANCE_IMPROVEMENT and self.config.performance_checkpointing:
            # Check if performance improved significantly
            history = self.performance_history.get(training_id, deque())
            if len(history) < 2:
                return True
            
            current_accuracy = version.metadata.performance_metrics.get("accuracy", 0.0)
            previous_best = max(h.get("accuracy", 0.0) for h in history)
            
            improvement = current_accuracy - previous_best
            return improvement > 0.01  # 1% improvement threshold
        
        return False
    
    async def _update_best_version(self, training_id: str, version: ModelVersion):
        """Update the best version candidate for rollback"""
        current_best = self.rollback_candidates.get(training_id)
        
        if not current_best:
            self.rollback_candidates[training_id] = version.metadata.version
            return
        
        # Load current best version
        best_version = await self.storage.load_model(training_id, current_best)
        if not best_version:
            self.rollback_candidates[training_id] = version.metadata.version
            return
        
        # Compare performance
        comparison = self.comparator.compare_performance(best_version, version)
        
        if comparison["is_improvement"]:
            self.rollback_candidates[training_id] = version.metadata.version
            self.logger.info(f"Updated best version for {training_id}: {version.metadata.version}")
    
    async def get_model_version(self, version_id: str, training_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a specific model version"""
        try:
            if training_id:
                model_id = training_id
            else:
                # Try to find training_id from version history
                model_id = None
                for tid, versions in self.version_history.items():
                    if version_id in versions:
                        model_id = tid
                        break
                
                if not model_id:
                    return None
            
            version = await self.storage.load_model(model_id, version_id)
            if not version:
                return None
            
            # Convert to serializable format
            model_data = {
                "metadata": version.metadata.to_dict(),
                "model_weights": {k: v.tolist() for k, v in version.model_weights.items()},
                "checksum": version.calculate_checksum()
            }
            
            if version.optimizer_state:
                model_data["optimizer_state"] = version.optimizer_state
            
            if version.training_config:
                model_data["training_config"] = version.training_config
            
            if version.validation_results:
                model_data["validation_results"] = version.validation_results
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Failed to get model version {version_id}: {e}")
            return None
    
    async def rollback_to_version(self, training_id: str, target_version: str = None) -> Optional[str]:
        """Rollback to a specific version or the best version"""
        try:
            if not target_version:
                target_version = self.rollback_candidates.get(training_id)
            
            if not target_version:
                self.logger.error(f"No rollback candidate for training {training_id}")
                return None
            
            # Load target version
            target_model = await self.storage.load_model(training_id, target_version)
            if not target_model:
                self.logger.error(f"Target version {target_version} not found")
                return None
            
            # Create rollback version
            rollback_version_id = f"rollback_{target_version}_{int(time.time())}"
            
            rollback_metadata = ModelMetadata(
                model_id=training_id,
                version=rollback_version_id,
                training_id=training_id,
                round_number=target_model.metadata.round_number,
                algorithm=target_model.metadata.algorithm,
                performance_metrics=target_model.metadata.performance_metrics,
                model_size_bytes=target_model.metadata.model_size_bytes,
                compression_ratio=target_model.metadata.compression_ratio,
                created_at=datetime.utcnow(),
                created_by="rollback_system",
                parent_version=target_version,
                checkpoint_trigger=CheckpointTrigger.ERROR_RECOVERY,
                description=f"Rollback to {target_version}",
                tags=["rollback"]
            )
            
            rollback_version = ModelVersion(
                metadata=rollback_metadata,
                model_weights=target_model.model_weights.copy(),
                optimizer_state=target_model.optimizer_state,
                training_config=target_model.training_config,
                validation_results=target_model.validation_results
            )
            
            # Save rollback version
            success = await self.storage.save_model(rollback_version)
            if not success:
                raise ValueError("Failed to save rollback version")
            
            # Update active version
            self.active_versions[training_id] = rollback_version_id
            self.version_history[training_id].append(rollback_version_id)
            
            self.logger.info(f"Rolled back {training_id} to version {target_version} "
                           f"(new version: {rollback_version_id})")
            
            return rollback_version_id
            
        except Exception as e:
            self.logger.error(f"Failed to rollback training {training_id}: {e}")
            return None
    
    async def compare_versions(self, training_id: str, version1: str, version2: str) -> Optional[Dict[str, Any]]:
        """Compare two model versions"""
        try:
            model1 = await self.storage.load_model(training_id, version1)
            model2 = await self.storage.load_model(training_id, version2)
            
            if not model1 or not model2:
                return None
            
            diff = self.comparator.compute_model_diff(model1, model2)
            return diff
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {e}")
            return None
    
    def setup_rollback_config(self, training_id: str, config: RollbackConfig):
        """Setup rollback configuration for a training session"""
        self.rollback_configs[training_id] = config
        self.logger.info(f"Setup rollback config for {training_id}")
    
    async def _monitor_performance_degradation(self):
        """Monitor for performance degradation and trigger rollbacks"""
        while not self._shutdown_event.is_set():
            try:
                for training_id, config in self.rollback_configs.items():
                    if not config.enabled:
                        continue
                    
                    await self._check_performance_degradation(training_id, config)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_performance_degradation(self, training_id: str, config: RollbackConfig):
        """Check for performance degradation and trigger rollback if needed"""
        try:
            history = self.performance_history.get(training_id, deque())
            if len(history) < config.monitoring_rounds:
                return
            
            # Get recent performance
            recent_performance = list(history)[-config.monitoring_rounds:]
            current_accuracy = recent_performance[-1].get("accuracy", 0.0)
            
            # Find best historical performance
            best_accuracy = max(h.get("accuracy", 0.0) for h in history)
            
            # Check degradation
            degradation = (best_accuracy - current_accuracy) / best_accuracy
            
            if degradation > config.performance_threshold:
                # Count consecutive degradations
                consecutive_count = 0
                for i in range(len(recent_performance) - 1, -1, -1):
                    perf = recent_performance[i].get("accuracy", 0.0)
                    if (best_accuracy - perf) / best_accuracy > config.performance_threshold:
                        consecutive_count += 1
                    else:
                        break
                
                if consecutive_count >= config.consecutive_degradations:
                    self.logger.warning(f"Performance degradation detected for {training_id}: "
                                      f"{degradation:.2%} over {consecutive_count} rounds")
                    
                    if config.rollback_to_best:
                        rollback_version = await self.rollback_to_version(training_id)
                        if rollback_version:
                            self.logger.info(f"Automatic rollback triggered for {training_id}")
        
        except Exception as e:
            self.logger.error(f"Error checking performance degradation: {e}")
    
    async def _cleanup_old_versions(self):
        """Clean up old model versions"""
        while not self._shutdown_event.is_set():
            try:
                if not self.config.auto_cleanup_enabled:
                    await asyncio.sleep(3600)  # Check every hour
                    continue
                
                cleanup_threshold = datetime.utcnow() - timedelta(days=self.config.cleanup_after_days)
                
                for training_id, versions in self.version_history.items():
                    versions_to_remove = []
                    
                    for version_id in versions:
                        # Load metadata to check creation date
                        version = await self.storage.load_model(training_id, version_id)
                        if version and version.metadata.created_at < cleanup_threshold:
                            # Don't remove active version or best version
                            if (version_id != self.active_versions.get(training_id) and 
                                version_id != self.rollback_candidates.get(training_id)):
                                versions_to_remove.append(version_id)
                    
                    # Remove old versions
                    for version_id in versions_to_remove:
                        success = await self.storage.delete_model(training_id, version_id)
                        if success:
                            self.version_history[training_id].remove(version_id)
                            self.logger.info(f"Cleaned up old version {training_id}:{version_id}")
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    # Public API methods
    def get_version_history(self, training_id: str) -> List[str]:
        """Get version history for a training session"""
        return self.version_history.get(training_id, []).copy()
    
    def get_active_version(self, training_id: str) -> Optional[str]:
        """Get active version for a training session"""
        return self.active_versions.get(training_id)
    
    def get_best_version(self, training_id: str) -> Optional[str]:
        """Get best version for a training session"""
        return self.rollback_candidates.get(training_id)
    
    async def list_all_models(self) -> Dict[str, List[str]]:
        """List all models and their versions"""
        result = {}
        for training_id in self.version_history:
            result[training_id] = await self.storage.list_versions(training_id)
        return result
    
    def get_version_stats(self) -> Dict[str, Any]:
        """Get versioning statistics"""
        total_versions = sum(len(versions) for versions in self.version_history.values())
        
        return {
            "total_trainings": len(self.version_history),
            "total_versions": total_versions,
            "active_trainings": len(self.active_versions),
            "storage_backend": self.config.storage_backend,
            "auto_cleanup_enabled": self.config.auto_cleanup_enabled,
            "checkpoint_interval": self.config.checkpoint_interval
        }
    
    async def shutdown(self):
        """Shutdown the version manager"""
        self.logger.info("Shutting down Model Version Manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Model Version Manager shutdown complete")