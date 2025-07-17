#!/usr/bin/env python3
"""
Neural Processor - Main neural processing engine for SutazAI
Integrates biological modeling, neuromorphic computing, and adaptive learning
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .biological_modeling import BiologicalNeuralNetwork, BiologicalConfig
from .neuromorphic_engine import NeuromorphicEngine, NeuromorphicConfig
from .adaptive_learning import AdaptiveLearningSystem, AdaptiveConfig
from .neural_optimizer import NeuralOptimizer, OptimizationConfig
from .synaptic_plasticity import SynapticPlasticityManager, PlasticityConfig
from .neural_memory import NeuralMemorySystem, MemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class NeuralConfig:
    """Configuration for neural processor"""
    # Core settings
    device: str = "auto"
    dtype: str = "float32"
    enable_cuda: bool = True
    enable_mixed_precision: bool = True
    
    # Architecture settings
    enable_biological_modeling: bool = True
    enable_neuromorphic_computing: bool = True
    enable_adaptive_learning: bool = True
    enable_synaptic_plasticity: bool = True
    enable_neural_memory: bool = True
    
    # Performance settings
    batch_size: int = 32
    num_workers: int = 4
    memory_efficient: bool = True
    compile_model: bool = True
    
    # Learning settings
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    activation_function: str = "gelu"
    
    # Biological settings
    biological_config: BiologicalConfig = field(default_factory=BiologicalConfig)
    neuromorphic_config: NeuromorphicConfig = field(default_factory=NeuromorphicConfig)
    adaptive_config: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    plasticity_config: PlasticityConfig = field(default_factory=PlasticityConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    
    # Storage settings
    model_dir: str = "/opt/sutazaiapp/backend/data/neural_models"
    checkpoint_dir: str = "/opt/sutazaiapp/backend/data/neural_checkpoints"
    log_dir: str = "/opt/sutazaiapp/backend/logs/neural"

class NeuralProcessor:
    """
    Main neural processing engine for SutazAI
    Combines biological modeling, neuromorphic computing, and adaptive learning
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize neural processor"""
        self.config = NeuralConfig(**config) if config else NeuralConfig()
        self.device = self._setup_device()
        self.dtype = getattr(torch, self.config.dtype)
        
        # Initialize subsystems
        self.biological_network: Optional[BiologicalNeuralNetwork] = None
        self.neuromorphic_engine: Optional[NeuromorphicEngine] = None
        self.adaptive_system: Optional[AdaptiveLearningSystem] = None
        self.optimizer: Optional[NeuralOptimizer] = None
        self.plasticity_manager: Optional[SynapticPlasticityManager] = None
        self.memory_system: Optional[NeuralMemorySystem] = None
        
        # State management
        self.is_initialized = False
        self.is_training = False
        self.is_running = False
        
        # Performance tracking
        self.metrics = {
            "processing_time": [],
            "memory_usage": [],
            "throughput": [],
            "accuracy": [],
            "loss": []
        }
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
        # Create directories
        self._create_directories()
        
        logger.info(f"Neural processor initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "auto":
            if torch.cuda.is_available() and self.config.enable_cuda:
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.config.model_dir,
            self.config.checkpoint_dir,
            self.config.log_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize neural processor subsystems"""
        try:
            with self._lock:
                if self.is_initialized:
                    logger.info("Neural processor already initialized")
                    return True
                
                logger.info("Initializing neural processor subsystems...")
                
                # Initialize biological neural network
                if self.config.enable_biological_modeling:
                    self.biological_network = BiologicalNeuralNetwork(
                        config=self.config.biological_config
                    )
                    await self.biological_network.initialize()
                    logger.info("Biological neural network initialized")
                
                # Initialize neuromorphic engine
                if self.config.enable_neuromorphic_computing:
                    self.neuromorphic_engine = NeuromorphicEngine(
                        config=self.config.neuromorphic_config
                    )
                    await self.neuromorphic_engine.initialize()
                    logger.info("Neuromorphic engine initialized")
                
                # Initialize adaptive learning system
                if self.config.enable_adaptive_learning:
                    self.adaptive_system = AdaptiveLearningSystem(
                        config=self.config.adaptive_config
                    )
                    await self.adaptive_system.initialize()
                    logger.info("Adaptive learning system initialized")
                
                # Initialize neural optimizer
                self.optimizer = NeuralOptimizer(
                    config=self.config.optimization_config
                )
                await self.optimizer.initialize()
                logger.info("Neural optimizer initialized")
                
                # Initialize synaptic plasticity manager
                if self.config.enable_synaptic_plasticity:
                    self.plasticity_manager = SynapticPlasticityManager(
                        config=self.config.plasticity_config
                    )
                    await self.plasticity_manager.initialize()
                    logger.info("Synaptic plasticity manager initialized")
                
                # Initialize neural memory system
                if self.config.enable_neural_memory:
                    self.memory_system = NeuralMemorySystem(
                        config=self.config.memory_config
                    )
                    await self.memory_system.initialize()
                    logger.info("Neural memory system initialized")
                
                self.is_initialized = True
                logger.info("Neural processor initialization completed")
                return True
                
        except Exception as e:
            logger.error(f"Neural processor initialization failed: {e}")
            return False
    
    async def process_input(self, input_data: Union[torch.Tensor, np.ndarray, List], 
                          processing_mode: str = "inference") -> Dict[str, Any]:
        """
        Process input through neural processing pipeline
        
        Args:
            input_data: Input data to process
            processing_mode: "inference", "training", or "adaptive"
            
        Returns:
            Dictionary containing processing results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Neural processor not initialized")
            
            start_time = datetime.now(timezone.utc)
            
            # Convert input to tensor
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(self.device, dtype=self.dtype)
            elif isinstance(input_data, list):
                input_tensor = torch.tensor(input_data, device=self.device, dtype=self.dtype)
            elif isinstance(input_data, torch.Tensor):
                input_tensor = input_data.to(self.device, dtype=self.dtype)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            # Process through neural pipeline
            results = {}
            
            # Biological processing
            if self.biological_network:
                bio_result = await self.biological_network.process(input_tensor)
                results["biological"] = bio_result
                input_tensor = bio_result.get("output", input_tensor)
            
            # Neuromorphic processing
            if self.neuromorphic_engine:
                neuro_result = await self.neuromorphic_engine.process(input_tensor)
                results["neuromorphic"] = neuro_result
                input_tensor = neuro_result.get("output", input_tensor)
            
            # Adaptive processing
            if self.adaptive_system and processing_mode in ["training", "adaptive"]:
                adaptive_result = await self.adaptive_system.adapt(input_tensor)
                results["adaptive"] = adaptive_result
                input_tensor = adaptive_result.get("output", input_tensor)
            
            # Apply synaptic plasticity
            if self.plasticity_manager and processing_mode == "training":
                plasticity_result = await self.plasticity_manager.update_plasticity(input_tensor)
                results["plasticity"] = plasticity_result
            
            # Store in neural memory
            if self.memory_system:
                await self.memory_system.store(input_tensor, results)
            
            # Final processing
            final_output = await self._final_processing(input_tensor, results)
            
            # Calculate metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.metrics["processing_time"].append(processing_time)
            
            return {
                "output": final_output,
                "results": results,
                "processing_time": processing_time,
                "device": str(self.device),
                "mode": processing_mode
            }
            
        except Exception as e:
            logger.error(f"Neural processing failed: {e}")
            raise
    
    async def _final_processing(self, input_tensor: torch.Tensor, 
                              intermediate_results: Dict[str, Any]) -> torch.Tensor:
        """Final processing step"""
        # Apply optimization if available
        if self.optimizer:
            optimized_tensor = await self.optimizer.optimize(input_tensor)
            return optimized_tensor
        
        return input_tensor
    
    async def train(self, training_data: List[Dict[str, Any]], 
                   validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Train neural processor on provided data
        
        Args:
            training_data: List of training samples
            validation_data: Optional validation samples
            
        Returns:
            Training results and metrics
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Neural processor not initialized")
            
            with self._lock:
                self.is_training = True
            
            logger.info(f"Starting training on {len(training_data)} samples")
            
            training_results = {
                "epoch_losses": [],
                "epoch_accuracies": [],
                "validation_losses": [],
                "validation_accuracies": [],
                "training_time": 0
            }
            
            start_time = datetime.now(timezone.utc)
            
            # Training loop
            for epoch in range(self.config.optimization_config.num_epochs):
                epoch_loss = 0.0
                epoch_accuracy = 0.0
                
                # Process training batches
                for batch in self._create_batches(training_data):
                    batch_result = await self.process_input(
                        batch["input"], 
                        processing_mode="training"
                    )
                    
                    # Calculate loss and accuracy
                    loss = self._calculate_loss(batch_result["output"], batch["target"])
                    accuracy = self._calculate_accuracy(batch_result["output"], batch["target"])
                    
                    epoch_loss += loss
                    epoch_accuracy += accuracy
                    
                    # Update neural parameters
                    if self.optimizer:
                        await self.optimizer.update_parameters(loss)
                
                # Average metrics
                epoch_loss /= len(training_data)
                epoch_accuracy /= len(training_data)
                
                training_results["epoch_losses"].append(epoch_loss)
                training_results["epoch_accuracies"].append(epoch_accuracy)
                
                # Validation
                if validation_data:
                    val_loss, val_accuracy = await self._validate(validation_data)
                    training_results["validation_losses"].append(val_loss)
                    training_results["validation_accuracies"].append(val_accuracy)
                
                logger.info(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
            
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            training_results["training_time"] = training_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            with self._lock:
                self.is_training = False
    
    async def _validate(self, validation_data: List[Dict[str, Any]]) -> Tuple[float, float]:
        """Validate model on validation data"""
        total_loss = 0.0
        total_accuracy = 0.0
        
        for batch in self._create_batches(validation_data):
            batch_result = await self.process_input(
                batch["input"], 
                processing_mode="inference"
            )
            
            loss = self._calculate_loss(batch_result["output"], batch["target"])
            accuracy = self._calculate_accuracy(batch_result["output"], batch["target"])
            
            total_loss += loss
            total_accuracy += accuracy
        
        return total_loss / len(validation_data), total_accuracy / len(validation_data)
    
    def _create_batches(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create batches from data"""
        batches = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            # Stack inputs and targets
            inputs = torch.stack([item["input"] for item in batch_data])
            targets = torch.stack([item["target"] for item in batch_data])
            
            batches.append({
                "input": inputs,
                "target": targets
            })
        
        return batches
    
    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate loss between output and target"""
        if output.dim() == 1:
            # Regression loss
            loss = nn.MSELoss()(output, target)
        else:
            # Classification loss
            loss = nn.CrossEntropyLoss()(output, target)
        
        return loss.item()
    
    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy between output and target"""
        if output.dim() == 1:
            # Regression accuracy (using MSE as inverted accuracy)
            mse = nn.MSELoss()(output, target)
            return 1.0 / (1.0 + mse.item())
        else:
            # Classification accuracy
            predictions = torch.argmax(output, dim=1)
            correct = (predictions == target).float()
            return correct.mean().item()
    
    async def save_model(self, model_path: str) -> bool:
        """Save neural processor state"""
        try:
            model_data = {
                "config": self.config.__dict__,
                "metrics": self.metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Save biological network
            if self.biological_network:
                bio_path = f"{model_path}_biological.pt"
                await self.biological_network.save(bio_path)
                model_data["biological_path"] = bio_path
            
            # Save neuromorphic engine
            if self.neuromorphic_engine:
                neuro_path = f"{model_path}_neuromorphic.pt"
                await self.neuromorphic_engine.save(neuro_path)
                model_data["neuromorphic_path"] = neuro_path
            
            # Save adaptive system
            if self.adaptive_system:
                adaptive_path = f"{model_path}_adaptive.pt"
                await self.adaptive_system.save(adaptive_path)
                model_data["adaptive_path"] = adaptive_path
            
            # Save main model data
            with open(f"{model_path}_processor.json", "w") as f:
                json.dump(model_data, f, indent=2)
            
            logger.info(f"Neural processor saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    async def load_model(self, model_path: str) -> bool:
        """Load neural processor state"""
        try:
            # Load main model data
            with open(f"{model_path}_processor.json", "r") as f:
                model_data = json.load(f)
            
            # Load biological network
            if "biological_path" in model_data and self.biological_network:
                await self.biological_network.load(model_data["biological_path"])
            
            # Load neuromorphic engine
            if "neuromorphic_path" in model_data and self.neuromorphic_engine:
                await self.neuromorphic_engine.load(model_data["neuromorphic_path"])
            
            # Load adaptive system
            if "adaptive_path" in model_data and self.adaptive_system:
                await self.adaptive_system.load(model_data["adaptive_path"])
            
            # Restore metrics
            self.metrics = model_data.get("metrics", self.metrics)
            
            logger.info(f"Neural processor loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get neural processor metrics"""
        return {
            "processing_times": self.metrics["processing_time"][-100:],  # Last 100
            "average_processing_time": np.mean(self.metrics["processing_time"]) if self.metrics["processing_time"] else 0,
            "memory_usage": self.metrics["memory_usage"][-100:],
            "throughput": self.metrics["throughput"][-100:],
            "is_initialized": self.is_initialized,
            "is_training": self.is_training,
            "is_running": self.is_running,
            "device": str(self.device),
            "config": self.config.__dict__
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        return {
            "biological_network": self.biological_network.get_status() if self.biological_network else None,
            "neuromorphic_engine": self.neuromorphic_engine.get_status() if self.neuromorphic_engine else None,
            "adaptive_system": self.adaptive_system.get_status() if self.adaptive_system else None,
            "optimizer": self.optimizer.get_status() if self.optimizer else None,
            "plasticity_manager": self.plasticity_manager.get_status() if self.plasticity_manager else None,
            "memory_system": self.memory_system.get_status() if self.memory_system else None
        }
    
    async def shutdown(self) -> bool:
        """Shutdown neural processor"""
        try:
            logger.info("Shutting down neural processor...")
            
            with self._lock:
                self.is_running = False
                self.is_training = False
            
            # Shutdown subsystems
            if self.biological_network:
                await self.biological_network.shutdown()
            
            if self.neuromorphic_engine:
                await self.neuromorphic_engine.shutdown()
            
            if self.adaptive_system:
                await self.adaptive_system.shutdown()
            
            if self.optimizer:
                await self.optimizer.shutdown()
            
            if self.plasticity_manager:
                await self.plasticity_manager.shutdown()
            
            if self.memory_system:
                await self.memory_system.shutdown()
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            logger.info("Neural processor shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Neural processor shutdown failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check neural processor health"""
        try:
            if not self.is_initialized:
                return False
            
            # Check subsystems
            if self.biological_network and not self.biological_network.health_check():
                return False
            
            if self.neuromorphic_engine and not self.neuromorphic_engine.health_check():
                return False
            
            if self.adaptive_system and not self.adaptive_system.health_check():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_neural_processor(config: Optional[Dict[str, Any]] = None) -> NeuralProcessor:
    """Create neural processor instance"""
    return NeuralProcessor(config=config)