"""
Federated Learning Client Framework
==================================

Client-side implementation for federated learning in SutazAI agents.
Provides local training, model updates, and communication with coordinator.

Features:
- Local model training with CPU optimization
- Model compression and differential privacy
- Asynchronous communication with coordinator
- Resource-aware training scheduling
- Byzantine-robust client behavior
- Integration with existing agent infrastructure
"""

import asyncio
import json
import logging
import time
import uuid
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import pickle
import zlib
from concurrent.futures import ThreadPoolExecutor

import aioredis
from pydantic import BaseModel

from agents.core.base_agent import BaseAgent, AgentMessage, AgentCapability, AgentStatus
from .privacy import PrivacyManager, DifferentialPrivacyConfig
from .aggregator import ClientUpdate


class TrainingMode(Enum):
    """Local training modes"""
    BATCH = "batch"
    MINI_BATCH = "mini_batch"
    ONLINE = "online"
    FEDERATED = "federated"


class ClientStatus(Enum):
    """Federated client status"""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class LocalTrainingConfig:
    """Configuration for local training"""
    model_type: str
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 5
    optimizer: str = "sgd"  # sgd, adam, rmsprop
    momentum: float = 0.9
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    early_stopping_patience: int = 3
    validation_split: float = 0.2
    shuffle_data: bool = True
    privacy_config: Optional[DifferentialPrivacyConfig] = None
    compression_ratio: float = 1.0
    max_grad_norm: Optional[float] = None


@dataclass
class TrainingJob:
    """Local training job"""
    job_id: str
    training_id: str
    round_id: str
    round_number: int
    global_model: Dict[str, np.ndarray]
    config: LocalTrainingConfig
    data_samples: int
    start_time: datetime
    deadline: datetime
    status: ClientStatus
    progress: float = 0.0
    error_message: Optional[str] = None
    result: Optional[ClientUpdate] = None


class LocalModel(ABC):
    """Abstract base class for local machine learning models"""
    
    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights"""
        pass
    
    @abstractmethod
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights"""
        pass
    
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              config: LocalTrainingConfig) -> Dict[str, float]:
        """Train the model locally"""
        pass
    
    @abstractmethod
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        pass


class SimpleNeuralNetwork(LocalModel):
    """Simple neural network implementation for federated learning"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Initialize weights
        self.weights = self._initialize_weights()
        
        # Training state
        self.optimizer_state = {}
        
    def _initialize_weights(self) -> Dict[str, np.ndarray]:
        """Initialize network weights"""
        weights = {}
        
        # Input to first hidden layer
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Xavier initialization
            limit = np.sqrt(6.0 / (prev_dim + hidden_dim))
            weights[f'W{i}'] = np.random.uniform(-limit, limit, (prev_dim, hidden_dim))
            weights[f'b{i}'] = np.zeros(hidden_dim)
            prev_dim = hidden_dim
        
        # Last hidden to output layer
        limit = np.sqrt(6.0 / (prev_dim + self.output_dim))
        weights[f'W{len(self.hidden_dims)}'] = np.random.uniform(-limit, limit, (prev_dim, self.output_dim))
        weights[f'b{len(self.hidden_dims)}'] = np.zeros(self.output_dim)
        
        return weights
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get model weights"""
        return {k: v.copy() for k, v in self.weights.items()}
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set model weights"""
        self.weights = {k: v.copy() for k, v in weights.items()}
    
    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass with activations stored for backprop"""
        activations = [x]
        current = x
        
        # Forward through hidden layers
        for i in range(len(self.hidden_dims)):
            z = np.dot(current, self.weights[f'W{i}']) + self.weights[f'b{i}']
            current = self._relu(z)
            activations.append(current)
        
        # Output layer
        z_out = np.dot(current, self.weights[f'W{len(self.hidden_dims)}']) + self.weights[f'b{len(self.hidden_dims)}']
        output = self._softmax(z_out)
        activations.append(output)
        
        return output, activations
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, 
              config: LocalTrainingConfig) -> Dict[str, float]:
        """Train the model using mini-batch SGD"""
        n_samples = x_train.shape[0]
        n_batches = max(1, n_samples // config.batch_size)
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Shuffle data if configured
            if config.shuffle_data:
                indices = np.random.permutation(n_samples)
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, n_samples)
                
                x_batch = x_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Forward pass
                predictions, activations = self._forward(x_batch)
                
                # Compute loss and accuracy
                loss = self._cross_entropy_loss(predictions, y_batch)
                accuracy = self._accuracy(predictions, y_batch)
                
                # Backward pass
                gradients = self._backward(predictions, y_batch, activations, x_batch)
                
                # Apply gradients
                if config.optimizer == "sgd":
                    self._apply_sgd(gradients, config.learning_rate, config.momentum)
                elif config.optimizer == "adam":
                    self._apply_adam(gradients, config.learning_rate, epoch * n_batches + batch_idx)
                
                epoch_loss += loss
                epoch_accuracy += accuracy
            
            # Average metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(avg_accuracy)
        
        return {
            "final_loss": history["loss"][-1],
            "final_accuracy": history["accuracy"][-1],
            "loss_history": history["loss"],
            "accuracy_history": history["accuracy"]
        }
    
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model"""
        predictions, _ = self._forward(x_test)
        
        loss = self._cross_entropy_loss(predictions, y_test)
        accuracy = self._accuracy(predictions, y_test)
        
        return {"loss": loss, "accuracy": accuracy}
    
    def _cross_entropy_loss(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss"""
        n_samples = predictions.shape[0]
        # Clip predictions to prevent log(0)
        predictions = np.clip(predictions, 1e-12, 1.0 - 1e-12)
        
        if len(y_true.shape) == 1:  # Integer labels
            loss = -np.sum(np.log(predictions[np.arange(n_samples), y_true]))
        else:  # One-hot encoded
            loss = -np.sum(y_true * np.log(predictions))
        
        return loss / n_samples
    
    def _accuracy(self, predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Classification accuracy"""
        pred_labels = np.argmax(predictions, axis=1)
        
        if len(y_true.shape) == 1:  # Integer labels
            true_labels = y_true
        else:  # One-hot encoded
            true_labels = np.argmax(y_true, axis=1)
        
        return np.mean(pred_labels == true_labels)
    
    def _backward(self, predictions: np.ndarray, y_true: np.ndarray, 
                  activations: List[np.ndarray], x_batch: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass to compute gradients"""
        gradients = {}
        n_samples = x_batch.shape[0]
        
        # Output layer gradient
        if len(y_true.shape) == 1:  # Integer labels
            y_one_hot = np.zeros_like(predictions)
            y_one_hot[np.arange(n_samples), y_true] = 1
        else:
            y_one_hot = y_true
        
        # Gradient of output layer
        delta = (predictions - y_one_hot) / n_samples
        
        # Backpropagate through layers
        for i in range(len(self.hidden_dims), -1, -1):
            # Gradient w.r.t. weights and biases
            if i == 0:
                gradients[f'W{i}'] = np.dot(activations[i].T, delta)
            else:
                gradients[f'W{i}'] = np.dot(activations[i].T, delta)
            
            gradients[f'b{i}'] = np.sum(delta, axis=0)
            
            # Gradient w.r.t. previous layer activations
            if i > 0:
                delta = np.dot(delta, self.weights[f'W{i}'].T)
                # Apply derivative of ReLU
                delta = delta * self._relu_derivative(activations[i])
        
        return gradients
    
    def _apply_sgd(self, gradients: Dict[str, np.ndarray], lr: float, momentum: float):
        """Apply SGD with momentum"""
        for param_name, grad in gradients.items():
            if param_name not in self.optimizer_state:
                self.optimizer_state[param_name] = np.zeros_like(grad)
            
            # Momentum update
            self.optimizer_state[param_name] = (momentum * self.optimizer_state[param_name] - 
                                               lr * grad)
            
            # Apply update
            self.weights[param_name] += self.optimizer_state[param_name]
    
    def _apply_adam(self, gradients: Dict[str, np.ndarray], lr: float, t: int):
        """Apply Adam optimizer"""
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        for param_name, grad in gradients.items():
            if param_name not in self.optimizer_state:
                self.optimizer_state[param_name] = {
                    'm': np.zeros_like(grad),
                    'v': np.zeros_like(grad)
                }
            
            m = self.optimizer_state[param_name]['m']
            v = self.optimizer_state[param_name]['v']
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** (t + 1))
            
            # Compute bias-corrected second raw moment estimate
            v_hat = v / (1 - beta2 ** (t + 1))
            
            # Update parameters
            self.weights[param_name] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Store updated moments
            self.optimizer_state[param_name]['m'] = m
            self.optimizer_state[param_name]['v'] = v


class FederatedClient:
    """
    Federated Learning Client
    
    Integrates with SutazAI agents to provide federated learning capabilities.
    Handles local training, model updates, and coordination with the federated server.
    """
    
    def __init__(self, 
                 agent_id: str,
                 redis_url: str = "redis://localhost:6379",
                 namespace: str = "sutazai:federated"):
        
        self.agent_id = agent_id
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        
        # Client state
        self.status = ClientStatus.IDLE
        self.capabilities = {
            "model_types": ["neural_network", "linear_regression", "logistic_regression"],
            "max_samples": 10000,
            "cpu_cores": 2,  # Per-client constraint
            "memory_mb": 1024
        }
        
        # Training management
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.job_queue: List[TrainingJob] = []
        self.models: Dict[str, LocalModel] = {}
        self.datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Privacy and security
        self.privacy_manager = PrivacyManager()
        
        # Performance tracking
        self.client_stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "average_training_time": 0.0,
            "total_samples_trained": 0,
            "contribution_score": 0.0
        }
        
        # Thread pool for CPU-intensive training
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger(f"federated_client_{agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize the federated client"""
        try:
            self.logger.info(f"Initializing Federated Client {self.agent_id}")
            
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            
            # Initialize privacy manager
            await self.privacy_manager.initialize()
            
            # Subscribe to coordinator messages
            await self._setup_message_handling()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Register capabilities with coordinator
            await self._register_capabilities()
            
            self.status = ClientStatus.IDLE
            self.logger.info(f"Federated Client {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize federated client: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background client tasks"""
        tasks = [
            self._job_executor(),
            self._heartbeat_sender(),
            self._message_processor(),
            self._performance_monitor()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _setup_message_handling(self):
        """Setup message handling from coordinator"""
        try:
            # Subscribe to agent-specific channel
            channel = f"{self.namespace}:agent:{self.agent_id}"
            
            # This would typically use Redis pub/sub, but for simplicity
            # we'll use a polling approach
            self.logger.info(f"Setup message handling for channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup message handling: {e}")
    
    async def _register_capabilities(self):
        """Register client capabilities with coordinator"""
        try:
            response_channel = f"{self.namespace}:coordinator:response:{self.agent_id}"
            
            capabilities_message = {
                "action": "register_capabilities",
                "agent_id": self.agent_id,
                "capabilities": self.capabilities,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis.lpush(response_channel, json.dumps(capabilities_message))
            self.logger.info(f"Registered capabilities: {self.capabilities}")
            
        except Exception as e:
            self.logger.error(f"Failed to register capabilities: {e}")
    
    async def handle_training_request(self, message_data: Dict[str, Any]) -> bool:
        """Handle federated training request from coordinator"""
        try:
            training_id = message_data["training_id"]
            round_id = message_data["round_id"]
            round_number = message_data["round_number"]
            global_model = message_data["global_model"]
            training_config_data = message_data.get("training_config", {})
            
            # Create local training configuration
            config = LocalTrainingConfig(
                model_type=global_model.get("model_type", "neural_network"),
                learning_rate=training_config_data.get("learning_rate", 0.01),
                batch_size=training_config_data.get("batch_size", 32),
                epochs=training_config_data.get("local_epochs", 5)
            )
            
            # Create training job
            job = TrainingJob(
                job_id=str(uuid.uuid4()),
                training_id=training_id,
                round_id=round_id,
                round_number=round_number,
                global_model=self._convert_model_format(global_model),
                config=config,
                data_samples=self._get_available_samples(),
                start_time=datetime.utcnow(),
                deadline=datetime.utcnow() + timedelta(minutes=30),  # 30 min deadline
                status=ClientStatus.WAITING
            )
            
            # Add to job queue
            self.job_queue.append(job)
            self.client_stats["total_jobs"] += 1
            
            self.logger.info(f"Accepted training job {job.job_id} for round {round_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to handle training request: {e}")
            return False
    
    def _convert_model_format(self, global_model: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Convert global model from serialized format to numpy arrays"""
        model_weights = {}
        
        for layer_name, weights in global_model.get("weights", {}).items():
            if isinstance(weights, list):
                model_weights[layer_name] = np.array(weights)
            else:
                model_weights[layer_name] = weights
        
        return model_weights
    
    def _get_available_samples(self) -> int:
        """Get number of available training samples"""
        total_samples = 0
        for dataset_name, (x, y) in self.datasets.items():
            total_samples += x.shape[0]
        
        return min(total_samples, self.capabilities["max_samples"])
    
    async def _job_executor(self):
        """Execute training jobs from the queue"""
        while not self._shutdown_event.is_set():
            try:
                if self.job_queue and self.status == ClientStatus.IDLE:
                    job = self.job_queue.pop(0)
                    await self._execute_training_job(job)
                
                await asyncio.sleep(1)  # Check for jobs every second
                
            except Exception as e:
                self.logger.error(f"Job executor error: {e}")
                await asyncio.sleep(5)
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute a single training job"""
        try:
            self.logger.info(f"Starting training job {job.job_id}")
            self.status = ClientStatus.TRAINING
            job.status = ClientStatus.TRAINING
            self.active_jobs[job.job_id] = job
            
            start_time = time.time()
            
            # Get or create model
            model = self._get_or_create_model(job.config.model_type)
            
            # Set global model weights
            model.set_weights(job.global_model)
            
            # Get training data
            x_train, y_train = self._get_training_data(job.training_id)
            
            if x_train is None or len(x_train) == 0:
                raise ValueError("No training data available")
            
            # Apply privacy if configured
            if job.config.privacy_config:
                x_train, y_train = await self.privacy_manager.apply_privacy(
                    x_train, y_train, job.config.privacy_config
                )
            
            # Train model locally
            training_result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, model.train, x_train, y_train, job.config
            )
            
            # Evaluate model
            eval_result = model.evaluate(x_train, y_train)  # On training data for simplicity
            
            # Create client update
            training_time = time.time() - start_time
            
            client_update = ClientUpdate(
                client_id=self.agent_id,
                model_weights=model.get_weights(),
                num_samples=len(x_train),
                loss=eval_result["loss"],
                accuracy=eval_result["accuracy"],
                computation_time=training_time,
                communication_time=0.0,  # Will be measured during upload
                metadata={
                    "training_result": training_result,
                    "job_id": job.job_id,
                    "round_number": job.round_number
                }
            )
            
            job.result = client_update
            job.status = ClientStatus.UPLOADING
            
            # Send update to coordinator
            await self._send_update_to_coordinator(job)
            
            # Complete job
            job.status = ClientStatus.IDLE
            self.status = ClientStatus.IDLE
            
            # Update statistics
            self.client_stats["completed_jobs"] += 1
            self.client_stats["total_samples_trained"] += len(x_train)
            self.client_stats["average_training_time"] = (
                (self.client_stats["average_training_time"] * (self.client_stats["completed_jobs"] - 1) + training_time) /
                self.client_stats["completed_jobs"]
            )
            
            self.logger.info(f"Completed training job {job.job_id} in {training_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Training job {job.job_id} failed: {e}")
            job.status = ClientStatus.ERROR
            job.error_message = str(e)
            self.status = ClientStatus.IDLE
            self.client_stats["failed_jobs"] += 1
        
        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    def _get_or_create_model(self, model_type: str) -> LocalModel:
        """Get or create a local model"""
        if model_type not in self.models:
            if model_type == "neural_network":
                # Simple neural network for demonstration
                self.models[model_type] = SimpleNeuralNetwork(
                    input_dim=784,  # MNIST-like
                    hidden_dims=[128, 64],
                    output_dim=10
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        return self.models[model_type]
    
    def _get_training_data(self, training_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get training data for the specified training session"""
        # For demonstration, generate synthetic data
        # In practice, this would load from local dataset storage
        
        if "synthetic" not in self.datasets:
            # Generate synthetic MNIST-like data
            n_samples = min(1000, self.capabilities["max_samples"])
            x = np.random.randn(n_samples, 784)
            y = np.random.randint(0, 10, n_samples)
            self.datasets["synthetic"] = (x, y)
        
        return self.datasets["synthetic"]
    
    async def _send_update_to_coordinator(self, job: TrainingJob):
        """Send model update to coordinator"""
        try:
            if not job.result:
                raise ValueError("No training result to send")
            
            upload_start = time.time()
            
            # Prepare update message
            update_message = {
                "client_id": self.agent_id,
                "training_id": job.training_id,
                "round_id": job.round_id,
                "model_weights": {k: v.tolist() for k, v in job.result.model_weights.items()},
                "num_samples": job.result.num_samples,
                "loss": job.result.loss,
                "accuracy": job.result.accuracy,
                "computation_time": job.result.computation_time,
                "communication_time": 0.0,  # Will update below
                "metadata": job.result.metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to coordinator's update channel
            update_channel = f"{self.namespace}:coordinator:updates:{job.round_id}"
            await self.redis.lpush(update_channel, json.dumps(update_message))
            
            # Update communication time
            communication_time = time.time() - upload_start
            job.result.communication_time = communication_time
            
            self.logger.info(f"Sent update for job {job.job_id} (upload time: {communication_time:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"Failed to send update for job {job.job_id}: {e}")
            raise
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat to coordinator"""
        while not self._shutdown_event.is_set():
            try:
                heartbeat_message = {
                    "action": "heartbeat",
                    "agent_id": self.agent_id,
                    "status": self.status.value,
                    "active_jobs": len(self.active_jobs),
                    "queue_length": len(self.job_queue),
                    "capabilities": self.capabilities,
                    "stats": self.client_stats,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Send heartbeat
                heartbeat_channel = f"{self.namespace}:coordinator:heartbeat"
                await self.redis.lpush(heartbeat_channel, json.dumps(heartbeat_message))
                
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)
    
    async def _message_processor(self):
        """Process incoming messages from coordinator"""
        while not self._shutdown_event.is_set():
            try:
                # Check for incoming messages
                channel = f"{self.namespace}:agent:{self.agent_id}"
                message = await self.redis.lpop(channel)
                
                if message:
                    message_data = json.loads(message)
                    await self._handle_coordinator_message(message_data)
                
                await asyncio.sleep(1)  # Check for messages every second
                
            except Exception as e:
                self.logger.error(f"Message processor error: {e}")
                await asyncio.sleep(5)
    
    async def _handle_coordinator_message(self, message_data: Dict[str, Any]):
        """Handle message from coordinator"""
        try:
            action = message_data.get("action")
            
            if action == "federated_training":
                await self.handle_training_request(message_data)
            elif action == "query_fl_capabilities":
                await self._send_capabilities_response()
            elif action == "cancel_training":
                await self._cancel_training(message_data.get("training_id"))
            else:
                self.logger.warning(f"Unknown action: {action}")
        
        except Exception as e:
            self.logger.error(f"Error handling coordinator message: {e}")
    
    async def _send_capabilities_response(self):
        """Send capabilities response to coordinator"""
        try:
            response_channel = f"{self.namespace}:coordinator:response:{self.agent_id}"
            
            response = {
                "action": "capabilities_response",
                "agent_id": self.agent_id,
                "capabilities": list(self.capabilities.keys()),
                "model_types": self.capabilities["model_types"],
                "max_samples": self.capabilities["max_samples"],
                "status": self.status.value,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis.lpush(response_channel, json.dumps(response))
            
        except Exception as e:
            self.logger.error(f"Failed to send capabilities response: {e}")
    
    async def _cancel_training(self, training_id: str):
        """Cancel training for specified training ID"""
        try:
            cancelled_jobs = []
            
            # Cancel queued jobs
            self.job_queue = [job for job in self.job_queue 
                            if job.training_id != training_id or cancelled_jobs.append(job.job_id)]
            
            # Cancel active jobs (simplified - in practice would need more sophisticated cancellation)
            for job_id, job in list(self.active_jobs.items()):
                if job.training_id == training_id:
                    job.status = ClientStatus.ERROR
                    job.error_message = "Training cancelled by coordinator"
                    cancelled_jobs.append(job_id)
            
            if cancelled_jobs:
                self.logger.info(f"Cancelled {len(cancelled_jobs)} jobs for training {training_id}")
        
        except Exception as e:
            self.logger.error(f"Error cancelling training {training_id}: {e}")
    
    async def _performance_monitor(self):
        """Monitor client performance"""
        while not self._shutdown_event.is_set():
            try:
                # Update contribution score based on successful completions
                if self.client_stats["total_jobs"] > 0:
                    success_rate = self.client_stats["completed_jobs"] / self.client_stats["total_jobs"]
                    self.client_stats["contribution_score"] = (
                        success_rate * self.client_stats["total_samples_trained"] / 1000.0
                    )
                
                # Log performance summary
                if self.client_stats["completed_jobs"] > 0:
                    self.logger.info(
                        f"Performance: {self.client_stats['completed_jobs']} jobs completed, "
                        f"avg time: {self.client_stats['average_training_time']:.2f}s, "
                        f"contribution score: {self.client_stats['contribution_score']:.2f}"
                    )
                
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
                await asyncio.sleep(300)
    
    # Public API methods
    def add_dataset(self, name: str, x: np.ndarray, y: np.ndarray):
        """Add a dataset for training"""
        self.datasets[name] = (x.copy(), y.copy())
        self.capabilities["max_samples"] = max(
            self.capabilities["max_samples"],
            sum(x.shape[0] for x, _ in self.datasets.values())
        )
        self.logger.info(f"Added dataset '{name}' with {x.shape[0]} samples")
    
    def get_client_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self.client_stats,
            "status": self.status.value,
            "active_jobs": len(self.active_jobs),
            "queued_jobs": len(self.job_queue),
            "available_datasets": list(self.datasets.keys()),
            "total_samples": sum(x.shape[0] for x, _ in self.datasets.values())
        }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        job = self.active_jobs.get(job_id)
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "training_id": job.training_id,
            "round_number": job.round_number,
            "status": job.status.value,
            "progress": job.progress,
            "start_time": job.start_time.isoformat(),
            "error_message": job.error_message
        }
    
    async def shutdown(self):
        """Shutdown the federated client"""
        self.logger.info(f"Shutting down Federated Client {self.agent_id}")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
        
        self.logger.info(f"Federated Client {self.agent_id} shutdown complete")


# Integration with SutazAI agent framework
class FederatedLearningCapability:
    """
    Federated Learning capability that can be added to existing SutazAI agents
    """
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.federated_client: Optional[FederatedClient] = None
        
    async def initialize(self) -> bool:
        """Initialize federated learning capability"""
        try:
            self.federated_client = FederatedClient(
                agent_id=self.agent.agent_id,
                redis_url=self.agent.redis_url if hasattr(self.agent, 'redis_url') else "redis://localhost:6379"
            )
            
            success = await self.federated_client.initialize()
            
            if success:
                # Add federated learning to agent capabilities
                if hasattr(self.agent, 'capabilities'):
                    self.agent.capabilities.add(AgentCapability.LEARNING)
                
                # Register message handler
                if hasattr(self.agent, 'register_message_handler'):
                    self.agent.register_message_handler(
                        "federated_training", 
                        self._handle_federated_message
                    )
            
            return success
            
        except Exception as e:
            logging.error(f"Failed to initialize federated learning capability: {e}")
            return False
    
    async def _handle_federated_message(self, message: AgentMessage) -> bool:
        """Handle federated learning messages"""
        if self.federated_client:
            return await self.federated_client.handle_training_request(message.content)
        return False
    
    def add_training_data(self, dataset_name: str, x: np.ndarray, y: np.ndarray):
        """Add training data to the federated client"""
        if self.federated_client:
            self.federated_client.add_dataset(dataset_name, x, y)
    
    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """Get federated training statistics"""
        if self.federated_client:
            return self.federated_client.get_client_stats()
        return None
    
    async def shutdown(self):
        """Shutdown federated learning capability"""
        if self.federated_client:
            await self.federated_client.shutdown()