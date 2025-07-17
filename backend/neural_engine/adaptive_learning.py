#!/usr/bin/env python3
"""
Adaptive Learning System
Implements meta-learning and adaptive optimization for neural networks
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive learning system"""
    # Learning parameters
    meta_learning_rate: float = 0.001
    adaptation_steps: int = 5
    adaptation_learning_rate: float = 0.01
    
    # Memory parameters
    memory_size: int = 1000
    experience_replay: bool = True
    prioritized_replay: bool = True
    
    # Optimization parameters
    optimizer_type: str = "adam"
    scheduler_type: str = "cosine"
    gradient_clipping: float = 1.0
    
    # Regularization
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    
    # Adaptation strategies
    enable_maml: bool = True  # Model-Agnostic Meta-Learning
    enable_reptile: bool = False  # Reptile algorithm
    enable_online_adaptation: bool = True
    
    # Device settings
    device: str = "auto"

class ExperienceBuffer:
    """Experience replay buffer for adaptive learning"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, experience: Dict[str, Any], priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = False) -> List[Dict[str, Any]]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized and len(self.priorities) == len(self.buffer):
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probabilities = priorities / priorities.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            return [self.buffer[i] for i in indices]
        else:
            # Random sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def update_priority(self, index: int, priority: float):
        """Update priority of experience"""
        if 0 <= index < len(self.priorities):
            self.priorities[index] = priority
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.priorities.clear()

class MAMLOptimizer:
    """Model-Agnostic Meta-Learning optimizer"""
    
    def __init__(self, model: nn.Module, config: AdaptiveConfig):
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(
            model.parameters(), 
            lr=config.meta_learning_rate
        )
        
    def meta_update(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform meta-learning update"""
        meta_loss = 0.0
        task_losses = []
        
        for task in tasks:
            # Clone model for task-specific adaptation
            task_model = self._clone_model()
            
            # Adapt to task
            task_loss = self._adapt_to_task(task_model, task)
            task_losses.append(task_loss)
            
            # Calculate meta-gradient
            meta_loss += task_loss
        
        # Average meta-loss
        meta_loss /= len(tasks)
        
        # Meta-optimization step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clipping
        )
        
        self.meta_optimizer.step()
        
        return {
            "meta_loss": meta_loss.item(),
            "task_losses": task_losses,
            "num_tasks": len(tasks)
        }
    
    def _clone_model(self) -> nn.Module:
        """Create a copy of the model for task adaptation"""
        # Create a deep copy of the model
        cloned_model = type(self.model)()
        cloned_model.load_state_dict(self.model.state_dict())
        return cloned_model
    
    def _adapt_to_task(self, task_model: nn.Module, task: Dict[str, Any]) -> torch.Tensor:
        """Adapt model to specific task"""
        # Task-specific optimizer
        task_optimizer = optim.SGD(
            task_model.parameters(), 
            lr=self.config.adaptation_learning_rate
        )
        
        support_data = task["support"]
        query_data = task["query"]
        
        # Adaptation steps on support set
        for _ in range(self.config.adaptation_steps):
            task_optimizer.zero_grad()
            
            # Forward pass on support data
            support_output = task_model(support_data["input"])
            support_loss = nn.functional.mse_loss(support_output, support_data["target"])
            
            # Backward pass
            support_loss.backward()
            task_optimizer.step()
        
        # Evaluate on query set
        with torch.no_grad():
            query_output = task_model(query_data["input"])
            query_loss = nn.functional.mse_loss(query_output, query_data["target"])
        
        return query_loss

class OnlineAdaptationSystem:
    """Online adaptation system for continuous learning"""
    
    def __init__(self, model: nn.Module, config: AdaptiveConfig):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Adaptation state
        self.adaptation_history = []
        self.performance_history = []
        self.current_lr = config.adaptation_learning_rate
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for adaptation"""
        if self.config.optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.adaptation_learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.adaptation_learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=100
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=50, 
                gamma=0.1
            )
        else:
            return None
    
    def adapt(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform online adaptation step"""
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.model(data["input"])
        loss = nn.functional.mse_loss(output, data["target"])
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clipping
        )
        
        # Optimization step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
            self.current_lr = self.scheduler.get_last_lr()[0]
        
        # Record adaptation
        adaptation_info = {
            "loss": loss.item(),
            "learning_rate": self.current_lr,
            "timestamp": datetime.now(timezone.utc)
        }
        
        self.adaptation_history.append(adaptation_info)
        
        # Performance tracking
        with torch.no_grad():
            accuracy = self._calculate_accuracy(output, data["target"])
            self.performance_history.append(accuracy)
        
        return adaptation_info
    
    def _calculate_accuracy(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy metric"""
        if output.dim() == 1:
            # Regression
            mse = nn.functional.mse_loss(output, target)
            return 1.0 / (1.0 + mse.item())
        else:
            # Classification
            predictions = torch.argmax(output, dim=1)
            targets = torch.argmax(target, dim=1)
            correct = (predictions == targets).float()
            return correct.mean().item()

class AdaptiveLearningSystem:
    """
    Main adaptive learning system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = AdaptiveConfig(**config) if config else AdaptiveConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Components
        self.model: Optional[nn.Module] = None
        self.maml_optimizer: Optional[MAMLOptimizer] = None
        self.online_adapter: Optional[OnlineAdaptationSystem] = None
        self.experience_buffer = ExperienceBuffer(self.config.memory_size)
        
        # State
        self.is_initialized = False
        self.adaptation_count = 0
        self.total_loss = 0.0
        
        # Metrics
        self.metrics = {
            "adaptations": 0,
            "average_loss": 0.0,
            "learning_rate": self.config.adaptation_learning_rate,
            "meta_updates": 0
        }
        
        logger.info("Adaptive learning system created")
    
    async def initialize(self, model: Optional[nn.Module] = None) -> bool:
        """Initialize adaptive learning system"""
        try:
            if self.is_initialized:
                return True
            
            # Create or use provided model
            if model is None:
                self.model = self._create_default_model()
            else:
                self.model = model
            
            self.model.to(self.device)
            
            # Initialize components
            if self.config.enable_maml:
                self.maml_optimizer = MAMLOptimizer(self.model, self.config)
            
            if self.config.enable_online_adaptation:
                self.online_adapter = OnlineAdaptationSystem(self.model, self.config)
            
            self.is_initialized = True
            logger.info("Adaptive learning system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive learning system initialization failed: {e}")
            return False
    
    def _create_default_model(self) -> nn.Module:
        """Create default neural network model"""
        return nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    async def adapt(self, input_data: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Perform adaptation step
        
        Args:
            input_data: Input tensor
            target: Target tensor (optional)
            
        Returns:
            Dictionary containing adaptation results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Adaptive learning system not initialized")
            
            input_data = input_data.to(self.device)
            if target is not None:
                target = target.to(self.device)
            
            # Prepare data
            data = {"input": input_data}
            if target is not None:
                data["target"] = target
            
            # Add to experience buffer
            if self.config.experience_replay:
                self.experience_buffer.add(data)
            
            # Perform adaptation
            adaptation_result = {}
            
            # Online adaptation
            if self.online_adapter and target is not None:
                online_result = self.online_adapter.adapt(data)
                adaptation_result["online"] = online_result
                
                # Update metrics
                self.metrics["adaptations"] += 1
                self.total_loss += online_result["loss"]
                self.metrics["average_loss"] = self.total_loss / self.metrics["adaptations"]
                self.metrics["learning_rate"] = online_result["learning_rate"]
            
            # Experience replay
            if self.config.experience_replay and len(self.experience_buffer.buffer) > 10:
                replay_result = await self._experience_replay()
                adaptation_result["replay"] = replay_result
            
            # Meta-learning (MAML)
            if self.maml_optimizer and len(self.experience_buffer.buffer) > 20:
                # Create tasks from experience buffer
                tasks = self._create_meta_tasks()
                if tasks:
                    meta_result = self.maml_optimizer.meta_update(tasks)
                    adaptation_result["meta"] = meta_result
                    self.metrics["meta_updates"] += 1
            
            # Generate output
            with torch.no_grad():
                output = self.model(input_data)
            
            return {
                "output": output,
                "adaptation_result": adaptation_result,
                "metrics": self.metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Adaptation failed: {e}")
            raise
    
    async def _experience_replay(self) -> Dict[str, Any]:
        """Perform experience replay"""
        try:
            # Sample batch from buffer
            batch = self.experience_buffer.sample(
                batch_size=min(32, len(self.experience_buffer.buffer)),
                prioritized=self.config.prioritized_replay
            )
            
            if not batch:
                return {"replayed_samples": 0}
            
            # Process batch
            replay_loss = 0.0
            for experience in batch:
                if "target" in experience:
                    output = self.model(experience["input"])
                    loss = nn.functional.mse_loss(output, experience["target"])
                    replay_loss += loss.item()
            
            replay_loss /= len(batch)
            
            return {
                "replayed_samples": len(batch),
                "replay_loss": replay_loss
            }
            
        except Exception as e:
            logger.error(f"Experience replay failed: {e}")
            return {"error": str(e)}
    
    def _create_meta_tasks(self) -> List[Dict[str, Any]]:
        """Create meta-learning tasks from experience buffer"""
        try:
            if len(self.experience_buffer.buffer) < 20:
                return []
            
            tasks = []
            
            # Create multiple tasks
            for _ in range(min(5, len(self.experience_buffer.buffer) // 10)):
                # Sample support and query sets
                support_samples = self.experience_buffer.sample(5)
                query_samples = self.experience_buffer.sample(3)
                
                # Filter samples with targets
                support_samples = [s for s in support_samples if "target" in s]
                query_samples = [q for q in query_samples if "target" in q]
                
                if support_samples and query_samples:
                    # Create support set
                    support_inputs = torch.stack([s["input"] for s in support_samples])
                    support_targets = torch.stack([s["target"] for s in support_samples])
                    
                    # Create query set
                    query_inputs = torch.stack([q["input"] for q in query_samples])
                    query_targets = torch.stack([q["target"] for q in query_samples])
                    
                    task = {
                        "support": {
                            "input": support_inputs,
                            "target": support_targets
                        },
                        "query": {
                            "input": query_inputs,
                            "target": query_targets
                        }
                    }
                    
                    tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Meta-task creation failed: {e}")
            return []
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        stats = {
            "total_adaptations": self.metrics["adaptations"],
            "average_loss": self.metrics["average_loss"],
            "current_learning_rate": self.metrics["learning_rate"],
            "meta_updates": self.metrics["meta_updates"],
            "experience_buffer_size": len(self.experience_buffer.buffer)
        }
        
        # Add performance history if available
        if self.online_adapter:
            stats["performance_history"] = self.online_adapter.performance_history[-100:]
            stats["adaptation_history"] = self.online_adapter.adaptation_history[-100:]
        
        return stats
    
    async def save(self, path: str) -> bool:
        """Save adaptive learning system"""
        try:
            save_data = {
                "config": self.config.__dict__,
                "model_state": self.model.state_dict() if self.model else None,
                "metrics": self.metrics,
                "adaptation_count": self.adaptation_count,
                "total_loss": self.total_loss
            }
            
            # Save optimizer states
            if self.online_adapter:
                save_data["online_optimizer_state"] = self.online_adapter.optimizer.state_dict()
                if self.online_adapter.scheduler:
                    save_data["scheduler_state"] = self.online_adapter.scheduler.state_dict()
            
            torch.save(save_data, path)
            logger.info(f"Adaptive learning system saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save adaptive learning system: {e}")
            return False
    
    async def load(self, path: str) -> bool:
        """Load adaptive learning system"""
        try:
            save_data = torch.load(path, map_location=self.device)
            
            # Load model state
            if save_data["model_state"] and self.model:
                self.model.load_state_dict(save_data["model_state"])
            
            # Load metrics
            self.metrics = save_data.get("metrics", self.metrics)
            self.adaptation_count = save_data.get("adaptation_count", 0)
            self.total_loss = save_data.get("total_loss", 0.0)
            
            # Load optimizer states
            if self.online_adapter and "online_optimizer_state" in save_data:
                self.online_adapter.optimizer.load_state_dict(save_data["online_optimizer_state"])
                
                if "scheduler_state" in save_data and self.online_adapter.scheduler:
                    self.online_adapter.scheduler.load_state_dict(save_data["scheduler_state"])
            
            logger.info(f"Adaptive learning system loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adaptive learning system: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get adaptive learning system status"""
        return {
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "config": self.config.__dict__,
            "metrics": self.metrics,
            "statistics": self.get_adaptation_statistics()
        }
    
    def health_check(self) -> bool:
        """Check adaptive learning system health"""
        try:
            return (
                self.is_initialized and
                self.model is not None and
                (self.online_adapter is not None or self.maml_optimizer is not None)
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown adaptive learning system"""
        try:
            # Clear experience buffer
            self.experience_buffer.clear()
            
            # Reset metrics
            self.metrics = {
                "adaptations": 0,
                "average_loss": 0.0,
                "learning_rate": self.config.adaptation_learning_rate,
                "meta_updates": 0
            }
            
            self.adaptation_count = 0
            self.total_loss = 0.0
            self.is_initialized = False
            
            logger.info("Adaptive learning system shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Adaptive learning system shutdown failed: {e}")
            return False