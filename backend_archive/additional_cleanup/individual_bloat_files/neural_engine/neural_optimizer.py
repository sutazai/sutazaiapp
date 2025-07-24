#!/usr/bin/env python3
"""
Neural Optimizer
Advanced optimization techniques for neural networks
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for neural optimization"""
    # Basic optimization
    optimizer_type: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Advanced optimization
    use_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    
    # Learning rate scheduling
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    max_lr: float = 0.01
    min_lr: float = 1e-6
    
    # Training settings
    num_epochs: int = 100
    batch_size: int = 32
    gradient_clipping: float = 1.0
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Device settings
    device: str = "auto"

class LookaheadOptimizer:
    """Lookahead optimizer wrapper"""
    
    def __init__(self, base_optimizer: optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_weights[p] = p.data.clone()
    
    def step(self, closure=None):
        """Perform optimization step"""
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        
        # Lookahead update
        if self.step_count % self.k == 0:
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad and p in self.slow_weights:
                        # Update slow weights
                        self.slow_weights[p] += self.alpha * (p.data - self.slow_weights[p])
                        # Set fast weights to slow weights
                        p.data.copy_(self.slow_weights[p])
        
        return loss
    
    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dictionary"""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'slow_weights': self.slow_weights,
            'step_count': self.step_count
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.slow_weights = state_dict['slow_weights']
        self.step_count = state_dict['step_count']

class WarmupScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, max_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
        
        # Store initial learning rates
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Decay phase
            progress = (self.step_count - self.warmup_steps) / (10000 - self.warmup_steps)  # Assume 10000 total steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        # Update optimizer
        for group in self.optimizer.param_groups:
            group['lr'] = lr
    
    def get_lr(self):
        """Get current learning rate"""
        return [group['lr'] for group in self.optimizer.param_groups]

class NeuralOptimizer:
    """
    Advanced neural network optimizer with multiple optimization techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = OptimizationConfig(**config) if config else OptimizationConfig()
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        # Components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[any] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # State
        self.is_initialized = False
        self.current_epoch = 0
        self.step_count = 0
        
        # Metrics
        self.metrics = {
            "learning_rate": self.config.learning_rate,
            "gradient_norm": 0.0,
            "parameter_norm": 0.0,
            "optimization_steps": 0
        }
        
        # Loss tracking
        self.loss_history = []
        self.lr_history = []
        
        logger.info("Neural optimizer created")
    
    async def initialize(self, model: nn.Module) -> bool:
        """Initialize neural optimizer"""
        try:
            if self.is_initialized:
                return True
            
            self.model = model
            self.model.to(self.device)
            
            # Create base optimizer
            self.optimizer = self._create_optimizer()
            
            # Wrap with Lookahead if enabled
            if self.config.use_lookahead:
                self.optimizer = LookaheadOptimizer(
                    self.optimizer, 
                    k=self.config.lookahead_k, 
                    alpha=self.config.lookahead_alpha
                )
            
            # Create scheduler
            self.scheduler = self._create_scheduler()
            
            # Create gradient scaler for mixed precision
            if self.device.type == "cuda":
                self.scaler = torch.cuda.amp.GradScaler()
            
            self.is_initialized = True
            logger.info("Neural optimizer initialized")
            return True
            
        except Exception as e:
            logger.error(f"Neural optimizer initialization failed: {e}")
            return False
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create base optimizer"""
        if self.config.optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[any]:
        """Create learning rate scheduler"""
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=self.config.num_epochs // 3, 
                gamma=0.1
            )
        elif self.config.scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=0.95
            )
        elif self.config.scheduler_type == "warmup":
            return WarmupScheduler(
                self.optimizer,
                self.config.warmup_steps,
                self.config.max_lr,
                self.config.min_lr
            )
        else:
            return None
    
    async def optimize(self, input_tensor: torch.Tensor, 
                      target_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Optimize neural network parameters
        
        Args:
            input_tensor: Input tensor
            target_tensor: Target tensor for supervised learning
            
        Returns:
            Optimized tensor
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Neural optimizer not initialized")
            
            input_tensor = input_tensor.to(self.device)
            if target_tensor is not None:
                target_tensor = target_tensor.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.model(input_tensor)
                    if target_tensor is not None:
                        loss = self._calculate_loss(output, target_tensor)
                    else:
                        loss = self._calculate_unsupervised_loss(output)
            else:
                output = self.model(input_tensor)
                if target_tensor is not None:
                    loss = self._calculate_loss(output, target_tensor)
                else:
                    loss = self._calculate_unsupervised_loss(output)
            
            # Backward pass
            await self._backward_pass(loss)
            
            # Update metrics
            self._update_metrics(loss)
            
            return output
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _calculate_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate supervised loss"""
        if output.dim() == 1 or output.size(-1) == 1:
            # Regression
            loss = nn.MSELoss()(output, target)
        else:
            # Classification
            if self.config.label_smoothing > 0:
                loss = self._label_smoothing_loss(output, target)
            else:
                loss = nn.CrossEntropyLoss()(output, target)
        
        return loss
    
    def _calculate_unsupervised_loss(self, output: torch.Tensor) -> torch.Tensor:
        """Calculate unsupervised loss (e.g., autoencoder reconstruction)"""
        # Simple reconstruction loss (modify based on specific task)
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)
        return loss
    
    def _label_smoothing_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss with label smoothing"""
        num_classes = output.size(-1)
        smooth_target = target * (1 - self.config.label_smoothing) + \
                       self.config.label_smoothing / num_classes
        
        log_probs = nn.functional.log_softmax(output, dim=-1)
        loss = -(smooth_target * log_probs).sum(dim=-1).mean()
        
        return loss
    
    async def _backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with optimization"""
        self.optimizer.zero_grad()
        
        if self.scaler:
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clipping
            )
            
            # Optimization step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
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
        
        self.step_count += 1
    
    def _update_metrics(self, loss: torch.Tensor):
        """Update optimization metrics"""
        # Update loss history
        self.loss_history.append(loss.item())
        if len(self.loss_history) > 1000:
            self.loss_history.pop(0)
        
        # Update learning rate history
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)
        if len(self.lr_history) > 1000:
            self.lr_history.pop(0)
        
        # Calculate gradient norm
        total_norm = 0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            self.metrics["gradient_norm"] = (total_norm / param_count) ** 0.5
        
        # Calculate parameter norm
        total_param_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_param_norm += param_norm.item() ** 2
        
        self.metrics["parameter_norm"] = total_param_norm ** 0.5
        
        # Update other metrics
        self.metrics["learning_rate"] = current_lr
        self.metrics["optimization_steps"] = self.step_count
    
    async def update_parameters(self, loss: torch.Tensor):
        """Update model parameters based on loss"""
        await self._backward_pass(loss)
        self._update_metrics(loss)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "current_epoch": self.current_epoch,
            "step_count": self.step_count,
            "current_lr": self.metrics["learning_rate"],
            "gradient_norm": self.metrics["gradient_norm"],
            "parameter_norm": self.metrics["parameter_norm"],
            "recent_losses": self.loss_history[-100:],
            "recent_lrs": self.lr_history[-100:],
            "average_loss": sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0
        }
    
    def adjust_learning_rate(self, factor: float):
        """Adjust learning rate by factor"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        
        self.metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate adjusted to {self.metrics['learning_rate']}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status"""
        return {
            "is_initialized": self.is_initialized,
            "device": str(self.device),
            "config": self.config.__dict__,
            "metrics": self.metrics,
            "statistics": self.get_optimization_statistics()
        }
    
    def health_check(self) -> bool:
        """Check optimizer health"""
        try:
            return (
                self.is_initialized and
                self.model is not None and
                self.optimizer is not None
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown neural optimizer"""
        try:
            # Clear histories
            self.loss_history.clear()
            self.lr_history.clear()
            
            # Reset metrics
            self.metrics = {
                "learning_rate": self.config.learning_rate,
                "gradient_norm": 0.0,
                "parameter_norm": 0.0,
                "optimization_steps": 0
            }
            
            self.current_epoch = 0
            self.step_count = 0
            self.is_initialized = False
            
            logger.info("Neural optimizer shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Neural optimizer shutdown failed: {e}")
            return False