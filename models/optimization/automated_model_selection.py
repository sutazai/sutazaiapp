"""
Automated Model Selection System for SutazAI
Implements intelligent model selection with resource-aware deployment
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import os
from pathlib import Path
import asyncio
import aiohttp
import sqlite3
import time
import hashlib
from collections import defaultdict, deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml

logger = logging.getLogger(__name__)

class SelectionStrategy(Enum):
    """Model selection strategies"""
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_BASED = "resource_based"
    COST_BASED = "cost_based"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class DeploymentMode(Enum):
    """Deployment modes"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    SHADOW = "shadow"

class ResourceConstraint(Enum):
    """Resource constraint types"""
    MEMORY = "memory"
    CPU = "cpu"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

@dataclass
class ModelProfile:
    """Profile information for a model"""
    name: str
    type: str = "llm"
    
    # Performance characteristics
    avg_latency: float = 0.0
    avg_throughput: float = 0.0
    avg_quality: float = 0.0
    success_rate: float = 1.0
    
    # Resource requirements
    memory_requirement_mb: float = 512.0
    cpu_requirement_cores: float = 1.0
    gpu_requirement_gb: float = 0.0
    
    # Operational metrics
    cold_start_time: float = 0.0
    warmup_time: float = 0.0
    scaling_factor: float = 1.0
    
    # Cost metrics
    compute_cost_per_token: float = 0.0
    memory_cost_per_mb_hour: float = 0.0
    
    # Capabilities
    supported_tasks: List[str] = field(default_factory=list)
    max_context_length: int = 2048
    supported_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Quality metrics by domain
    domain_performance: Dict[str, float] = field(default_factory=dict)
    
    # Deployment constraints
    min_instances: int = 1
    max_instances: int = 10
    auto_scaling: bool = True

@dataclass
class ResourceConstraints:
    """System resource constraints"""
    max_memory_mb: float = 4096.0
    max_cpu_cores: float = 4.0
    max_gpu_memory_gb: float = 0.0
    
    # Performance constraints
    max_latency_ms: float = 5000.0
    min_throughput_tps: float = 1.0
    
    # Cost constraints
    max_cost_per_hour: float = 10.0
    max_cost_per_token: float = 0.01
    
    # Quality constraints
    min_quality_score: float = 0.7
    min_success_rate: float = 0.95

@dataclass
class SelectionContext:
    """Context for model selection"""
    task_type: str
    domain: str = "general"
    language: str = "en"
    
    # Request characteristics
    expected_tokens: int = 256
    max_latency_ms: float = 5000.0
    quality_priority: float = 0.5  # 0=speed, 1=quality
    
    # User/session context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    
    # Load context
    current_load: float = 0.0
    peak_hours: bool = False
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelSelectionEngine:
    """Core model selection engine"""
    
    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.HYBRID):
        self.strategy = strategy
        self.model_profiles = {}
        self.performance_history = defaultdict(list)
        self.selection_history = []
        self.resource_monitor = None
        
        # Learning components
        self.selection_rewards = defaultdict(list)
        self.exploration_rate = 0.1
        self.learning_rate = 0.01
        
        # Model weights for different criteria
        self.selection_weights = {
            'performance': 0.4,
            'resource_efficiency': 0.3,
            'cost': 0.2,
            'reliability': 0.1
        }
    
    def add_model_profile(self, profile: ModelProfile):
        """Add a model profile to the selection engine"""
        self.model_profiles[profile.name] = profile
        logger.info(f"Added model profile: {profile.name}")
    
    def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a model"""
        if model_name not in self.model_profiles:
            logger.warning(f"Model {model_name} not found in profiles")
            return
        
        profile = self.model_profiles[model_name]
        
        # Update performance metrics with exponential moving average
        alpha = 0.1  # Learning rate
        
        if 'latency' in metrics:
            profile.avg_latency = (1 - alpha) * profile.avg_latency + alpha * metrics['latency']
        
        if 'throughput' in metrics:
            profile.avg_throughput = (1 - alpha) * profile.avg_throughput + alpha * metrics['throughput']
        
        if 'quality' in metrics:
            profile.avg_quality = (1 - alpha) * profile.avg_quality + alpha * metrics['quality']
        
        if 'success_rate' in metrics:
            profile.success_rate = (1 - alpha) * profile.success_rate + alpha * metrics['success_rate']
        
        # Store performance history
        self.performance_history[model_name].append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
        
        # Limit history size
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][-500:]
    
    def select_model(self, context: SelectionContext, 
                    constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select the best model for given context and constraints"""
        if not self.model_profiles:
            raise ValueError("No model profiles available")
        
        if self.strategy == SelectionStrategy.PERFORMANCE_BASED:
            return self._select_by_performance(context, constraints)
        elif self.strategy == SelectionStrategy.RESOURCE_BASED:
            return self._select_by_resources(context, constraints)
        elif self.strategy == SelectionStrategy.COST_BASED:
            return self._select_by_cost(context, constraints)
        elif self.strategy == SelectionStrategy.HYBRID:
            return self._select_hybrid(context, constraints)
        elif self.strategy == SelectionStrategy.ADAPTIVE:
            return self._select_adaptive(context, constraints)
        elif self.strategy == SelectionStrategy.REINFORCEMENT_LEARNING:
            return self._select_reinforcement_learning(context, constraints)
        else:
            return self._select_hybrid(context, constraints)
    
    def _select_by_performance(self, context: SelectionContext, 
                             constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model based on performance metrics"""
        best_model = None
        best_score = -1.0
        
        for model_name, profile in self.model_profiles.items():
            # Check basic constraints
            if not self._meets_constraints(profile, constraints):
                continue
            
            if not self._suitable_for_task(profile, context):
                continue
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(profile, context)
            
            if performance_score > best_score:
                best_score = performance_score
                best_model = model_name
        
        if best_model is None:
            # Fallback to least resource-intensive model
            best_model = min(self.model_profiles.keys(), 
                           key=lambda m: self.model_profiles[m].memory_requirement_mb)
            best_score = 0.5
        
        return best_model, best_score
    
    def _select_by_resources(self, context: SelectionContext, 
                           constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model based on resource efficiency"""
        best_model = None
        best_efficiency = -1.0
        
        for model_name, profile in self.model_profiles.items():
            if not self._meets_constraints(profile, constraints):
                continue
            
            if not self._suitable_for_task(profile, context):
                continue
            
            # Calculate resource efficiency
            efficiency = self._calculate_resource_efficiency(profile, context)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_model = model_name
        
        return best_model or list(self.model_profiles.keys())[0], best_efficiency
    
    def _select_by_cost(self, context: SelectionContext, 
                       constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model based on cost optimization"""
        best_model = None
        best_cost_score = -1.0
        
        for model_name, profile in self.model_profiles.items():
            if not self._meets_constraints(profile, constraints):
                continue
            
            if not self._suitable_for_task(profile, context):
                continue
            
            # Calculate cost efficiency
            cost_score = self._calculate_cost_efficiency(profile, context)
            
            if cost_score > best_cost_score:
                best_cost_score = cost_score
                best_model = model_name
        
        return best_model or list(self.model_profiles.keys())[0], best_cost_score
    
    def _select_hybrid(self, context: SelectionContext, 
                      constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model using hybrid approach combining multiple factors"""
        best_model = None
        best_combined_score = -1.0
        
        for model_name, profile in self.model_profiles.items():
            if not self._meets_constraints(profile, constraints):
                continue
            
            if not self._suitable_for_task(profile, context):
                continue
            
            # Calculate component scores
            performance_score = self._calculate_performance_score(profile, context)
            resource_score = self._calculate_resource_efficiency(profile, context)
            cost_score = self._calculate_cost_efficiency(profile, context)
            reliability_score = self._calculate_reliability_score(profile)
            
            # Combine scores using weights
            combined_score = (
                self.selection_weights['performance'] * performance_score +
                self.selection_weights['resource_efficiency'] * resource_score +
                self.selection_weights['cost'] * cost_score +
                self.selection_weights['reliability'] * reliability_score
            )
            
            # Apply context-specific adjustments
            combined_score = self._apply_context_adjustments(
                combined_score, profile, context
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_model = model_name
        
        return best_model or list(self.model_profiles.keys())[0], best_combined_score
    
    def _select_adaptive(self, context: SelectionContext, 
                        constraints: ResourceConstraints) -> Tuple[str, float]:
        """Adaptive selection that learns from past performance"""
        # Start with hybrid selection
        base_model, base_score = self._select_hybrid(context, constraints)
        
        # Apply learning-based adjustments
        if len(self.selection_history) > 50:  # Need history for adaptation
            # Analyze recent selection patterns
            recent_selections = self.selection_history[-50:]
            
            # Calculate success rates for different contexts
            context_performance = defaultdict(list)
            
            for selection in recent_selections:
                key = f"{selection['context']['task_type']}_{selection['context']['domain']}"
                reward = selection.get('reward', 0.5)
                context_performance[key].append({
                    'model': selection['selected_model'],
                    'reward': reward
                })
            
            # Adjust selection based on historical performance
            current_context_key = f"{context.task_type}_{context.domain}"
            
            if current_context_key in context_performance:
                history = context_performance[current_context_key]
                
                # Calculate model performance for this context
                model_rewards = defaultdict(list)
                for entry in history:
                    model_rewards[entry['model']].append(entry['reward'])
                
                # Find best performing model for this context
                best_historical_model = None
                best_avg_reward = -1.0
                
                for model_name, rewards in model_rewards.items():
                    if model_name in self.model_profiles:
                        avg_reward = np.mean(rewards)
                        if avg_reward > best_avg_reward:
                            best_avg_reward = avg_reward
                            best_historical_model = model_name
                
                # Blend historical and current selection
                if best_historical_model and best_avg_reward > base_score * 0.8:
                    return best_historical_model, best_avg_reward
        
        return base_model, base_score
    
    def _select_reinforcement_learning(self, context: SelectionContext,
                                     constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model using reinforcement learning approach"""
        # Epsilon-greedy strategy
        if np.random.random() < self.exploration_rate:
            # Exploration: select random valid model
            valid_models = [
                name for name, profile in self.model_profiles.items()
                if self._meets_constraints(profile, constraints) and
                   self._suitable_for_task(profile, context)
            ]
            
            if valid_models:
                selected_model = np.random.choice(valid_models)
                return selected_model, 0.5  # Neutral confidence for exploration
        
        # Exploitation: select model with highest Q-value
        return self._select_by_q_values(context, constraints)
    
    def _select_by_q_values(self, context: SelectionContext,
                           constraints: ResourceConstraints) -> Tuple[str, float]:
        """Select model based on learned Q-values"""
        best_model = None
        best_q_value = -float('inf')
        
        context_key = self._get_context_key(context)
        
        for model_name, profile in self.model_profiles.items():
            if not self._meets_constraints(profile, constraints):
                continue
            
            if not self._suitable_for_task(profile, context):
                continue
            
            # Get Q-value for this model-context pair
            q_value = self._get_q_value(model_name, context_key)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_model = model_name
        
        return best_model or list(self.model_profiles.keys())[0], best_q_value
    
    def _get_q_value(self, model_name: str, context_key: str) -> float:
        """Get Q-value for model-context pair"""
        # Simple Q-learning implementation
        rewards = self.selection_rewards.get(f"{model_name}_{context_key}", [])
        
        if not rewards:
            return 0.5  # Neutral value for unseen combinations
        
        # Return discounted average of recent rewards
        recent_rewards = rewards[-10:]  # Last 10 rewards
        weights = np.array([0.9 ** i for i in range(len(recent_rewards))][::-1])
        
        return np.average(recent_rewards, weights=weights)
    
    def _get_context_key(self, context: SelectionContext) -> str:
        """Generate context key for learning"""
        return f"{context.task_type}_{context.domain}_{context.priority}"
    
    def _meets_constraints(self, profile: ModelProfile, 
                          constraints: ResourceConstraints) -> bool:
        """Check if model meets resource constraints"""
        if profile.memory_requirement_mb > constraints.max_memory_mb:
            return False
        
        if profile.cpu_requirement_cores > constraints.max_cpu_cores:
            return False
        
        if profile.gpu_requirement_gb > constraints.max_gpu_memory_gb:
            return False
        
        if profile.avg_latency * 1000 > constraints.max_latency_ms:
            return False
        
        if profile.avg_throughput < constraints.min_throughput_tps:
            return False
        
        if profile.avg_quality < constraints.min_quality_score:
            return False
        
        if profile.success_rate < constraints.min_success_rate:
            return False
        
        return True
    
    def _suitable_for_task(self, profile: ModelProfile, context: SelectionContext) -> bool:
        """Check if model is suitable for the task"""
        # Check supported tasks
        if profile.supported_tasks and context.task_type not in profile.supported_tasks:
            return False
        
        # Check language support
        if context.language not in profile.supported_languages:
            return False
        
        # Check context length requirements
        if context.expected_tokens > profile.max_context_length:
            return False
        
        return True
    
    def _calculate_performance_score(self, profile: ModelProfile, 
                                   context: SelectionContext) -> float:
        """Calculate performance score for a model"""
        score = 0.0
        
        # Base performance metrics
        latency_score = max(0, 1.0 - profile.avg_latency / 10.0)  # Normalize to 10s max
        throughput_score = min(1.0, profile.avg_throughput / 100.0)  # Scale to reasonable max
        quality_score = profile.avg_quality
        reliability_score = profile.success_rate
        
        # Weight based on context priority
        if context.quality_priority > 0.7:
            # Quality-focused
            score = 0.5 * quality_score + 0.3 * reliability_score + 0.2 * throughput_score
        elif context.quality_priority < 0.3:
            # Speed-focused
            score = 0.5 * latency_score + 0.3 * throughput_score + 0.2 * quality_score
        else:
            # Balanced
            score = 0.3 * quality_score + 0.3 * latency_score + 0.2 * throughput_score + 0.2 * reliability_score
        
        # Domain-specific adjustments
        if context.domain in profile.domain_performance:
            domain_score = profile.domain_performance[context.domain]
            score = 0.8 * score + 0.2 * domain_score
        
        return score
    
    def _calculate_resource_efficiency(self, profile: ModelProfile, 
                                     context: SelectionContext) -> float:
        """Calculate resource efficiency score"""
        # Performance per resource unit
        memory_efficiency = profile.avg_throughput / max(profile.memory_requirement_mb, 1)
        cpu_efficiency = profile.avg_throughput / max(profile.cpu_requirement_cores, 1)
        
        # Quality per resource unit
        quality_per_memory = profile.avg_quality / max(profile.memory_requirement_mb, 1)
        
        # Combine efficiency metrics
        efficiency = 0.4 * memory_efficiency + 0.3 * cpu_efficiency + 0.3 * quality_per_memory
        
        # Normalize to 0-1 range (approximate)
        return min(1.0, efficiency / 0.01)  # Adjust scaling factor as needed
    
    def _calculate_cost_efficiency(self, profile: ModelProfile, 
                                 context: SelectionContext) -> float:
        """Calculate cost efficiency score"""
        # Cost per token
        token_cost = profile.compute_cost_per_token
        
        # Cost per quality unit
        quality_cost = token_cost / max(profile.avg_quality, 0.1)
        
        # Memory cost for expected usage
        memory_cost = profile.memory_cost_per_mb_hour * profile.memory_requirement_mb
        
        # Total cost efficiency (inverse of cost)
        total_cost = token_cost + quality_cost + memory_cost
        
        if total_cost == 0:
            return 1.0
        
        # Normalize cost efficiency
        return max(0, 1.0 - total_cost * 1000)  # Adjust scaling as needed
    
    def _calculate_reliability_score(self, profile: ModelProfile) -> float:
        """Calculate reliability score"""
        base_reliability = profile.success_rate
        
        # Factor in cold start and warmup times
        startup_penalty = min(0.2, (profile.cold_start_time + profile.warmup_time) / 60.0)
        
        reliability_score = base_reliability - startup_penalty
        
        return max(0, min(1.0, reliability_score))
    
    def _apply_context_adjustments(self, base_score: float, profile: ModelProfile,
                                 context: SelectionContext) -> float:
        """Apply context-specific adjustments to the score"""
        adjusted_score = base_score
        
        # Priority adjustments
        if context.priority == "critical":
            # Prefer more reliable models for critical tasks
            adjusted_score += 0.1 * profile.success_rate
        elif context.priority == "low":
            # Prefer more cost-efficient models for low priority
            cost_factor = 1.0 - profile.compute_cost_per_token * 1000
            adjusted_score += 0.1 * cost_factor
        
        # Load adjustments
        if context.current_load > 0.8:
            # Under high load, prefer faster models
            latency_bonus = max(0, 1.0 - profile.avg_latency / 5.0)
            adjusted_score += 0.1 * latency_bonus
        
        # Peak hours adjustments
        if context.peak_hours:
            # During peak hours, balance performance and efficiency
            efficiency = self._calculate_resource_efficiency(profile, context)
            adjusted_score += 0.05 * efficiency
        
        return max(0, min(1.0, adjusted_score))
    
    def record_selection_outcome(self, model_name: str, context: SelectionContext,
                               outcome_metrics: Dict[str, float]):
        """Record the outcome of a model selection for learning"""
        # Calculate reward based on outcome
        reward = self._calculate_reward(outcome_metrics, context)
        
        # Store selection record
        selection_record = {
            'timestamp': time.time(),
            'selected_model': model_name,
            'context': context.__dict__.copy(),
            'outcome_metrics': outcome_metrics.copy(),
            'reward': reward
        }
        
        self.selection_history.append(selection_record)
        
        # Update Q-values for reinforcement learning
        context_key = self._get_context_key(context)
        q_key = f"{model_name}_{context_key}"
        
        if q_key not in self.selection_rewards:
            self.selection_rewards[q_key] = []
        
        self.selection_rewards[q_key].append(reward)
        
        # Limit history size
        if len(self.selection_rewards[q_key]) > 100:
            self.selection_rewards[q_key] = self.selection_rewards[q_key][-50:]
        
        # Update model profile with new performance data
        self.update_model_performance(model_name, outcome_metrics)
        
        # Adapt selection weights based on outcomes
        self._adapt_selection_weights()
    
    def _calculate_reward(self, outcome_metrics: Dict[str, float], 
                         context: SelectionContext) -> float:
        """Calculate reward for reinforcement learning"""
        reward = 0.0
        
        # Performance-based rewards
        if 'latency' in outcome_metrics:
            target_latency = context.max_latency_ms / 1000.0
            latency_reward = max(0, 1.0 - outcome_metrics['latency'] / target_latency)
            reward += 0.3 * latency_reward
        
        if 'quality' in outcome_metrics:
            reward += 0.4 * outcome_metrics['quality']
        
        if 'success' in outcome_metrics:
            reward += 0.2 * (1.0 if outcome_metrics['success'] else 0.0)
        
        if 'user_satisfaction' in outcome_metrics:
            reward += 0.1 * outcome_metrics['user_satisfaction']
        
        return max(0, min(1.0, reward))
    
    def _adapt_selection_weights(self):
        """Adapt selection weights based on recent outcomes"""
        if len(self.selection_history) < 20:
            return
        
        recent_selections = self.selection_history[-20:]
        
        # Analyze which factors correlate with better outcomes
        performance_rewards = []
        resource_rewards = []
        cost_rewards = []
        reliability_rewards = []
        
        for selection in recent_selections:
            model_name = selection['selected_model']
            reward = selection['reward']
            
            if model_name in self.model_profiles:
                profile = self.model_profiles[model_name]
                
                # Correlate reward with different factors
                performance_rewards.append((profile.avg_quality, reward))
                resource_rewards.append((profile.memory_requirement_mb, -reward))  # Negative for efficiency
                cost_rewards.append((profile.compute_cost_per_token, -reward))  # Negative for cost
                reliability_rewards.append((profile.success_rate, reward))
        
        # Update weights based on correlations (simplified approach)
        if len(performance_rewards) > 5:
            performance_corr = np.corrcoef([p[0] for p in performance_rewards], 
                                         [p[1] for p in performance_rewards])[0, 1]
            
            if not np.isnan(performance_corr) and abs(performance_corr) > 0.3:
                self.selection_weights['performance'] *= (1 + 0.1 * performance_corr)
        
        # Normalize weights
        total_weight = sum(self.selection_weights.values())
        if total_weight > 0:
            for key in self.selection_weights:
                self.selection_weights[key] /= total_weight
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics and insights"""
        stats = {
            'total_selections': len(self.selection_history),
            'model_usage': defaultdict(int),
            'average_reward': 0.0,
            'selection_weights': self.selection_weights.copy(),
            'model_profiles_count': len(self.model_profiles)
        }
        
        if self.selection_history:
            # Model usage statistics
            for selection in self.selection_history:
                stats['model_usage'][selection['selected_model']] += 1
            
            # Average reward
            rewards = [s.get('reward', 0) for s in self.selection_history]
            stats['average_reward'] = np.mean(rewards)
            
            # Recent performance trend
            if len(rewards) >= 10:
                recent_rewards = rewards[-10:]
                earlier_rewards = rewards[-20:-10] if len(rewards) >= 20 else rewards[:-10]
                
                if earlier_rewards:
                    recent_avg = np.mean(recent_rewards)
                    earlier_avg = np.mean(earlier_rewards)
                    stats['performance_trend'] = 'improving' if recent_avg > earlier_avg else 'declining'
                else:
                    stats['performance_trend'] = 'stable'
        
        return dict(stats)

class AutoDeployer:
    """Handles automated model deployment"""
    
    def __init__(self, ollama_host: str = "http://localhost:10104"):
        self.ollama_host = ollama_host
        self.deployed_models = set()
        self.deployment_history = []
        self.session = None
    
    async def initialize(self):
        """Initialize the auto deployer"""
        self.session = aiohttp.ClientSession()
        
        # Check current deployments
        await self._check_current_deployments()
    
    async def _check_current_deployments(self):
        """Check which models are currently deployed"""
        try:
            async with self.session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    for model in models:
                        model_name = model.get("name", "")
                        if model_name:
                            self.deployed_models.add(model_name)
                    
                    logger.info(f"Found {len(self.deployed_models)} deployed models")
        except Exception as e:
            logger.error(f"Error checking deployments: {e}")
    
    async def deploy_model(self, model_name: str, mode: DeploymentMode = DeploymentMode.IMMEDIATE) -> bool:
        """Deploy a model with specified mode"""
        if model_name in self.deployed_models:
            logger.info(f"Model {model_name} already deployed")
            return True
        
        deployment_record = {
            'model_name': model_name,
            'mode': mode.value,
            'timestamp': time.time(),
            'status': 'deploying'
        }
        
        try:
            if mode == DeploymentMode.IMMEDIATE:
                success = await self._deploy_immediate(model_name)
            elif mode == DeploymentMode.GRADUAL:
                success = await self._deploy_gradual(model_name)
            elif mode == DeploymentMode.CANARY:
                success = await self._deploy_canary(model_name)
            else:
                success = await self._deploy_immediate(model_name)
            
            deployment_record['status'] = 'success' if success else 'failed'
            deployment_record['duration'] = time.time() - deployment_record['timestamp']
            
            if success:
                self.deployed_models.add(model_name)
                logger.info(f"Successfully deployed model: {model_name}")
            else:
                logger.error(f"Failed to deploy model: {model_name}")
            
            self.deployment_history.append(deployment_record)
            return success
            
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {e}")
            deployment_record['status'] = 'error'
            deployment_record['error'] = str(e)
            self.deployment_history.append(deployment_record)
            return False
    
    async def _deploy_immediate(self, model_name: str) -> bool:
        """Deploy model immediately"""
        try:
            # Pull the model
            data = {"name": model_name}
            async with self.session.post(
                f"{self.ollama_host}/api/pull",
                json=data,
                timeout=aiohttp.ClientTimeout(total=1800)  # 30 minute timeout
            ) as response:
                if response.status == 200:
                    # Warm up the model
                    await self._warmup_model(model_name)
                    return True
                else:
                    logger.error(f"Pull failed for {model_name}: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Immediate deployment failed for {model_name}: {e}")
            return False
    
    async def _deploy_gradual(self, model_name: str) -> bool:
        """Deploy model gradually with health checks"""
        try:
            # Start deployment
            success = await self._deploy_immediate(model_name)
            
            if success:
                # Perform gradual health checks
                for i in range(5):
                    await asyncio.sleep(10)  # Wait between checks
                    
                    # Test model health
                    healthy = await self._check_model_health(model_name)
                    
                    if not healthy:
                        logger.warning(f"Health check {i+1} failed for {model_name}")
                        return False
                
                logger.info(f"Gradual deployment successful for {model_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Gradual deployment failed for {model_name}: {e}")
            return False
    
    async def _deploy_canary(self, model_name: str) -> bool:
        """Deploy model using canary deployment"""
        try:
            # Deploy as canary (limited traffic)
            success = await self._deploy_immediate(model_name)
            
            if success:
                # Monitor canary performance
                canary_metrics = await self._monitor_canary(model_name, duration=60)
                
                # Evaluate canary success
                if self._evaluate_canary_metrics(canary_metrics):
                    logger.info(f"Canary deployment successful for {model_name}")
                    return True
                else:
                    logger.warning(f"Canary metrics failed for {model_name}")
                    # Could implement rollback here
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Canary deployment failed for {model_name}: {e}")
            return False
    
    async def _warmup_model(self, model_name: str):
        """Warm up a deployed model"""
        try:
            warmup_data = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 5}
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=warmup_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    await response.json()
                    logger.info(f"Warmed up model: {model_name}")
                else:
                    logger.warning(f"Warmup failed for {model_name}")
        except Exception as e:
            logger.warning(f"Error warming up {model_name}: {e}")
    
    async def _check_model_health(self, model_name: str) -> bool:
        """Check if a model is healthy"""
        try:
            test_data = {
                "model": model_name,
                "prompt": "Test health check",
                "stream": False,
                "options": {"num_predict": 10}
            }
            
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=test_data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    return len(response_text.strip()) > 0
                return False
        except Exception as e:
            logger.error(f"Health check failed for {model_name}: {e}")
            return False
    
    async def _monitor_canary(self, model_name: str, duration: int = 60) -> Dict[str, float]:
        """Monitor canary deployment metrics"""
        start_time = time.time()
        metrics = {
            'requests': 0,
            'successes': 0,
            'total_latency': 0.0,
            'errors': 0
        }
        
        # Simulate canary monitoring (in real implementation, this would collect actual metrics)
        while time.time() - start_time < duration:
            try:
                # Test request
                test_start = time.time()
                healthy = await self._check_model_health(model_name)
                test_latency = time.time() - test_start
                
                metrics['requests'] += 1
                
                if healthy:
                    metrics['successes'] += 1
                    metrics['total_latency'] += test_latency
                else:
                    metrics['errors'] += 1
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                metrics['errors'] += 1
                logger.error(f"Canary monitoring error: {e}")
        
        return metrics
    
    def _evaluate_canary_metrics(self, metrics: Dict[str, float]) -> bool:
        """Evaluate if canary metrics are acceptable"""
        if metrics['requests'] == 0:
            return False
        
        success_rate = metrics['successes'] / metrics['requests']
        avg_latency = metrics['total_latency'] / max(metrics['successes'], 1)
        
        # Canary success criteria
        return success_rate >= 0.95 and avg_latency <= 5.0 and metrics['errors'] <= 2
    
    async def undeploy_model(self, model_name: str) -> bool:
        """Undeploy a model"""
        if model_name not in self.deployed_models:
            return True
        
        try:
            # In Ollama, we can't directly "undeploy" but we can remove the model
            data = {"name": model_name}
            async with self.session.delete(
                f"{self.ollama_host}/api/delete",
                json=data
            ) as response:
                if response.status == 200:
                    self.deployed_models.discard(model_name)
                    logger.info(f"Undeployed model: {model_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error undeploying {model_name}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup deployer resources"""
        if self.session:
            await self.session.close()

class ModelSelectionOrchestrator:
    """Orchestrates the entire model selection and deployment process"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "model_selection_config.yaml"
        self.selection_engine = ModelSelectionEngine()
        self.auto_deployer = AutoDeployer()
        self.resource_constraints = ResourceConstraints()
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Load model profiles
                if 'models' in config:
                    for model_config in config['models']:
                        profile = ModelProfile(**model_config)
                        self.selection_engine.add_model_profile(profile)
                
                # Load resource constraints
                if 'constraints' in config:
                    self.resource_constraints = ResourceConstraints(**config['constraints'])
                
                # Load selection strategy
                if 'strategy' in config:
                    strategy_name = config['strategy'].upper()
                    if hasattr(SelectionStrategy, strategy_name):
                        self.selection_engine.strategy = getattr(SelectionStrategy, strategy_name)
                
                logger.info("Configuration loaded successfully")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self._create_default_configuration()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default configuration"""
        # Default model profiles
        default_models = [
            ModelProfile(
                name="tinyllama",
                avg_latency=1.5,
                avg_throughput=25.0,
                avg_quality=0.7,
                memory_requirement_mb=800,
                cpu_requirement_cores=1.0,
                supported_tasks=["general", "simple_qa", "summarization"],
                max_context_length=2048
            ),
            ModelProfile(
                name="tinyllama2.5-coder:7b",
                avg_latency=3.0,
                avg_throughput=15.0,
                avg_quality=0.85,
                memory_requirement_mb=2500,
                cpu_requirement_cores=2.0,
                supported_tasks=["coding", "analysis", "complex_reasoning"],
                max_context_length=4096
            )
        ]
        
        for profile in default_models:
            self.selection_engine.add_model_profile(profile)
        
        logger.info("Created default configuration")
    
    async def initialize(self):
        """Initialize the orchestrator"""
        await self.auto_deployer.initialize()
        logger.info("Model selection orchestrator initialized")
    
    async def select_and_deploy(self, context: SelectionContext) -> Tuple[str, float, bool]:
        """Select and deploy the best model for the context"""
        # Select model
        selected_model, confidence = self.selection_engine.select_model(
            context, self.resource_constraints
        )
        
        # Deploy if not already deployed
        deployed = selected_model in self.auto_deployer.deployed_models
        
        if not deployed:
            deployment_success = await self.auto_deployer.deploy_model(
                selected_model, DeploymentMode.IMMEDIATE
            )
        else:
            deployment_success = True
        
        return selected_model, confidence, deployment_success
    
    async def optimize_deployment(self):
        """Optimize current deployments based on usage patterns"""
        stats = self.selection_engine.get_selection_statistics()
        
        # Get model usage statistics
        model_usage = stats['model_usage']
        
        # Identify underutilized models
        total_selections = stats['total_selections']
        underutilized_threshold = 0.05  # Less than 5% usage
        
        underutilized_models = []
        for model_name in self.auto_deployer.deployed_models:
            usage_ratio = model_usage.get(model_name, 0) / max(total_selections, 1)
            if usage_ratio < underutilized_threshold:
                underutilized_models.append(model_name)
        
        # Undeploy underutilized models to save resources
        for model_name in underutilized_models:
            logger.info(f"Undeploying underutilized model: {model_name}")
            await self.auto_deployer.undeploy_model(model_name)
        
        logger.info(f"Deployment optimization complete. Undeployed {len(underutilized_models)} models")
    
    def update_performance(self, model_name: str, context: SelectionContext,
                          outcome_metrics: Dict[str, float]):
        """Update model performance and record selection outcome"""
        self.selection_engine.record_selection_outcome(
            model_name, context, outcome_metrics
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the model selection system"""
        return {
            'selection_engine': self.selection_engine.get_selection_statistics(),
            'deployed_models': list(self.auto_deployer.deployed_models),
            'deployment_history': len(self.auto_deployer.deployment_history),
            'resource_constraints': self.resource_constraints.__dict__,
            'model_profiles': len(self.selection_engine.model_profiles)
        }
    
    async def cleanup(self):
        """Cleanup orchestrator resources"""
        await self.auto_deployer.cleanup()

# Factory functions and utilities
async def create_model_selection_system(config_path: str = None) -> ModelSelectionOrchestrator:
    """Create and initialize a model selection system"""
    orchestrator = ModelSelectionOrchestrator(config_path)
    await orchestrator.initialize()
    return orchestrator

# Example usage
async def example_model_selection():
    """Example usage of the model selection system"""
    # Create orchestrator
    orchestrator = await create_model_selection_system()
    
    # Example contexts
    contexts = [
        SelectionContext(
            task_type="coding",
            domain="python",
            expected_tokens=500,
            quality_priority=0.8,
            priority="high"
        ),
        SelectionContext(
            task_type="general",
            domain="qa",
            expected_tokens=100,
            quality_priority=0.3,
            priority="low"
        ),
        SelectionContext(
            task_type="analysis",
            domain="technical",
            expected_tokens=300,
            quality_priority=0.6,
            priority="normal"
        )
    ]
    
    # Test selections
    results = []
    for i, context in enumerate(contexts):
        selected_model, confidence, deployed = await orchestrator.select_and_deploy(context)
        
        results.append({
            'context': context.__dict__,
            'selected_model': selected_model,
            'confidence': confidence,
            'deployed': deployed
        })
        
        # Simulate outcome metrics
        outcome_metrics = {
            'latency': np.random.normal(2.0, 0.5),
            'quality': np.random.normal(0.8, 0.1),
            'success': 1.0,
            'user_satisfaction': np.random.normal(0.75, 0.15)
        }
        
        # Update performance
        orchestrator.update_performance(selected_model, context, outcome_metrics)
        
        print(f"Selection {i+1}: {selected_model} (confidence: {confidence:.3f})")
    
    # Get status
    status = orchestrator.get_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Optimize deployments
    await orchestrator.optimize_deployment()
    
    # Cleanup
    await orchestrator.cleanup()
    
    return results

if __name__ == "__main__":
    # Run example
    async def main():
        results = await example_model_selection()
        return results
    
    # asyncio.run(main())