"""
Model Ensemble Optimization System for SutazAI
Implements advanced ensemble techniques for improved performance and reliability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from concurrent.futures import ThreadPoolExecutor
import time
import hashlib
from collections import defaultdict, OrderedDict
import random

logger = logging.getLogger(__name__)

class EnsembleStrategy(Enum):
    """Types of ensemble strategies"""
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking" 
    BOOSTING = "boosting"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    DYNAMIC_ROUTING = "dynamic_routing"
    ADAPTIVE_WEIGHTING = "adaptive_weighting"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble optimization"""
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE
    
    # Model selection
    models: List[str] = field(default_factory=lambda: ["tinyllama2.5-coder:7b", "tinyllama"])
    model_weights: Optional[List[float]] = None
    
    # Ensemble parameters
    voting_threshold: float = 0.5
    confidence_threshold: float = 0.8
    diversity_weight: float = 0.3
    performance_weight: float = 0.7
    
    # Dynamic routing
    routing_strategy: str = "confidence"  # confidence, expertise, load
    expertise_domains: Dict[str, List[str]] = field(default_factory=dict)
    
    # Adaptive parameters
    adaptation_rate: float = 0.1
    adaptation_window: int = 100
    min_samples_for_adaptation: int = 10
    
    # Performance optimization
    parallel_execution: bool = True
    max_concurrent_models: int = 3
    timeout_seconds: float = 30.0
    fallback_model: str = "tinyllama"
    
    # Quality control
    consensus_requirement: float = 0.6
    outlier_detection: bool = True
    outlier_threshold: float = 2.0
    
    # Caching and efficiency
    enable_caching: bool = True
    cache_ttl: int = 3600
    response_similarity_threshold: float = 0.95

class ModelProxy:
    """Proxy for individual models in the ensemble"""
    
    def __init__(self, model_name: str, ollama_host: str = None):
        self.model_name = model_name
        # Use environment variable or Docker service name
        self.ollama_host = ollama_host if ollama_host else os.getenv("OLLAMA_HOST", "http://ollama:10104")
        self.session = None
        
        # Performance tracking
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
        self.confidence_scores = []
        
        # Expertise tracking
        self.domain_performance = defaultdict(list)
        self.recent_performance = OrderedDict()
        
    async def initialize(self):
        """Initialize the model proxy"""
        self.session = aiohttp.ClientSession()
        await self._warmup()
        logger.info(f"Model proxy {self.model_name} initialized")
    
    async def _warmup(self):
        """Warm up the model"""
        try:
            await self.generate("Hello", max_tokens=5, temperature=0.1)
        except Exception as e:
            logger.warning(f"Warmup failed for {self.model_name}: {e}")
    
    async def generate(self, prompt: str, max_tokens: int = 512, 
                      temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """Generate response from the model"""
        start_time = time.time()
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "num_ctx": 2048,
                **kwargs
            }
        }
        
        try:
            async with self.session.post(
                f"{self.ollama_host}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('response', '')
                    
                    # Calculate confidence based on response characteristics
                    confidence = self._calculate_confidence(
                        prompt, response_text, response_time, result
                    )
                    
                    # Update statistics
                    self.response_times.append(response_time)
                    self.success_count += 1
                    self.confidence_scores.append(confidence)
                    
                    # Keep only recent history
                    if len(self.response_times) > 1000:
                        self.response_times = self.response_times[-500:]
                        self.confidence_scores = self.confidence_scores[-500:]
                    
                    return {
                        'response': response_text,
                        'confidence': confidence,
                        'response_time': response_time,
                        'model': self.model_name,
                        'tokens_generated': len(response_text.split()),
                        'total_duration': result.get('total_duration', 0),
                        'success': True
                    }
                else:
                    self.error_count += 1
                    logger.error(f"Model {self.model_name} failed: {response.status}")
                    return {
                        'response': '',
                        'confidence': 0.0,
                        'response_time': response_time,
                        'model': self.model_name,
                        'success': False,
                        'error': f"HTTP {response.status}"
                    }
                    
        except Exception as e:
            end_time = time.time()
            self.error_count += 1
            logger.error(f"Error in {self.model_name}: {e}")
            return {
                'response': '',
                'confidence': 0.0,
                'response_time': end_time - start_time,
                'model': self.model_name,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_confidence(self, prompt: str, response: str, 
                           response_time: float, result_data: Dict) -> float:
        """Calculate confidence score for the response"""
        if not response or len(response.strip()) == 0:
            return 0.0
        
        confidence_factors = []
        
        # Length factor (reasonable response length)
        response_length = len(response.split())
        if 10 <= response_length <= 500:
            length_factor = 1.0
        elif response_length < 10:
            length_factor = response_length / 10.0
        else:
            length_factor = max(0.5, 1.0 - (response_length - 500) / 1000.0)
        confidence_factors.append(length_factor)
        
        # Response time factor (faster is often better for simple tasks)
        avg_response_time = np.mean(self.response_times) if self.response_times else 5.0
        time_factor = min(1.0, avg_response_time / max(response_time, 0.1))
        confidence_factors.append(time_factor * 0.3)  # Lower weight for time
        
        # Content quality indicators
        quality_score = 0.0
        
        # Check for coherence indicators
        if response.count('.') > 0:  # Contains sentences
            quality_score += 0.2
        if response.count('\n') > 0:  # Has structure
            quality_score += 0.1
        if len(set(response.lower().split())) / len(response.split()) > 0.7:  # Vocabulary diversity
            quality_score += 0.2
        if not any(marker in response.lower() for marker in ['error', 'sorry', "don't know", "unclear"]):
            quality_score += 0.3
        
        confidence_factors.append(quality_score)
        
        # Model-specific reliability
        if self.success_count > 0:
            reliability = self.success_count / (self.success_count + self.error_count)
            confidence_factors.append(reliability * 0.2)
        
        # Combine factors
        confidence = np.mean(confidence_factors)
        return max(0.0, min(1.0, confidence))
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        return np.mean(self.response_times) if self.response_times else 0.0
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    def get_average_confidence(self) -> float:
        """Get average confidence score"""
        return np.mean(self.confidence_scores) if self.confidence_scores else 0.0
    
    def update_domain_performance(self, domain: str, performance_score: float):
        """Update performance for a specific domain"""
        self.domain_performance[domain].append(performance_score)
        
        # Keep only recent performance data
        if len(self.domain_performance[domain]) > 100:
            self.domain_performance[domain] = self.domain_performance[domain][-50:]
    
    def get_domain_expertise(self, domain: str) -> float:
        """Get expertise score for a domain"""
        if domain not in self.domain_performance:
            return 0.5  # Neutral expertise
        
        scores = self.domain_performance[domain]
        return np.mean(scores) if scores else 0.5
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

class EnsembleWeightOptimizer:
    """Optimizes ensemble weights based on performance data"""
    
    def __init__(self):
        self.weight_history = []
        self.performance_history = []
    
    def optimize_weights(self, models: List[ModelProxy], 
                        performance_data: List[Dict[str, float]]) -> List[float]:
        """Optimize ensemble weights using performance data"""
        if len(models) != len(performance_data):
            # Return equal weights if data mismatch
            return [1.0 / len(models)] * len(models)
        
        # Extract performance metrics
        confidence_scores = [data.get('confidence', 0.5) for data in performance_data]
        response_times = [data.get('response_time', 5.0) for data in performance_data]
        success_rates = [model.get_success_rate() for model in models]
        
        # Normalize metrics
        confidence_scores = self._normalize_scores(confidence_scores)
        speed_scores = self._normalize_scores([1.0 / max(rt, 0.1) for rt in response_times])
        success_rates = self._normalize_scores(success_rates)
        
        # Combine metrics
        combined_scores = []
        for i in range(len(models)):
            score = (
                0.4 * confidence_scores[i] +
                0.3 * success_rates[i] +
                0.3 * speed_scores[i]
            )
            combined_scores.append(score)
        
        # Convert to weights
        total_score = sum(combined_scores)
        if total_score == 0:
            return [1.0 / len(models)] * len(models)
        
        weights = [score / total_score for score in combined_scores]
        
        # Apply minimum weight threshold
        min_weight = 0.05
        adjusted_weights = [max(w, min_weight) for w in weights]
        
        # Renormalize
        total_adjusted = sum(adjusted_weights)
        final_weights = [w / total_adjusted for w in adjusted_weights]
        
        return final_weights
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def adaptive_weight_update(self, current_weights: List[float], 
                             performance_feedback: List[float],
                             learning_rate: float = 0.1) -> List[float]:
        """Update weights based on recent performance feedback"""
        if len(current_weights) != len(performance_feedback):
            return current_weights
        
        # Normalize performance feedback
        feedback_normalized = self._normalize_scores(performance_feedback)
        
        # Update weights using gradient-like update
        updated_weights = []
        for i, (weight, feedback) in enumerate(zip(current_weights, feedback_normalized)):
            # Increase weight for good performance, decrease for poor performance
            adjustment = learning_rate * (feedback - 0.5)  # Center around 0.5
            new_weight = weight + adjustment * weight  # Proportional adjustment
            updated_weights.append(max(0.01, new_weight))  # Minimum weight
        
        # Renormalize
        total_weight = sum(updated_weights)
        return [w / total_weight for w in updated_weights]

class EnsembleRouter:
    """Routes queries to appropriate models based on content and context"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.routing_history = []
        
        # Domain classification keywords
        self.domain_keywords = {
            'code': ['python', 'javascript', 'function', 'class', 'algorithm', 'debug', 'code', 'programming'],
            'math': ['calculate', 'equation', 'formula', 'mathematics', 'solve', 'number', 'statistics'],
            'analysis': ['analyze', 'explain', 'describe', 'compare', 'evaluate', 'discuss'],
            'creative': ['write', 'story', 'creative', 'generate', 'imagine', 'design'],
            'technical': ['system', 'architecture', 'infrastructure', 'deployment', 'configuration']
        }
    
    def route_query(self, prompt: str, models: List[ModelProxy],
                   current_load: Dict[str, float] = None) -> List[ModelProxy]:
        """Route query to appropriate models"""
        if self.config.routing_strategy == "confidence":
            return self._route_by_confidence(prompt, models)
        elif self.config.routing_strategy == "expertise":
            return self._route_by_expertise(prompt, models)
        elif self.config.routing_strategy == "load":
            return self._route_by_load(prompt, models, current_load or {})
        else:
            return models  # Use all models
    
    def _route_by_confidence(self, prompt: str, models: List[ModelProxy]) -> List[ModelProxy]:
        """Route based on model confidence scores"""
        # Select models with high average confidence
        confident_models = [
            model for model in models
            if model.get_average_confidence() > self.config.confidence_threshold
        ]
        
        if not confident_models:
            # Fall back to best performing model
            best_model = max(models, key=lambda m: m.get_average_confidence())
            return [best_model]
        
        return confident_models[:self.config.max_concurrent_models]
    
    def _route_by_expertise(self, prompt: str, models: List[ModelProxy]) -> List[ModelProxy]:
        """Route based on domain expertise"""
        # Classify the prompt domain
        domain = self._classify_domain(prompt)
        
        # Score models by domain expertise
        model_scores = []
        for model in models:
            expertise_score = model.get_domain_expertise(domain)
            general_performance = model.get_average_confidence()
            combined_score = 0.7 * expertise_score + 0.3 * general_performance
            model_scores.append((model, combined_score))
        
        # Sort by score and select top models
        model_scores.sort(key=lambda x: x[1], reverse=True)
        selected_models = [model for model, _ in model_scores[:self.config.max_concurrent_models]]
        
        return selected_models
    
    def _route_by_load(self, prompt: str, models: List[ModelProxy],
                      current_load: Dict[str, float]) -> List[ModelProxy]:
        """Route based on current model load"""
        # Select models with lowest load
        available_models = []
        for model in models:
            load = current_load.get(model.model_name, 0.0)
            if load < 0.8:  # Not overloaded
                available_models.append((model, load))
        
        if not available_models:
            # All models are loaded, use fastest responding model
            fastest_model = min(models, key=lambda m: m.get_average_response_time())
            return [fastest_model]
        
        # Sort by load and select least loaded models
        available_models.sort(key=lambda x: x[1])
        selected_models = [model for model, _ in available_models[:self.config.max_concurrent_models]]
        
        return selected_models
    
    def _classify_domain(self, prompt: str) -> str:
        """Classify prompt into domain categories"""
        prompt_lower = prompt.lower()
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                domain_scores[domain] = score / len(keywords)  # Normalize by keyword count
        
        if not domain_scores:
            return 'general'
        
        # Return domain with highest score
        return max(domain_scores, key=domain_scores.get)

class EnsembleAggregator:
    """Aggregates responses from multiple models"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.weight_optimizer = EnsembleWeightOptimizer()
    
    async def aggregate_responses(self, responses: List[Dict[str, Any]], 
                                models: List[ModelProxy]) -> Dict[str, Any]:
        """Aggregate responses from multiple models"""
        if not responses:
            return {'response': '', 'confidence': 0.0, 'models_used': []}
        
        # Filter successful responses
        successful_responses = [r for r in responses if r.get('success', False)]
        
        if not successful_responses:
            # Return fallback response
            return {
                'response': 'I apologize, but I was unable to generate a response at this time.',
                'confidence': 0.0,
                'models_used': [r.get('model', 'unknown') for r in responses],
                'error': 'All models failed'
            }
        
        if self.config.strategy == EnsembleStrategy.VOTING:
            return await self._voting_aggregation(successful_responses)
        elif self.config.strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_aggregation(successful_responses, models)
        elif self.config.strategy == EnsembleStrategy.MIXTURE_OF_EXPERTS:
            return await self._mixture_of_experts(successful_responses, models)
        elif self.config.strategy == EnsembleStrategy.ADAPTIVE_WEIGHTING:
            return await self._adaptive_weighting(successful_responses, models)
        else:
            # Default to confidence-based selection
            return await self._confidence_based_selection(successful_responses)
    
    async def _voting_aggregation(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate using voting mechanism"""
        if len(responses) == 1:
            return responses[0]
        
        # For text responses, use confidence-weighted voting
        confidence_scores = [r.get('confidence', 0.5) for r in responses]
        
        # Select response with highest confidence
        best_idx = np.argmax(confidence_scores)
        best_response = responses[best_idx]
        
        # Calculate consensus score
        avg_confidence = np.mean(confidence_scores)
        consensus_score = 1.0 - np.std(confidence_scores)
        
        return {
            'response': best_response['response'],
            'confidence': avg_confidence * consensus_score,
            'models_used': [r.get('model', 'unknown') for r in responses],
            'consensus_score': consensus_score,
            'aggregation_method': 'voting'
        }
    
    async def _weighted_aggregation(self, responses: List[Dict[str, Any]], 
                                  models: List[ModelProxy]) -> Dict[str, Any]:
        """Aggregate using weighted combination"""
        if len(responses) == 1:
            return responses[0]
        
        # Get or compute weights
        if self.config.model_weights and len(self.config.model_weights) == len(responses):
            weights = self.config.model_weights
        else:
            # Optimize weights based on current performance
            performance_data = [
                {
                    'confidence': r.get('confidence', 0.5),
                    'response_time': r.get('response_time', 5.0)
                }
                for r in responses
            ]
            weights = self.weight_optimizer.optimize_weights(models[:len(responses)], performance_data)
        
        # For text responses, select based on weighted confidence
        weighted_confidences = [
            r.get('confidence', 0.5) * weight 
            for r, weight in zip(responses, weights)
        ]
        
        best_idx = np.argmax(weighted_confidences)
        best_response = responses[best_idx]
        
        # Calculate weighted average confidence
        total_weight = sum(weights)
        avg_confidence = sum(
            r.get('confidence', 0.5) * weight 
            for r, weight in zip(responses, weights)
        ) / total_weight if total_weight > 0 else 0.5
        
        return {
            'response': best_response['response'],
            'confidence': avg_confidence,
            'models_used': [r.get('model', 'unknown') for r in responses],
            'weights_used': weights,
            'aggregation_method': 'weighted_average'
        }
    
    async def _mixture_of_experts(self, responses: List[Dict[str, Any]], 
                                models: List[ModelProxy]) -> Dict[str, Any]:
        """Mixture of experts aggregation"""
        # Select expert based on response quality and domain
        expert_scores = []
        
        for i, response in enumerate(responses):
            # Score based on multiple factors
            confidence = response.get('confidence', 0.5)
            response_time = response.get('response_time', 5.0)
            response_length = len(response.get('response', '').split())
            
            # Normalize factors
            time_score = min(1.0, 5.0 / max(response_time, 0.1))
            length_score = min(1.0, max(0.1, response_length / 100.0))
            
            expert_score = (
                0.5 * confidence +
                0.3 * time_score +
                0.2 * length_score
            )
            expert_scores.append(expert_score)
        
        # Select best expert
        best_expert_idx = np.argmax(expert_scores)
        best_response = responses[best_expert_idx]
        
        return {
            'response': best_response['response'],
            'confidence': best_response.get('confidence', 0.5),
            'models_used': [r.get('model', 'unknown') for r in responses],
            'expert_selected': best_response.get('model', 'unknown'),
            'expert_scores': expert_scores,
            'aggregation_method': 'mixture_of_experts'
        }
    
    async def _adaptive_weighting(self, responses: List[Dict[str, Any]], 
                                models: List[ModelProxy]) -> Dict[str, Any]:
        """Adaptive weighting based on recent performance"""
        # Start with equal weights if no history
        if not hasattr(self, 'adaptive_weights'):
            self.adaptive_weights = [1.0 / len(responses)] * len(responses)
        
        # Get current performance feedback
        performance_feedback = [r.get('confidence', 0.5) for r in responses]
        
        # Update weights based on feedback
        self.adaptive_weights = self.weight_optimizer.adaptive_weight_update(
            self.adaptive_weights, performance_feedback, self.config.adaptation_rate
        )
        
        # Use updated weights for aggregation
        weighted_confidences = [
            r.get('confidence', 0.5) * weight 
            for r, weight in zip(responses, self.adaptive_weights)
        ]
        
        best_idx = np.argmax(weighted_confidences)
        best_response = responses[best_idx]
        
        return {
            'response': best_response['response'],
            'confidence': best_response.get('confidence', 0.5),
            'models_used': [r.get('model', 'unknown') for r in responses],
            'adaptive_weights': self.adaptive_weights.copy(),
            'aggregation_method': 'adaptive_weighting'
        }
    
    async def _confidence_based_selection(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select response based on confidence scores"""
        confidences = [r.get('confidence', 0.5) for r in responses]
        best_idx = np.argmax(confidences)
        best_response = responses[best_idx]
        
        return {
            'response': best_response['response'],
            'confidence': best_response.get('confidence', 0.5),
            'models_used': [r.get('model', 'unknown') for r in responses],
            'all_confidences': confidences,
            'aggregation_method': 'confidence_based'
        }

class ModelEnsemble:
    """Main ensemble orchestrator"""
    
    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = {}
        self.router = EnsembleRouter(self.config)
        self.aggregator = EnsembleAggregator(self.config)
        self.performance_tracker = {}
        self.response_cache = {}
        
        # Load balancing
        self.current_load = defaultdict(float)
        self.load_history = defaultdict(list)
    
    async def initialize(self):
        """Initialize the ensemble"""
        logger.info("Initializing model ensemble...")
        
        # Initialize model proxies
        for model_name in self.config.models:
            proxy = ModelProxy(model_name)
            await proxy.initialize()
            self.models[model_name] = proxy
        
        # Initialize performance tracking
        for model_name in self.config.models:
            self.performance_tracker[model_name] = {
                'requests': 0,
                'successes': 0,
                'total_response_time': 0.0,
                'confidence_scores': []
            }
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
    
    async def generate(self, prompt: str, max_tokens: int = 512, 
                      temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        """Generate response using ensemble"""
        # Check cache first
        if self.config.enable_caching:
            cached_response = self._get_cached_response(prompt, kwargs)
            if cached_response:
                return cached_response
        
        # Route query to appropriate models
        selected_models = self.router.route_query(
            prompt, 
            list(self.models.values()),
            self.current_load
        )
        
        # Generate responses from selected models
        if self.config.parallel_execution:
            responses = await self._parallel_generate(
                selected_models, prompt, max_tokens, temperature, **kwargs
            )
        else:
            responses = await self._sequential_generate(
                selected_models, prompt, max_tokens, temperature, **kwargs
            )
        
        # Update load tracking
        await self._update_load_tracking(selected_models, responses)
        
        # Aggregate responses
        final_response = await self.aggregator.aggregate_responses(
            responses, selected_models
        )
        
        # Cache the response
        if self.config.enable_caching and final_response.get('confidence', 0) > 0.7:
            self._cache_response(prompt, kwargs, final_response)
        
        # Update performance tracking
        self._update_performance_tracking(responses)
        
        return final_response
    
    async def _parallel_generate(self, models: List[ModelProxy], prompt: str,
                               max_tokens: int, temperature: float, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses in parallel"""
        tasks = []
        for model in models:
            task = asyncio.create_task(
                model.generate(prompt, max_tokens, temperature, **kwargs)
            )
            tasks.append(task)
        
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds
            )
            
            # Handle exceptions
            valid_responses = []
            for response in responses:
                if isinstance(response, Exception):
                    logger.error(f"Model generation failed: {response}")
                    valid_responses.append({
                        'response': '',
                        'confidence': 0.0,
                        'success': False,
                        'error': str(response)
                    })
                else:
                    valid_responses.append(response)
            
            return valid_responses
            
        except asyncio.TimeoutError:
            logger.warning("Ensemble generation timed out")
            # Return fallback response
            fallback_model = self.models.get(self.config.fallback_model)
            if fallback_model:
                fallback_response = await fallback_model.generate(
                    prompt, max_tokens, temperature, **kwargs
                )
                return [fallback_response]
            return []
    
    async def _sequential_generate(self, models: List[ModelProxy], prompt: str,
                                 max_tokens: int, temperature: float, **kwargs) -> List[Dict[str, Any]]:
        """Generate responses sequentially"""
        responses = []
        
        for model in models:
            try:
                response = await asyncio.wait_for(
                    model.generate(prompt, max_tokens, temperature, **kwargs),
                    timeout=self.config.timeout_seconds
                )
                responses.append(response)
                
                # Early stopping if we get a high-confidence response
                if response.get('confidence', 0) > 0.9:
                    break
                    
            except asyncio.TimeoutError:
                logger.warning(f"Model {model.model_name} timed out")
                responses.append({
                    'response': '',
                    'confidence': 0.0,
                    'success': False,
                    'error': 'timeout',
                    'model': model.model_name
                })
        
        return responses
    
    async def _update_load_tracking(self, models: List[ModelProxy], 
                                  responses: List[Dict[str, Any]]):
        """Update load tracking for models"""
        for model, response in zip(models, responses):
            response_time = response.get('response_time', 0)
            
            # Update current load (simple moving average)
            current_load = self.current_load[model.model_name]
            load_factor = min(1.0, response_time / 5.0)  # Normalize to 5s baseline
            
            self.current_load[model.model_name] = 0.9 * current_load + 0.1 * load_factor
            
            # Track load history
            self.load_history[model.model_name].append(load_factor)
            if len(self.load_history[model.model_name]) > 100:
                self.load_history[model.model_name] = self.load_history[model.model_name][-50:]
    
    def _get_cached_response(self, prompt: str, kwargs: Dict) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        cache_key = self._generate_cache_key(prompt, kwargs)
        
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            
            # Check TTL
            if time.time() - cached_item['timestamp'] < self.config.cache_ttl:
                logger.debug("Using cached response")
                cached_item['cached'] = True
                return cached_item
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, prompt: str, kwargs: Dict, response: Dict[str, Any]):
        """Cache response"""
        cache_key = self._generate_cache_key(prompt, kwargs)
        
        cached_item = response.copy()
        cached_item['timestamp'] = time.time()
        
        self.response_cache[cache_key] = cached_item
        
        # Limit cache size
        if len(self.response_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]['timestamp']
            )[:100]
            
            for key in oldest_keys:
                del self.response_cache[key]
    
    def _generate_cache_key(self, prompt: str, kwargs: Dict) -> str:
        """Generate cache key for prompt and parameters"""
        key_data = {
            'prompt': prompt,
            'params': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_performance_tracking(self, responses: List[Dict[str, Any]]):
        """Update performance tracking statistics"""
        for response in responses:
            model_name = response.get('model', 'unknown')
            
            if model_name in self.performance_tracker:
                tracker = self.performance_tracker[model_name]
                tracker['requests'] += 1
                
                if response.get('success', False):
                    tracker['successes'] += 1
                    tracker['total_response_time'] += response.get('response_time', 0)
                    tracker['confidence_scores'].append(response.get('confidence', 0))
                    
                    # Limit history size
                    if len(tracker['confidence_scores']) > 1000:
                        tracker['confidence_scores'] = tracker['confidence_scores'][-500:]
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        stats = {
            'models': {},
            'overall': {
                'total_requests': 0,
                'total_successes': 0,
                'cache_hit_rate': 0,
                'current_load': dict(self.current_load)
            }
        }
        
        total_requests = 0
        total_successes = 0
        
        for model_name, tracker in self.performance_tracker.items():
            requests = tracker['requests']
            successes = tracker['successes']
            
            total_requests += requests
            total_successes += successes
            
            if requests > 0:
                success_rate = successes / requests
                avg_response_time = tracker['total_response_time'] / successes if successes > 0 else 0
                avg_confidence = np.mean(tracker['confidence_scores']) if tracker['confidence_scores'] else 0
            else:
                success_rate = 0
                avg_response_time = 0
                avg_confidence = 0
            
            stats['models'][model_name] = {
                'requests': requests,
                'successes': successes,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'avg_confidence': avg_confidence,
                'current_load': self.current_load[model_name]
            }
        
        stats['overall']['total_requests'] = total_requests
        stats['overall']['total_successes'] = total_successes
        stats['overall']['success_rate'] = total_successes / total_requests if total_requests > 0 else 0
        
        return stats
    
    async def cleanup(self):
        """Cleanup ensemble resources"""
        for model in self.models.values():
            await model.cleanup()

# Factory function for easy integration
async def create_ensemble(models: List[str] = None, 
                         strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
                         config: EnsembleConfig = None) -> ModelEnsemble:
    """Create and initialize a model ensemble"""
    if config is None:
        config = EnsembleConfig(
            models=models or ["tinyllama2.5-coder:7b", "tinyllama"],
            strategy=strategy
        )
    
    ensemble = ModelEnsemble(config)
    await ensemble.initialize()
    
    return ensemble

# Example usage
async def example_ensemble_usage():
    """Example usage of the ensemble system"""
    # Create ensemble
    ensemble = await create_ensemble(
        models=["tinyllama2.5-coder:7b", "tinyllama"],
        strategy=EnsembleStrategy.ADAPTIVE_WEIGHTING
    )
    
    # Generate response
    result = await ensemble.generate(
        "Explain machine learning in simple terms",
        max_tokens=256,
        temperature=0.7
    )
    
    logger.info(f"Response: {result['response']}")
    logger.info(f"Confidence: {result['confidence']}")
    logger.info(f"Models used: {result['models_used']}")
    
    # Get statistics
    stats = ensemble.get_ensemble_statistics()
    logger.info("Ensemble Statistics:", json.dumps(stats, indent=2))
    
    # Cleanup
    await ensemble.cleanup()
    
    return result

if __name__ == "__main__":
    # Run example
    import asyncio
    
    async def main():
        result = await example_ensemble_usage()
        return result
    
    # asyncio.run(main())