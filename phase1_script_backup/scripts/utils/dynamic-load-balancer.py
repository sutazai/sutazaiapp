#!/usr/bin/env python3
"""
Dynamic Load Balancer for SutazAI
=================================

Purpose: Advanced dynamic load balancing with ML-based predictions and real-time optimization
Usage: python scripts/dynamic-load-balancer.py [--algorithm adaptive] [--enable-ml]
Requirements: Python 3.8+, scikit-learn, numpy, asyncio

Features:
- Multiple load balancing algorithms (Round Robin, Least Connections, Weighted, Adaptive)
- Machine learning-based load prediction
- Health-based routing decisions
- Dynamic weight adjustment
- Circuit breaker pattern implementation
- Real-time performance optimization
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import math
import random
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import psutil
import weakref

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/dynamic_load_balancer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DynamicLoadBalancer')

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"
    ML_PREDICTIVE = "ml_predictive"
    CONSISTENT_HASHING = "consistent_hashing"

class HealthStatus(Enum):
    """Health status of backend services"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    CIRCUIT_OPEN = "circuit_open"

class RequestType(Enum):
    """Types of requests for different handling"""
    COMPUTE_HEAVY = "compute_heavy"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_BOUND = "io_bound"
    STREAMING = "streaming"
    BATCH = "batch"
    INTERACTIVE = "interactive"

@dataclass
class BackendService:
    """Backend service representation"""
    service_id: str
    host: str
    port: int
    weight: float = 1.0
    current_connections: int = 0
    max_connections: int = 1000
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_health_check: float = 0.0
    health_status: HealthStatus = HealthStatus.HEALTHY
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    load_average: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    circuit_breaker_state: str = "closed"
    circuit_breaker_failures: int = 0
    circuit_breaker_last_failure: float = 0.0

@dataclass
class RequestMetrics:
    """Request metrics for analysis"""
    request_id: str
    timestamp: float
    request_type: RequestType
    size_bytes: int
    processing_time: float
    backend_service: str
    success: bool
    response_size: int = 0
    queue_time: float = 0.0
    error_type: Optional[str] = None

@dataclass
class LoadBalanceDecision:
    """Load balancing decision record"""
    timestamp: float
    algorithm_used: LoadBalancingAlgorithm
    selected_backend: str
    all_candidates: List[str]
    decision_factors: Dict[str, float]
    prediction_confidence: float = 0.0

class CircuitBreaker:
    """Circuit breaker implementation for backend services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, 
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset()
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                self.half_open_calls = 0
            else:
                raise Exception("Circuit breaker is open")
        
        if self.state == "half_open":
            if self.half_open_calls >= self.half_open_max_calls:
                raise Exception("Circuit breaker half-open limit exceeded")
            self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "half_open":
            self.reset()
        else:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class LoadPredictor:
    """Machine learning-based load predictor"""
    
    def __init__(self):
        self.models = {
            'response_time': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'cpu_usage': RandomForestRegressor(n_estimators=50, random_state=42),
            'memory_usage': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        self.scalers = {
            'response_time': StandardScaler(),
            'cpu_usage': StandardScaler(),
            'memory_usage': StandardScaler()
        }
        self.is_trained = {key: False for key in self.models.keys()}
        self.training_data = defaultdict(list)
        self.predictions_cache = {}
        self.cache_ttl = 10.0  # Cache predictions for 10 seconds
    
    def extract_features(self, service: BackendService, request_type: RequestType, 
                        current_time: float) -> np.ndarray:
        """Extract features for ML prediction"""
        # Time-based features
        hour_of_day = time.localtime(current_time).tm_hour
        minute_of_hour = time.localtime(current_time).tm_min
        day_of_week = time.localtime(current_time).tm_wday
        
        # Service features
        connection_ratio = service.current_connections / max(service.max_connections, 1)
        error_rate = service.error_count / max(service.error_count + service.success_count, 1)
        avg_response_time = statistics.mean(service.response_times) if service.response_times else 0
        
        # Request type encoding (one-hot)
        request_type_features = [0] * len(RequestType)
        request_type_features[list(RequestType).index(request_type)] = 1
        
        # Historical load patterns
        recent_load = service.cpu_usage
        load_trend = 0  # Would calculate trend from historical data
        
        features = np.array([
            hour_of_day / 24.0,
            minute_of_hour / 60.0,
            day_of_week / 7.0,
            connection_ratio,
            error_rate,
            avg_response_time / 1000.0,  # Normalize to seconds
            service.cpu_usage / 100.0,
            service.memory_usage / 100.0,
            service.load_average / 5.0,  # Assuming max load of 5
            recent_load / 100.0,
            load_trend,
            *request_type_features
        ])
        
        return features
    
    def train_models(self, training_samples: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train prediction models"""
        if len(training_samples) < 50:  # Need minimum samples
            return {}
        
        # Prepare training data
        features_list = []
        targets = defaultdict(list)
        
        for sample in training_samples:
            features = sample['features']
            features_list.append(features)
            
            targets['response_time'].append(sample['response_time'])
            targets['cpu_usage'].append(sample['cpu_usage'])
            targets['memory_usage'].append(sample['memory_usage'])
        
        X = np.array(features_list)
        training_scores = {}
        
        # Train each model
        for target_name, model in self.models.items():
            try:
                y = np.array(targets[target_name])
                
                # Scale features
                X_scaled = self.scalers[target_name].fit_transform(X)
                
                # Train model
                model.fit(X_scaled, y)
                self.is_trained[target_name] = True
                
                # Evaluate training performance
                y_pred = model.predict(X_scaled)
                score = 1.0 - (mean_squared_error(y, y_pred) / max(np.var(y), 1e-8))
                training_scores[target_name] = max(score, 0.0)
                
                logger.info(f"Trained {target_name} model with score: {score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to train {target_name} model: {e}")
                training_scores[target_name] = 0.0
        
        return training_scores
    
    def predict_load(self, service: BackendService, request_type: RequestType) -> Dict[str, float]:
        """Predict load metrics for a service"""
        current_time = time.time()
        cache_key = f"{service.service_id}_{request_type.value}_{int(current_time / self.cache_ttl)}"
        
        # Check cache
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
        
        features = self.extract_features(service, request_type, current_time)
        predictions = {}
        
        for target_name, model in self.models.items():
            if self.is_trained[target_name]:
                try:
                    features_scaled = self.scalers[target_name].transform(features.reshape(1, -1))
                    prediction = model.predict(features_scaled)[0]
                    predictions[target_name] = max(prediction, 0.0)
                except Exception as e:
                    logger.error(f"Prediction failed for {target_name}: {e}")
                    predictions[target_name] = self._fallback_prediction(service, target_name)
            else:
                predictions[target_name] = self._fallback_prediction(service, target_name)
        
        # Cache predictions
        self.predictions_cache[cache_key] = predictions
        
        # Clean old cache entries
        if len(self.predictions_cache) > 1000:
            old_keys = [k for k in self.predictions_cache.keys() 
                       if int(k.split('_')[-1]) < int(current_time / self.cache_ttl) - 10]
            for k in old_keys:
                del self.predictions_cache[k]
        
        return predictions
    
    def _fallback_prediction(self, service: BackendService, target_name: str) -> float:
        """Fallback prediction when ML model is not available"""
        if target_name == 'response_time':
            return statistics.mean(service.response_times) if service.response_times else 100.0
        elif target_name == 'cpu_usage':
            return service.cpu_usage
        elif target_name == 'memory_usage':
            return service.memory_usage
        else:
            return 0.0
    
    def update_training_data(self, sample: Dict[str, Any]):
        """Add sample to training data"""
        for target in self.models.keys():
            self.training_data[target].append(sample)
            
            # Keep only recent samples
            if len(self.training_data[target]) > 10000:
                self.training_data[target].pop(0)

class ConsistentHashRing:
    """Consistent hashing implementation for load balancing"""
    
    def __init__(self, replicas: int = 100):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []
    
    def add_node(self, node: str):
        """Add node to hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def remove_node(self, node: str):
        """Remove node from hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
        
        self.sorted_keys = sorted(self.ring.keys())
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for given key"""
        if not self.ring:
            return None
        
        hash_key = self._hash(key)
        
        # Find the first node with hash >= hash_key
        for ring_key in self.sorted_keys:
            if ring_key >= hash_key:
                return self.ring[ring_key]
        
        # If no node found, wrap around to first node
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class DynamicLoadBalancer:
    """Main dynamic load balancer implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = LoadBalancingAlgorithm(config.get("algorithm", "adaptive"))
        self.services = {}  # service_id -> BackendService
        self.service_groups = defaultdict(list)  # group_name -> [service_ids]
        
        # Algorithm-specific state
        self.round_robin_counters = defaultdict(int)
        self.consistent_hash_rings = {}  # group_name -> ConsistentHashRing
        
        # Machine learning components
        self.enable_ml = config.get("enable_ml", False)
        self.predictor = LoadPredictor() if self.enable_ml else None
        
        # Circuit breakers
        self.circuit_breakers = {}  # service_id -> CircuitBreaker
        
        # Metrics and monitoring
        self.request_history = deque(maxlen=10000)
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # Background tasks
        self.running = False
        self.health_check_task = None
        self.ml_training_task = None
        self.metrics_collection_task = None
        
        # Adaptive algorithm parameters
        self.adaptive_weights = {
            "response_time": 0.3,
            "cpu_usage": 0.25,
            "memory_usage": 0.2,
            "connections": 0.15,
            "error_rate": 0.1
        }
        
        logger.info(f"Dynamic Load Balancer initialized with algorithm: {self.algorithm.value}")
    
    async def start(self):
        """Start the load balancer"""
        self.running = True
        
        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        if self.enable_ml:
            self.ml_training_task = asyncio.create_task(self._ml_training_loop())
        
        logger.info("Dynamic Load Balancer started")
    
    async def stop(self):
        """Stop the load balancer"""
        self.running = False
        
        # Cancel background tasks
        for task in [self.health_check_task, self.ml_training_task, self.metrics_collection_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Dynamic Load Balancer stopped")
    
    def register_service(self, service: BackendService, group: str = "default"):
        """Register a backend service"""
        self.services[service.service_id] = service
        self.service_groups[group].append(service.service_id)
        self.circuit_breakers[service.service_id] = CircuitBreaker()
        
        # Add to consistent hash ring if needed
        if group not in self.consistent_hash_rings:
            self.consistent_hash_rings[group] = ConsistentHashRing()
        self.consistent_hash_rings[group].add_node(service.service_id)
        
        logger.info(f"Registered service {service.service_id} in group {group}")
    
    def unregister_service(self, service_id: str, group: str = "default"):
        """Unregister a backend service"""
        if service_id in self.services:
            del self.services[service_id]
            
            if service_id in self.service_groups[group]:
                self.service_groups[group].remove(service_id)
            
            if service_id in self.circuit_breakers:
                del self.circuit_breakers[service_id]
            
            # Remove from consistent hash ring
            if group in self.consistent_hash_rings:
                self.consistent_hash_rings[group].remove_node(service_id)
            
            logger.info(f"Unregistered service {service_id} from group {group}")
    
    async def select_backend(self, request_type: RequestType = RequestType.INTERACTIVE,
                           group: str = "default", session_id: Optional[str] = None,
                           request_size: int = 0) -> Optional[BackendService]:
        """Select the best backend service for a request"""
        available_services = [
            self.services[sid] for sid in self.service_groups[group]
            if sid in self.services and self._is_service_available(self.services[sid])
        ]
        
        if not available_services:
            logger.warning(f"No available services in group {group}")
            return None
        
        # Apply load balancing algorithm
        selected_service = None
        decision_factors = {}
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected_service = await self._round_robin_select(available_services, group)
            decision_factors["algorithm"] = "round_robin"
            
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected_service = await self._least_connections_select(available_services)
            decision_factors["connections"] = selected_service.current_connections
            
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            selected_service = await self._weighted_round_robin_select(available_services, group)
            decision_factors["weight"] = selected_service.weight
            
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            selected_service = await self._least_response_time_select(available_services)
            decision_factors["avg_response_time"] = statistics.mean(selected_service.response_times) if selected_service.response_times else 0
            
        elif self.algorithm == LoadBalancingAlgorithm.RESOURCE_BASED:
            selected_service = await self._resource_based_select(available_services)
            decision_factors["cpu_usage"] = selected_service.cpu_usage
            decision_factors["memory_usage"] = selected_service.memory_usage
            
        elif self.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            selected_service = await self._adaptive_select(available_services, request_type)
            decision_factors["adaptive_score"] = await self._calculate_adaptive_score(selected_service, request_type)
            
        elif self.algorithm == LoadBalancingAlgorithm.ML_PREDICTIVE:
            selected_service = await self._ml_predictive_select(available_services, request_type)
            if self.enable_ml and self.predictor:
                predictions = self.predictor.predict_load(selected_service, request_type)
                decision_factors["predicted_response_time"] = predictions.get("response_time", 0)
                decision_factors["predicted_cpu"] = predictions.get("cpu_usage", 0)
            
        elif self.algorithm == LoadBalancingAlgorithm.CONSISTENT_HASHING:
            selected_service = await self._consistent_hash_select(available_services, session_id or str(request_size), group)
            decision_factors["hash_key"] = session_id or str(request_size)
        
        # Record decision
        if selected_service:
            decision = LoadBalanceDecision(
                timestamp=time.time(),
                algorithm_used=self.algorithm,
                selected_backend=selected_service.service_id,
                all_candidates=[s.service_id for s in available_services],
                decision_factors=decision_factors
            )
            self.decision_history.append(decision)
            
            # Update connection count
            selected_service.current_connections += 1
        
        return selected_service
    
    def _is_service_available(self, service: BackendService) -> bool:
        """Check if service is available for load balancing"""
        return (service.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
                service.circuit_breaker_state != "open" and
                service.current_connections < service.max_connections)
    
    async def _round_robin_select(self, services: List[BackendService], group: str) -> BackendService:
        """Round robin selection"""
        counter = self.round_robin_counters[group]
        selected = services[counter % len(services)]
        self.round_robin_counters[group] = (counter + 1) % len(services)
        return selected
    
    async def _least_connections_select(self, services: List[BackendService]) -> BackendService:
        """Least connections selection"""
        return min(services, key=lambda s: s.current_connections)
    
    async def _weighted_round_robin_select(self, services: List[BackendService], group: str) -> BackendService:
        """Weighted round robin selection"""
        # Calculate cumulative weights
        total_weight = sum(s.weight for s in services)
        if total_weight == 0:
            return await self._round_robin_select(services, group)
        
        # Generate random number and select based on weight
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for service in services:
            cumulative_weight += service.weight
            if rand <= cumulative_weight:
                return service
        
        return services[-1]  # Fallback
    
    async def _least_response_time_select(self, services: List[BackendService]) -> BackendService:
        """Least response time selection"""
        def avg_response_time(service):
            return statistics.mean(service.response_times) if service.response_times else float('inf')
        
        return min(services, key=avg_response_time)
    
    async def _resource_based_select(self, services: List[BackendService]) -> BackendService:
        """Resource-based selection (least loaded)"""
        def resource_score(service):
            # Lower score = less loaded
            cpu_score = service.cpu_usage / 100.0
            memory_score = service.memory_usage / 100.0
            connection_score = service.current_connections / max(service.max_connections, 1)
            return cpu_score + memory_score + connection_score
        
        return min(services, key=resource_score)
    
    async def _adaptive_select(self, services: List[BackendService], request_type: RequestType) -> BackendService:
        """Adaptive selection based on multiple factors"""
        best_service = None
        best_score = float('inf')
        
        for service in services:
            score = await self._calculate_adaptive_score(service, request_type)
            if score < best_score:
                best_score = score
                best_service = service
        
        return best_service or services[0]
    
    async def _calculate_adaptive_score(self, service: BackendService, request_type: RequestType) -> float:
        """Calculate adaptive score for service selection"""
        # Normalize metrics to 0-1 range
        response_time_score = statistics.mean(service.response_times) / 1000.0 if service.response_times else 0.1
        cpu_score = service.cpu_usage / 100.0
        memory_score = service.memory_usage / 100.0
        connection_score = service.current_connections / max(service.max_connections, 1)
        error_rate = service.error_count / max(service.error_count + service.success_count, 1)
        
        # Weight scores based on request type
        if request_type == RequestType.COMPUTE_HEAVY:
            weights = {"response_time": 0.2, "cpu_usage": 0.4, "memory_usage": 0.2, "connections": 0.1, "error_rate": 0.1}
        elif request_type == RequestType.MEMORY_INTENSIVE:
            weights = {"response_time": 0.2, "cpu_usage": 0.2, "memory_usage": 0.4, "connections": 0.1, "error_rate": 0.1}
        elif request_type == RequestType.IO_BOUND:
            weights = {"response_time": 0.4, "cpu_usage": 0.1, "memory_usage": 0.1, "connections": 0.3, "error_rate": 0.1}
        else:
            weights = self.adaptive_weights
        
        # Calculate weighted score
        score = (response_time_score * weights["response_time"] +
                cpu_score * weights["cpu_usage"] +
                memory_score * weights["memory_usage"] +
                connection_score * weights["connections"] +
                error_rate * weights["error_rate"])
        
        return score
    
    async def _ml_predictive_select(self, services: List[BackendService], request_type: RequestType) -> BackendService:
        """ML-based predictive selection"""
        if not self.enable_ml or not self.predictor:
            return await self._adaptive_select(services, request_type)
        
        best_service = None
        best_score = float('inf')
        
        for service in services:
            predictions = self.predictor.predict_load(service, request_type)
            
            # Calculate predicted performance score
            pred_response_time = predictions.get("response_time", 100) / 1000.0
            pred_cpu = predictions.get("cpu_usage", 50) / 100.0
            pred_memory = predictions.get("memory_usage", 50) / 100.0
            
            # Combine predictions with current state
            current_score = await self._calculate_adaptive_score(service, request_type)
            predicted_score = pred_response_time * 0.5 + pred_cpu * 0.3 + pred_memory * 0.2
            
            # Weighted combination of current and predicted
            combined_score = current_score * 0.6 + predicted_score * 0.4
            
            if combined_score < best_score:
                best_score = combined_score
                best_service = service
        
        return best_service or services[0]
    
    async def _consistent_hash_select(self, services: List[BackendService], key: str, group: str) -> BackendService:
        """Consistent hashing selection"""
        ring = self.consistent_hash_rings.get(group)
        if not ring:
            return await self._round_robin_select(services, group)
        
        selected_service_id = ring.get_node(key)
        
        # Find the service object
        for service in services:
            if service.service_id == selected_service_id:
                return service
        
        # Fallback if service not available
        return await self._round_robin_select(services, group)
    
    async def record_request_completion(self, service_id: str, request_metrics: RequestMetrics):
        """Record completion of a request for metrics and learning"""
        if service_id not in self.services:
            return
        
        service = self.services[service_id]
        
        # Update service metrics
        service.current_connections = max(0, service.current_connections - 1)
        service.response_times.append(request_metrics.processing_time)
        
        if request_metrics.success:
            service.success_count += 1
            # Reset circuit breaker failure count on success
            if service_id in self.circuit_breakers:
                self.circuit_breakers[service_id]._on_success()
        else:
            service.error_count += 1
            # Increment circuit breaker failure count
            if service_id in self.circuit_breakers:
                self.circuit_breakers[service_id]._on_failure()
        
        # Add to request history
        self.request_history.append(request_metrics)
        
        # Update ML training data if enabled
        if self.enable_ml and self.predictor:
            features = self.predictor.extract_features(service, request_metrics.request_type, request_metrics.timestamp)
            
            training_sample = {
                'features': features,
                'response_time': request_metrics.processing_time,
                'cpu_usage': service.cpu_usage,
                'memory_usage': service.memory_usage,
                'timestamp': request_metrics.timestamp
            }
            
            self.predictor.update_training_data(training_sample)
    
    async def update_service_health(self, service_id: str, health_metrics: Dict[str, Any]):
        """Update service health metrics"""
        if service_id not in self.services:
            return
        
        service = self.services[service_id]
        
        # Update metrics
        service.cpu_usage = health_metrics.get("cpu_usage", service.cpu_usage)
        service.memory_usage = health_metrics.get("memory_usage", service.memory_usage)
        service.load_average = health_metrics.get("load_average", service.load_average)
        service.last_health_check = time.time()
        
        # Determine health status
        if health_metrics.get("responsive", True):
            if service.cpu_usage > 90 or service.memory_usage > 90:
                service.health_status = HealthStatus.DEGRADED
            else:
                service.health_status = HealthStatus.HEALTHY
        else:
            service.health_status = HealthStatus.UNHEALTHY
        
        # Update circuit breaker state
        if service_id in self.circuit_breakers:
            cb = self.circuit_breakers[service_id]
            service.circuit_breaker_state = cb.state
            service.circuit_breaker_failures = cb.failure_count
    
    def adjust_adaptive_weights(self, performance_feedback: Dict[str, float]):
        """Adjust adaptive algorithm weights based on performance feedback"""
        learning_rate = 0.1
        
        for metric, feedback in performance_feedback.items():
            if metric in self.adaptive_weights:
                # Positive feedback increases weight, negative decreases
                adjustment = learning_rate * feedback
                self.adaptive_weights[metric] = max(0.05, min(0.8, 
                    self.adaptive_weights[metric] + adjustment))
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        for metric in self.adaptive_weights:
            self.adaptive_weights[metric] /= total_weight
        
        logger.info(f"Updated adaptive weights: {self.adaptive_weights}")
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        while self.running:
            try:
                for service in self.services.values():
                    # Simulate health check (in real implementation, would make HTTP request)
                    if time.time() - service.last_health_check > 30:  # 30 seconds since last update
                        # Simulate degraded health for services with no recent updates
                        if service.health_status == HealthStatus.HEALTHY:
                            service.health_status = HealthStatus.DEGRADED
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)
    
    async def _ml_training_loop(self):
        """Background ML model training loop"""
        if not self.enable_ml or not self.predictor:
            return
        
        while self.running:
            try:
                # Collect training samples from all targets
                all_samples = []
                for target in self.predictor.training_data.keys():
                    all_samples.extend(self.predictor.training_data[target])
                
                if len(all_samples) >= 100:  # Minimum samples for training
                    scores = self.predictor.train_models(all_samples)
                    logger.info(f"ML model training completed with scores: {scores}")
                
                await asyncio.sleep(300)  # Train every 5 minutes
                
            except Exception as e:
                logger.error(f"ML training loop error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                # Collect performance metrics
                current_time = time.time()
                
                # Calculate system-wide metrics
                total_requests = len(self.request_history)
                if total_requests > 0:
                    recent_requests = [r for r in self.request_history 
                                     if current_time - r.timestamp < 60]  # Last minute
                    
                    if recent_requests:
                        avg_response_time = statistics.mean([r.processing_time for r in recent_requests])
                        success_rate = sum(1 for r in recent_requests if r.success) / len(recent_requests)
                        
                        self.performance_metrics["avg_response_time"].append(avg_response_time)
                        self.performance_metrics["success_rate"].append(success_rate)
                        
                        # Keep only recent metrics
                        for metric_list in self.performance_metrics.values():
                            if len(metric_list) > 1000:
                                metric_list.pop(0)
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(120)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        stats = {
            "algorithm": self.algorithm.value,
            "enable_ml": self.enable_ml,
            "services": {},
            "performance_metrics": dict(self.performance_metrics),
            "adaptive_weights": dict(self.adaptive_weights),
            "request_stats": {
                "total_requests": len(self.request_history),
                "recent_requests": len([r for r in self.request_history 
                                      if time.time() - r.timestamp < 3600])  # Last hour
            }
        }
        
        # Service statistics
        for service_id, service in self.services.items():
            circuit_breaker = self.circuit_breakers.get(service_id)
            
            stats["services"][service_id] = {
                "host": service.host,
                "port": service.port,
                "weight": service.weight,
                "current_connections": service.current_connections,
                "max_connections": service.max_connections,
                "health_status": service.health_status.value,
                "cpu_usage": service.cpu_usage,
                "memory_usage": service.memory_usage,
                "load_average": service.load_average,
                "success_count": service.success_count,
                "error_count": service.error_count,
                "avg_response_time": statistics.mean(service.response_times) if service.response_times else 0,
                "circuit_breaker_state": circuit_breaker.state if circuit_breaker else "unknown",
                "circuit_breaker_failures": circuit_breaker.failure_count if circuit_breaker else 0
            }
        
        # ML model statistics
        if self.enable_ml and self.predictor:
            stats["ml_models"] = {
                "trained_models": list(self.predictor.is_trained.keys()),
                "training_samples": {k: len(v) for k, v in self.predictor.training_data.items()},
                "prediction_cache_size": len(self.predictor.predictions_cache)
            }
        
        return stats
    
    def export_stats(self, filepath: str):
        """Export comprehensive statistics"""
        stats = self.get_comprehensive_stats()
        stats["export_timestamp"] = time.time()
        
        # Add recent request history
        stats["recent_requests"] = [
            asdict(r) for r in list(self.request_history)[-1000:]  # Last 1000 requests
        ]
        
        # Add recent decisions
        stats["recent_decisions"] = [
            asdict(d) for d in list(self.decision_history)[-100:]  # Last 100 decisions
        ]
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported load balancer statistics to {filepath}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Dynamic Load Balancer")
    parser.add_argument("--algorithm", choices=[alg.value for alg in LoadBalancingAlgorithm],
                       default="adaptive", help="Load balancing algorithm")
    parser.add_argument("--enable-ml", action="store_true", help="Enable ML-based predictions")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--test", action="store_true", help="Run test simulation")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")
    parser.add_argument("--export-stats", type=str, help="Export statistics to file")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "algorithm": args.algorithm,
        "enable_ml": args.enable_ml,
        "health_check_interval": 10,
        "ml_training_interval": 300
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create load balancer
    load_balancer = DynamicLoadBalancer(config)
    
    try:
        await load_balancer.start()
        
        if args.test:
            # Run test simulation
            logger.info("Running load balancer test simulation...")
            
            # Register test services
            for i in range(5):
                service = BackendService(
                    service_id=f"service_{i}",
                    host=f"192.168.1.{100+i}",
                    port=8080 + i,
                    weight=random.uniform(0.5, 2.0),
                    max_connections=100
                )
                load_balancer.register_service(service)
            
            # Simulate requests
            for i in range(1000):
                request_type = random.choice(list(RequestType))
                selected_service = await load_balancer.select_backend(request_type)
                
                if selected_service:
                    # Simulate request processing
                    processing_time = random.uniform(50, 500)  # 50-500ms
                    success = random.random() > 0.05  # 95% success rate
                    
                    request_metrics = RequestMetrics(
                        request_id=f"req_{i}",
                        timestamp=time.time(),
                        request_type=request_type,
                        size_bytes=random.randint(1024, 10240),
                        processing_time=processing_time,
                        backend_service=selected_service.service_id,
                        success=success
                    )
                    
                    await load_balancer.record_request_completion(
                        selected_service.service_id, request_metrics
                    )
                    
                    # Update service health randomly
                    if random.random() < 0.1:  # 10% chance
                        health_metrics = {
                            "cpu_usage": random.uniform(10, 95),
                            "memory_usage": random.uniform(20, 80),
                            "load_average": random.uniform(0.5, 4.0),
                            "responsive": random.random() > 0.02  # 98% responsive
                        }
                        await load_balancer.update_service_health(
                            selected_service.service_id, health_metrics
                        )
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} test requests")
            
            # Print final statistics
            stats = load_balancer.get_comprehensive_stats()
            print(json.dumps(stats, indent=2, default=str))
        
        elif args.monitor:
            logger.info("Starting load balancer monitoring. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(60)
                    stats = load_balancer.get_comprehensive_stats()
                    
                    # Print summary
                    total_services = len(stats["services"])
                    healthy_services = len([s for s in stats["services"].values() 
                                          if s["health_status"] == "healthy"])
                    total_requests = stats["request_stats"]["total_requests"]
                    
                    print(f"Load Balancer Status: {healthy_services}/{total_services} healthy services, "
                          f"{total_requests} total requests processed")
                    
            except KeyboardInterrupt:
                logger.info("Stopping monitoring...")
        
        if args.export_stats:
            load_balancer.export_stats(args.export_stats)
            print(f"Statistics exported to {args.export_stats}")
    
    finally:
        await load_balancer.stop()

if __name__ == "__main__":
    asyncio.run(main())