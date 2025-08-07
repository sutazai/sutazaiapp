#!/usr/bin/env python3
"""
Feature Flags and Graceful Degradation for SutazAI Self-Healing Architecture

This module implements feature flags and graceful degradation capabilities
to maintain system functionality even when services are unavailable.

Author: SutazAI Infrastructure Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import redis
from contextlib import asynccontextmanager
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureState(Enum):
    """Feature flag states"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"
    EMERGENCY_DISABLED = "emergency_disabled"

class DegradationLevel(Enum):
    """Degradation levels for graceful fallback"""
    FULL = "full"           # Full functionality
    REDUCED = "reduced"     # Reduced functionality
    MINIMAL = "minimal"     # Minimal functionality
    OFFLINE = "offline"     # Offline mode

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    name: str
    state: FeatureState = FeatureState.ENABLED
    rollout_percentage: float = 100.0
    dependencies: List[str] = field(default_factory=list)
    fallback_config: Dict[str, Any] = field(default_factory=dict)
    emergency_contacts: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    description: str = ""

@dataclass
class FallbackStrategy:
    """Fallback strategy configuration"""
    name: str
    priority: int
    handler: Callable
    requirements: List[str] = field(default_factory=list)
    degradation_level: DegradationLevel = DegradationLevel.REDUCED
    timeout: float = 30.0
    cache_ttl: int = 300

class FeatureFlagManager:
    """
    Feature flag manager with Redis backend and graceful degradation
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._create_redis_client()
        self.flags: Dict[str, FeatureFlag] = {}
        self.fallback_strategies: Dict[str, List[FallbackStrategy]] = defaultdict(list)
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Load initial flags
        self._load_flags_from_redis()
        self._initialize_default_flags()
    
    def _create_redis_client(self) -> Optional[redis.Redis]:
        """Create Redis client with fallback"""
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis not available for feature flags: {e}")
            return None
    
    def _load_flags_from_redis(self):
        """Load feature flags from Redis"""
        if not self.redis_client:
            return
        
        try:
            keys = self.redis_client.keys("feature_flag:*")
            for key in keys:
                flag_data = self.redis_client.get(key)
                if flag_data:
                    flag_dict = json.loads(flag_data)
                    flag = FeatureFlag(**flag_dict)
                    self.flags[flag.name] = flag
            
            logger.info(f"Loaded {len(self.flags)} feature flags from Redis")
        except Exception as e:
            logger.warning(f"Failed to load feature flags from Redis: {e}")
    
    def _save_flag_to_redis(self, flag: FeatureFlag):
        """Save feature flag to Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"feature_flag:{flag.name}"
            flag_dict = {
                'name': flag.name,
                'state': flag.state.value,
                'rollout_percentage': flag.rollout_percentage,
                'dependencies': flag.dependencies,
                'fallback_config': flag.fallback_config,
                'emergency_contacts': flag.emergency_contacts,
                'created_at': flag.created_at,
                'updated_at': flag.updated_at,
                'description': flag.description
            }
            self.redis_client.setex(key, 86400, json.dumps(flag_dict))  # 24 hour TTL
        except Exception as e:
            logger.warning(f"Failed to save feature flag to Redis: {e}")
    
    def _initialize_default_flags(self):
        """Initialize default feature flags for SutazAI"""
        default_flags = [
            # AI Agent Features
            FeatureFlag(
                name="ai_agents_enabled",
                state=FeatureState.ENABLED,
                dependencies=["ollama_service", "backend_service"],
                description="Enable AI agent functionality"
            ),
            FeatureFlag(
                name="letta_agent",
                state=FeatureState.ENABLED,
                dependencies=["ai_agents_enabled", "postgres_service"],
                description="Letta (MemGPT) agent functionality"
            ),
            FeatureFlag(
                name="autogpt_agent",
                state=FeatureState.ENABLED,
                dependencies=["ai_agents_enabled", "redis_service"],
                description="AutoGPT agent functionality"
            ),
            FeatureFlag(
                name="langchain_workflows",
                state=FeatureState.ENABLED,
                dependencies=["ai_agents_enabled", "chromadb_service"],
                description="LangChain workflow functionality"
            ),
            
            # Core Features
            FeatureFlag(
                name="vector_search",
                state=FeatureState.ENABLED,
                dependencies=["chromadb_service"],
                description="Vector similarity search"
            ),
            FeatureFlag(
                name="memory_persistence",
                state=FeatureState.ENABLED,
                dependencies=["postgres_service"],
                description="Persistent memory storage"
            ),
            FeatureFlag(
                name="real_time_collaboration",
                state=FeatureState.ENABLED,
                dependencies=["redis_service", "backend_service"],
                description="Real-time collaboration features"
            ),
            
            # Advanced Features
            FeatureFlag(
                name="advanced_analytics",
                state=FeatureState.ROLLOUT,
                rollout_percentage=50.0,
                dependencies=["prometheus_service"],
                description="Advanced analytics and insights"
            ),
            FeatureFlag(
                name="multi_modal_processing",
                state=FeatureState.ROLLOUT,
                rollout_percentage=25.0,
                dependencies=["ollama_service"],
                description="Multi-modal AI processing"
            ),
            
            # Experimental Features
            FeatureFlag(
                name="quantum_optimization",
                state=FeatureState.DISABLED,
                description="Experimental quantum optimization features"
            ),
            FeatureFlag(
                name="federated_learning",
                state=FeatureState.DISABLED,
                description="Federated learning capabilities"
            )
        ]
        
        for flag in default_flags:
            if flag.name not in self.flags:
                self.flags[flag.name] = flag
                self._save_flag_to_redis(flag)
    
    def create_flag(self, 
                   name: str, 
                   state: FeatureState = FeatureState.DISABLED,
                   dependencies: List[str] = None,
                   description: str = "") -> FeatureFlag:
        """Create new feature flag"""
        with self._lock:
            if name in self.flags:
                raise ValueError(f"Feature flag {name} already exists")
            
            flag = FeatureFlag(
                name=name,
                state=state,
                dependencies=dependencies or [],
                description=description
            )
            
            self.flags[name] = flag
            self._save_flag_to_redis(flag)
            
            logger.info(f"Created feature flag: {name}")
            return flag
    
    def update_flag(self, 
                   name: str, 
                   state: Optional[FeatureState] = None,
                   rollout_percentage: Optional[float] = None) -> FeatureFlag:
        """Update existing feature flag"""
        with self._lock:
            if name not in self.flags:
                raise ValueError(f"Feature flag {name} not found")
            
            flag = self.flags[name]
            
            if state is not None:
                flag.state = state
            if rollout_percentage is not None:
                flag.rollout_percentage = rollout_percentage
            
            flag.updated_at = time.time()
            self._save_flag_to_redis(flag)
            
            logger.info(f"Updated feature flag: {name} -> {flag.state.value}")
            return flag
    
    def emergency_disable(self, name: str, reason: str = ""):
        """Emergency disable feature flag"""
        with self._lock:
            if name in self.flags:
                flag = self.flags[name]
                old_state = flag.state
                flag.state = FeatureState.EMERGENCY_DISABLED
                flag.updated_at = time.time()
                self._save_flag_to_redis(flag)
                
                logger.critical(
                    f"EMERGENCY DISABLE: {name} ({old_state.value} -> "
                    f"{flag.state.value}) - Reason: {reason}"
                )
                
                # Notify emergency contacts if configured
                self._notify_emergency_contacts(flag, reason)
    
    def _notify_emergency_contacts(self, flag: FeatureFlag, reason: str):
        """Notify emergency contacts about flag changes"""
        # This would integrate with alerting systems
        # For now, just log the notification
        for contact in flag.emergency_contacts:
            logger.critical(
                f"EMERGENCY NOTIFICATION to {contact}: "
                f"Feature {flag.name} emergency disabled - {reason}"
            )
    
    def is_enabled(self, name: str, user_id: str = None) -> bool:
        """Check if feature flag is enabled"""
        with self._lock:
            if name not in self.flags:
                logger.warning(f"Feature flag {name} not found, defaulting to disabled")
                return False
            
            flag = self.flags[name]
            
            # Check flag state
            if flag.state == FeatureState.DISABLED:
                return False
            elif flag.state == FeatureState.EMERGENCY_DISABLED:
                return False
            elif flag.state == FeatureState.ENABLED:
                return self._check_dependencies(flag)
            elif flag.state == FeatureState.ROLLOUT:
                # Check rollout percentage
                if user_id:
                    # Use consistent hash for user-based rollout
                    import hashlib
                    hash_value = int(hashlib.md5(f"{name}:{user_id}".encode()).hexdigest(), 16)
                    percentage = (hash_value % 100) + 1
                    return percentage <= flag.rollout_percentage and self._check_dependencies(flag)
                else:
                    # Random rollout without user ID
                    import random
                    return random.random() * 100 <= flag.rollout_percentage and self._check_dependencies(flag)
            
            return False
    
    def _check_dependencies(self, flag: FeatureFlag) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in flag.dependencies:
            if dep.endswith("_service"):
                # Service dependency - check if service is healthy
                if not self._is_service_healthy(dep):
                    return False
            else:
                # Feature flag dependency
                if not self.is_enabled(dep):
                    return False
        return True
    
    def _is_service_healthy(self, service_name: str) -> bool:
        """Check if service is healthy (integrates with circuit breaker)"""
        try:
            if self.redis_client:
                # Remove _service suffix for health check
                service_key = service_name.replace("_service", "")
                health_key = f"health:{service_key}"
                health_data = self.redis_client.get(health_key)
                
                if health_data:
                    health = json.loads(health_data)
                    return health.get('status') == 'healthy'
            
            # If no health data available, assume healthy
            return True
        except Exception as e:
            logger.warning(f"Failed to check service health for {service_name}: {e}")
            return True  # Fail open
    
    def register_fallback(self, 
                         feature_name: str, 
                         strategy: FallbackStrategy):
        """Register fallback strategy for feature"""
        with self._lock:
            self.fallback_strategies[feature_name].append(strategy)
            # Sort by priority (lower number = higher priority)
            self.fallback_strategies[feature_name].sort(key=lambda x: x.priority)
            
            logger.info(f"Registered fallback strategy {strategy.name} for {feature_name}")
    
    async def execute_with_fallback(self, 
                                  feature_name: str, 
                                  primary_func: Callable,
                                  *args, 
                                  user_id: str = None,
                                  **kwargs) -> Any:
        """Execute function with fallback strategies"""
        # Check if feature is enabled
        if self.is_enabled(feature_name, user_id):
            try:
                return await self._execute_function(primary_func, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed for {feature_name}: {e}")
                # Continue to fallback strategies
        
        # Execute fallback strategies
        strategies = self.fallback_strategies.get(feature_name, [])
        
        for strategy in strategies:
            try:
                # Check strategy requirements
                if not self._check_strategy_requirements(strategy):
                    continue
                
                logger.info(
                    f"Executing fallback strategy {strategy.name} for {feature_name}"
                )
                
                result = await asyncio.wait_for(
                    self._execute_function(strategy.handler, *args, **kwargs),
                    timeout=strategy.timeout
                )
                
                # Cache successful fallback result
                self._cache_result(feature_name, result, strategy.cache_ttl)
                
                return result
                
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy.name} failed: {e}")
                continue
        
        # All strategies failed, try cache
        cached_result = self._get_cached_result(feature_name)
        if cached_result is not None:
            logger.info(f"Returning cached result for {feature_name}")
            return cached_result
        
        # No fallback available
        raise RuntimeError(f"All fallback strategies failed for {feature_name}")
    
    def _check_strategy_requirements(self, strategy: FallbackStrategy) -> bool:
        """Check if strategy requirements are met"""
        for req in strategy.requirements:
            if req.endswith("_service"):
                if not self._is_service_healthy(req):
                    return False
            elif not self.is_enabled(req):
                return False
        return True
    
    async def _execute_function(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function (sync or async)"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _cache_result(self, key: str, result: Any, ttl: int):
        """Cache result with TTL"""
        with self._lock:
            self.cache[key] = result
            self.cache_timestamps[key] = time.time() + ttl
    
    def _get_cached_result(self, key: str) -> Any:
        """Get cached result if not expired"""
        with self._lock:
            if key in self.cache:
                if time.time() < self.cache_timestamps.get(key, 0):
                    return self.cache[key]
                else:
                    # Expired, remove from cache
                    self.cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
            return None
    
    def get_degradation_level(self) -> DegradationLevel:
        """Calculate current system degradation level"""
        critical_features = [
            "ai_agents_enabled",
            "vector_search", 
            "memory_persistence"
        ]
        
        healthy_critical = sum(
            1 for feature in critical_features 
            if self.is_enabled(feature)
        )
        
        total_critical = len(critical_features)
        health_percentage = healthy_critical / total_critical
        
        if health_percentage >= 0.9:
            return DegradationLevel.FULL
        elif health_percentage >= 0.7:
            return DegradationLevel.REDUCED
        elif health_percentage >= 0.3:
            return DegradationLevel.MINIMAL
        else:
            return DegradationLevel.OFFLINE
    
    def get_feature_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feature flags"""
        with self._lock:
            status = {}
            for name, flag in self.flags.items():
                status[name] = {
                    'state': flag.state.value,
                    'enabled': self.is_enabled(name),
                    'rollout_percentage': flag.rollout_percentage,
                    'dependencies': flag.dependencies,
                    'dependencies_healthy': self._check_dependencies(flag),
                    'description': flag.description,
                    'updated_at': flag.updated_at
                }
            return status

# Graceful degradation decorators and context managers
class GracefulDegradation:
    """Graceful degradation utilities"""
    
    def __init__(self, feature_manager: FeatureFlagManager):
        self.feature_manager = feature_manager
    
    def feature_flag(self, feature_name: str, fallback_result: Any = None):
        """Decorator for feature flag protection"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    if self.feature_manager.is_enabled(feature_name):
                        try:
                            return await func(*args, **kwargs)
                        except Exception as e:
                            logger.warning(f"Feature {feature_name} failed: {e}")
                            return fallback_result
                    return fallback_result
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    if self.feature_manager.is_enabled(feature_name):
                        try:
                            return func(*args, **kwargs)
                        except Exception as e:
                            logger.warning(f"Feature {feature_name} failed: {e}")
                            return fallback_result
                    return fallback_result
                return sync_wrapper
        return decorator
    
    @asynccontextmanager
    async def degradation_context(self, feature_name: str):
        """Context manager for graceful degradation"""
        enabled = self.feature_manager.is_enabled(feature_name)
        degradation_level = self.feature_manager.get_degradation_level()
        
        context = {
            'feature_enabled': enabled,
            'degradation_level': degradation_level,
            'should_execute': enabled
        }
        
        try:
            yield context
        except Exception as e:
            logger.warning(f"Feature {feature_name} failed in context: {e}")
            context['error'] = e
            context['should_execute'] = False

# Global feature manager instance
feature_manager = FeatureFlagManager()
graceful_degradation = GracefulDegradation(feature_manager)

# Example usage
async def example_usage():
    """Example usage of feature flags and graceful degradation"""
    
    # Check if feature is enabled
    if feature_manager.is_enabled("ai_agents_enabled"):
        print("AI agents are enabled")
    
    # Use with fallback
    async def primary_ai_function():
        # Simulate AI processing
        await asyncio.sleep(0.1)
        return {"result": "AI processed successfully"}
    
    async def fallback_function():
        return {"result": "Fallback: Basic processing", "degraded": True}
    
    # Register fallback strategy
    fallback_strategy = FallbackStrategy(
        name="basic_processing",
        priority=1,
        handler=fallback_function,
        degradation_level=DegradationLevel.REDUCED
    )
    feature_manager.register_fallback("ai_agents_enabled", fallback_strategy)
    
    # Execute with fallback
    try:
        result = await feature_manager.execute_with_fallback(
            "ai_agents_enabled",
            primary_ai_function
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"All fallbacks failed: {e}")
    
    # Get system status
    degradation_level = feature_manager.get_degradation_level()
    print(f"System degradation level: {degradation_level.value}")
    
    feature_status = feature_manager.get_feature_status()
    print(f"Feature status: {json.dumps(feature_status, indent=2)}")

if __name__ == "__main__":
    asyncio.run(example_usage())