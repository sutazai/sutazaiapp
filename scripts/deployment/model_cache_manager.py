#!/usr/bin/env python3
"""
Purpose: Intelligent model caching and sharing system for multi-agent deployment
Usage: Manages model loading, caching, and sharing across 69 agents
Requirements: asyncio, psutil, mmap, redis
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import hashlib
import mmap
import pickle
import psutil
from pathlib import Path
import numpy as np
from collections import OrderedDict, defaultdict
import weakref
import gc
import aioredis

logger = logging.getLogger('model-cache-manager')


@dataclass
class CachedModel:
    """Represents a cached model in memory"""
    model_id: str
    model_name: str
    size_mb: float
    load_time: datetime
    last_access: datetime
    access_count: int = 0
    agents_using: Set[str] = field(default_factory=set)
    memory_mapped: bool = False
    shared_memory_key: Optional[str] = None
    weak_refs: List[weakref.ref] = field(default_factory=list)
    
    @property
    def time_since_access(self) -> float:
        """Time in seconds since last access"""
        return (datetime.utcnow() - self.last_access).total_seconds()
    
    @property
    def memory_efficiency_score(self) -> float:
        """Score for memory efficiency (higher is better)"""
        # Consider: usage frequency, number of agents, size
        usage_score = min(self.access_count / 100, 1.0)
        sharing_score = min(len(self.agents_using) / 10, 1.0)
        size_penalty = 1.0 / (1 + self.size_mb / 100)
        
        return (usage_score * 0.4 + sharing_score * 0.4 + size_penalty * 0.2)


@dataclass
class CacheMetrics:
    """Metrics for cache performance"""
    total_models_loaded: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_memory_used_mb: float = 0.0
    peak_memory_used_mb: float = 0.0
    models_evicted: int = 0
    average_load_time_ms: float = 0.0
    memory_saved_by_sharing_mb: float = 0.0
    

class ModelCacheManager:
    """
    Intelligent model caching and sharing system
    
    Features:
    - Memory-mapped model sharing
    - LRU eviction with intelligent scoring
    - Agent affinity tracking
    - Predictive preloading
    - Memory pressure handling
    - Zero-copy model sharing
    """
    
    def __init__(self,
                 max_memory_mb: int = 8192,
                 cache_dir: str = "/opt/sutazaiapp/model_cache",
                 redis_url: str = "redis://localhost:6379"):
        
        self.max_memory_mb = max_memory_mb
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.redis_url = redis_url
        
        # Cache storage
        self.model_cache: OrderedDict[str, CachedModel] = OrderedDict()
        self.model_data: Dict[str, Any] = {}  # Actual model data
        
        # Memory mapping
        self.memory_maps: Dict[str, mmap.mmap] = {}
        
        # Agent tracking
        self.agent_models: Dict[str, Set[str]] = defaultdict(set)  # agent -> models
        self.model_affinity: Dict[str, Dict[str, float]] = defaultdict(dict)  # model -> agent -> score
        
        # Metrics
        self.metrics = CacheMetrics()
        
        # Background tasks
        self._monitor_task = None
        self._preload_task = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize the cache manager"""
        logger.info("Initializing model cache manager")
        
        # Connect to Redis for distributed coordination
        self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._memory_monitor())
        self._preload_task = asyncio.create_task(self._predictive_preloader())
        
        # Load cache metadata
        await self._load_cache_metadata()
        
        logger.info(f"Cache initialized with {self.max_memory_mb}MB limit")
        
    async def get_model(self, 
                       model_name: str,
                       agent_id: str,
                       required_memory_mb: float = 100) -> Optional[Any]:
        """
        Get a model from cache or load it
        
        Args:
            model_name: Name of the model
            agent_id: ID of requesting agent
            required_memory_mb: Estimated memory requirement
            
        Returns:
            Model object or None if cannot load
        """
        model_id = self._get_model_id(model_name)
        
        # Check if in cache
        if model_id in self.model_cache:
            # Cache hit
            self.metrics.cache_hits += 1
            cached_model = self.model_cache[model_id]
            
            # Update access info
            cached_model.last_access = datetime.utcnow()
            cached_model.access_count += 1
            cached_model.agents_using.add(agent_id)
            
            # Move to end (LRU)
            self.model_cache.move_to_end(model_id)
            
            # Update affinity
            self.model_affinity[model_id][agent_id] = self.model_affinity[model_id].get(agent_id, 0) + 1
            self.agent_models[agent_id].add(model_id)
            
            logger.debug(f"Cache hit for {model_name} requested by {agent_id}")
            
            return self.model_data.get(model_id)
        
        # Cache miss
        self.metrics.cache_misses += 1
        
        # Check memory availability
        if not await self._ensure_memory_available(required_memory_mb):
            logger.warning(f"Insufficient memory to load {model_name}")
            return None
        
        # Load model
        model = await self._load_model(model_name, model_id)
        
        if model:
            # Create cache entry
            cached_model = CachedModel(
                model_id=model_id,
                model_name=model_name,
                size_mb=required_memory_mb,
                load_time=datetime.utcnow(),
                last_access=datetime.utcnow(),
                access_count=1,
                agents_using={agent_id}
            )
            
            # Add to cache
            self.model_cache[model_id] = cached_model
            self.model_data[model_id] = model
            
            # Update tracking
            self.agent_models[agent_id].add(model_id)
            self.model_affinity[model_id][agent_id] = 1
            
            # Update metrics
            self.metrics.total_models_loaded += 1
            self.metrics.total_memory_used_mb += required_memory_mb
            
            logger.info(f"Loaded {model_name} for {agent_id} ({required_memory_mb}MB)")
        
        return model
    
    async def share_model(self,
                         model_name: str,
                         source_agent: str,
                         target_agents: List[str]) -> bool:
        """
        Share a model between agents using zero-copy techniques
        
        Args:
            model_name: Model to share
            source_agent: Agent that has the model
            target_agents: Agents to share with
            
        Returns:
            Success status
        """
        model_id = self._get_model_id(model_name)
        
        if model_id not in self.model_cache:
            logger.error(f"Model {model_name} not in cache")
            return False
        
        cached_model = self.model_cache[model_id]
        
        # Create memory-mapped file if not already done
        if not cached_model.memory_mapped:
            success = await self._create_memory_map(model_id)
            if not success:
                return False
        
        # Share with target agents
        for agent_id in target_agents:
            cached_model.agents_using.add(agent_id)
            self.agent_models[agent_id].add(model_id)
            self.model_affinity[model_id][agent_id] = self.model_affinity[model_id].get(agent_id, 0) + 0.5
        
        # Calculate memory saved
        memory_saved = cached_model.size_mb * (len(target_agents) - 1)
        self.metrics.memory_saved_by_sharing_mb += memory_saved
        
        logger.info(f"Shared {model_name} with {len(target_agents)} agents, saved {memory_saved}MB")
        
        return True
    
    async def _ensure_memory_available(self, required_mb: float) -> bool:
        """Ensure enough memory is available, evicting if necessary"""
        
        current_usage = self.metrics.total_memory_used_mb
        available = self.max_memory_mb - current_usage
        
        if available >= required_mb:
            return True
        
        # Need to evict
        needed_mb = required_mb - available
        evicted_mb = 0
        
        # Get eviction candidates sorted by score (lower score = evict first)
        candidates = self._get_eviction_candidates()
        
        for model_id, score in candidates:
            if evicted_mb >= needed_mb:
                break
            
            cached_model = self.model_cache[model_id]
            
            # Check if model is actively used
            if len(cached_model.agents_using) > 2 and cached_model.time_since_access < 60:
                continue  # Skip actively used models
            
            # Evict model
            await self._evict_model(model_id)
            evicted_mb += cached_model.size_mb
            
        return evicted_mb >= needed_mb
    
    def _get_eviction_candidates(self) -> List[Tuple[str, float]]:
        """Get models sorted by eviction priority (lower score = evict first)"""
        
        candidates = []
        
        for model_id, cached_model in self.model_cache.items():
            # Calculate eviction score
            # Higher score = keep in cache
            
            # Base score on LRU (time since access)
            lru_score = 1.0 / (1 + cached_model.time_since_access / 3600)  # Decay over hours
            
            # Factor in usage frequency
            usage_score = min(cached_model.access_count / 100, 1.0)
            
            # Factor in number of agents using
            sharing_score = min(len(cached_model.agents_using) / 5, 1.0)
            
            # Size penalty (prefer evicting larger models)
            size_score = 1.0 - (cached_model.size_mb / self.max_memory_mb)
            
            # Combined score
            total_score = (
                lru_score * 0.3 +
                usage_score * 0.3 +
                sharing_score * 0.3 +
                size_score * 0.1
            )
            
            candidates.append((model_id, total_score))
        
        # Sort by score (ascending = evict first)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    async def _evict_model(self, model_id: str):
        """Evict a model from cache"""
        
        if model_id not in self.model_cache:
            return
        
        cached_model = self.model_cache[model_id]
        
        logger.info(f"Evicting model {cached_model.model_name} "
                   f"(used by {len(cached_model.agents_using)} agents)")
        
        # Clean up memory map
        if model_id in self.memory_maps:
            self.memory_maps[model_id].close()
            del self.memory_maps[model_id]
        
        # Remove from agent tracking
        for agent_id in cached_model.agents_using:
            self.agent_models[agent_id].discard(model_id)
        
        # Update metrics
        self.metrics.total_memory_used_mb -= cached_model.size_mb
        self.metrics.models_evicted += 1
        
        # Remove from cache
        del self.model_cache[model_id]
        del self.model_data[model_id]
        
        # Force garbage collection
        gc.collect()
    
    async def _load_model(self, model_name: str, model_id: str) -> Optional[Any]:
        """Load a model from disk"""
        
        start_time = datetime.utcnow()
        
        try:
            # Simulate model loading
            # In practice, this would load actual model files
            
            model_path = self.cache_dir / f"{model_name}.model"
            
            # Check if model file exists
            if not model_path.exists():
                # Download or create model
                await self._download_model(model_name, model_path)
            
            # Load model (simulated)
            await asyncio.sleep(0.1)  # Simulate load time
            
            model_data = {"name": model_name, "data": np.random.randn(1000, 1000)}
            
            # Update load time metric
            load_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.average_load_time_ms = (
                (self.metrics.average_load_time_ms * (self.metrics.total_models_loaded - 1) + load_time_ms) /
                self.metrics.total_models_loaded
            )
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    async def _create_memory_map(self, model_id: str) -> bool:
        """Create memory-mapped file for zero-copy sharing"""
        
        try:
            model_data = self.model_data.get(model_id)
            if not model_data:
                return False
            
            # Serialize model data
            serialized = pickle.dumps(model_data)
            
            # Create memory-mapped file
            mmap_path = self.cache_dir / f"{model_id}.mmap"
            
            with open(mmap_path, 'wb') as f:
                f.write(serialized)
            
            # Open as memory map
            with open(mmap_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                self.memory_maps[model_id] = mm
            
            # Update cache entry
            self.model_cache[model_id].memory_mapped = True
            self.model_cache[model_id].shared_memory_key = str(mmap_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create memory map for {model_id}: {e}")
            return False
    
    async def _memory_monitor(self):
        """Monitor memory usage and system pressure"""
        
        while True:
            try:
                # Get system memory info
                memory = psutil.virtual_memory()
                
                # Check if under memory pressure
                if memory.percent > 90:
                    logger.warning(f"High memory pressure: {memory.percent}%")
                    
                    # Aggressive eviction
                    target_reduction = self.metrics.total_memory_used_mb * 0.2
                    await self._reduce_memory_usage(target_reduction)
                
                # Update peak memory
                self.metrics.peak_memory_used_mb = max(
                    self.metrics.peak_memory_used_mb,
                    self.metrics.total_memory_used_mb
                )
                
                # Sleep
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                await asyncio.sleep(10)
    
    async def _predictive_preloader(self):
        """Predictively preload models based on usage patterns"""
        
        while True:
            try:
                # Analyze usage patterns
                predictions = self._predict_model_usage()
                
                for model_name, probability in predictions.items():
                    if probability > 0.7:  # High probability of use
                        model_id = self._get_model_id(model_name)
                        
                        if model_id not in self.model_cache:
                            # Preload if we have memory
                            if self.metrics.total_memory_used_mb < self.max_memory_mb * 0.8:
                                logger.info(f"Preloading {model_name} (probability: {probability:.2f})")
                                await self._load_model(model_name, model_id)
                
                # Sleep for prediction interval
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in predictive preloader: {e}")
                await asyncio.sleep(60)
    
    def _predict_model_usage(self) -> Dict[str, float]:
        """Predict which models will be needed soon"""
        
        predictions = {}
        
        # Simple prediction based on affinity scores
        for model_id, agent_scores in self.model_affinity.items():
            if model_id in self.model_cache:
                continue  # Already loaded
            
            # Calculate probability based on agent activity
            total_score = sum(agent_scores.values())
            active_agents = len([a for a in agent_scores if a in self.agent_models])
            
            probability = min(total_score / 100 + active_agents * 0.1, 1.0)
            
            model_name = self._get_model_name(model_id)
            predictions[model_name] = probability
        
        return predictions
    
    async def _reduce_memory_usage(self, target_mb: float):
        """Reduce memory usage by target amount"""
        
        reduced = 0
        candidates = self._get_eviction_candidates()
        
        for model_id, _ in candidates:
            if reduced >= target_mb:
                break
            
            cached_model = self.model_cache.get(model_id)
            if cached_model:
                await self._evict_model(model_id)
                reduced += cached_model.size_mb
    
    def _get_model_id(self, model_name: str) -> str:
        """Generate consistent model ID"""
        return hashlib.md5(model_name.encode()).hexdigest()
    
    def _get_model_name(self, model_id: str) -> str:
        """Get model name from ID"""
        # In practice, maintain a mapping
        return f"model_{model_id[:8]}"
    
    async def _download_model(self, model_name: str, model_path: Path):
        """Download or create model file"""
        # Simulate model download
        await asyncio.sleep(0.5)
        
        # Create dummy file
        model_path.write_bytes(b"dummy model data")
    
    async def _load_cache_metadata(self):
        """Load cache metadata from Redis"""
        if self.redis_client:
            # Load persistent cache info
            pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        hit_rate = (self.metrics.cache_hits / 
                   (self.metrics.cache_hits + self.metrics.cache_misses)
                   if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0)
        
        return {
            'models_cached': len(self.model_cache),
            'memory_used_mb': round(self.metrics.total_memory_used_mb, 2),
            'memory_limit_mb': self.max_memory_mb,
            'memory_utilization': round(self.metrics.total_memory_used_mb / self.max_memory_mb * 100, 2),
            'cache_hit_rate': round(hit_rate, 3),
            'total_hits': self.metrics.cache_hits,
            'total_misses': self.metrics.cache_misses,
            'models_evicted': self.metrics.models_evicted,
            'memory_saved_by_sharing_mb': round(self.metrics.memory_saved_by_sharing_mb, 2),
            'average_load_time_ms': round(self.metrics.average_load_time_ms, 2),
            'peak_memory_mb': round(self.metrics.peak_memory_used_mb, 2),
            'top_models': self._get_top_models(5),
            'agent_model_count': {
                agent: len(models) for agent, models in self.agent_models.items()
            }
        }
    
    def _get_top_models(self, n: int) -> List[Dict[str, Any]]:
        """Get top N most used models"""
        
        models = []
        for model_id, cached_model in self.model_cache.items():
            models.append({
                'name': cached_model.model_name,
                'size_mb': cached_model.size_mb,
                'access_count': cached_model.access_count,
                'agents_using': len(cached_model.agents_using),
                'efficiency_score': cached_model.memory_efficiency_score
            })
        
        # Sort by efficiency score
        models.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        return models[:n]
    
    async def cleanup(self):
        """Clean up resources"""
        
        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._preload_task:
            self._preload_task.cancel()
        
        # Close memory maps
        for mm in self.memory_maps.values():
            mm.close()
        
        # Close Redis
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        
        logger.info("Model cache manager cleaned up")


async def main():
    """Demo model cache manager"""
    
    cache = ModelCacheManager(
        max_memory_mb=4096,
        cache_dir="/opt/sutazaiapp/model_cache"
    )
    
    await cache.initialize()
    
    # Simulate agent requests
    agents = [f"agent_{i}" for i in range(10)]
    models = ["tinyllama"]  # Only using GPT-OSS model
    
    # Random access pattern
    for _ in range(50):
        agent = np.random.choice(agents)
        model = np.random.choice(models)
        
        result = await cache.get_model(model, agent, required_memory_mb=np.random.randint(50, 200))
        
        await asyncio.sleep(0.1)
    
    # Test model sharing
    await cache.share_model("tinyllama", "agent_0", ["agent_1", "agent_2", "agent_3"])
    
    # Get stats
    stats = cache.get_cache_stats()
    logger.info(json.dumps(stats, indent=2))
    
    await cache.cleanup()


if __name__ == "__main__":
    asyncio.run(main())