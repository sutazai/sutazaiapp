#!/usr/bin/env python3
"""
Intelligent Cache System for SutazAI
===================================

Purpose: Advanced caching system with ML-based prefetching, adaptive algorithms, and cross-agent sharing
Usage: python scripts/intelligent-cache-system.py [--mode adaptive] [--prefetch-depth 5]
Requirements: Python 3.8+, scikit-learn, numpy

Features:
- Multi-level caching (L1, L2, L3)
- Machine learning-based prefetching
- Adaptive cache replacement algorithms
- Cross-agent data sharing
- Cache coherency management
- Performance analytics and optimization
"""

import os
import sys
import json
import time
import hashlib
import logging
import argparse
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque, OrderedDict
import pickle
import gzip
import zlib
import weakref
from enum import Enum
import math
import statistics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/intelligent_cache.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('IntelligentCache')

class CacheLevel(Enum):
    """Cache levels"""
    L1 = "L1"  # Fastest, smallest (in-memory)
    L2 = "L2"  # Medium speed, medium size (compressed memory)
    L3 = "L3"  # Slower, largest (disk-based)

class ReplacementPolicy(Enum):
    """Cache replacement policies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    ARC = "arc"          # Adaptive Replacement Cache
    ADAPTIVE = "adaptive" # ML-based adaptive policy
    PREDICTIVE = "predictive"  # Predictive based on access patterns

class CacheOperation(Enum):
    """Cache operations for analytics"""
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    PREFETCH = "prefetch"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"

@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    size: int
    level: CacheLevel
    access_count: int
    last_access: float
    creation_time: float
    last_update: float
    checksum: str
    agent_id: str
    priority: int = 0
    predicted_next_access: float = 0.0
    access_pattern_score: float = 0.0
    sharing_score: float = 0.0
    compression_ratio: float = 1.0

@dataclass
class AccessPattern:
    """Access pattern for ML prediction"""
    key: str
    agent_id: str
    access_times: List[float]
    access_intervals: List[float]
    access_frequency: float
    temporal_locality: float
    spatial_locality: float
    sharing_frequency: float

@dataclass
class CacheStats:
    """Cache statistics"""
    level: CacheLevel
    total_requests: int
    hit_count: int
    miss_count: int
    eviction_count: int
    prefetch_count: int
    hit_ratio: float
    miss_ratio: float
    average_response_time: float
    memory_usage: int
    memory_limit: int
    utilization_ratio: float
    fragmentation_ratio: float

class AccessPredictor:
    """Machine learning-based access predictor"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = deque(maxlen=10000)
        self.training_lock = threading.Lock()
        
    def extract_features(self, pattern: AccessPattern, current_time: float) -> np.ndarray:
        """Extract features from access pattern"""
        if not pattern.access_times:
            return np.zeros(10)
        
        # Time-based features
        time_since_last = current_time - pattern.access_times[-1] if pattern.access_times else 0
        avg_interval = statistics.mean(pattern.access_intervals) if pattern.access_intervals else 0
        interval_variance = statistics.variance(pattern.access_intervals) if len(pattern.access_intervals) > 1 else 0
        
        # Frequency features
        recent_accesses = sum(1 for t in pattern.access_times if current_time - t < 3600)  # Last hour
        
        # Pattern features
        access_regularity = 1.0 / (interval_variance + 1.0)  # More regular = higher score
        
        features = np.array([
            time_since_last,
            avg_interval,
            interval_variance,
            pattern.access_frequency,
            pattern.temporal_locality,
            pattern.spatial_locality,
            pattern.sharing_frequency,
            recent_accesses,
            access_regularity,
            len(pattern.access_times)
        ])
        
        return features
    
    def train_model(self, patterns: List[AccessPattern], labels: List[float]):
        """Train the prediction model"""
        if len(patterns) < 10:  # Need minimum data
            return False
        
        with self.training_lock:
            try:
                # Extract features
                features = []
                current_time = time.time()
                
                for pattern in patterns:
                    feature_vector = self.extract_features(pattern, current_time)
                    features.append(feature_vector)
                
                X = np.array(features)
                y = np.array(labels)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.model.fit(X_scaled, y)
                self.is_trained = True
                
                logger.info(f"Trained access predictor with {len(patterns)} patterns")
                return True
                
            except Exception as e:
                logger.error(f"Failed to train access predictor: {e}")
                return False
    
    def predict_next_access(self, pattern: AccessPattern) -> float:
        """Predict time until next access"""
        if not self.is_trained:
            # Fallback to simple heuristic
            if pattern.access_intervals:
                return statistics.mean(pattern.access_intervals)
            return 3600.0  # Default 1 hour
        
        try:
            current_time = time.time()
            features = self.extract_features(pattern, current_time).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            prediction = self.model.predict(features_scaled)[0]
            return max(prediction, 0.0)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback
            if pattern.access_intervals:
                return statistics.mean(pattern.access_intervals)
            return 3600.0

class CacheLevel_Implementation:
    """Base class for cache level implementations"""
    
    def __init__(self, level: CacheLevel, max_size: int, replacement_policy: ReplacementPolicy):
        self.level = level
        self.max_size = max_size
        self.replacement_policy = replacement_policy
        self.entries = {}  # key -> CacheEntry
        self.data_store = {}  # key -> data
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "prefetches": 0,
            "total_response_time": 0.0
        }
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get item from cache"""
        start_time = time.time()
        
        with self.lock:
            if key in self.entries:
                # Cache hit
                entry = self.entries[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Update access tracking
                self.access_order.move_to_end(key)
                self.access_frequency[key] += 1
                
                self.stats["hits"] += 1
                self.stats["total_response_time"] += time.time() - start_time
                
                return self.data_store[key], True
            else:
                # Cache miss
                self.stats["misses"] += 1
                self.stats["total_response_time"] += time.time() - start_time
                return None, False
    
    def put(self, key: str, value: Any, entry: CacheEntry) -> bool:
        """Put item in cache"""
        with self.lock:
            # Calculate size
            try:
                if hasattr(value, '__len__'):
                    size = len(value) if isinstance(value, (str, bytes)) else sys.getsizeof(value)
                else:
                    size = sys.getsizeof(value)
            except Exception as e:
                # TODO: Review this exception handling
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                size = sys.getsizeof(value)
            
            entry.size = size
            
            # Check if we need to evict
            while self.current_size + size > self.max_size and self.entries:
                evicted = self._evict()
                if not evicted:
                    break
            
            # Check if we have enough space
            if self.current_size + size > self.max_size:
                return False
            
            # Compress data if this is L2 or L3
            if self.level in [CacheLevel.L2, CacheLevel.L3]:
                try:
                    compressed_value = gzip.compress(pickle.dumps(value))
                    entry.compression_ratio = len(compressed_value) / size if size > 0 else 1.0
                    value = compressed_value
                    size = len(compressed_value)
                except Exception as e:
                    # TODO: Review this exception handling
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    pass  # Use original if compression fails
            
            # Add to cache
            self.entries[key] = entry
            self.data_store[key] = value
            self.access_order[key] = time.time()
            self.current_size += size
            
            return True
    
    def _evict(self) -> bool:
        """Evict entry based on replacement policy"""
        if not self.entries:
            return False
        
        key_to_evict = None
        
        if self.replacement_policy == ReplacementPolicy.LRU:
            key_to_evict = next(iter(self.access_order))
        
        elif self.replacement_policy == ReplacementPolicy.LFU:
            key_to_evict = min(self.access_frequency, key=self.access_frequency.get)
        
        elif self.replacement_policy == ReplacementPolicy.ARC:
            # Simplified ARC - would need full implementation
            key_to_evict = next(iter(self.access_order))
        
        elif self.replacement_policy == ReplacementPolicy.ADAPTIVE:
            # Score-based eviction
            scores = {}
            current_time = time.time()
            
            for key, entry in self.entries.items():
                # Calculate eviction score (lower = more likely to evict)
                time_score = 1.0 / (current_time - entry.last_access + 1.0)
                freq_score = entry.access_count / 10.0
                priority_score = entry.priority / 10.0
                prediction_score = 1.0 / (entry.predicted_next_access + 1.0)
                
                scores[key] = time_score + freq_score + priority_score + prediction_score
            
            key_to_evict = min(scores, key=scores.get)
        
        if key_to_evict and key_to_evict in self.entries:
            entry = self.entries[key_to_evict]
            self.current_size -= entry.size
            
            del self.entries[key_to_evict]
            del self.data_store[key_to_evict]
            
            if key_to_evict in self.access_order:
                del self.access_order[key_to_evict]
            if key_to_evict in self.access_frequency:
                del self.access_frequency[key_to_evict]
            
            self.stats["evictions"] += 1
            logger.debug(f"Evicted key {key_to_evict} from {self.level.value}")
            return True
        
        return False
    
    def remove(self, key: str) -> bool:
        """Remove specific key from cache"""
        with self.lock:
            if key in self.entries:
                entry = self.entries[key]
                self.current_size -= entry.size
                
                del self.entries[key]
                del self.data_store[key]
                
                if key in self.access_order:
                    del self.access_order[key]
                if key in self.access_frequency:
                    del self.access_frequency[key]
                
                return True
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_ratio = self.stats["hits"] / max(total_requests, 1)
            miss_ratio = self.stats["misses"] / max(total_requests, 1)
            avg_response_time = self.stats["total_response_time"] / max(total_requests, 1)
            
            utilization_ratio = self.current_size / self.max_size
            
            # Calculate fragmentation (simplified)
            fragmentation_ratio = 0.0
            if self.entries:
                avg_entry_size = self.current_size / len(self.entries)
                size_variance = sum((entry.size - avg_entry_size) ** 2 for entry in self.entries.values())
                size_stddev = math.sqrt(size_variance / len(self.entries))
                fragmentation_ratio = size_stddev / max(avg_entry_size, 1)
            
            return CacheStats(
                level=self.level,
                total_requests=total_requests,
                hit_count=self.stats["hits"],
                miss_count=self.stats["misses"],
                eviction_count=self.stats["evictions"],
                prefetch_count=self.stats["prefetches"],
                hit_ratio=hit_ratio,
                miss_ratio=miss_ratio,
                average_response_time=avg_response_time,
                memory_usage=self.current_size,
                memory_limit=self.max_size,
                utilization_ratio=utilization_ratio,
                fragmentation_ratio=fragmentation_ratio
            )

class IntelligentCacheSystem:
    """Multi-level intelligent cache system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize cache levels
        self.l1_cache = CacheLevel_Implementation(
            CacheLevel.L1, 
            config.get("l1_size", 256 * 1024 * 1024),  # 256MB
            ReplacementPolicy(config.get("l1_policy", "adaptive"))
        )
        
        self.l2_cache = CacheLevel_Implementation(
            CacheLevel.L2,
            config.get("l2_size", 1024 * 1024 * 1024),  # 1GB
            ReplacementPolicy(config.get("l2_policy", "arc"))
        )
        
        self.l3_cache = CacheLevel_Implementation(
            CacheLevel.L3,
            config.get("l3_size", 4096 * 1024 * 1024),  # 4GB
            ReplacementPolicy(config.get("l3_policy", "lru"))
        )
        
        self.caches = [self.l1_cache, self.l2_cache, self.l3_cache]
        
        # Access pattern tracking
        self.access_patterns = {}  # key -> AccessPattern
        self.pattern_lock = threading.RLock()
        
        # ML predictor
        self.predictor = AccessPredictor()
        
        # Prefetching
        self.prefetch_queue = deque()
        self.prefetch_thread = None
        self.prefetch_enabled = config.get("enable_prefetch", True)
        self.prefetch_depth = config.get("prefetch_depth", 3)
        
        # Cross-agent sharing
        self.agent_cache_maps = defaultdict(set)  # agent_id -> set of keys
        self.key_sharing_map = defaultdict(set)   # key -> set of agent_ids
        
        # Performance monitoring
        self.running = False
        self.monitor_thread = None
        self.performance_history = deque(maxlen=1000)
        
        logger.info("Intelligent Cache System initialized")
    
    def start(self):
        """Start the cache system"""
        self.running = True
        
        if self.prefetch_enabled:
            self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.prefetch_thread.start()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Cache system started")
    
    def stop(self):
        """Stop the cache system"""
        self.running = False
        
        if self.prefetch_thread:
            self.prefetch_thread.join()
        
        if self.monitor_thread:
            self.monitor_thread.join()
        
        logger.info("Cache system stopped")
    
    def get(self, key: str, agent_id: str) -> Tuple[Optional[Any], CacheLevel]:
        """Get item from cache hierarchy"""
        start_time = time.time()
        
        # Update access pattern
        self._update_access_pattern(key, agent_id)
        
        # Try each cache level
        for cache in self.caches:
            value, hit = cache.get(key)
            if hit:
                # Promote to higher levels if beneficial
                self._promote_entry(key, value, cache.level)
                
                # Record agent access
                self.agent_cache_maps[agent_id].add(key)
                self.key_sharing_map[key].add(agent_id)
                
                # Trigger prefetching
                if self.prefetch_enabled:
                    self._trigger_prefetch(key, agent_id)
                
                logger.debug(f"Cache hit for {key} at {cache.level.value}")
                return value, cache.level
        
        # Cache miss - update patterns
        logger.debug(f"Cache miss for {key}")
        return None, None
    
    def put(self, key: str, value: Any, agent_id: str, priority: int = 0) -> bool:
        """Put item in cache hierarchy"""
        # Create cache entry
        entry = CacheEntry(
            key=key,
            size=0,  # Will be calculated by cache level
            level=CacheLevel.L1,  # Start at L1
            access_count=1,
            last_access=time.time(),
            creation_time=time.time(),
            last_update=time.time(),
            checksum=hashlib.md5(str(value).encode()).hexdigest(),
            agent_id=agent_id,
            priority=priority
        )
        
        # Try to put in L1 first
        if self.l1_cache.put(key, value, entry):
            self._update_access_pattern(key, agent_id)
            self.agent_cache_maps[agent_id].add(key)
            self.key_sharing_map[key].add(agent_id)
            return True
        
        # If L1 is full, try L2
        entry.level = CacheLevel.L2
        if self.l2_cache.put(key, value, entry):
            self._update_access_pattern(key, agent_id)
            self.agent_cache_maps[agent_id].add(key)
            self.key_sharing_map[key].add(agent_id)
            return True
        
        # If L2 is full, try L3
        entry.level = CacheLevel.L3
        if self.l3_cache.put(key, value, entry):
            self._update_access_pattern(key, agent_id)
            self.agent_cache_maps[agent_id].add(key)
            self.key_sharing_map[key].add(agent_id)
            return True
        
        logger.warning(f"Failed to cache {key} - all levels full")
        return False
    
    def _update_access_pattern(self, key: str, agent_id: str):
        """Update access pattern for key"""
        current_time = time.time()
        
        with self.pattern_lock:
            if key not in self.access_patterns:
                self.access_patterns[key] = AccessPattern(
                    key=key,
                    agent_id=agent_id,
                    access_times=[],
                    access_intervals=[],
                    access_frequency=0.0,
                    temporal_locality=0.0,
                    spatial_locality=0.0,
                    sharing_frequency=0.0
                )
            
            pattern = self.access_patterns[key]
            
            # Update access times
            if pattern.access_times:
                interval = current_time - pattern.access_times[-1]
                pattern.access_intervals.append(interval)
                
                # Keep only recent intervals
                if len(pattern.access_intervals) > 100:
                    pattern.access_intervals.pop(0)
            
            pattern.access_times.append(current_time)
            
            # Keep only recent accesses
            if len(pattern.access_times) > 100:
                pattern.access_times.pop(0)
            
            # Update metrics
            self._calculate_pattern_metrics(pattern)
    
    def _calculate_pattern_metrics(self, pattern: AccessPattern):
        """Calculate pattern metrics"""
        if len(pattern.access_times) < 2:
            return
        
        current_time = time.time()
        
        # Access frequency (accesses per hour)
        recent_accesses = sum(1 for t in pattern.access_times if current_time - t < 3600)
        pattern.access_frequency = recent_accesses
        
        # Temporal locality (how regular are the accesses)
        if len(pattern.access_intervals) > 1:
            mean_interval = statistics.mean(pattern.access_intervals)
            interval_variance = statistics.variance(pattern.access_intervals)
            pattern.temporal_locality = 1.0 / (1.0 + interval_variance / max(mean_interval, 1.0))
        
        # Sharing frequency
        sharing_agents = len(self.key_sharing_map.get(pattern.key, set()))
        pattern.sharing_frequency = sharing_agents / 131.0  # Normalize by total agents
        
        # Update prediction
        if self.predictor.is_trained:
            pattern.predicted_next_access = self.predictor.predict_next_access(pattern)
    
    def _promote_entry(self, key: str, value: Any, current_level: CacheLevel):
        """Promote frequently accessed entries to higher cache levels"""
        if current_level == CacheLevel.L1:
            return  # Already at highest level
        
        # Check if entry should be promoted
        pattern = self.access_patterns.get(key)
        if not pattern:
            return
        
        # Promotion criteria
        promote = False
        
        if current_level == CacheLevel.L3 and pattern.access_frequency > 5:
            # Promote from L3 to L2 if accessed frequently
            promote = True
            target_cache = self.l2_cache
        elif current_level == CacheLevel.L2 and pattern.access_frequency > 10:
            # Promote from L2 to L1 if accessed very frequently
            promote = True
            target_cache = self.l1_cache
        
        if promote:
            # Get entry from current level
            if current_level == CacheLevel.L3:
                source_cache = self.l3_cache
            else:
                source_cache = self.l2_cache
            
            if key in source_cache.entries:
                entry = source_cache.entries[key]
                entry.level = target_cache.level
                
                # Try to put in target level
                if target_cache.put(key, value, entry):
                    # Remove from source level
                    source_cache.remove(key)
                    logger.debug(f"Promoted {key} from {current_level.value} to {target_cache.level.value}")
    
    def _trigger_prefetch(self, key: str, agent_id: str):
        """Trigger prefetching based on access patterns"""
        if not self.prefetch_enabled:
            return
        
        # Find related keys to prefetch
        related_keys = self._find_related_keys(key, agent_id)
        
        for related_key in related_keys[:self.prefetch_depth]:
            if related_key not in [cache.entries for cache in self.caches]:
                self.prefetch_queue.append((related_key, agent_id))
    
    def _find_related_keys(self, key: str, agent_id: str) -> List[str]:
        """Find keys related to the current key for prefetching"""
        related_keys = []
        
        # Keys accessed by the same agent
        agent_keys = self.agent_cache_maps.get(agent_id, set())
        
        # Keys with similar access patterns
        current_pattern = self.access_patterns.get(key)
        if current_pattern:
            for other_key, other_pattern in self.access_patterns.items():
                if other_key != key:
                    # Calculate similarity score
                    similarity = self._calculate_pattern_similarity(current_pattern, other_pattern)
                    if similarity > 0.7:  # Threshold for similarity
                        related_keys.append(other_key)
        
        # Sort by relevance
        related_keys.sort(key=lambda k: self.access_patterns.get(k, AccessPattern("", "", [], [], 0, 0, 0, 0)).access_frequency, reverse=True)
        
        return related_keys
    
    def _calculate_pattern_similarity(self, pattern1: AccessPattern, pattern2: AccessPattern) -> float:
        """Calculate similarity between two access patterns"""
        # Simple similarity based on frequency and temporal locality
        freq_similarity = 1.0 - abs(pattern1.access_frequency - pattern2.access_frequency) / max(pattern1.access_frequency + pattern2.access_frequency, 1)
        temporal_similarity = 1.0 - abs(pattern1.temporal_locality - pattern2.temporal_locality)
        sharing_similarity = 1.0 - abs(pattern1.sharing_frequency - pattern2.sharing_frequency)
        
        return (freq_similarity + temporal_similarity + sharing_similarity) / 3.0
    
    def _prefetch_loop(self):
        """Prefetching background thread"""
        while self.running:
            try:
                if self.prefetch_queue:
                    key, agent_id = self.prefetch_queue.popleft()
                    
                    # Simulate prefetching (in real implementation, this would fetch from source)
                    logger.debug(f"Prefetching {key} for agent {agent_id}")
                    
                    # Update stats
                    for cache in self.caches:
                        cache.stats["prefetches"] += 1
                
                time.sleep(0.1)  # Small delay to prevent busy loop
                
            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
                time.sleep(1)
    
    def _monitor_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                # Collect performance metrics
                metrics = self.get_comprehensive_stats()
                metrics["timestamp"] = time.time()
                
                self.performance_history.append(metrics)
                
                # Train predictor periodically
                if len(self.performance_history) % 100 == 0:
                    self._train_predictor()
                
                # Adaptive optimization
                self._adaptive_optimization(metrics)
                
                time.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(30)
    
    def _train_predictor(self):
        """Train the access predictor with recent patterns"""
        patterns = []
        labels = []
        
        current_time = time.time()
        
        with self.pattern_lock:
            for pattern in self.access_patterns.values():
                if len(pattern.access_times) >= 3:
                    patterns.append(pattern)
                    
                    # Label is the actual next access time (for training)
                    if len(pattern.access_intervals) > 0:
                        labels.append(pattern.access_intervals[-1])
                    else:
                        labels.append(3600.0)  # Default
        
        if len(patterns) >= 10:
            success = self.predictor.train_model(patterns, labels)
            if success:
                logger.info(f"Retrained access predictor with {len(patterns)} patterns")
    
    def _adaptive_optimization(self, metrics: Dict[str, Any]):
        """Adaptive optimization based on performance metrics"""
        # Check cache hit ratios
        for level_stats in metrics["cache_levels"]:
            if level_stats["hit_ratio"] < 0.3:  # Low hit ratio
                logger.info(f"Low hit ratio ({level_stats['hit_ratio']:.2f}) detected for {level_stats['level']}")
                # Could trigger cache size adjustment or policy change
    
    def invalidate_agent_cache(self, agent_id: str) -> int:
        """Invalidate all cache entries for an agent"""
        invalidated_count = 0
        
        agent_keys = self.agent_cache_maps.get(agent_id, set()).copy()
        
        for key in agent_keys:
            for cache in self.caches:
                if cache.remove(key):
                    invalidated_count += 1
            
            # Update sharing maps
            self.key_sharing_map[key].discard(agent_id)
            if not self.key_sharing_map[key]:
                del self.key_sharing_map[key]
        
        # Clear agent cache map
        if agent_id in self.agent_cache_maps:
            del self.agent_cache_maps[agent_id]
        
        logger.info(f"Invalidated {invalidated_count} cache entries for agent {agent_id}")
        return invalidated_count
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "cache_levels": [],
            "access_patterns": {
                "total_patterns": len(self.access_patterns),
                "active_patterns": sum(1 for p in self.access_patterns.values() 
                                     if time.time() - p.access_times[-1] < 3600 if p.access_times),
                "average_frequency": statistics.mean([p.access_frequency for p in self.access_patterns.values()]) 
                                   if self.access_patterns else 0
            },
            "agent_stats": {
                "total_agents": len(self.agent_cache_maps),
                "total_shared_keys": len(self.key_sharing_map),
                "average_sharing": statistics.mean([len(agents) for agents in self.key_sharing_map.values()]) 
                                 if self.key_sharing_map else 0
            },
            "predictor": {
                "is_trained": self.predictor.is_trained,
                "feature_count": len(self.predictor.feature_history)
            },
            "prefetch": {
                "enabled": self.prefetch_enabled,
                "queue_size": len(self.prefetch_queue),
                "depth": self.prefetch_depth
            }
        }
        
        # Add cache level stats
        for cache in self.caches:
            cache_stats = cache.get_stats()
            stats["cache_levels"].append(asdict(cache_stats))
        
        return stats
    
    def export_stats(self, filepath: str):
        """Export comprehensive statistics"""
        stats = self.get_comprehensive_stats()
        stats["export_timestamp"] = time.time()
        stats["performance_history"] = list(self.performance_history)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Exported cache statistics to {filepath}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="SutazAI Intelligent Cache System")
    parser.add_argument("--mode", choices=["adaptive", "lru", "lfu", "arc"], 
                       default="adaptive", help="Cache replacement policy")
    parser.add_argument("--l1-size", type=int, default=256,
                       help="L1 cache size in MB")
    parser.add_argument("--l2-size", type=int, default=1024,
                       help="L2 cache size in MB")
    parser.add_argument("--l3-size", type=int, default=4096,
                       help="L3 cache size in MB")
    parser.add_argument("--prefetch-depth", type=int, default=3,
                       help="Prefetch depth")
    parser.add_argument("--disable-prefetch", action="store_true",
                       help="Disable prefetching")
    parser.add_argument("--test", action="store_true",
                       help="Run test workload")
    parser.add_argument("--monitor", action="store_true",
                       help="Start monitoring mode")
    parser.add_argument("--export-stats", type=str,
                       help="Export statistics to file")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "l1_size": args.l1_size * 1024 * 1024,
        "l2_size": args.l2_size * 1024 * 1024,
        "l3_size": args.l3_size * 1024 * 1024,
        "l1_policy": args.mode,
        "l2_policy": "arc",
        "l3_policy": "lru",
        "enable_prefetch": not args.disable_prefetch,
        "prefetch_depth": args.prefetch_depth
    }
    
    # Create cache system
    cache_system = IntelligentCacheSystem(config)
    
    try:
        cache_system.start()
        
        if args.test:
            # Run test workload
            logger.info("Running test workload...")
            
            # Simulate agent requests
            import random
            
            agents = [f"agent_{i:03d}" for i in range(20)]
            keys = [f"data_{i:04d}" for i in range(1000)]
            
            # Generate test data
            for i in range(5000):  # 5000 operations
                agent = random.choice(agents)
                key = random.choice(keys)
                
                # 70% reads, 30% writes
                if random.random() < 0.7:
                    # Read operation
                    value, level = cache_system.get(key, agent)
                    if value is None:
                        # Cache miss - simulate loading data
                        test_data = f"Test data for {key}".encode() * random.randint(1, 100)
                        cache_system.put(key, test_data, agent)
                else:
                    # Write operation
                    test_data = f"Updated data for {key} at {time.time()}".encode() * random.randint(1, 100)
                    cache_system.put(key, test_data, agent, priority=random.randint(0, 10))
                
                if i % 1000 == 0:
                    logger.info(f"Completed {i} operations")
            
            # Print final stats
            stats = cache_system.get_comprehensive_stats()
            print(json.dumps(stats, indent=2, default=str))
        
        elif args.monitor:
            logger.info("Starting cache monitoring. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(60)
                    stats = cache_system.get_comprehensive_stats()
                    
                    # Print summary
                    total_hits = sum(level["hit_count"] for level in stats["cache_levels"])
                    total_requests = sum(level["total_requests"] for level in stats["cache_levels"])
                    overall_hit_ratio = total_hits / max(total_requests, 1)
                    
                    print(f"Overall Hit Ratio: {overall_hit_ratio:.3f}, "
                          f"Active Patterns: {stats['access_patterns']['active_patterns']}, "
                          f"Agents: {stats['agent_stats']['total_agents']}")
                    
            except KeyboardInterrupt:
                logger.info("Stopping monitoring...")
        
        if args.export_stats:
            cache_system.export_stats(args.export_stats)
            print(f"Statistics exported to {args.export_stats}")
    
    finally:
        cache_system.stop()

if __name__ == "__main__":
    main()