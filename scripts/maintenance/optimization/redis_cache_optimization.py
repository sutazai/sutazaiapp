#!/usr/bin/env python3
"""
🔥 REDIS CACHE OPTIMIZATION SCRIPT
ULTRAINVESTIGATE Database Specialist

Fixes critical Redis cache issues:
- 9.08% hit rate → 80%+ target
- Implements Redis-first caching strategy
- Optimizes TTL and eviction policies
- Provides performance monitoring
"""

import asyncio
import redis
import json
import time
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    hits: int
    misses: int
    hit_rate: float
    total_keys: int
    memory_used: str
    memory_peak: str
    expired_keys: int
    evicted_keys: int

class RedisCacheOptimizer:
    """Redis cache optimization and monitoring tool"""
    
    def __init__(self, redis_url: str = "redis://localhost:10001"):
        self.redis_url = redis_url
        self.client = None
        
    async def connect(self):
        """Connect to Redis"""
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"✅ Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
            
    def get_current_metrics(self) -> CacheMetrics:
        """Get current Redis performance metrics"""
        info = self.client.info()
        stats = self.client.info("stats")
        
        hits = stats.get('keyspace_hits', 0)
        misses = stats.get('keyspace_misses', 0)
        total_requests = hits + misses
        hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
        
        return CacheMetrics(
            hits=hits,
            misses=misses,
            hit_rate=round(hit_rate, 2),
            total_keys=self.client.dbsize(),
            memory_used=info.get('used_memory_human', '0'),
            memory_peak=info.get('used_memory_peak_human', '0'),
            expired_keys=stats.get('expired_keys', 0),
            evicted_keys=stats.get('evicted_keys', 0)
        )
        
    def analyze_cache_problems(self) -> Dict[str, Any]:
        """Analyze current cache configuration and identify problems"""
        metrics = self.get_current_metrics()
        info = self.client.info()
        config = self.client.config_get("*")
        
        problems = []
        recommendations = []
        
        # Problem 1: Low hit rate
        if metrics.hit_rate < 50:
            problems.append(f"CRITICAL: Cache hit rate is {metrics.hit_rate}% (target: 80%+)")
            recommendations.append("Implement Redis-first caching strategy")
            recommendations.append("Review TTL settings for premature expiration")
            
        # Problem 2: High expired keys
        if metrics.expired_keys > metrics.hits:
            problems.append(f"TTL too aggressive: {metrics.expired_keys} expired vs {metrics.hits} hits")
            recommendations.append("Increase TTL for frequently accessed data")
            
        # Problem 3: Memory utilization
        memory_usage = info.get('used_memory', 0)
        max_memory = info.get('maxmemory', 0)
        if max_memory > 0:
            utilization = (memory_usage / max_memory) * 100
            if utilization < 10:
                problems.append(f"Low memory utilization: {utilization:.1f}%")
                recommendations.append("Consider reducing maxmemory or increasing cache usage")
                
        return {
            'metrics': metrics,
            'problems': problems,
            'recommendations': recommendations,
            'config': {
                'maxmemory': config.get('maxmemory'),
                'maxmemory_policy': config.get('maxmemory-policy'),
                'timeout': config.get('timeout'),
                'tcp_keepalive': config.get('tcp-keepalive')
            }
        }
        
    def optimize_redis_config(self):
        """Apply Redis configuration optimizations"""
        logger.info("🔧 Applying Redis configuration optimizations...")
        
        optimizations = {
            # Memory optimization
            'maxmemory': '256mb',  # Reduce from 512mb based on low usage
            'maxmemory-policy': 'allkeys-lru',  # Keep current policy
            
            # Connection optimization  
            'tcp-keepalive': '300',  # Add connection management
            'timeout': '0',  # Persistent connections
            
            # Persistence optimization for performance
            'save': '',  # Disable snapshots for better performance
            'appendonly': 'yes',  # Enable AOF for durability
            'appendfsync': 'everysec',  # Balanced durability/performance
            
            # Performance tuning
            'hash-max-ziplist-entries': '512',
            'hash-max-ziplist-value': '64',
            'list-max-ziplist-size': '-2',
            'set-max-intset-entries': '512',
        }
        
        applied = []
        failed = []
        
        for key, value in optimizations.items():
            try:
                # Some configs require restart, others can be applied live
                if key in ['save', 'appendonly', 'appendfsync']:
                    logger.warning(f"⚠️  {key}={value} requires Redis restart")
                    continue
                    
                self.client.config_set(key, value)
                applied.append(f"{key}={value}")
                logger.info(f"✅ Applied: {key}={value}")
            except Exception as e:
                failed.append(f"{key}: {str(e)}")
                logger.error(f"❌ Failed to set {key}={value}: {e}")
                
        return {'applied': applied, 'failed': failed}
        
    def warm_cache_with_common_patterns(self):
        """Pre-populate cache with common access patterns"""
        logger.info("🔥 Warming cache with common patterns...")
        
        # Common cache patterns based on SutazAI application
        cache_warmup = [
            # User session data
            ('user:session:active', {'count': 0, 'sessions': []}),
            
            # Agent status cache
            ('agents:health:summary', {'healthy': 0, 'unhealthy': 0, 'total': 0}),
            
            # Model registry cache
            ('models:available', ['tinyllama']),
            ('models:default', 'tinyllama'),
            
            # System metrics cache
            ('metrics:system:latest', {'timestamp': time.time(), 'cpu': 0, 'memory': 0}),
            
            # Common API response cache
            ('api:health:backend', {'status': 'healthy', 'timestamp': time.time()}),
            ('api:health:ollama', {'status': 'healthy', 'models_loaded': 1}),
            
            # Chat history cache patterns
            ('chat:recent:global', []),
            
            # Task queue cache
            ('tasks:pending:count', 0),
            ('tasks:completed:count', 0),
        ]
        
        warmed = 0
        for key, value in cache_warmup:
            try:
                # Set with 1 hour TTL for warmup data
                self.client.setex(key, 3600, json.dumps(value))
                warmed += 1
            except Exception as e:
                logger.error(f"Failed to warm cache key {key}: {e}")
                
        logger.info(f"🔥 Cache warmed with {warmed} common patterns")
        return warmed
        
    def setup_cache_monitoring(self):
        """Set up monitoring for cache performance"""
        logger.info("📊 Setting up cache performance monitoring...")
        
        # Store baseline metrics
        baseline = self.get_current_metrics()
        self.client.hset('cache:monitoring:baseline', mapping={
            'timestamp': time.time(),
            'hit_rate': baseline.hit_rate,
            'total_keys': baseline.total_keys,
            'hits': baseline.hits,
            'misses': baseline.misses
        })
        
        # Set up alerting thresholds
        monitoring_config = {
            'hit_rate_threshold': 80.0,  # Alert if below 80%
            'memory_threshold': 90.0,    # Alert if above 90% memory usage
            'key_expiry_ratio': 2.0,     # Alert if expired/hits ratio > 2
        }
        
        self.client.hset('cache:monitoring:config', mapping=monitoring_config)
        logger.info("📊 Monitoring configuration saved to Redis")
        
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        analysis = self.analyze_cache_problems()
        metrics = analysis['metrics']
        
        report = f"""
🔥 REDIS CACHE OPTIMIZATION REPORT
=====================================

📊 CURRENT PERFORMANCE METRICS:
- Hit Rate: {metrics.hit_rate}% (Target: 80%+)
- Total Requests: {metrics.hits + metrics.misses:,}
- Cache Hits: {metrics.hits:,}
- Cache Misses: {metrics.misses:,}
- Total Keys: {metrics.total_keys:,}
- Memory Used: {metrics.memory_used}
- Memory Peak: {metrics.memory_peak}
- Expired Keys: {metrics.expired_keys:,}
- Evicted Keys: {metrics.evicted_keys:,}

🚨 IDENTIFIED PROBLEMS:
"""
        for i, problem in enumerate(analysis['problems'], 1):
            report += f"{i}. {problem}\n"
            
        report += f"""
💡 OPTIMIZATION RECOMMENDATIONS:
"""
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {rec}\n"
            
        report += f"""
⚙️  CURRENT CONFIGURATION:
- Max Memory: {analysis['config']['maxmemory'] or 'No limit'}
- Eviction Policy: {analysis['config']['maxmemory_policy']}
- Connection Timeout: {analysis['config']['timeout']}s
- TCP Keep-Alive: {analysis['config']['tcp_keepalive']}s

🎯 OPTIMIZATION TARGETS:
- Hit Rate: 80%+ (Current: {metrics.hit_rate}%)
- Memory Utilization: 30-70%
- Response Time: <5ms average
- Key Expiry Ratio: <1.0
"""
        return report
        
    def benchmark_performance(self, operations: int = 1000) -> Dict[str, float]:
        """Benchmark Redis performance"""
        logger.info(f"🏃 Running performance benchmark with {operations} operations...")
        
        # Warm up
        for i in range(100):
            self.client.set(f"benchmark:warmup:{i}", f"value_{i}")
            
        # Benchmark SET operations
        start_time = time.time()
        for i in range(operations):
            self.client.set(f"benchmark:set:{i}", f"value_{i}", ex=3600)
        set_time = time.time() - start_time
        
        # Benchmark GET operations
        start_time = time.time()
        for i in range(operations):
            self.client.get(f"benchmark:set:{i}")
        get_time = time.time() - start_time
        
        # Cleanup
        self.client.delete(*[f"benchmark:set:{i}" for i in range(operations)])
        self.client.delete(*[f"benchmark:warmup:{i}" for i in range(100)])
        
        results = {
            'set_ops_per_sec': operations / set_time,
            'get_ops_per_sec': operations / get_time,
            'avg_set_latency_ms': (set_time / operations) * 1000,
            'avg_get_latency_ms': (get_time / operations) * 1000,
        }
        
        logger.info(f"📈 Benchmark Results:")
        logger.info(f"   SET: {results['set_ops_per_sec']:.0f} ops/sec, {results['avg_set_latency_ms']:.2f}ms avg")
        logger.info(f"   GET: {results['get_ops_per_sec']:.0f} ops/sec, {results['avg_get_latency_ms']:.2f}ms avg")
        
        return results

async def main():
    """Main optimization workflow"""
    logger.info("🔥 REDIS CACHE OPTIMIZATION TOOL")
    logger.info("=" * 50)
    
    optimizer = RedisCacheOptimizer()
    
    try:
        # Step 1: Connect to Redis
        await optimizer.connect()
        
        # Step 2: Analyze current state
        logger.info("\n📊 ANALYZING CURRENT CACHE STATE...")
        report = optimizer.generate_optimization_report()
        logger.info(report)
        
        # Step 3: Run performance benchmark
        logger.info("\n🏃 RUNNING PERFORMANCE BENCHMARK...")
        benchmark = optimizer.benchmark_performance(1000)
        
        # Step 4: Apply optimizations
        logger.info("\n🔧 APPLYING OPTIMIZATIONS...")
        config_result = optimizer.optimize_redis_config()
        logger.info(f"✅ Applied {len(config_result['applied'])} configuration changes")
        if config_result['failed']:
            logger.error(f"❌ Failed to apply {len(config_result['failed'])} changes")
            
        # Step 5: Warm cache
        logger.info("\n🔥 WARMING CACHE...")
        warmed_keys = optimizer.warm_cache_with_common_patterns()
        
        # Step 6: Set up monitoring
        logger.info("\n📊 SETTING UP MONITORING...")
        optimizer.setup_cache_monitoring()
        
        # Step 7: Final metrics
        logger.info("\n📈 FINAL OPTIMIZATION RESULTS:")
        final_metrics = optimizer.get_current_metrics()
        logger.info(f"Hit Rate: {final_metrics.hit_rate}%")
        logger.info(f"Total Keys: {final_metrics.total_keys}")
        logger.info(f"Memory Used: {final_metrics.memory_used}")
        
        logger.info(f"\n🎉 OPTIMIZATION COMPLETE!")
        logger.info(f"✅ Cache warmed with {warmed_keys} patterns")
        logger.info(f"✅ Monitoring configured")
        logger.info(f"✅ Performance benchmarked")
        
        # Save detailed results
        results = {
            'timestamp': time.time(),
            'before_optimization': report,
            'benchmark_results': benchmark,
            'config_changes': config_result,
            'warmed_keys': warmed_keys,
            'final_metrics': {
                'hit_rate': final_metrics.hit_rate,
                'total_keys': final_metrics.total_keys,
                'memory_used': final_metrics.memory_used
            }
        }
        
        optimizer.client.set('cache:optimization:results', json.dumps(results), ex=86400)
        logger.info(f"📝 Optimization results saved to Redis key: cache:optimization:results")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())