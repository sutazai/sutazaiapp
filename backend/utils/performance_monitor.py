"""
ULTRA-PERFORMANCE MONITORING SUITE
Created by: PERF-MASTER-001 with ULTRA-THINKING
Purpose: Comprehensive performance monitoring with real-time insights
"""

import time
import asyncio
import logging
import psutil
import redis
import asyncpg
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from functools import wraps
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data class"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class UltraPerformanceMonitor:
    """
    ULTRA-PERFORMANCE monitoring system
    Tracks: Response times, Cache performance, Database queries, Resource usage
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[PerformanceMetric]] = {}
        self.thresholds = {
            'api_response_ms': 100,
            'db_query_ms': 50,
            'cache_hit_rate': 0.85,
            'redis_ops_per_sec': 1000,
            'cpu_percent': 80,
            'memory_percent': 85
        }
        self.alerts: List[Dict] = []
        self._redis_client = None
        self._db_pool = None
        
    async def initialize(self, redis_host='redis', db_config=None):
        """Initialize monitoring connections"""
        try:
            # Redis connection for metrics storage
            self._redis_client = redis.Redis(
                host=redis_host,
                port=6379,
                decode_responses=True,
                socket_keepalive=True
            )
            
            # Database connection for query monitoring
            if db_config:
                self._db_pool = await asyncpg.create_pool(
                    **db_config,
                    min_size=2,
                    max_size=5
                )
                
            logger.info("Ultra Performance Monitor initialized")
            
        except Exception as e:
            logger.error(f"Monitor initialization failed: {e}")
            
    def measure_time(self, metric_name: str, tags: Dict[str, str] = None):
        """Decorator to measure function execution time"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    
                    # Record metric
                    metric = PerformanceMetric(
                        name=f"{metric_name}_time",
                        value=elapsed_ms,
                        unit="ms",
                        timestamp=datetime.now(),
                        tags=tags or {}
                    )
                    self.record_metric(metric)
                    
                    # Check threshold
                    self._check_threshold(metric_name, elapsed_ms)
                    
                    # Log if slow
                    if elapsed_ms > self.thresholds.get(f'{metric_name}_ms', 1000):
                        logger.warning(f"Slow {metric_name}: {elapsed_ms:.2f}ms")
                        
                    return result
                    
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.error(f"Error in {metric_name} after {elapsed_ms:.2f}ms: {e}")
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    
                    metric = PerformanceMetric(
                        name=f"{metric_name}_time",
                        value=elapsed_ms,
                        unit="ms",
                        timestamp=datetime.now(),
                        tags=tags or {}
                    )
                    self.record_metric(metric)
                    
                    return result
                    
                except Exception as e:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.error(f"Error in {metric_name} after {elapsed_ms:.2f}ms: {e}")
                    raise
                    
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
            
        return decorator
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            
        # Keep last 1000 metrics per name
        self.metrics[metric.name].append(metric)
        if len(self.metrics[metric.name]) > 1000:
            self.metrics[metric.name] = self.metrics[metric.name][-1000:]
            
        # Store in Redis for persistence
        if self._redis_client:
            try:
                key = f"perf:{metric.name}:{metric.timestamp.timestamp()}"
                self._redis_client.setex(
                    key,
                    3600,  # 1 hour TTL
                    json.dumps(metric.to_dict())
                )
            except Exception as e:
                logger.debug(f"Failed to store metric in Redis: {e}")
                
    def _check_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds threshold"""
        threshold_key = f"{metric_name}_ms"
        if threshold_key in self.thresholds:
            if value > self.thresholds[threshold_key]:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'metric': metric_name,
                    'value': value,
                    'threshold': self.thresholds[threshold_key],
                    'severity': 'warning' if value < self.thresholds[threshold_key] * 2 else 'critical'
                }
                self.alerts.append(alert)
                logger.warning(f"Performance threshold exceeded: {alert}")
                
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache performance statistics"""
        if not self._redis_client:
            return {}
            
        try:
            info = self._redis_client.info('stats')
            
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses
            
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            stats = {
                'hit_rate': round(hit_rate, 2),
                'hits': hits,
                'misses': misses,
                'total_requests': total,
                'ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'performance_grade': self._grade_performance(hit_rate)
            }
            
            # Record as metric
            self.record_metric(PerformanceMetric(
                name="cache_hit_rate",
                value=hit_rate,
                unit="percent",
                timestamp=datetime.now()
            ))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
            
    def _grade_performance(self, hit_rate: float) -> str:
        """Grade cache performance"""
        if hit_rate >= 95:
            return "Excellent"
        elif hit_rate >= 85:
            return "Good"
        elif hit_rate >= 70:
            return "Fair"
        elif hit_rate >= 50:
            return "Poor"
        else:
            return "Critical"
            
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL performance statistics"""
        if not self._db_pool:
            return {}
            
        try:
            async with self._db_pool.acquire() as conn:
                # Get database stats
                stats = await conn.fetchrow("""
                    SELECT 
                        numbackends as connections,
                        xact_commit as commits,
                        xact_rollback as rollbacks,
                        blks_hit as cache_hits,
                        blks_read as disk_reads,
                        tup_returned as rows_returned,
                        tup_fetched as rows_fetched
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
                
                cache_ratio = 0
                if stats['cache_hits'] + stats['disk_reads'] > 0:
                    cache_ratio = stats['cache_hits'] / (stats['cache_hits'] + stats['disk_reads']) * 100
                    
                return {
                    'connections': stats['connections'],
                    'commits': stats['commits'],
                    'rollbacks': stats['rollbacks'],
                    'cache_hit_ratio': round(cache_ratio, 2),
                    'rows_returned': stats['rows_returned'],
                    'rows_fetched': stats['rows_fetched']
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_percent': disk.percent,
                'disk_free_gb': round(disk.free / (1024**3), 2)
            }
            
            # Check thresholds
            if cpu_percent > self.thresholds['cpu_percent']:
                logger.warning(f"High CPU usage: {cpu_percent}%")
                
            if memory.percent > self.thresholds['memory_percent']:
                logger.warning(f"High memory usage: {memory.percent}%")
                
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'health_score': 100
        }
        
        # Calculate average response times
        for metric_name, metrics in self.metrics.items():
            if metrics:
                recent = metrics[-100:]  # Last 100 measurements
                values = [m.value for m in recent]
                summary['metrics'][metric_name] = {
                    'avg': round(sum(values) / len(values), 2),
                    'min': round(min(values), 2),
                    'max': round(max(values), 2),
                    'count': len(values),
                    'unit': recent[0].unit
                }
                
        # Calculate health score
        deductions = len(self.alerts) * 5
        summary['health_score'] = max(0, 100 - deductions)
        
        return summary
        
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test Redis performance
        if self._redis_client:
            start = time.perf_counter()
            for i in range(100):
                self._redis_client.set(f"perf_test_{i}", f"value_{i}", ex=60)
                self._redis_client.get(f"perf_test_{i}")
            redis_time = (time.perf_counter() - start) * 1000
            results['tests']['redis_100_ops'] = f"{redis_time:.2f}ms"
            
        # Test database performance
        if self._db_pool:
            start = time.perf_counter()
            async with self._db_pool.acquire() as conn:
                await conn.fetch("SELECT 1")
            db_time = (time.perf_counter() - start) * 1000
            results['tests']['db_connection'] = f"{db_time:.2f}ms"
            
        # Get current stats
        results['cache_stats'] = await self.get_cache_stats()
        results['db_stats'] = await self.get_database_stats()
        results['system_stats'] = self.get_system_stats()
        
        return results
        
    def clear_metrics(self):
        """Clear all recorded metrics"""
        self.metrics.clear()
        self.alerts.clear()
        logger.info("Performance metrics cleared")


# Global monitor instance
_monitor: Optional[UltraPerformanceMonitor] = None


async def get_performance_monitor() -> UltraPerformanceMonitor:
    """Get or create the global performance monitor"""
    global _monitor
    
    if _monitor is None:
        _monitor = UltraPerformanceMonitor()
        await _monitor.initialize()
        
    return _monitor


# Convenience decorators
def monitor_api_performance(endpoint: str):
    """Monitor API endpoint performance"""
    monitor = asyncio.get_event_loop().run_until_complete(get_performance_monitor())
    return monitor.measure_time(f"api_{endpoint}", tags={'type': 'api'})


def monitor_db_query(query_name: str):
    """Monitor database query performance"""
    monitor = asyncio.get_event_loop().run_until_complete(get_performance_monitor())
    return monitor.measure_time(f"db_{query_name}", tags={'type': 'database'})


def monitor_cache_operation(operation: str):
    """Monitor cache operation performance"""
    monitor = asyncio.get_event_loop().run_until_complete(get_performance_monitor())
    return monitor.measure_time(f"cache_{operation}", tags={'type': 'cache'})