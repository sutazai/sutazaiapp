#!/usr/bin/env python3
"""
Database Connection Pool Optimizer
================================

Optimizes database connection settings for peak performance:
- PostgreSQL connection pool tuning
- Redis connection optimization
- Connection pool monitoring and alerting
- Automatic connection pool scaling
- Performance benchmark testing
"""

import asyncio
import asyncpg
import redis.asyncio as redis
import psutil
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics"""
    database: str
    pool_size: int
    active_connections: int
    idle_connections: int
    max_connections: int
    utilization_percent: float
    avg_response_time_ms: float
    errors_per_minute: float
    queue_size: int
    pool_exhaustion_count: int


@dataclass
class OptimizationRecommendation:
    """Database optimization recommendation"""
    database: str
    current_setting: str
    recommended_setting: str
    reason: str
    impact_level: str  # high, medium, low


class DatabaseConnectionOptimizer:
    """Database connection pool optimizer for operational excellence"""
    
    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # System resources
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Performance baselines
        self.target_response_time_ms = 50  # Target response time
        self.target_pool_utilization = 0.7  # 70% utilization target
        
    async def initialize_connections(self):
        """Initialize database connections for testing"""
        try:
            # PostgreSQL connection with current settings
            pg_user = os.getenv("POSTGRES_USER", "sutazai")
            pg_pass = os.getenv("POSTGRES_PASSWORD", "sutazai")
            pg_host = os.getenv("POSTGRES_HOST", "localhost")
            pg_port = os.getenv("POSTGRES_PORT", "10000")
            pg_db = os.getenv("POSTGRES_DB", "sutazai")
            pg_dsn = os.getenv(
                'DATABASE_URL',
                f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'
            )
            
            self.pg_pool = await asyncpg.create_pool(
                pg_dsn,
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={'jit': 'off'}
            )
            
            # Redis connection
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:10001/0')
            self.redis_client = redis.from_url(
                redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                connection_pool_kwargs={'max_connections': 20}
            )
            
            logger.info("Database connections initialized for optimization testing")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def benchmark_connection_pool(self, pool_size: int, concurrent_requests: int) -> Dict[str, float]:
        """Benchmark connection pool performance"""
        logger.info(f"Benchmarking pool_size={pool_size}, concurrent={concurrent_requests}")
        
        # Create test pool
        pg_user = os.getenv("POSTGRES_USER", "sutazai")
        pg_pass = os.getenv("POSTGRES_PASSWORD", "sutazai")
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "10000")
        pg_db = os.getenv("POSTGRES_DB", "sutazai")
        pg_dsn = os.getenv(
            'DATABASE_URL',
            f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'
        )
        
        test_pool = await asyncpg.create_pool(
            pg_dsn,
            min_size=min(5, pool_size),
            max_size=pool_size,
            command_timeout=10,
            server_settings={'jit': 'off'}
        )
        
        try:
            # Run concurrent queries
            async def test_query():
                start_time = asyncio.get_event_loop().time()
                try:
                    async with test_pool.acquire() as conn:
                        await conn.fetchval('SELECT count(*) FROM pg_stat_activity')
                    end_time = asyncio.get_event_loop().time()
                    return (end_time - start_time) * 1000, None  # Response time in ms, no error
                except Exception as e:
                    end_time = asyncio.get_event_loop().time()
                    return (end_time - start_time) * 1000, str(e)
            
            start_time = asyncio.get_event_loop().time()
            
            # Run concurrent queries
            tasks = [test_query() for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = asyncio.get_event_loop().time()
            
            # Analyze results
            response_times = []
            errors = 0
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    continue
                
                response_time, error = result
                if error:
                    errors += 1
                else:
                    response_times.append(response_time)
            
            total_time = (end_time - start_time) * 1000
            
            metrics = {
                'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
                'max_response_time_ms': max(response_times) if response_times else 0,
                'min_response_time_ms': min(response_times) if response_times else 0,
                'total_time_ms': total_time,
                'requests_per_second': concurrent_requests / (total_time / 1000) if total_time > 0 else 0,
                'error_rate': errors / concurrent_requests,
                'successful_requests': len(response_times),
                'failed_requests': errors
            }
            
            return metrics
            
        finally:
            await test_pool.close()
    
    async def get_current_pool_metrics(self) -> ConnectionPoolMetrics:
        """Get current PostgreSQL connection pool metrics"""
        async with self.pg_pool.acquire() as conn:
            # Get connection statistics
            stats = await conn.fetchrow("""
                SELECT 
                    (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections,
                    (SELECT count(*) FROM pg_stat_activity) as total_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections
            """)
            
            # Estimate pool metrics (simplified)
            pool_size = self.pg_pool._maxsize
            active_connections = stats['active_connections']
            idle_connections = stats['idle_connections'] 
            
            utilization = active_connections / pool_size if pool_size > 0 else 0
            
            return ConnectionPoolMetrics(
                database='postgresql',
                pool_size=pool_size,
                active_connections=active_connections,
                idle_connections=idle_connections,
                max_connections=stats['max_connections'],
                utilization_percent=utilization * 100,
                avg_response_time_ms=0,  # Would need historical data
                errors_per_minute=0,     # Would need monitoring data
                queue_size=0,           # Not easily available
                pool_exhaustion_count=0  # Would need monitoring data
            )
    
    def calculate_optimal_pool_size(self) -> Tuple[int, int]:
        """Calculate optimal connection pool sizes based on system resources"""
        
        # PostgreSQL pool sizing formula
        # Based on: CPU cores, available memory, and expected concurrency
        
        # Conservative: 2 connections per CPU core + buffer for background tasks
        base_pg_pool_size = self.cpu_count * 2
        
        # Adjust based on available memory
        # Assume ~10MB per connection (conservative estimate)
        memory_limited_connections = int(self.memory_gb * 1024 / 10)  # MB to connections
        
        # Take the smaller of CPU-based and memory-based limits
        optimal_pg_pool_size = min(base_pg_pool_size, memory_limited_connections)
        
        # Ensure minimum viable pool size
        optimal_pg_pool_size = max(optimal_pg_pool_size, 10)
        
        # Cap at reasonable maximum for single-instance deployment
        optimal_pg_pool_size = min(optimal_pg_pool_size, 100)
        
        # Redis pool sizing (generally can be higher since Redis is lightweight)
        optimal_redis_pool_size = optimal_pg_pool_size * 2
        optimal_redis_pool_size = max(optimal_redis_pool_size, 20)
        optimal_redis_pool_size = min(optimal_redis_pool_size, 200)
        
        return optimal_pg_pool_size, optimal_redis_pool_size
    
    async def run_optimization_analysis(self) -> List[OptimizationRecommendation]:
        """Analyze current setup and generate optimization recommendations"""
        recommendations = []
        
        # Get current metrics
        current_metrics = await self.get_current_pool_metrics()
        optimal_pg_size, optimal_redis_size = self.calculate_optimal_pool_size()
        
        # Connection pool size recommendations
        if current_metrics.pool_size < optimal_pg_size:
            recommendations.append(OptimizationRecommendation(
                database='postgresql',
                current_setting=f'pool_size={current_metrics.pool_size}',
                recommended_setting=f'pool_size={optimal_pg_size}',
                reason=f'Current pool size is suboptimal for {self.cpu_count} CPU cores and {self.memory_gb:.1f}GB RAM',
                impact_level='high',
            ))
        
        # Utilization recommendations
        if current_metrics.utilization_percent > 80:
            recommendations.append(OptimizationRecommendation(
                database='postgresql',
                current_setting=f'utilization={current_metrics.utilization_percent:.1f}%',
                recommended_setting=f'increase_pool_size_to={int(current_metrics.pool_size * 1.5)}',
                reason='High connection pool utilization may cause queuing and performance degradation',
                impact_level='high',
            ))
        
        # PostgreSQL configuration recommendations
        async with self.pg_pool.acquire() as conn:
            # Check shared_buffers setting
            shared_buffers = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'shared_buffers'")
            shared_buffers_mb = int(shared_buffers) * 8 / 1024  # Convert from 8KB blocks to MB
            
            optimal_shared_buffers_mb = self.memory_gb * 1024 * 0.25  # 25% of RAM
            
            if shared_buffers_mb < optimal_shared_buffers_mb * 0.8:  # Less than 80% of optimal
                recommendations.append(OptimizationRecommendation(
                    database='postgresql',
                    current_setting=f'shared_buffers={shared_buffers_mb:.0f}MB',
                    recommended_setting=f'shared_buffers={optimal_shared_buffers_mb:.0f}MB',
                    reason='Increase shared_buffers to improve cache hit ratio',
                    impact_level='high',
                ))
            
            # Check work_mem setting
            work_mem = await conn.fetchval("SELECT setting FROM pg_settings WHERE name = 'work_mem'")
            work_mem_mb = int(work_mem) / 1024  # Convert from KB to MB
            
            # Optimal work_mem: available_memory / max_connections / 4
            optimal_work_mem_mb = (self.memory_gb * 1024) / current_metrics.max_connections / 4
            
            if work_mem_mb < optimal_work_mem_mb * 0.5:  # Less than 50% of optimal
                recommendations.append(OptimizationRecommendation(
                    database='postgresql',
                    current_setting=f'work_mem={work_mem_mb:.0f}MB',
                    recommended_setting=f'work_mem={optimal_work_mem_mb:.0f}MB',
                    reason='Increase work_mem to improve sort and hash operation performance',
                    impact_level='medium',
                ))
        
        # Performance benchmark recommendations
        logger.info("Running performance benchmarks...")
        
        # Test current pool size
        current_benchmark = await self.benchmark_connection_pool(
            current_metrics.pool_size, 
            current_metrics.pool_size * 2
        )
        
        if current_benchmark['avg_response_time_ms'] > self.target_response_time_ms:
            recommendations.append(OptimizationRecommendation(
                database='postgresql',
                current_setting=f'avg_response_time={current_benchmark["avg_response_time_ms"]:.1f}ms',
                recommended_setting=f'target_response_time={self.target_response_time_ms}ms',
                reason='Current response time exceeds target performance',
                impact_level='high',
            ))
        
        if current_benchmark['error_rate'] > 0.01:  # More than 1% errors
            recommendations.append(OptimizationRecommendation(
                database='postgresql',
                current_setting=f'error_rate={current_benchmark["error_rate"]:.2%}',
                recommended_setting='error_rate<1%',
                reason='High error rate indicates connection issues or resource exhaustion',
                impact_level='critical',
            ))
        
        return recommendations
    
    def generate_configuration_files(self, recommendations: List[OptimizationRecommendation]):
        """Generate optimized configuration files"""
        
        # Generate PostgreSQL configuration
        optimal_pg_size, optimal_redis_size = self.calculate_optimal_pool_size()
        
        # PostgreSQL optimization settings
        postgres_config = f"""# Optimized PostgreSQL Configuration
# Generated by Database Connection Optimizer
# System: {self.cpu_count} CPU cores, {self.memory_gb:.1f}GB RAM
# Date: {datetime.now().isoformat()}

# Connection Settings
max_connections = {optimal_pg_size + 50}  # Pool size + buffer for admin connections
superuser_reserved_connections = 3

# Memory Settings
shared_buffers = {int(self.memory_gb * 1024 * 0.25)}MB
effective_cache_size = {int(self.memory_gb * 1024 * 0.75)}MB
work_mem = {int((self.memory_gb * 1024) / optimal_pg_size / 4)}MB
maintenance_work_mem = {int(self.memory_gb * 1024 * 0.05)}MB

# Connection Pool Optimization
tcp_keepalives_idle = 600
tcp_keepalives_interval = 30
tcp_keepalives_count = 3

# Performance Settings
random_page_cost = 1.1
effective_io_concurrency = {self.cpu_count * 50}
max_worker_processes = {self.cpu_count}
max_parallel_workers = {min(self.cpu_count, 8)}
max_parallel_workers_per_gather = {min(self.cpu_count // 2, 4)}

# WAL Settings for Performance
wal_buffers = 16MB
checkpoint_completion_target = 0.9
"""
        
        # Connection pool Python configuration
        pool_config = f"""# Optimized Connection Pool Configuration
# Generated by Database Connection Optimizer
# Date: {datetime.now().isoformat()}

# PostgreSQL Pool Settings
POSTGRES_POOL_MIN_SIZE = {max(5, optimal_pg_size // 4)}
POSTGRES_POOL_MAX_SIZE = {optimal_pg_size}
POSTGRES_POOL_TIMEOUT = 30
POSTGRES_COMMAND_TIMEOUT = 30

# Redis Pool Settings
REDIS_POOL_MAX_CONNECTIONS = {optimal_redis_size}
REDIS_SOCKET_TIMEOUT = 5
REDIS_SOCKET_CONNECT_TIMEOUT = 5

# Performance Targets
TARGET_RESPONSE_TIME_MS = {self.target_response_time_ms}
TARGET_POOL_UTILIZATION = {self.target_pool_utilization}
"""
        
        # Save configurations
        config_dir = "/opt/sutazaiapp/configs/optimized"
        os.makedirs(config_dir, exist_ok=True)
        
        with open(f"{config_dir}/postgresql-optimized.conf", "w") as f:
            f.write(postgres_config)
        
        with open(f"{config_dir}/connection-pool-optimized.env", "w") as f:
            f.write(pool_config)
        
        logger.info(f"Generated optimized configurations in {config_dir}/")
    
    async def create_monitoring_script(self):
        """Create monitoring script for connection pool health"""
        
        monitoring_script = '''#!/usr/bin/env python3
"""
Real-time Database Connection Pool Monitor
"""
import asyncio
import asyncpg
import redis.asyncio as redis
import time
import json
from datetime import datetime

async def monitor_pools():
    """Monitor connection pools continuously"""
    
    # Initialize connections
    pg_user = os.getenv("POSTGRES_USER", "sutazai")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "sutazai")
    pg_host = os.getenv("POSTGRES_HOST", "localhost")
    pg_port = os.getenv("POSTGRES_PORT", "10000")
    pg_db = os.getenv("POSTGRES_DB", "sutazai")
    pg_pool = await asyncpg.create_pool(
        f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}',
        min_size=5, max_size=20
    )
    
    redis_client = redis.from_url('redis://localhost:10001/0')
    
    try:
        while True:
            timestamp = datetime.now().isoformat()
            
            # PostgreSQL metrics
            async with pg_pool.acquire() as conn:
                pg_stats = await conn.fetchrow("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections
                    FROM pg_stat_activity
                """)
            
            # Redis metrics
            redis_info = await redis_client.info()
            
            metrics = {
                'timestamp': timestamp,
                'postgresql': {
                    'total_connections': pg_stats['total_connections'],
                    'active_connections': pg_stats['active_connections'],
                    'idle_connections': pg_stats['idle_connections'],
                    'pool_size': pg_pool._maxsize,
                    'pool_utilization': pg_stats['active_connections'] / pg_pool._maxsize
                },
                'redis': {
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'total_commands_processed': redis_info.get('total_commands_processed', 0)
                }
            }
            
            # Log metrics
            print(json.dumps(metrics, indent=2))
            
            # Check for alerts
            if metrics['postgresql']['pool_utilization'] > 0.8:
                print(f"ALERT: High PostgreSQL pool utilization: {metrics['postgresql']['pool_utilization']:.1%}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
    finally:
        await pg_pool.close()
        await redis_client.close()

if __name__ == "__main__":
    asyncio.run(monitor_pools())
'''
        
        monitor_file = "/opt/sutazaiapp/scripts/monitoring/connection-pool-monitor.py"
        with open(monitor_file, "w") as f:
            f.write(monitoring_script)
        
        os.chmod(monitor_file, 0o755)
        logger.info(f"Created connection pool monitoring script: {monitor_file}")
    
    async def cleanup(self):
        """Clean up connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        if self.redis_client:
            await self.redis_client.close()


async def main():
    """Main optimization function"""
    optimizer = DatabaseConnectionOptimizer()
    
    try:
        await optimizer.initialize_connections()
        
        print("Database Connection Pool Optimization Analysis")
        print("=" * 50)
        print(f"System Resources: {optimizer.cpu_count} CPU cores, {optimizer.memory_gb:.1f}GB RAM")
        
        # Get current metrics
        current_metrics = await optimizer.get_current_pool_metrics()
        print(f"Current Pool Size: {current_metrics.pool_size}")
        print(f"Current Utilization: {current_metrics.utilization_percent:.1f}%")
        
        # Calculate optimal settings
        optimal_pg_size, optimal_redis_size = optimizer.calculate_optimal_pool_size()
        print(f"Recommended PostgreSQL Pool Size: {optimal_pg_size}")
        print(f"Recommended Redis Pool Size: {optimal_redis_size}")
        
        # Run full analysis
        print("\nRunning optimization analysis...")
        recommendations = await optimizer.run_optimization_analysis()
        
        # Display recommendations
        print(f"\nOptimization Recommendations ({len(recommendations)} found):")
        print("-" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.database.upper()} - {rec.impact_level.upper()} IMPACT")
            print(f"   Current: {rec.current_setting}")
            print(f"   Recommended: {rec.recommended_setting}")
            print(f"   Reason: {rec.reason}")
            print()
        
        # Generate configuration files
        print("Generating optimized configuration files...")
        optimizer.generate_configuration_files(recommendations)
        
        # Create monitoring script
        print("Creating connection pool monitoring script...")
        await optimizer.create_monitoring_script()
        
        # Save full report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': {
                'cpu_cores': optimizer.cpu_count,
                'memory_gb': optimizer.memory_gb
            },
            'current_metrics': asdict(current_metrics),
            'optimal_pool_sizes': {
                'postgresql': optimal_pg_size,
                'redis': optimal_redis_size
            },
            'recommendations': [asdict(rec) for rec in recommendations]
        }
        
        report_file = f"/opt/sutazaiapp/logs/connection_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFull optimization report saved to: {report_file}")
        print("\nNext Steps:")
        print("1. Review generated configuration files in /opt/sutazaiapp/configs/optimized/")
        print("2. Apply PostgreSQL configuration and restart container")
        print("3. Update connection pool settings in application code")
        print("4. Monitor performance using the generated monitoring script")
        print("5. Run this analysis periodically to maintain optimal performance")
        
    except Exception as e:
        logger.error(f"Optimization analysis failed: {e}")
        raise
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
