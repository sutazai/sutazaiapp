#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRAFIX: Database Performance Testing and Benchmarking Script
Tests connection pooling, query performance, and Redis cache hit rates

Usage:
    python test_database_performance.py
    python test_database_performance.py --benchmark-duration 60
    python test_database_performance.py --load-test --concurrent-users 100
"""

import asyncio
import asyncpg
import redis.asyncio as redis
import time
import statistics
import argparse
from typing import List, Dict, Any
from datetime import datetime, timedelta
import uuid
import json
import sys
import os

# Add backend to path
sys.path.append('/opt/sutazaiapp/backend')

class DatabasePerformanceTester:
    """ULTRAFIX: Comprehensive database performance testing"""
    
    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.test_results = {}
        
    async def setup(self):
        """Initialize database connections"""
        # PostgreSQL connection pool
        self.db_pool = await asyncpg.create_pool(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '10000')),
            user=os.getenv('POSTGRES_USER', 'sutazai'),
            password=os.getenv('POSTGRES_PASSWORD', 'sutazai'),
            database=os.getenv('POSTGRES_DB', 'sutazai'),
            min_size=10,
            max_size=20,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
        
        # Redis connection  
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '10001')),
            db=0,
            decode_responses=True
        )
        
        logger.info("✅ Database connections initialized")
        
    async def test_connection_pool_performance(self) -> Dict[str, Any]:
        """Test PostgreSQL connection pooling performance"""
        logger.info("\n🔍 Testing PostgreSQL connection pool performance...")
        
        query_times = []
        connection_times = []
        
        # Test with multiple concurrent connections
        async def single_query():
            start = time.time()
            async with self.db_pool.acquire() as conn:
                conn_time = time.time() - start
                connection_times.append(conn_time)
                
                query_start = time.time()
                await conn.fetchval("SELECT 1")
                query_time = time.time() - query_start
                query_times.append(query_time)
        
        # Run 100 concurrent queries
        start_time = time.time()
        tasks = [single_query() for _ in range(100)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        results = {
            'total_queries': 100,
            'total_time': round(total_time, 3),
            'queries_per_second': round(100 / total_time, 2),
            'avg_connection_time': round(statistics.mean(connection_times) * 1000, 2),  # ms
            'avg_query_time': round(statistics.mean(query_times) * 1000, 2),  # ms
            'max_query_time': round(max(query_times) * 1000, 2),  # ms
            'pool_size': self.db_pool.get_size(),
            'pool_free': self.db_pool.get_idle_size(),
            'status': 'EXCELLENT' if statistics.mean(query_times) < 0.01 else 'GOOD' if statistics.mean(query_times) < 0.05 else 'NEEDS_IMPROVEMENT'
        }
        
        logger.info(f"   📊 Queries per second: {results['queries_per_second']}")
        logger.info(f"   ⚡ Avg query time: {results['avg_query_time']}ms")
        logger.info(f"   🔗 Avg connection time: {results['avg_connection_time']}ms")
        logger.info(f"   🎯 Status: {results['status']}")
        
        return results
    
    async def test_redis_cache_performance(self) -> Dict[str, Any]:
        """Test Redis cache performance and hit rates"""
        logger.info("\n🔍 Testing Redis cache performance...")
        
        set_times = []
        get_times = []
        hits = 0
        misses = 0
        
        # Test data
        test_keys = [f"test:performance:{i}" for i in range(100)]
        test_values = [f"test_value_{i}_{uuid.uuid4()}" for i in range(100)]
        
        # Test SET operations
        start_time = time.time()
        for key, value in zip(test_keys, test_values):
            set_start = time.time()
            await self.redis_client.set(key, value, ex=3600)
            set_times.append(time.time() - set_start)
        set_total_time = time.time() - start_time
        
        # Test GET operations (should all hit)
        start_time = time.time()
        for key in test_keys:
            get_start = time.time()
            result = await self.redis_client.get(key)
            get_times.append(time.time() - get_start)
            if result:
                hits += 1
            else:
                misses += 1
        get_total_time = time.time() - start_time
        
        # Test cache misses
        miss_start = time.time()
        for i in range(20):
            miss_key = f"nonexistent:key:{i}"
            result = await self.redis_client.get(miss_key)
            if result:
                hits += 1
            else:
                misses += 1
        miss_total_time = time.time() - miss_start
        
        # Clean up
        await self.redis_client.delete(*test_keys)
        
        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
        
        results = {
            'total_operations': len(test_keys) * 2 + 20,
            'set_ops_per_second': round(len(test_keys) / set_total_time, 2),
            'get_ops_per_second': round((len(test_keys) + 20) / (get_total_time + miss_total_time), 2),
            'avg_set_time': round(statistics.mean(set_times) * 1000, 2),  # ms
            'avg_get_time': round(statistics.mean(get_times) * 1000, 2),  # ms
            'cache_hits': hits,
            'cache_misses': misses,
            'hit_rate': round(hit_rate, 4),
            'hit_rate_percent': round(hit_rate * 100, 2),
            'status': 'EXCELLENT' if hit_rate > 0.85 else 'GOOD' if hit_rate > 0.7 else 'NEEDS_IMPROVEMENT'
        }
        
        logger.info(f"   📈 Cache hit rate: {results['hit_rate_percent']}%")
        logger.info(f"   ⚡ GET ops/sec: {results['get_ops_per_second']}")
        logger.info(f"   💾 SET ops/sec: {results['set_ops_per_second']}")
        logger.info(f"   🎯 Status: {results['status']}")
        
        return results
    
    async def test_uuid_vs_serial_performance(self) -> Dict[str, Any]:
        """Test UUID vs SERIAL primary key performance"""
        logger.info("\n🔍 Testing UUID vs SERIAL performance...")
        
        # Create test tables
        await self.create_test_tables()
        
        # Test SERIAL inserts
        serial_times = []
        for i in range(100):
            start = time.time()
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO test_serial (name, data) VALUES ($1, $2)",
                    f"test_{i}", f"data_{i}"
                )
            serial_times.append(time.time() - start)
        
        # Test UUID inserts
        uuid_times = []
        for i in range(100):
            start = time.time()
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO test_uuid (id, name, data) VALUES ($1, $2, $3)",
                    str(uuid.uuid4()), f"test_{i}", f"data_{i}"
                )
            uuid_times.append(time.time() - start)
        
        # Test query performance
        serial_query_times = []
        uuid_query_times = []
        
        # Get some IDs for testing
        async with self.db_pool.acquire() as conn:
            serial_ids = await conn.fetch("SELECT id FROM test_serial LIMIT 20")
            uuid_ids = await conn.fetch("SELECT id FROM test_uuid LIMIT 20")
        
        # Test SERIAL queries
        for row in serial_ids:
            start = time.time()
            async with self.db_pool.acquire() as conn:
                await conn.fetchrow("SELECT * FROM test_serial WHERE id = $1", row['id'])
            serial_query_times.append(time.time() - start)
        
        # Test UUID queries
        for row in uuid_ids:
            start = time.time()
            async with self.db_pool.acquire() as conn:
                await conn.fetchrow("SELECT * FROM test_uuid WHERE id = $1", row['id'])
            uuid_query_times.append(time.time() - start)
        
        # Clean up
        await self.cleanup_test_tables()
        
        results = {
            'serial_avg_insert_time': round(statistics.mean(serial_times) * 1000, 2),  # ms
            'uuid_avg_insert_time': round(statistics.mean(uuid_times) * 1000, 2),  # ms
            'serial_avg_query_time': round(statistics.mean(serial_query_times) * 1000, 2),  # ms
            'uuid_avg_query_time': round(statistics.mean(uuid_query_times) * 1000, 2),  # ms
            'insert_performance_impact': round(((statistics.mean(uuid_times) - statistics.mean(serial_times)) / statistics.mean(serial_times)) * 100, 2),
            'query_performance_impact': round(((statistics.mean(uuid_query_times) - statistics.mean(serial_query_times)) / statistics.mean(serial_query_times)) * 100, 2),
            'recommendation': 'UUID' if statistics.mean(uuid_query_times) < statistics.mean(serial_query_times) * 1.1 else 'SERIAL'
        }
        
        logger.info(f"   📊 SERIAL insert: {results['serial_avg_insert_time']}ms")
        logger.info(f"   🆔 UUID insert: {results['uuid_avg_insert_time']}ms")
        logger.info(f"   📊 SERIAL query: {results['serial_avg_query_time']}ms") 
        logger.info(f"   🆔 UUID query: {results['uuid_avg_query_time']}ms")
        logger.info(f"   💡 Recommendation: {results['recommendation']}")
        
        return results
    
    async def create_test_tables(self):
        """Create test tables for performance comparison"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_serial (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_uuid (
                    id UUID PRIMARY KEY,
                    name VARCHAR(100),
                    data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    async def cleanup_test_tables(self):
        """Clean up test tables"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("DROP TABLE IF EXISTS test_serial")
            await conn.execute("DROP TABLE IF EXISTS test_uuid")
    
    async def load_test(self, concurrent_users: int = 50, duration: int = 30):
        """Run load test with multiple concurrent users"""
        logger.info(f"\n🚀 Running load test with {concurrent_users} concurrent users for {duration}s...")
        
        results = {
            'concurrent_users': concurrent_users,
            'duration': duration,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'requests_per_second': 0,
            'errors': []
        }
        
        start_time = time.time()
        response_times = []
        
        async def user_simulation():
            """Simulate a single user's database operations"""
            user_requests = 0
            user_successes = 0
            user_failures = 0
            
            while time.time() - start_time < duration:
                try:
                    # Random database operations
                    operation_start = time.time()
                    
                    # 70% reads, 30% writes
                    if asyncio.get_event_loop().time() % 10 < 7:
                        # Read operation
                        async with self.db_pool.acquire() as conn:
                            await conn.fetchval("SELECT COUNT(*) FROM users")
                        
                        # Cache operation
                        await self.redis_client.get(f"user:test:{user_requests}")
                    else:
                        # Write operation
                        test_id = str(uuid.uuid4())
                        async with self.db_pool.acquire() as conn:
                            await conn.execute(
                                "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                                f"test_{test_id[:8]}", f"test_{test_id[:8]}@example.com", "hash"
                            )
                        
                        # Cache operation
                        await self.redis_client.set(f"user:test:{user_requests}", test_id, ex=3600)
                    
                    operation_time = time.time() - operation_start
                    response_times.append(operation_time)
                    user_requests += 1
                    user_successes += 1
                    
                except Exception as e:
                    user_failures += 1
                    results['errors'].append(str(e))
                
                # Small delay between requests
                await asyncio.sleep(0.1)
            
            return {
                'requests': user_requests,
                'successes': user_successes,
                'failures': user_failures
            }
        
        # Run concurrent users
        tasks = [user_simulation() for _ in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        total_time = time.time() - start_time
        
        for user_result in user_results:
            if isinstance(user_result, dict):
                results['total_requests'] += user_result['requests']
                results['successful_requests'] += user_result['successes']
                results['failed_requests'] += user_result['failures']
        
        if response_times:
            results['avg_response_time'] = round(statistics.mean(response_times) * 1000, 2)  # ms
            results['requests_per_second'] = round(results['total_requests'] / total_time, 2)
        
        logger.info(f"   📊 Total requests: {results['total_requests']}")
        logger.info(f"   ✅ Success rate: {round((results['successful_requests'] / results['total_requests']) * 100, 2)}%")
        logger.info(f"   ⚡ Requests/sec: {results['requests_per_second']}")
        logger.info(f"   🕒 Avg response: {results['avg_response_time']}ms")
        
        return results
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info("\n🎯 ULTRAFIX DATABASE PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        # Run all tests
        pool_results = await self.test_connection_pool_performance()
        redis_results = await self.test_redis_cache_performance()
        uuid_results = await self.test_uuid_vs_serial_performance()
        
        # Overall assessment
        overall_score = 0
        max_score = 0
        
        # Connection pool scoring
        if pool_results['status'] == 'EXCELLENT':
            overall_score += 30
        elif pool_results['status'] == 'GOOD':
            overall_score += 20
        else:
            overall_score += 10
        max_score += 30
        
        # Redis cache scoring
        if redis_results['status'] == 'EXCELLENT':
            overall_score += 35
        elif redis_results['status'] == 'GOOD':
            overall_score += 25
        else:
            overall_score += 15
        max_score += 35
        
        # UUID performance scoring (higher is better for distributed systems)
        if uuid_results['recommendation'] == 'UUID':
            overall_score += 35
        else:
            overall_score += 20
        max_score += 35
        
        overall_percentage = round((overall_score / max_score) * 100, 1)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_percentage,
            'overall_grade': 'A+' if overall_percentage >= 90 else 'A' if overall_percentage >= 80 else 'B' if overall_percentage >= 70 else 'C',
            'connection_pool': pool_results,
            'redis_cache': redis_results,
            'uuid_performance': uuid_results,
            'recommendations': []
        }
        
        # Generate recommendations
        if pool_results['status'] != 'EXCELLENT':
            report['recommendations'].append("Optimize connection pool settings (increase pool size or reduce query complexity)")
        
        if redis_results['hit_rate_percent'] < 85:
            report['recommendations'].append("Implement Redis-first caching strategy to improve hit rates")
        
        if uuid_results['recommendation'] == 'SERIAL' and uuid_results['query_performance_impact'] > 20:
            report['recommendations'].append("Consider keeping SERIAL PKs or optimize UUID indexing")
        elif uuid_results['recommendation'] == 'UUID':
            report['recommendations'].append("Migrate to UUID primary keys for better distributed performance")
        
        if not report['recommendations']:
            report['recommendations'].append("System performance is optimal - no immediate improvements needed")
        
        # Print final report
        logger.info(f"\n🏆 OVERALL PERFORMANCE GRADE: {report['overall_grade']} ({report['overall_score']}%)")
        logger.info(f"🔧 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"   {i}. {rec}")
        
        return report
    
    async def cleanup(self):
        """Clean up connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            await self.redis_client.close()

async def main():
    parser = argparse.ArgumentParser(description='ULTRAFIX Database Performance Testing')
    parser.add_argument('--benchmark-duration', type=int, default=30, help='Benchmark duration in seconds')
    parser.add_argument('--load-test', action='store_true', help='Run load test')
    parser.add_argument('--concurrent-users', type=int, default=50, help='Number of concurrent users for load test')
    parser.add_argument('--output-file', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    tester = DatabasePerformanceTester()
    
    try:
        await tester.setup()
        
        # Generate comprehensive report
        report = await tester.generate_performance_report()
        
        # Run load test if requested
        if args.load_test:
            load_results = await tester.load_test(args.concurrent_users, args.benchmark_duration)
            report['load_test'] = load_results
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"\n💾 Results saved to {args.output_file}")
        
        logger.info(f"\n✅ Performance testing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during performance testing: {e}")
        sys.exit(1)
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())