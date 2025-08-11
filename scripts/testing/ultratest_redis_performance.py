#!/usr/bin/env python3
"""
ULTRATEST Redis Cache Performance Validation
Tests Redis cache hit rate and performance improvements.
"""

import redis
import time
import json
import random
import string
import statistics
from datetime import datetime
from typing import Dict, List, Any

class UltratestRedisValidator:
    def __init__(self):
        self.redis_client = None
        self.test_results = {}
        self.performance_metrics = {}
        
    def connect_to_redis(self) -> bool:
        """Connect to Redis instance"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=10001, db=0)
            self.redis_client.ping()
            print("✅ Connected to Redis successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information and statistics"""
        try:
            info = self.redis_client.info()
            return {
                'version': info.get('redis_version'),
                'uptime_seconds': info.get('uptime_in_seconds'),
                'connected_clients': info.get('connected_clients'),
                'used_memory': info.get('used_memory'),
                'used_memory_human': info.get('used_memory_human'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec'),
                'total_commands_processed': info.get('total_commands_processed')
            }
        except Exception as e:
            print(f"❌ Failed to get Redis info: {e}")
            return {}
    
    def calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        info = self.get_redis_info()
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        
        if hits + misses == 0:
            return 0.0
        
        hit_rate = (hits / (hits + misses)) * 100
        return hit_rate
    
    def test_cache_performance(self, num_operations: int = 1000) -> Dict[str, float]:
        """Test cache read/write performance"""
        print(f"🚀 Testing Redis performance with {num_operations} operations...")
        
        # Generate test data
        test_keys = [f"test_key_{i}" for i in range(num_operations)]
        test_values = [''.join(random.choices(string.ascii_letters, k=100)) for _ in range(num_operations)]
        
        # Test write performance
        write_times = []
        for i, (key, value) in enumerate(zip(test_keys, test_values)):
            start_time = time.time()
            self.redis_client.set(key, value)
            end_time = time.time()
            write_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Test read performance
        read_times = []
        for key in test_keys:
            start_time = time.time()
            self.redis_client.get(key)
            end_time = time.time()
            read_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Clean up test data
        self.redis_client.delete(*test_keys)
        
        return {
            'avg_write_time_ms': statistics.mean(write_times),
            'avg_read_time_ms': statistics.mean(read_times),
            'min_write_time_ms': min(write_times),
            'max_write_time_ms': max(write_times),
            'min_read_time_ms': min(read_times),
            'max_read_time_ms': max(read_times),
            'operations_tested': num_operations
        }
    
    def simulate_cache_usage(self, duration_seconds: int = 30) -> Dict[str, Any]:
        """Simulate realistic cache usage patterns"""
        print(f"🎯 Simulating cache usage for {duration_seconds} seconds...")
        
        # Common cache keys that would be used frequently
        common_keys = [f"user_session_{i}" for i in range(100)]
        frequent_keys = common_keys[:20]  # 20% of keys accessed 80% of the time
        
        # Pre-populate cache with some data
        for key in common_keys:
            self.redis_client.setex(key, 300, f"cached_data_for_{key}")  # 5 minute TTL
        
        start_time = time.time()
        operations = 0
        
        while time.time() - start_time < duration_seconds:
            # 80% chance to access frequent keys, 20% for others
            if random.random() < 0.8:
                key = random.choice(frequent_keys)
            else:
                key = random.choice(common_keys)
            
            # 70% reads, 30% writes (typical cache pattern)
            if random.random() < 0.7:
                self.redis_client.get(key)
            else:
                self.redis_client.setex(key, 300, f"updated_data_{int(time.time())}")
            
            operations += 1
            time.sleep(0.001)  # Small delay to simulate real usage
        
        # Clean up
        self.redis_client.delete(*common_keys)
        
        return {
            'total_operations': operations,
            'operations_per_second': operations / duration_seconds,
            'duration_seconds': duration_seconds
        }
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test Redis memory usage patterns"""
        print("💾 Testing Redis memory usage...")
        
        initial_info = self.get_redis_info()
        initial_memory = initial_info.get('used_memory', 0)
        
        # Add 10,000 keys with 1KB data each
        test_data = 'x' * 1024  # 1KB of data
        test_keys = []
        
        for i in range(10000):
            key = f"memory_test_{i}"
            self.redis_client.set(key, test_data)
            test_keys.append(key)
        
        after_write_info = self.get_redis_info()
        after_write_memory = after_write_info.get('used_memory', 0)
        
        # Clean up
        self.redis_client.delete(*test_keys)
        
        final_info = self.get_redis_info()
        final_memory = final_info.get('used_memory', 0)
        
        return {
            'initial_memory_bytes': initial_memory,
            'after_write_memory_bytes': after_write_memory,
            'final_memory_bytes': final_memory,
            'memory_increase_bytes': after_write_memory - initial_memory,
            'memory_increase_mb': (after_write_memory - initial_memory) / (1024 * 1024),
            'keys_added': len(test_keys),
            'avg_memory_per_key_bytes': (after_write_memory - initial_memory) / len(test_keys) if test_keys else 0
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive Redis performance validation"""
        print("\n🔧 ULTRATEST: Redis Cache Performance Validation")
        print("=" * 70)
        
        if not self.connect_to_redis():
            return {'error': 'Failed to connect to Redis'}
        
        # Get initial Redis information
        initial_info = self.get_redis_info()
        print(f"Redis Version: {initial_info.get('redis_version', 'Unknown')}")
        print(f"Memory Usage: {initial_info.get('used_memory_human', 'Unknown')}")
        
        # Test 1: Current cache hit rate
        initial_hit_rate = self.calculate_cache_hit_rate()
        print(f"Initial Cache Hit Rate: {initial_hit_rate:.2f}%")
        
        # Test 2: Performance benchmarking
        perf_metrics = self.test_cache_performance(1000)
        print(f"Average Write Time: {perf_metrics['avg_write_time_ms']:.2f}ms")
        print(f"Average Read Time: {perf_metrics['avg_read_time_ms']:.2f}ms")
        
        # Test 3: Simulate realistic usage
        usage_simulation = self.simulate_cache_usage(30)
        print(f"Simulated Operations/sec: {usage_simulation['operations_per_second']:.2f}")
        
        # Test 4: Memory usage patterns
        memory_test = self.test_memory_usage()
        print(f"Memory efficiency: {memory_test['avg_memory_per_key_bytes']:.2f} bytes per key")
        
        # Calculate final hit rate after tests
        final_info = self.get_redis_info()
        final_hit_rate = self.calculate_cache_hit_rate()
        
        # Compile comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'redis_info': final_info,
            'initial_hit_rate': initial_hit_rate,
            'final_hit_rate': final_hit_rate,
            'performance_metrics': perf_metrics,
            'usage_simulation': usage_simulation,
            'memory_test': memory_test,
            'target_hit_rate': 85.0,
            'hit_rate_achieved': final_hit_rate >= 85.0,
            'performance_targets': {
                'target_read_time_ms': 1.0,
                'target_write_time_ms': 2.0,
                'read_performance_achieved': perf_metrics['avg_read_time_ms'] <= 1.0,
                'write_performance_achieved': perf_metrics['avg_write_time_ms'] <= 2.0
            }
        }
        
        return results
    
    def generate_performance_report(self, results: Dict[str, Any]):
        """Generate detailed performance report"""
        print("\n" + "=" * 80)
        print("📊 ULTRATEST REDIS PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Test Execution Time: {results.get('timestamp', 'Unknown')}")
        
        # Cache Hit Rate Analysis
        print(f"\n🎯 CACHE HIT RATE ANALYSIS:")
        print("-" * 40)
        initial_rate = results.get('initial_hit_rate', 0)
        final_rate = results.get('final_hit_rate', 0)
        target_rate = results.get('target_hit_rate', 85)
        
        print(f"Initial Hit Rate: {initial_rate:.2f}%")
        print(f"Final Hit Rate: {final_rate:.2f}%")
        print(f"Target Hit Rate: {target_rate:.2f}%")
        
        if results.get('hit_rate_achieved', False):
            print("✅ CACHE HIT RATE TARGET ACHIEVED!")
        else:
            print(f"❌ Cache hit rate below target (gap: {target_rate - final_rate:.2f}%)")
        
        # Performance Analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 40)
        perf = results.get('performance_metrics', {})
        perf_targets = results.get('performance_targets', {})
        
        print(f"Average Read Time: {perf.get('avg_read_time_ms', 0):.2f}ms (target: {perf_targets.get('target_read_time_ms', 1):.2f}ms)")
        print(f"Average Write Time: {perf.get('avg_write_time_ms', 0):.2f}ms (target: {perf_targets.get('target_write_time_ms', 2):.2f}ms)")
        
        if perf_targets.get('read_performance_achieved', False):
            print("✅ READ PERFORMANCE TARGET ACHIEVED!")
        else:
            print("❌ Read performance below target")
            
        if perf_targets.get('write_performance_achieved', False):
            print("✅ WRITE PERFORMANCE TARGET ACHIEVED!")
        else:
            print("❌ Write performance below target")
        
        # Throughput Analysis
        print(f"\n🚀 THROUGHPUT ANALYSIS:")
        print("-" * 40)
        usage_sim = results.get('usage_simulation', {})
        ops_per_sec = usage_sim.get('operations_per_second', 0)
        print(f"Operations per Second: {ops_per_sec:.2f}")
        print(f"Total Operations: {usage_sim.get('total_operations', 0)}")
        
        if ops_per_sec >= 1000:
            print("✅ HIGH THROUGHPUT ACHIEVED (1000+ ops/sec)")
        elif ops_per_sec >= 500:
            print("⚠️  MEDIUM THROUGHPUT (500+ ops/sec)")
        else:
            print("❌ LOW THROUGHPUT (<500 ops/sec)")
        
        # Memory Efficiency
        print(f"\n💾 MEMORY EFFICIENCY:")
        print("-" * 40)
        memory = results.get('memory_test', {})
        memory_per_key = memory.get('avg_memory_per_key_bytes', 0)
        memory_increase_mb = memory.get('memory_increase_mb', 0)
        
        print(f"Memory per Key: {memory_per_key:.2f} bytes")
        print(f"Memory Increase: {memory_increase_mb:.2f} MB for {memory.get('keys_added', 0)} keys")
        
        # Overall Assessment
        print(f"\n🏆 OVERALL ASSESSMENT:")
        print("-" * 40)
        
        achievements = []
        issues = []
        
        if results.get('hit_rate_achieved', False):
            achievements.append("Cache hit rate target met (85%+)")
        else:
            issues.append(f"Cache hit rate below 85% ({final_rate:.2f}%)")
            
        if perf_targets.get('read_performance_achieved', False):
            achievements.append("Read performance target met (<1ms)")
        else:
            issues.append(f"Read performance slow ({perf.get('avg_read_time_ms', 0):.2f}ms)")
            
        if perf_targets.get('write_performance_achieved', False):
            achievements.append("Write performance target met (<2ms)")
        else:
            issues.append(f"Write performance slow ({perf.get('avg_write_time_ms', 0):.2f}ms)")
        
        if ops_per_sec >= 1000:
            achievements.append(f"High throughput achieved ({ops_per_sec:.2f} ops/sec)")
        
        print("🎉 ACHIEVEMENTS:")
        for achievement in achievements:
            print(f"   ✅ {achievement}")
        
        if issues:
            print("\n⚠️  AREAS FOR IMPROVEMENT:")
            for issue in issues:
                print(f"   ❌ {issue}")
        
        success_rate = (len(achievements) / (len(achievements) + len(issues))) * 100 if (achievements or issues) else 0
        print(f"\n📈 Redis Performance Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 80  # Consider 80%+ success rate as passing

def main():
    """Run comprehensive Redis performance validation"""
    print("🚀 Starting ULTRATEST Redis Performance Validation")
    
    validator = UltratestRedisValidator()
    results = validator.run_comprehensive_test()
    
    if 'error' in results:
        print(f"❌ Test failed: {results['error']}")
        return 1
    
    # Generate report
    success = validator.generate_performance_report(results)
    
    # Save results
    with open('/opt/sutazaiapp/tests/ultratest_redis_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full report saved to: /opt/sutazaiapp/tests/ultratest_redis_report.json")
    
    if success:
        print("\n🎉 REDIS PERFORMANCE VALIDATION SUCCESSFUL!")
        return 0
    else:
        print("\n⚠️  REDIS PERFORMANCE NEEDS IMPROVEMENT")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())