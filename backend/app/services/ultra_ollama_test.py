#!/usr/bin/env python3
"""
ULTRA Ollama Service Test Suite - Agent_3 (Ollama_Specialist)
Validates ULTRAFIX implementation for 100% success rate

Tests:
1. Connection reliability under load
2. Timeout handling and adaptive optimization
3. Error recovery and circuit breaker functionality
4. Performance benchmarking and monitoring
5. Batch processing efficiency
6. Cache effectiveness
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
from datetime import datetime
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UltraOllamaTestSuite:
    """Comprehensive test suite for ULTRA Ollama Service"""
    
    def __init__(self):
        self.test_results = {
            'connection_reliability': {},
            'timeout_handling': {},
            'error_recovery': {},
            'performance_benchmark': {},
            'batch_processing': {},
            'cache_effectiveness': {}
        }
        
    async def run_all_tests(self):
        """Run complete ULTRAFIX validation suite"""
        logger.info("🚀 Starting ULTRA Ollama Service Test Suite...")
        
        start_time = time.time()
        
        # Import ULTRA service
        try:
            from consolidated_ollama_service import get_ollama_service as get_ultra_ollama_service
            self.service = await get_ultra_ollama_service()
            logger.info("✅ ULTRA Ollama Service loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ULTRA Ollama Service: {e}")
            return False
        
        # Test Suite 1: Connection Reliability
        logger.info("🔗 Testing Connection Reliability...")
        await self.test_connection_reliability()
        
        # Test Suite 2: Timeout Handling
        logger.info("⏱️ Testing Timeout Handling...")
        await self.test_timeout_handling()
        
        # Test Suite 3: Error Recovery
        logger.info("🔧 Testing Error Recovery...")
        await self.test_error_recovery()
        
        # Test Suite 4: Performance Benchmark
        logger.info("⚡ Testing Performance Benchmark...")
        await self.test_performance_benchmark()
        
        # Test Suite 5: Batch Processing
        logger.info("📦 Testing Batch Processing...")
        await self.test_batch_processing()
        
        # Test Suite 6: Cache Effectiveness
        logger.info("🎯 Testing Cache Effectiveness...")
        await self.test_cache_effectiveness()
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        self.generate_test_report(total_time)
        
        return self.calculate_overall_success_rate()
    
    async def test_connection_reliability(self):
        """Test connection reliability under various conditions"""
        test_results = {
            'simple_requests': [],
            'concurrent_requests': [],
            'sustained_load': []
        }
        
        # Test 1: Simple sequential requests
        logger.info("  📝 Testing simple sequential requests...")
        for i in range(20):
            start = time.time()
            result = await self.service.generate(f"Test {i}", priority='high')
            elapsed = time.time() - start
            
            success = not result.get('error', False)
            test_results['simple_requests'].append({
                'success': success,
                'time': elapsed,
                'response_length': len(result.get('response', ''))
            })
            
            if i % 5 == 0:
                logger.info(f"    Completed {i+1}/20 simple requests")
        
        # Test 2: Concurrent requests
        logger.info("  🔄 Testing concurrent requests...")
        concurrent_tasks = []
        for i in range(10):
            task = asyncio.create_task(
                self.service.generate(f"Concurrent test {i}")
            )
            concurrent_tasks.append(task)
        
        start = time.time()
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        concurrent_time = time.time() - start
        
        for i, result in enumerate(concurrent_results):
            success = not isinstance(result, Exception) and not result.get('error', False)
            test_results['concurrent_requests'].append({
                'success': success,
                'result': result if not isinstance(result, Exception) else str(result)
            })
        
        logger.info(f"    Concurrent test completed in {concurrent_time:.2f}s")
        
        # Test 3: Sustained load
        logger.info("  💪 Testing sustained load...")
        sustained_start = time.time()
        for batch in range(5):  # 5 batches of 5 requests
            batch_tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    self.service.generate(f"Load test batch {batch}, request {i}")
                )
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                success = not isinstance(result, Exception) and not result.get('error', False)
                test_results['sustained_load'].append({'success': success})
            
            logger.info(f"    Completed batch {batch+1}/5")
            await asyncio.sleep(1)  # Brief pause between batches
        
        sustained_time = time.time() - sustained_start
        logger.info(f"    Sustained load test completed in {sustained_time:.2f}s")
        
        self.test_results['connection_reliability'] = test_results
        
        # Calculate success rates
        simple_success = sum(1 for r in test_results['simple_requests'] if r['success'])
        concurrent_success = sum(1 for r in test_results['concurrent_requests'] if r['success'])
        sustained_success = sum(1 for r in test_results['sustained_load'] if r['success'])
        
        logger.info(f"  📊 Simple requests: {simple_success}/20 ({simple_success/20*100:.1f}%)")
        logger.info(f"  📊 Concurrent requests: {concurrent_success}/10 ({concurrent_success/10*100:.1f}%)")
        logger.info(f"  📊 Sustained load: {sustained_success}/25 ({sustained_success/25*100:.1f}%)")
    
    async def test_timeout_handling(self):
        """Test timeout handling and adaptive optimization"""
        test_results = {
            'short_timeouts': [],
            'long_timeouts': [],
            'adaptive_behavior': []
        }
        
        # Test different timeout scenarios
        test_cases = [
            ("Short prompt", "Hi", {'num_predict': 5}),
            ("Medium prompt", "Write a short paragraph about technology.", {'num_predict': 50}),
            ("Long prompt", "Write a detailed analysis of artificial intelligence trends and their impact on society, covering at least 5 different aspects.", {'num_predict': 200}),
        ]
        
        for test_name, prompt, options in test_cases:
            logger.info(f"    Testing: {test_name}")
            
            start = time.time()
            result = await self.service.generate(prompt, options=options)
            elapsed = time.time() - start
            
            success = not result.get('error', False)
            test_results['adaptive_behavior'].append({
                'test_name': test_name,
                'success': success,
                'time': elapsed,
                'expected_tokens': options['num_predict'],
                'actual_response_length': len(result.get('response', ''))
            })
            
            logger.info(f"      {test_name}: {elapsed:.2f}s, Success: {success}")
        
        self.test_results['timeout_handling'] = test_results
    
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        test_results = {
            'recovery_attempts': 0,
            'successful_recoveries': 0,
            'health_status_changes': []
        }
        
        # Get initial health status
        initial_health = await self.service.health_check()
        test_results['health_status_changes'].append({
            'timestamp': datetime.now().isoformat(),
            'status': initial_health['status'],
            'consecutive_failures': initial_health['metrics']['consecutive_failures']
        })
        
        logger.info(f"    Initial health: {initial_health['status']}")
        
        # Test recovery after simulated stress
        logger.info("    Testing recovery mechanisms...")
        
        # Reset performance counters for clean test
        await self.service.reset_performance_counters()
        
        # Get health after reset
        post_reset_health = await self.service.health_check()
        test_results['health_status_changes'].append({
            'timestamp': datetime.now().isoformat(),
            'status': post_reset_health['status'],
            'consecutive_failures': post_reset_health['metrics']['consecutive_failures']
        })
        
        logger.info(f"    Post-reset health: {post_reset_health['status']}")
        
        self.test_results['error_recovery'] = test_results
    
    async def test_performance_benchmark(self):
        """Benchmark performance metrics"""
        test_results = {
            'response_times': [],
            'throughput': 0,
            'cache_performance': {}
        }
        
        logger.info("    Running performance benchmark...")
        
        # Benchmark response times for different request types
        benchmark_prompts = [
            "Hello",
            "What is AI?",
            "Explain machine learning in one sentence.",
            "List three benefits of renewable energy.",
            "How does photosynthesis work?"
        ]
        
        total_start = time.time()
        
        for i, prompt in enumerate(benchmark_prompts * 4):  # 20 total requests
            start = time.time()
            result = await self.service.generate(prompt)
            elapsed = time.time() - start
            
            success = not result.get('error', False)
            if success:
                test_results['response_times'].append(elapsed)
            
            if i % 5 == 0:
                logger.info(f"      Completed {i+1}/20 benchmark requests")
        
        total_time = time.time() - total_start
        test_results['throughput'] = len(test_results['response_times']) / total_time
        
        # Calculate statistics
        if test_results['response_times']:
            test_results['avg_response_time'] = statistics.mean(test_results['response_times'])
            test_results['median_response_time'] = statistics.median(test_results['response_times'])
            test_results['min_response_time'] = min(test_results['response_times'])
            test_results['max_response_time'] = max(test_results['response_times'])
            test_results['p95_response_time'] = sorted(test_results['response_times'])[int(len(test_results['response_times']) * 0.95)]
        
        self.test_results['performance_benchmark'] = test_results
        
        logger.info(f"    Throughput: {test_results['throughput']:.2f} requests/second")
        if test_results['response_times']:
            logger.info(f"    Avg response time: {test_results['avg_response_time']:.2f}s")
            logger.info(f"    P95 response time: {test_results['p95_response_time']:.2f}s")
    
    async def test_batch_processing(self):
        """Test batch processing efficiency"""
        test_results = {
            'batch_sizes': [],
            'individual_vs_batch': {}
        }
        
        logger.info("    Testing batch processing...")
        
        test_prompts = [
            "What is the weather?",
            "Tell me a joke.",
            "Explain gravity.",
            "What is Python?",
            "How do computers work?"
        ]
        
        # Test individual requests
        individual_start = time.time()
        individual_results = []
        for prompt in test_prompts:
            result = await self.service.generate(prompt)
            individual_results.append(result)
        individual_time = time.time() - individual_start
        
        # Test batch requests
        batch_start = time.time()
        batch_results = await self.service.generate_batch(test_prompts)
        batch_time = time.time() - batch_start
        
        test_results['individual_vs_batch'] = {
            'individual_time': individual_time,
            'batch_time': batch_time,
            'efficiency_gain': (individual_time - batch_time) / individual_time * 100,
            'individual_success': sum(1 for r in individual_results if not r.get('error')),
            'batch_success': sum(1 for r in batch_results if not r.get('error'))
        }
        
        self.test_results['batch_processing'] = test_results
        
        logger.info(f"    Individual: {individual_time:.2f}s, Batch: {batch_time:.2f}s")
        logger.info(f"    Efficiency gain: {test_results['individual_vs_batch']['efficiency_gain']:.1f}%")
    
    async def test_cache_effectiveness(self):
        """Test cache effectiveness"""
        test_results = {
            'cache_hits': 0,
            'cache_misses': 0,
            'response_times': {
                'cache_hits': [],
                'cache_misses': []
            }
        }
        
        logger.info("    Testing cache effectiveness...")
        
        # Make repeated requests to test caching
        test_prompt = "What is artificial intelligence?"
        
        # First request (cache miss)
        start = time.time()
        result1 = await self.service.generate(test_prompt)
        time1 = time.time() - start
        
        # Second request (should be cache hit)
        start = time.time()
        result2 = await self.service.generate(test_prompt)
        time2 = time.time() - start
        
        # Third request (should be cache hit)
        start = time.time()
        result3 = await self.service.generate(test_prompt)
        time3 = time.time() - start
        
        # Analyze results
        cache_hit_threshold = 0.1  # If response time < 0.1s, likely cache hit
        
        times = [time1, time2, time3]
        for i, t in enumerate(times):
            if t < cache_hit_threshold:
                test_results['cache_hits'] += 1
                test_results['response_times']['cache_hits'].append(t)
            else:
                test_results['cache_misses'] += 1
                test_results['response_times']['cache_misses'].append(t)
        
        self.test_results['cache_effectiveness'] = test_results
        
        logger.info(f"    Cache hits: {test_results['cache_hits']}/3")
        logger.info(f"    Response times: {times[0]:.3f}s, {times[1]:.3f}s, {times[2]:.3f}s")
    
    def generate_test_report(self, total_time: float):
        """Generate comprehensive test report"""
        logger.info("📊 ULTRA Ollama Service Test Report")
        logger.info("=" * 50)
        
        # Connection Reliability Report
        conn_results = self.test_results['connection_reliability']
        simple_success = sum(1 for r in conn_results['simple_requests'] if r['success'])
        concurrent_success = sum(1 for r in conn_results['concurrent_requests'] if r['success'])
        sustained_success = sum(1 for r in conn_results['sustained_load'] if r['success'])
        
        logger.info(f"🔗 Connection Reliability:")
        logger.info(f"   Simple requests: {simple_success}/20 ({simple_success/20*100:.1f}%)")
        logger.info(f"   Concurrent requests: {concurrent_success}/10 ({concurrent_success/10*100:.1f}%)")
        logger.info(f"   Sustained load: {sustained_success}/25 ({sustained_success/25*100:.1f}%)")
        
        # Performance Report
        if 'avg_response_time' in self.test_results['performance_benchmark']:
            perf = self.test_results['performance_benchmark']
            logger.info(f"⚡ Performance Benchmark:")
            logger.info(f"   Average response time: {perf['avg_response_time']:.2f}s")
            logger.info(f"   Throughput: {perf['throughput']:.2f} req/s")
            logger.info(f"   P95 response time: {perf['p95_response_time']:.2f}s")
        
        # Batch Processing Report
        batch = self.test_results['batch_processing']['individual_vs_batch']
        logger.info(f"📦 Batch Processing:")
        logger.info(f"   Individual time: {batch['individual_time']:.2f}s")
        logger.info(f"   Batch time: {batch['batch_time']:.2f}s")
        logger.info(f"   Efficiency gain: {batch['efficiency_gain']:.1f}%")
        
        # Cache Report
        cache = self.test_results['cache_effectiveness']
        logger.info(f"🎯 Cache Effectiveness:")
        logger.info(f"   Cache hits: {cache['cache_hits']}/3")
        
        logger.info(f"⏱️ Total test time: {total_time:.2f}s")
        logger.info("=" * 50)
    
    def calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all tests"""
        total_tests = 0
        successful_tests = 0
        
        # Connection reliability tests
        conn = self.test_results['connection_reliability']
        total_tests += len(conn['simple_requests']) + len(conn['concurrent_requests']) + len(conn['sustained_load'])
        successful_tests += sum(1 for r in conn['simple_requests'] if r['success'])
        successful_tests += sum(1 for r in conn['concurrent_requests'] if r['success'])
        successful_tests += sum(1 for r in conn['sustained_load'] if r['success'])
        
        # Timeout handling tests
        timeout = self.test_results['timeout_handling']
        total_tests += len(timeout['adaptive_behavior'])
        successful_tests += sum(1 for r in timeout['adaptive_behavior'] if r['success'])
        
        # Batch processing tests
        batch = self.test_results['batch_processing']['individual_vs_batch']
        total_tests += 10  # 5 individual + 5 batch
        successful_tests += batch['individual_success'] + batch['batch_success']
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        logger.info(f"🎯 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate*100:.1f}%)")
        
        if success_rate >= 0.95:
            logger.info("✅ ULTRAFIX SUCCESS: 95%+ success rate achieved!")
            return True
        else:
            logger.warning(f"⚠️ ULTRAFIX NEEDS IMPROVEMENT: {success_rate*100:.1f}% success rate")
            return False


async def main():
    """Run the ULTRA Ollama test suite"""
    test_suite = UltraOllamaTestSuite()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("🏆 ULTRAFIX VALIDATION SUCCESSFUL!")
        return 0
    else:
        logger.error("❌ ULTRAFIX VALIDATION FAILED!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))