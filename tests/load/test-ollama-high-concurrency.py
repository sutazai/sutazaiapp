#!/usr/bin/env python3
"""
Ollama High-Concurrency Load Testing Script
Tests the optimized Ollama setup with 174+ concurrent connections
"""

import asyncio
import aiohttp
import json
import time
import logging
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
import random
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/ollama_load_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestRequest:
    """Individual test request configuration."""
    id: int
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    priority: int = 5

@dataclass
class TestResult:
    """Result of a single test request."""
    request_id: int
    success: bool
    response_time_ms: float
    queue_wait_time_ms: float
    response_size: int
    error_message: Optional[str] = None
    timestamp: float = 0.0
    status_code: int = 0

@dataclass
class LoadTestMetrics:
    """Aggregated load test metrics."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    total_test_duration_s: float
    concurrent_users: int
    errors_by_type: Dict[str, int]

class OllamaLoadTester:
    """
    High-concurrency load tester for Ollama.
    Simulates 174+ concurrent AI agent connections.
    """
    
    def __init__(self, 
                 ollama_urls: List[str] = None,
                 max_concurrent: int = 200,
                 request_timeout: int = 120,
                 connection_pool_size: int = 300):
        
        self.ollama_urls = ollama_urls or ["http://localhost:11434"]
        self.max_concurrent = max_concurrent
        self.request_timeout = request_timeout
        self.connection_pool_size = connection_pool_size
        
        # Load balancing state
        self.current_url_index = 0
        
        # Test state
        self.is_running = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestResult] = []
        
        # Test prompts for variety
        self.test_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate Fibonacci numbers.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does machine learning work?",
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "Write a short story about a robot.",
            "List the top 10 programming languages.",
            "How do neural networks function?",
            "What causes climate change?",
            "Explain blockchain technology.",
            "Write a haiku about coding.",
            "What is artificial intelligence?",
            "How do computers process information?",
            "Explain the water cycle.",
            "What are the principles of good design?",
            "How does HTTP work?",
            "What is database normalization?",
            "Explain the concept of recursion."
        ]
        
        logger.info(f"Load tester initialized for {len(self.ollama_urls)} Ollama instances")

    async def initialize(self):
        """Initialize the load tester."""
        # Create aiohttp session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=self.connection_pool_size,
            limit_per_host=self.connection_pool_size // len(self.ollama_urls),
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            headers={"Content-Type": "application/json"}
        )
        
        self.is_running = True
        logger.info("Load tester initialized")

    async def shutdown(self):
        """Shutdown the load tester."""
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        logger.info("Load tester shutdown complete")

    def get_next_url(self) -> str:
        """Get next Ollama URL using round-robin load balancing."""
        url = self.ollama_urls[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.ollama_urls)
        return url

    def generate_test_request(self, request_id: int) -> TestRequest:
        """Generate a test request with varied parameters."""
        return TestRequest(
            id=request_id,
            model="gpt-oss",  # Using GPT-OSS model
            prompt=random.choice(self.test_prompts),
            max_tokens=random.randint(50, 200),
            temperature=random.uniform(0.1, 0.8),
            priority=random.randint(1, 10)
        )

    async def execute_request(self, request: TestRequest) -> TestResult:
        """Execute a single test request."""
        start_time = time.time()
        
        try:
            url = self.get_next_url()
            
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": 0.9,
                    "top_k": 20
                },
                "stream": False
            }
            
            request_start = time.time()
            
            async with self.session.post(f"{url}/api/generate", 
                                       json=payload) as response:
                
                response_time_ms = (time.time() - request_start) * 1000
                
                if response.status == 200:
                    response_data = await response.json()
                    response_size = len(json.dumps(response_data))
                    
                    return TestResult(
                        request_id=request.id,
                        success=True,
                        response_time_ms=response_time_ms,
                        queue_wait_time_ms=0,  # TODO: Extract from response if available
                        response_size=response_size,
                        timestamp=time.time(),
                        status_code=response.status
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        request_id=request.id,
                        success=False,
                        response_time_ms=response_time_ms,
                        queue_wait_time_ms=0,
                        response_size=0,
                        error_message=f"HTTP {response.status}: {error_text}",
                        timestamp=time.time(),
                        status_code=response.status
                    )
                    
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                request_id=request.id,
                success=False,
                response_time_ms=response_time_ms,
                queue_wait_time_ms=0,
                response_size=0,
                error_message="Request timeout",
                timestamp=time.time(),
                status_code=0
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return TestResult(
                request_id=request.id,
                success=False,
                response_time_ms=response_time_ms,
                queue_wait_time_ms=0,
                response_size=0,
                error_message=str(e),
                timestamp=time.time(),
                status_code=0
            )

    async def run_concurrent_test(self, 
                                concurrent_users: int,
                                requests_per_user: int,
                                ramp_up_time: int = 60) -> LoadTestMetrics:
        """Run a concurrent load test."""
        logger.info(f"Starting concurrent test: {concurrent_users} users, "
                   f"{requests_per_user} requests each, "
                   f"{ramp_up_time}s ramp-up")
        
        test_start_time = time.time()
        self.results = []
        
        # Create all test requests
        all_requests = []
        for user_id in range(concurrent_users):
            for req_num in range(requests_per_user):
                request_id = user_id * requests_per_user + req_num
                all_requests.append(self.generate_test_request(request_id))
        
        total_requests = len(all_requests)
        logger.info(f"Generated {total_requests} test requests")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(request: TestRequest) -> TestResult:
            async with semaphore:
                return await self.execute_request(request)
        
        # Execute requests with gradual ramp-up
        tasks = []
        requests_per_second = concurrent_users / ramp_up_time if ramp_up_time > 0 else concurrent_users
        
        for i, request in enumerate(all_requests):
            # Calculate delay for ramp-up
            if ramp_up_time > 0:
                delay = i / requests_per_second
                if delay > 0:
                    await asyncio.sleep(min(delay - (time.time() - test_start_time), 0.1))
            
            task = asyncio.create_task(execute_with_semaphore(request))
            tasks.append(task)
        
        logger.info(f"All {total_requests} requests dispatched, waiting for completion...")
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = [r for r in results if isinstance(r, TestResult)]
        self.results = valid_results
        
        test_duration = time.time() - test_start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_duration, concurrent_users)
        
        logger.info(f"Test completed: {metrics.success_rate:.1f}% success rate, "
                   f"{metrics.avg_response_time_ms:.1f}ms avg response time, "
                   f"{metrics.throughput_rps:.1f} RPS")
        
        return metrics

    async def run_sustained_test(self,
                               concurrent_users: int,
                               duration_seconds: int,
                               requests_per_second: int = None) -> LoadTestMetrics:
        """Run a sustained load test for a specific duration."""
        if requests_per_second is None:
            requests_per_second = concurrent_users // 2  # Default rate
        
        logger.info(f"Starting sustained test: {concurrent_users} concurrent users, "
                   f"{duration_seconds}s duration, {requests_per_second} RPS target")
        
        test_start_time = time.time()
        self.results = []
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def execute_with_semaphore(request: TestRequest) -> TestResult:
            async with semaphore:
                return await self.execute_request(request)
        
        # Request generation loop
        request_id = 0
        active_tasks = []
        
        while time.time() - test_start_time < duration_seconds:
            loop_start = time.time()
            
            # Generate requests for this second
            for _ in range(requests_per_second):
                if time.time() - test_start_time >= duration_seconds:
                    break
                
                request = self.generate_test_request(request_id)
                request_id += 1
                
                task = asyncio.create_task(execute_with_semaphore(request))
                active_tasks.append(task)
            
            # Clean up completed tasks
            active_tasks = [task for task in active_tasks if not task.done()]
            
            # Rate limiting - sleep to maintain target RPS
            loop_duration = time.time() - loop_start
            sleep_time = max(0, 1.0 - loop_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        logger.info(f"Stopping request generation, waiting for {len(active_tasks)} active requests...")
        
        # Wait for remaining requests to complete
        if active_tasks:
            completed_results = await asyncio.gather(*active_tasks, return_exceptions=True)
            valid_results = [r for r in completed_results if isinstance(r, TestResult)]
            self.results.extend(valid_results)
        
        test_duration = time.time() - test_start_time
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_duration, concurrent_users)
        
        logger.info(f"Sustained test completed: {metrics.success_rate:.1f}% success rate, "
                   f"{metrics.throughput_rps:.1f} actual RPS")
        
        return metrics

    def calculate_metrics(self, test_duration: float, concurrent_users: int) -> LoadTestMetrics:
        """Calculate comprehensive test metrics."""
        if not self.results:
            return LoadTestMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0,
                avg_response_time_ms=0,
                median_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                throughput_rps=0,
                total_test_duration_s=test_duration,
                concurrent_users=concurrent_users,
                errors_by_type={}
            )
        
        # Basic counts
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100
        
        # Response time metrics
        response_times = [r.response_time_ms for r in self.results]
        response_times.sort()
        
        avg_response_time = statistics.mean(response_times)
        median_response_time = statistics.median(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Percentiles
        p95_index = int(len(response_times) * 0.95)
        p99_index = int(len(response_times) * 0.99)
        p95_response_time = response_times[min(p95_index, len(response_times) - 1)]
        p99_response_time = response_times[min(p99_index, len(response_times) - 1)]
        
        # Throughput
        throughput_rps = total_requests / test_duration if test_duration > 0 else 0
        
        # Error analysis
        errors_by_type = {}
        for result in self.results:
            if not result.success and result.error_message:
                error_type = result.error_message.split(':')[0].strip()
                errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        return LoadTestMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            throughput_rps=throughput_rps,
            total_test_duration_s=test_duration,
            concurrent_users=concurrent_users,
            errors_by_type=errors_by_type
        )

    def print_metrics_report(self, metrics: LoadTestMetrics):
        """Print a comprehensive metrics report."""
        print("\n" + "="*80)
        print("OLLAMA HIGH-CONCURRENCY LOAD TEST RESULTS")
        print("="*80)
        
        print(f"\nTest Configuration:")
        print(f"  Concurrent Users: {metrics.concurrent_users}")
        print(f"  Total Duration: {metrics.total_test_duration_s:.1f}s")
        print(f"  Ollama Instances: {len(self.ollama_urls)}")
        print(f"  Target Model: gpt-oss")
        
        print(f"\nRequest Statistics:")
        print(f"  Total Requests: {metrics.total_requests}")
        print(f"  Successful: {metrics.successful_requests}")
        print(f"  Failed: {metrics.failed_requests}")
        print(f"  Success Rate: {metrics.success_rate:.2f}%")
        print(f"  Throughput: {metrics.throughput_rps:.2f} requests/second")
        
        print(f"\nResponse Time Statistics (ms):")
        print(f"  Average: {metrics.avg_response_time_ms:.1f}")
        print(f"  Median: {metrics.median_response_time_ms:.1f}")
        print(f"  95th Percentile: {metrics.p95_response_time_ms:.1f}")
        print(f"  99th Percentile: {metrics.p99_response_time_ms:.1f}")
        print(f"  Minimum: {metrics.min_response_time_ms:.1f}")
        print(f"  Maximum: {metrics.max_response_time_ms:.1f}")
        
        if metrics.errors_by_type:
            print(f"\nError Analysis:")
            for error_type, count in sorted(metrics.errors_by_type.items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} ({count/metrics.total_requests*100:.1f}%)")
        
        # Performance assessment
        print(f"\nPerformance Assessment:")
        if metrics.success_rate >= 99:
            print("  ✅ EXCELLENT: >99% success rate")
        elif metrics.success_rate >= 95:
            print("  ✅ GOOD: >95% success rate")
        elif metrics.success_rate >= 90:
            print("  ⚠️  ACCEPTABLE: >90% success rate")
        else:
            print("  ❌ POOR: <90% success rate")
        
        if metrics.p95_response_time_ms <= 2000:
            print("  ✅ EXCELLENT: P95 response time ≤2s")
        elif metrics.p95_response_time_ms <= 5000:
            print("  ✅ GOOD: P95 response time ≤5s")
        elif metrics.p95_response_time_ms <= 10000:
            print("  ⚠️  ACCEPTABLE: P95 response time ≤10s")
        else:
            print("  ❌ POOR: P95 response time >10s")
        
        if metrics.throughput_rps >= 50:
            print("  ✅ EXCELLENT: Throughput ≥50 RPS")
        elif metrics.throughput_rps >= 25:
            print("  ✅ GOOD: Throughput ≥25 RPS")
        elif metrics.throughput_rps >= 10:
            print("  ⚠️  ACCEPTABLE: Throughput ≥10 RPS")
        else:
            print("  ❌ POOR: Throughput <10 RPS")
        
        print("\n" + "="*80)

    async def save_results(self, metrics: LoadTestMetrics, filename: str):
        """Save test results to a JSON file."""
        results_data = {
            "timestamp": time.time(),
            "test_config": {
                "ollama_urls": self.ollama_urls,
                "max_concurrent": self.max_concurrent,
                "request_timeout": self.request_timeout
            },
            "metrics": asdict(metrics),
            "individual_results": [asdict(r) for r in self.results[:1000]]  # Limit to first 1000
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")

async def main():
    """Main entry point for load testing."""
    parser = argparse.ArgumentParser(description="Ollama High-Concurrency Load Tester")
    parser.add_argument("--urls", nargs="+", default=["http://localhost:11434"],
                       help="Ollama instance URLs")
    parser.add_argument("--concurrent-users", type=int, default=174,
                       help="Number of concurrent users (default: 174)")
    parser.add_argument("--requests-per-user", type=int, default=10,
                       help="Requests per user for concurrent test")
    parser.add_argument("--duration", type=int, default=300,
                       help="Duration for sustained test (seconds)")
    parser.add_argument("--test-type", choices=["concurrent", "sustained", "both"], 
                       default="both", help="Type of test to run")
    parser.add_argument("--max-concurrent", type=int, default=200,
                       help="Maximum concurrent requests")
    parser.add_argument("--ramp-up-time", type=int, default=60,
                       help="Ramp-up time for concurrent test (seconds)")
    parser.add_argument("--output-file", 
                       default="/opt/sutazaiapp/logs/load_test_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize load tester
    tester = OllamaLoadTester(
        ollama_urls=args.urls,
        max_concurrent=args.max_concurrent
    )
    
    # Graceful shutdown handler
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(tester.shutdown())
    
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda s, f: signal_handler())
    
    try:
        await tester.initialize()
        
        print(f"\n🚀 Starting Ollama High-Concurrency Load Test")
        print(f"Target: {args.concurrent_users} concurrent connections")
        print(f"Ollama instances: {args.urls}")
        print(f"Model: gpt-oss")
        
        # Run tests based on type
        if args.test_type in ["concurrent", "both"]:
            print(f"\n📊 Running Concurrent Load Test...")
            concurrent_metrics = await tester.run_concurrent_test(
                concurrent_users=args.concurrent_users,
                requests_per_user=args.requests_per_user,
                ramp_up_time=args.ramp_up_time
            )
            tester.print_metrics_report(concurrent_metrics)
            
            if args.test_type == "concurrent":
                await tester.save_results(concurrent_metrics, args.output_file)
        
        if args.test_type in ["sustained", "both"]:
            print(f"\n⏱️  Running Sustained Load Test...")
            sustained_metrics = await tester.run_sustained_test(
                concurrent_users=args.concurrent_users,
                duration_seconds=args.duration
            )
            tester.print_metrics_report(sustained_metrics)
            
            if args.test_type == "sustained":
                await tester.save_results(sustained_metrics, args.output_file)
        
        if args.test_type == "both":
            # Save combined results
            combined_data = {
                "concurrent_test": asdict(concurrent_metrics),
                "sustained_test": asdict(sustained_metrics),
                "timestamp": time.time()
            }
            with open(args.output_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
            logger.info(f"Combined results saved to {args.output_file}")
        
        print(f"\n✅ Load testing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        sys.exit(1)
    finally:
        await tester.shutdown()

if __name__ == "__main__":
    asyncio.run(main())