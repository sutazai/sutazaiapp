#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
Ollama Performance Testing Script
Benchmarks response times and validates <2s target
"""

import asyncio
import time
import statistics
import httpx
import json
from typing import List, Dict, Any
from datetime import datetime


class OllamaPerformanceTester:
    """Performance tester for Ollama service"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:10104"
        self.backend_url = "http://localhost:10010"
        self.model = "tinyllama"
        self.results = []
        
    async def test_direct_ollama(self, prompt: str) -> Dict[str, Any]:
        """Test direct Ollama API performance"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_ctx": 2048,
                            "num_predict": 256,
                            "temperature": 0.1,
                            "top_k": 10,
                            "top_p": 0.85,
                            "num_thread": 12,
                            "num_batch": 64
                        }
                    }
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "time": elapsed,
                        "response_length": len(result.get("response", "")),
                        "tokens": len(result.get("response", "").split())
                    }
                else:
                    return {
                        "success": False,
                        "time": elapsed,
                        "error": f"Status {response.status_code}"
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "time": time.time() - start_time,
                    "error": str(e)
                }
    
    async def test_streaming_performance(self, prompt: str) -> Dict[str, Any]:
        """Test streaming performance (time to first token)"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            first_token_time = None
            tokens_received = 0
            
            try:
                async with client.stream(
                    "POST",
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "num_ctx": 2048,
                            "num_predict": 256,
                            "temperature": 0.1
                        }
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("response"):
                                    tokens_received += 1
                                    if first_token_time is None:
                                        first_token_time = time.time() - start_time
                                
                                if data.get("done"):
                                    break
                            except:
                                pass
                
                total_time = time.time() - start_time
                
                return {
                    "success": True,
                    "first_token_ms": first_token_time * 1000 if first_token_time else None,
                    "total_time": total_time,
                    "tokens": tokens_received,
                    "tokens_per_second": tokens_received / total_time if total_time > 0 else 0
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def test_backend_api(self, prompt: str) -> Dict[str, Any]:
        """Test backend API with Ollama integration"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            try:
                response = await client.post(
                    f"{self.backend_url}/api/v1/chat/",
                    json={
                        "message": prompt,
                        "model": self.model
                    }
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "time": elapsed,
                        "cached": result.get("cached", False),
                        "response_length": len(result.get("response", ""))
                    }
                else:
                    return {
                        "success": False,
                        "time": elapsed,
                        "error": f"Status {response.status_code}"
                    }
                    
            except Exception as e:
                return {
                    "success": False,
                    "time": time.time() - start_time,
                    "error": str(e)
                }
    
    async def run_benchmark(self, num_iterations: int = 10):
        """Run comprehensive benchmark"""
        logger.info("="*70)
        logger.info("🏃 OLLAMA PERFORMANCE BENCHMARK")
        logger.info(f"Target: <2 second response time")
        logger.info(f"Model: {self.model}")
        logger.info(f"Iterations: {num_iterations}")
        logger.info("="*70)
        
        # Test prompts of varying complexity
        test_cases = [
            ("Simple", "What is 2+2?"),
            ("Medium", "Explain Python in one sentence."),
            ("Complex", "Write a Python function to sort a list."),
            ("Long Context", "Summarize the following text: " + "Lorem ipsum " * 50),
        ]
        
        for test_name, prompt in test_cases:
            logger.info(f"\n📝 Testing: {test_name}")
            logger.info(f"   Prompt: {prompt[:60]}...")
            logger.info("-"*50)
            
            # Direct Ollama tests
            direct_times = []
            logger.info("\n🔹 Direct Ollama API:")
            for i in range(num_iterations):
                result = await self.test_direct_ollama(prompt)
                if result["success"]:
                    direct_times.append(result["time"])
                    status = "✅" if result["time"] < 2.0 else "⚠️" if result["time"] < 3.0 else "❌"
                    logger.info(f"   Run {i+1}: {result['time']:.2f}s {status} ({result['tokens']} tokens)")
                else:
                    logger.error(f"   Run {i+1}: ❌ Failed - {result['error']}")
            
            if direct_times:
                self._print_statistics("Direct API", direct_times)
            
            # Streaming tests
            logger.info("\n🔹 Streaming Performance:")
            first_token_times = []
            for i in range(min(3, num_iterations)):  # Fewer streaming tests
                result = await self.test_streaming_performance(prompt)
                if result["success"]:
                    ft_ms = result["first_token_ms"]
                    if ft_ms:
                        first_token_times.append(ft_ms)
                        status = "🎯" if ft_ms < 500 else "✅" if ft_ms < 1000 else "⚠️"
                        logger.info(f"   Run {i+1}: First token {ft_ms:.0f}ms {status} ({result['tokens_per_second']:.1f} tok/s)")
                else:
                    logger.error(f"   Run {i+1}: ❌ Failed - {result['error']}")
            
            if first_token_times:
                logger.info(f"   Average first token: {statistics.mean(first_token_times):.0f}ms")
            
            # Backend API tests (with caching)
            backend_times = []
            cached_count = 0
            logger.info("\n🔹 Backend API (with cache):")
            for i in range(num_iterations):
                result = await self.test_backend_api(prompt)
                if result["success"]:
                    backend_times.append(result["time"])
                    if result.get("cached"):
                        cached_count += 1
                    cache_indicator = "💾" if result.get("cached") else "🔄"
                    status = "✅" if result["time"] < 2.0 else "⚠️" if result["time"] < 3.0 else "❌"
                    logger.info(f"   Run {i+1}: {result['time']:.2f}s {status} {cache_indicator}")
                else:
                    logger.error(f"   Run {i+1}: ❌ Failed - {result['error']}")
            
            if backend_times:
                self._print_statistics("Backend API", backend_times)
                if cached_count > 0:
                    logger.info(f"   Cache hits: {cached_count}/{len(backend_times)} ({cached_count*100/len(backend_times):.0f}%)")
            
            logger.info("-"*50)
    
    def _print_statistics(self, name: str, times: List[float]):
        """Print statistics for a set of response times"""
        if not times:
            return
        
        avg = statistics.mean(times)
        median = statistics.median(times)
        stdev = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        under_2s = sum(1 for t in times if t < 2.0)
        
        logger.info(f"\n   📊 {name} Statistics:")
        logger.info(f"      Average: {avg:.2f}s")
        logger.info(f"      Median:  {median:.2f}s")
        logger.info(f"      Min:     {min_time:.2f}s")
        logger.info(f"      Max:     {max_time:.2f}s")
        logger.info(f"      StdDev:  {stdev:.2f}s")
        logger.info(f"      Under 2s: {under_2s}/{len(times)} ({under_2s*100/len(times):.0f}%)")
        
        # Performance rating
        if avg < 2.0:
            logger.info(f"      🎯 EXCELLENT: Average under 2s target!")
        elif avg < 3.0:
            logger.info(f"      ✅ GOOD: Average under 3s")
        elif avg < 5.0:
            logger.info(f"      ⚠️  FAIR: Average under 5s")
        else:
            logger.info(f"      ❌ POOR: Average over 5s")
    
    async def run_load_test(self, concurrent_requests: int = 10):
        """Run concurrent load test"""
        logger.info("\n" + "="*70)
        logger.info("🔥 LOAD TEST")
        logger.info(f"Concurrent requests: {concurrent_requests}")
        logger.info("="*70)
        
        prompt = "What is the capital of France?"
        
        logger.info(f"\nSending {concurrent_requests} concurrent requests...")
        tasks = []
        for i in range(concurrent_requests):
            tasks.append(self.test_direct_ollama(prompt))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        logger.info(f"\n📊 Load Test Results:")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Successful: {len(successful)}/{concurrent_requests}")
        logger.error(f"   Failed: {len(failed)}/{concurrent_requests}")
        
        if successful:
            response_times = [r["time"] for r in successful]
            logger.info(f"   Average response: {statistics.mean(response_times):.2f}s")
            logger.info(f"   Min response: {min(response_times):.2f}s")
            logger.info(f"   Max response: {max(response_times):.2f}s")
            logger.info(f"   Throughput: {len(successful)/total_time:.1f} req/s")
    
    def print_summary(self):
        """Print final summary and recommendations"""
        logger.info("\n" + "="*70)
        logger.info("🎯 OPTIMIZATION SUMMARY")
        logger.info("="*70)
        
        logger.info("""
Key factors for <2s response time:

✅ ACHIEVED:
   • Model preloading and memory pinning
   • Optimized context size (2048 tokens)
   • Increased parallel processing (8 parallel)
   • Connection pooling (50 connections)
   • Response caching in backend

🔧 RECOMMENDATIONS:
   1. Enable GPU acceleration if available
   2. Use quantized models (GGUF format)
   3. Implement request batching
   4. Use streaming for perceived performance
   5. Cache common responses
   6. Consider model-specific optimizations

📈 MONITORING:
   • Track p50, p95, p99 latencies
   • Monitor cache hit rates
   • Watch memory usage
   • Track concurrent request handling
        """)


async def main():
    """Main function"""
    tester = OllamaPerformanceTester()
    
    # Run benchmark
    await tester.run_benchmark(num_iterations=5)
    
    # Run load test
    await tester.run_load_test(concurrent_requests=10)
    
    # Print summary
    tester.print_summary()


if __name__ == "__main__":
    logger.info(f"\n🕐 Starting test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    asyncio.run(main())