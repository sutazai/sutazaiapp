#!/usr/bin/env python3
"""
AI Agents Concurrent Load Testing Script
Tests all 8 AI agents with configurable concurrent load
Measures: response times, error rates, resource usage
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class LoadTestResult:
    agent_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx]

AGENT_ENDPOINTS = {
    "letta": "http://localhost:11401",
    "crewai": "http://localhost:11403",
    "aider": "http://localhost:11404",
    "langchain": "http://localhost:11405",
    "finrobot": "http://localhost:11410",
    "shellgpt": "http://localhost:11413",
    "documind": "http://localhost:11414",
    "gpt-engineer": "http://localhost:11416"
}

TEST_PROMPTS = [
    "Hello, how are you?",
    "What is the weather today?",
    "Tell me a joke",
    "Explain quantum computing",
    "Write a haiku about AI"
]

async def make_request(session: aiohttp.ClientSession, agent_name: str, url: str, prompt: str) -> tuple:
    """Make single request to agent and measure response time"""
    start_time = time.time()
    try:
        payload = {"prompt": prompt, "model": "tinyllama"}
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with session.post(
            f"{url}/generate",
            json=payload,
            timeout=timeout
        ) as response:
            response_time = time.time() - start_time
            
            if response.status == 200:
                data = await response.json()
                return True, response_time, None
            else:
                error_text = await response.text()
                return False, response_time, f"HTTP {response.status}: {error_text}"
                
    except asyncio.TimeoutError:
        response_time = time.time() - start_time
        return False, response_time, "Request timeout"
    except Exception as e:
        response_time = time.time() - start_time
        return False, response_time, str(e)

async def test_agent(agent_name: str, url: str, num_requests: int, concurrent_requests: int) -> LoadTestResult:
    """Test single agent with specified load"""
    print(f"🔄 Testing {agent_name} ({num_requests} requests, {concurrent_requests} concurrent)...")
    
    result = LoadTestResult(
        agent_name=agent_name,
        total_requests=num_requests,
        successful_requests=0,
        failed_requests=0
    )
    
    # Check agent health first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    print(f"⚠️  {agent_name} health check failed")
                    result.failed_requests = num_requests
                    result.errors.append("Agent unhealthy")
                    return result
    except Exception as e:
        print(f"❌ {agent_name} not accessible: {e}")
        result.failed_requests = num_requests
        result.errors.append(f"Agent not accessible: {e}")
        return result
    
    # Create request queue
    request_queue = []
    for i in range(num_requests):
        prompt = TEST_PROMPTS[i % len(TEST_PROMPTS)]
        request_queue.append((agent_name, url, prompt))
    
    # Execute requests in batches (concurrent_requests at a time)
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(request_queue), concurrent_requests):
            batch = request_queue[i:i + concurrent_requests]
            tasks = [
                make_request(session, agent_name, url, prompt)
                for agent_name, url, prompt in batch
            ]
            
            results = await asyncio.gather(*tasks)
            
            for success, response_time, error in results:
                result.response_times.append(response_time)
                if success:
                    result.successful_requests += 1
                else:
                    result.failed_requests += 1
                    if error:
                        result.errors.append(error)
    
    return result

async def run_load_test(num_requests: int = 20, concurrent_requests: int = 5):
    """Run load test on all agents"""
    print(f"\n{'='*80}")
    print(f"🚀 AI AGENTS CONCURRENT LOAD TEST")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  - Total requests per agent: {num_requests}")
    print(f"  - Concurrent requests: {concurrent_requests}")
    print(f"  - Agents under test: {len(AGENT_ENDPOINTS)}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Test all agents concurrently
    tasks = [
        test_agent(name, url, num_requests, concurrent_requests)
        for name, url in AGENT_ENDPOINTS.items()
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 LOAD TEST RESULTS")
    print(f"{'='*80}\n")
    
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "total_duration_seconds": round(total_time, 2),
        "configuration": {
            "requests_per_agent": num_requests,
            "concurrent_requests": concurrent_requests
        },
        "agents": []
    }
    
    all_success = True
    
    for result in sorted(results, key=lambda x: x.agent_name):
        status_icon = "✅" if result.success_rate >= 80 else "⚠️" if result.success_rate >= 50 else "❌"
        
        print(f"{status_icon} {result.agent_name.upper()}")
        print(f"   Success Rate: {result.success_rate:.1f}% ({result.successful_requests}/{result.total_requests})")
        print(f"   Avg Response Time: {result.avg_response_time*1000:.0f}ms")
        print(f"   P95 Response Time: {result.p95_response_time*1000:.0f}ms")
        print(f"   P99 Response Time: {result.p99_response_time*1000:.0f}ms")
        
        if result.failed_requests > 0:
            print(f"   ⚠️  Failures: {result.failed_requests}")
            all_success = False
            if result.errors:
                unique_errors = list(set(result.errors[:3]))  # Show up to 3 unique errors
                for error in unique_errors:
                    print(f"      - {error[:80]}")
        print()
        
        summary["agents"].append({
            "name": result.agent_name,
            "success_rate": round(result.success_rate, 2),
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "avg_response_time_ms": round(result.avg_response_time * 1000, 2),
            "p95_response_time_ms": round(result.p95_response_time * 1000, 2),
            "p99_response_time_ms": round(result.p99_response_time * 1000, 2),
            "errors": list(set(result.errors[:5]))  # Keep up to 5 unique errors
        })
    
    # Overall statistics
    total_requests = sum(r.total_requests for r in results)
    total_successful = sum(r.successful_requests for r in results)
    overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
    
    all_response_times = []
    for r in results:
        all_response_times.extend(r.response_times)
    
    print(f"{'='*80}")
    print(f"📈 OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Successful: {total_successful} ({overall_success_rate:.1f}%)")
    print(f"  Failed: {total_requests - total_successful}")
    print(f"  Total Duration: {total_time:.2f}s")
    print(f"  Requests/Second: {total_requests/total_time:.1f}")
    
    if all_response_times:
        print(f"  Avg Response Time: {statistics.mean(all_response_times)*1000:.0f}ms")
        print(f"  Min Response Time: {min(all_response_times)*1000:.0f}ms")
        print(f"  Max Response Time: {max(all_response_times)*1000:.0f}ms")
    
    print(f"{'='*80}\n")
    
    summary["overall"] = {
        "total_requests": total_requests,
        "successful_requests": total_successful,
        "failed_requests": total_requests - total_successful,
        "success_rate": round(overall_success_rate, 2),
        "total_duration_seconds": round(total_time, 2),
        "requests_per_second": round(total_requests / total_time, 2),
        "all_agents_passing": all_success
    }
    
    # Save results
    output_file = f"/opt/sutazaiapp/test-results/agent_load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}\n")
    
    return summary

if __name__ == "__main__":
    import sys
    
    # Parse command line args
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    concurrent_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Run the load test
    asyncio.run(run_load_test(num_requests, concurrent_requests))
