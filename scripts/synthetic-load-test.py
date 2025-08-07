#!/usr/bin/env python3
"""
Synthetic Load Testing for Agent Services
Generates various types of loads to test metrics and alerting
"""

import asyncio
import aiohttp
import random
import time
import argparse
import json
from typing import List, Dict, Any
from datetime import datetime

# Agent endpoints to test
AGENTS = {
    "ollama-integration": {
        "url": "http://localhost:8090",
        "endpoints": [
            {"path": "/generate", "method": "POST", "data": {"prompt": "Hello world", "temperature": 0.7}},
            {"path": "/health", "method": "GET"},
            {"path": "/models", "method": "GET"}
        ]
    },
    "task-assignment": {
        "url": "http://localhost:8551", 
        "endpoints": [
            {"path": "/health", "method": "GET"},
            {"path": "/queue", "method": "GET"},
            {"path": "/stats", "method": "GET"}
        ]
    },
    "resource-arbitration": {
        "url": "http://localhost:8588",
        "endpoints": [
            {"path": "/health", "method": "GET"}
        ]
    },
    "ai-orchestrator": {
        "url": "http://localhost:8589",
        "endpoints": [
            {"path": "/health", "method": "GET"}
        ]
    }
}

class LoadTester:
    def __init__(self, duration: int = 60, rate: int = 10, error_rate: float = 0.0):
        """
        Initialize load tester
        
        Args:
            duration: Test duration in seconds
            rate: Requests per second
            error_rate: Percentage of requests that should fail (0.0-1.0)
        """
        self.duration = duration
        self.rate = rate
        self.error_rate = error_rate
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": []
        }
        
    async def make_request(
        self, 
        session: aiohttp.ClientSession,
        url: str,
        endpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a single request to an endpoint"""
        
        # Simulate errors based on error rate
        if random.random() < self.error_rate:
            # Simulate different types of errors
            error_type = random.choice(["timeout", "invalid_data", "server_error"])
            
            if error_type == "timeout":
                await asyncio.sleep(5)  # Simulate timeout
                raise asyncio.TimeoutError("Simulated timeout")
            elif error_type == "invalid_data":
                if endpoint["method"] == "POST":
                    endpoint = endpoint.copy()
                    endpoint["data"] = {"invalid": "data"}
            # server_error will be natural from bad data
        
        start_time = time.time()
        result = {
            "endpoint": f"{url}{endpoint['path']}",
            "method": endpoint["method"],
            "status": None,
            "response_time": None,
            "error": None
        }
        
        try:
            if endpoint["method"] == "GET":
                async with session.get(f"{url}{endpoint['path']}") as response:
                    result["status"] = response.status
                    result["response_time"] = time.time() - start_time
                    
            elif endpoint["method"] == "POST":
                async with session.post(
                    f"{url}{endpoint['path']}", 
                    json=endpoint.get("data", {})
                ) as response:
                    result["status"] = response.status
                    result["response_time"] = time.time() - start_time
                    
            if result["status"] >= 200 and result["status"] < 300:
                self.stats["successful_requests"] += 1
            else:
                self.stats["failed_requests"] += 1
                
        except Exception as e:
            result["error"] = str(e)
            self.stats["failed_requests"] += 1
            self.stats["errors"].append({
                "endpoint": result["endpoint"],
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        self.stats["total_requests"] += 1
        if result["response_time"]:
            self.stats["response_times"].append(result["response_time"])
            
        return result
        
    async def test_agent(
        self, 
        agent_name: str,
        agent_config: Dict[str, Any]
    ):
        """Run load test for a specific agent"""
        
        print(f"\n=== Testing {agent_name} ===")
        print(f"URL: {agent_config['url']}")
        print(f"Rate: {self.rate} req/s")
        print(f"Duration: {self.duration}s")
        print(f"Error rate: {self.error_rate*100:.1f}%")
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            start_time = time.time()
            tasks = []
            
            while time.time() - start_time < self.duration:
                # Select random endpoint
                endpoint = random.choice(agent_config["endpoints"])
                
                # Create request task
                task = asyncio.create_task(
                    self.make_request(session, agent_config["url"], endpoint)
                )
                tasks.append(task)
                
                # Control rate
                await asyncio.sleep(1.0 / self.rate)
                
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Print statistics
        self.print_stats(agent_name)
        
    def print_stats(self, agent_name: str):
        """Print test statistics"""
        
        print(f"\n--- Results for {agent_name} ---")
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        
        if self.stats['response_times']:
            avg_time = sum(self.stats['response_times']) / len(self.stats['response_times'])
            min_time = min(self.stats['response_times'])
            max_time = max(self.stats['response_times'])
            
            # Calculate percentiles
            sorted_times = sorted(self.stats['response_times'])
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
            
            print(f"\nResponse times:")
            print(f"  Average: {avg_time*1000:.2f}ms")
            print(f"  Min: {min_time*1000:.2f}ms")
            print(f"  Max: {max_time*1000:.2f}ms")
            print(f"  P50: {p50*1000:.2f}ms")
            print(f"  P95: {p95*1000:.2f}ms")
            print(f"  P99: {p99*1000:.2f}ms")
            
        if self.stats['errors']:
            print(f"\nErrors (first 5):")
            for error in self.stats['errors'][:5]:
                print(f"  - {error['endpoint']}: {error['error']}")
                
        # Check for alert conditions
        error_rate = self.stats['failed_requests'] / max(1, self.stats['total_requests'])
        if error_rate > 0.05:
            print(f"\n⚠️  ALERT: Error rate {error_rate*100:.1f}% exceeds 5% threshold!")
            
        if self.stats['response_times'] and p95 > 0.3:
            print(f"⚠️  ALERT: P95 latency {p95*1000:.0f}ms exceeds 300ms threshold!")
            
    async def run_all_tests(self):
        """Run tests for all agents"""
        
        for agent_name, agent_config in AGENTS.items():
            # Reset stats for each agent
            self.stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "response_times": [],
                "errors": []
            }
            
            try:
                await self.test_agent(agent_name, agent_config)
            except Exception as e:
                print(f"Error testing {agent_name}: {e}")
                
            # Small delay between agents
            await asyncio.sleep(2)
            
def main():
    parser = argparse.ArgumentParser(description="Synthetic Load Testing for Agents")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rate", type=int, default=10, help="Requests per second")
    parser.add_argument("--error-rate", type=float, default=0.0, help="Error rate (0.0-1.0)")
    parser.add_argument("--agent", type=str, help="Test specific agent only")
    parser.add_argument("--spike", action="store_true", help="Simulate spike load")
    parser.add_argument("--gradual", action="store_true", help="Gradually increase load")
    
    args = parser.parse_args()
    
    print("=== SutazAI Agent Load Testing ===")
    print(f"Start time: {datetime.utcnow().isoformat()}")
    
    # Create load tester
    tester = LoadTester(
        duration=args.duration,
        rate=args.rate,
        error_rate=args.error_rate
    )
    
    # Run tests
    if args.agent:
        if args.agent in AGENTS:
            asyncio.run(tester.test_agent(args.agent, AGENTS[args.agent]))
        else:
            print(f"Unknown agent: {args.agent}")
            print(f"Available agents: {', '.join(AGENTS.keys())}")
    else:
        asyncio.run(tester.run_all_tests())
        
    print(f"\n=== Test Complete ===")
    print(f"End time: {datetime.utcnow().isoformat()}")
    
if __name__ == "__main__":
    main()