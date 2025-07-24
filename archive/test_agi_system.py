#!/usr/bin/env python3
"""
SutazAI AGI/ASI System - Comprehensive Test Suite
Tests all components and ensures 100% functionality
"""

import asyncio
import httpx
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:8501"
NEO4J_URL = "http://localhost:7474"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3003"

class AGISystemTester:
    """Comprehensive test suite for SutazAI AGI/ASI System"""
    
    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "test_details": []
        }
        
    async def run_all_tests(self):
        """Run all system tests"""
        logger.info("Starting SutazAI AGI/ASI System Tests...")
        logger.info("=" * 60)
        
        # Test categories
        await self.test_infrastructure()
        await self.test_api_endpoints()
        await self.test_agi_brain()
        await self.test_agent_orchestration()
        await self.test_knowledge_management()
        await self.test_reasoning_engine()
        await self.test_self_improvement()
        await self.test_monitoring()
        await self.test_frontend()
        await self.test_performance()
        
        # Print results
        self.print_results()
        
    async def test_infrastructure(self):
        """Test core infrastructure services"""
        logger.info("\n🔧 Testing Infrastructure Services...")
        
        services = [
            ("PostgreSQL", 5432),
            ("Redis", 6379),
            ("Neo4j", 7687),
            ("ChromaDB", 8001),
            ("Qdrant", 6333),
            ("Ollama", 11434),
        ]
        
        for service_name, port in services:
            await self.test_service_connection(service_name, port)
            
    async def test_service_connection(self, service_name: str, port: int):
        """Test if a service is accessible"""
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            
            if result == 0:
                self.record_test(f"{service_name} Connection", True, f"Port {port} is accessible")
            else:
                self.record_test(f"{service_name} Connection", False, f"Port {port} is not accessible")
        except Exception as e:
            self.record_test(f"{service_name} Connection", False, str(e))
            
    async def test_api_endpoints(self):
        """Test all API endpoints"""
        logger.info("\n🌐 Testing API Endpoints...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health endpoint
            try:
                response = await client.get(f"{BASE_URL}/health")
                if response.status_code == 200:
                    data = response.json()
                    self.record_test("API Health Check", True, f"Status: {data.get('status')}")
                    
                    # Check specific health metrics
                    if data.get('gpu_available'):
                        self.record_test("GPU Detection", True, "GPU is available")
                    else:
                        self.record_test("GPU Detection", True, "No GPU detected (CPU mode)", warning=True)
                        
                    agents_healthy = data.get('agents_healthy', 0)
                    agents_total = data.get('agents_total', 0)
                    if agents_healthy == agents_total:
                        self.record_test("Agent Health", True, f"{agents_healthy}/{agents_total} agents healthy")
                    else:
                        self.record_test("Agent Health", False, f"Only {agents_healthy}/{agents_total} agents healthy")
                else:
                    self.record_test("API Health Check", False, f"Status code: {response.status_code}")
            except Exception as e:
                self.record_test("API Health Check", False, str(e))
                
            # Test root endpoint
            try:
                response = await client.get(f"{BASE_URL}/")
                self.record_test("API Root Endpoint", response.status_code == 200, 
                               f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("API Root Endpoint", False, str(e))
                
    async def test_agi_brain(self):
        """Test AGI Brain functionality"""
        logger.info("\n🧠 Testing AGI Brain...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Test thinking capability
            try:
                test_query = "What is the meaning of artificial general intelligence?"
                response = await client.post(
                    f"{BASE_URL}/think",
                    json={"query": test_query, "trace_enabled": True}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('response'):
                        self.record_test("AGI Brain - Thinking", True, 
                                       f"Response length: {len(data['response'])} chars")
                        
                        # Check cognitive trace
                        if data.get('cognitive_trace'):
                            self.record_test("AGI Brain - Cognitive Trace", True, 
                                           f"{len(data['cognitive_trace'])} trace items")
                        else:
                            self.record_test("AGI Brain - Cognitive Trace", False, "No trace returned")
                    else:
                        self.record_test("AGI Brain - Thinking", False, "Empty response")
                else:
                    self.record_test("AGI Brain - Thinking", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("AGI Brain - Thinking", False, str(e))
                
    async def test_agent_orchestration(self):
        """Test Agent Orchestrator"""
        logger.info("\n🤖 Testing Agent Orchestration...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # List available agents
            try:
                response = await client.get(f"{BASE_URL}/agents")
                if response.status_code == 200:
                    agents = response.json()
                    agent_count = len(agents.get('agents', []))
                    self.record_test("Agent Listing", True, f"{agent_count} agents available")
                    
                    # Check specific agents
                    expected_agents = ["autogpt", "crewai", "langchain", "aider", "semgrep"]
                    for agent_name in expected_agents:
                        agent_found = any(a['name'] == agent_name for a in agents.get('agents', []))
                        self.record_test(f"Agent - {agent_name}", agent_found, 
                                       "Registered" if agent_found else "Not found")
                else:
                    self.record_test("Agent Listing", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Agent Listing", False, str(e))
                
            # Test task execution
            try:
                test_task = {
                    "description": "Write a simple Python hello world function",
                    "type": "code",
                    "parameters": {"language": "python"}
                }
                
                response = await client.post(f"{BASE_URL}/execute", json=test_task)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'completed':
                        self.record_test("Task Execution", True, "Task completed successfully")
                    else:
                        self.record_test("Task Execution", False, f"Status: {result.get('status')}")
                else:
                    self.record_test("Task Execution", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Task Execution", False, str(e))
                
    async def test_knowledge_management(self):
        """Test Knowledge Manager"""
        logger.info("\n📚 Testing Knowledge Management...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Add knowledge
            try:
                test_knowledge = {
                    "content": "SutazAI is an AGI/ASI system that runs 100% locally",
                    "type": "system",
                    "metadata": {"importance": "high"}
                }
                
                response = await client.post(f"{BASE_URL}/learn", json=test_knowledge)
                if response.status_code == 200:
                    result = response.json()
                    knowledge_id = result.get('knowledge_id')
                    if knowledge_id:
                        self.record_test("Knowledge Addition", True, f"ID: {knowledge_id}")
                        
                        # Test knowledge search
                        search_response = await client.get(
                            f"{BASE_URL}/knowledge/search",
                            params={"query": "SutazAI", "limit": 10}
                        )
                        
                        if search_response.status_code == 200:
                            search_results = search_response.json()
                            result_count = len(search_results.get('results', []))
                            self.record_test("Knowledge Search", True, f"{result_count} results found")
                        else:
                            self.record_test("Knowledge Search", False, 
                                           f"Status: {search_response.status_code}")
                    else:
                        self.record_test("Knowledge Addition", False, "No ID returned")
                else:
                    self.record_test("Knowledge Addition", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Knowledge Addition", False, str(e))
                
    async def test_reasoning_engine(self):
        """Test Reasoning Engine"""
        logger.info("\n🔍 Testing Reasoning Engine...")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Test reasoning
            try:
                test_problem = {
                    "problem": "If all humans are mortal, and Socrates is human, what can we conclude?",
                    "reasoning_type": "DEDUCTIVE"
                }
                
                response = await client.post(f"{BASE_URL}/reason", json=test_problem)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('solution'):
                        self.record_test("Reasoning - Deductive", True, 
                                       f"Certainty: {result.get('certainty', 0):.2f}")
                    else:
                        self.record_test("Reasoning - Deductive", False, "No solution provided")
                else:
                    self.record_test("Reasoning - Deductive", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Reasoning - Deductive", False, str(e))
                
    async def test_self_improvement(self):
        """Test Self-Improvement System"""
        logger.info("\n🔄 Testing Self-Improvement...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Check if self-improvement is enabled
                response = await client.post(
                    f"{BASE_URL}/improve",
                    json={"target": "system", "max_improvements": 1}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    improvements = result.get('improvements', [])
                    self.record_test("Self-Improvement", True, 
                                   f"{len(improvements)} improvements suggested")
                elif response.status_code == 403:
                    self.record_test("Self-Improvement", True, 
                                   "Disabled in production (expected)", warning=True)
                else:
                    self.record_test("Self-Improvement", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Self-Improvement", False, str(e))
                
    async def test_monitoring(self):
        """Test Monitoring Stack"""
        logger.info("\n📊 Testing Monitoring...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test Prometheus
            try:
                response = await client.get(f"{PROMETHEUS_URL}/api/v1/query?query=up")
                self.record_test("Prometheus", response.status_code == 200, 
                               f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Prometheus", False, str(e))
                
            # Test Grafana
            try:
                response = await client.get(GRAFANA_URL)
                self.record_test("Grafana", response.status_code in [200, 302], 
                               f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Grafana", False, str(e))
                
            # Test metrics endpoint
            try:
                response = await client.get(f"{BASE_URL}/metrics")
                if response.status_code == 200:
                    metrics = response.text
                    metric_lines = metrics.strip().split('\n')
                    self.record_test("Metrics Endpoint", True, f"{len(metric_lines)} metrics exposed")
                else:
                    self.record_test("Metrics Endpoint", False, f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Metrics Endpoint", False, str(e))
                
    async def test_frontend(self):
        """Test Frontend UI"""
        logger.info("\n🎨 Testing Frontend...")
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(FRONTEND_URL)
                self.record_test("Frontend UI", response.status_code == 200, 
                               f"Status: {response.status_code}")
            except Exception as e:
                self.record_test("Frontend UI", False, str(e))
                
    async def test_performance(self):
        """Test system performance"""
        logger.info("\n⚡ Testing Performance...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test API response time
            try:
                start_time = time.time()
                response = await client.get(f"{BASE_URL}/health")
                response_time = (time.time() - start_time) * 1000  # ms
                
                if response_time < 100:
                    self.record_test("API Response Time", True, f"{response_time:.2f}ms")
                elif response_time < 500:
                    self.record_test("API Response Time", True, 
                                   f"{response_time:.2f}ms (acceptable)", warning=True)
                else:
                    self.record_test("API Response Time", False, f"{response_time:.2f}ms (too slow)")
            except Exception as e:
                self.record_test("API Response Time", False, str(e))
                
            # Test concurrent requests
            try:
                tasks = []
                for _ in range(10):
                    tasks.append(client.get(f"{BASE_URL}/health"))
                    
                start_time = time.time()
                responses = await asyncio.gather(*tasks)
                total_time = (time.time() - start_time) * 1000
                
                success_count = sum(1 for r in responses if r.status_code == 200)
                if success_count == 10:
                    self.record_test("Concurrent Requests", True, 
                                   f"All 10 succeeded in {total_time:.2f}ms")
                else:
                    self.record_test("Concurrent Requests", False, 
                                   f"Only {success_count}/10 succeeded")
            except Exception as e:
                self.record_test("Concurrent Requests", False, str(e))
                
    def record_test(self, test_name: str, passed: bool, details: str = "", warning: bool = False):
        """Record test result"""
        self.results["total_tests"] += 1
        
        if passed and not warning:
            self.results["passed"] += 1
            status = "✅ PASS"
            color = "\033[92m"
        elif passed and warning:
            self.results["passed"] += 1
            self.results["warnings"] += 1
            status = "⚠️  WARN"
            color = "\033[93m"
        else:
            self.results["failed"] += 1
            status = "❌ FAIL"
            color = "\033[91m"
            
        logger.info(f"{color}{status}\033[0m - {test_name}: {details}")
        
        self.results["test_details"].append({
            "test": test_name,
            "passed": passed,
            "warning": warning,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
    def print_results(self):
        """Print test results summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total = self.results["total_tests"]
        passed = self.results["passed"]
        failed = self.results["failed"]
        warnings = self.results["warnings"]
        
        # Calculate pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ({pass_rate:.1f}%)")
        logger.info(f"Failed: {failed}")
        logger.info(f"Warnings: {warnings}")
        
        # Overall status
        if failed == 0:
            logger.info("\n🎉 ALL TESTS PASSED! The SutazAI AGI/ASI System is fully functional!")
        else:
            logger.info(f"\n⚠️  {failed} tests failed. Please check the failing components.")
            
        # Save detailed results
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Exit code
        sys.exit(0 if failed == 0 else 1)

async def main():
    """Main test execution"""
    tester = AGISystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main()) 