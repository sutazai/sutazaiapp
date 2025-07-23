#!/usr/bin/env python3
"""
Comprehensive AI Agent Testing System for SutazAI
Tests all agents with optimized configurations and proper error handling
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AgentTest:
    name: str
    port: int
    health_endpoint: str
    test_endpoint: str
    test_payload: Dict[str, Any]
    timeout: int = 30
    expected_status: str = "success"

class SutazAIAgentTester:
    """Comprehensive agent testing system"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": [],
            "summary": {"passed": 0, "failed": 0, "timeout": 0}
        }
        
        # Define all agent tests with lightweight model
        self.agent_tests = [
            AgentTest(
                name="Letta (Memory Agent)",
                port=8094,
                health_endpoint="/health",
                test_endpoint="/chat",
                test_payload={
                    "message": "Hello, can you remember that my favorite color is blue?",
                    "agent_name": "test_agent",
                    "session_id": "test_session"
                },
                timeout=15
            ),
            AgentTest(
                name="CrewAI (Multi-Agent)",
                port=8096,
                health_endpoint="/health",
                test_endpoint="/execute",
                test_payload={
                    "agents": [
                        {
                            "role": "Writer",
                            "goal": "Write clear, concise content",
                            "backstory": "You write simple, clear text"
                        }
                    ],
                    "tasks": [
                        {
                            "description": "Write a single sentence about AI",
                            "agent_role": "Writer",
                            "expected_output": "One sentence about AI"
                        }
                    ],
                    "process": "sequential"
                },
                timeout=45
            ),
            AgentTest(
                name="Aider (Code Assistant)",
                port=8095,
                health_endpoint="/health",
                test_endpoint="/code",
                test_payload={
                    "message": "Create a simple hello function",
                    "files": [],
                    "model": "llama3.2:1b",
                    "workspace": "/app/workspace"
                },
                timeout=60
            ),
            AgentTest(
                name="AutoGPT (Autonomous)",
                port=8092,
                health_endpoint="/health",
                test_endpoint="/execute",
                test_payload={
                    "task": "Create a simple text file with hello world",
                    "max_iterations": 1,
                    "workspace": "/app/workspace"
                },
                timeout=45
            ),
            AgentTest(
                name="GPT-Engineer (Code Gen)",
                port=8097,
                health_endpoint="/health",
                test_endpoint="/generate",
                test_payload={
                    "prompt": "Create a simple Python function that returns 'Hello World'",
                    "workspace": "/app/workspace"
                },
                timeout=60
            )
        ]
        
    async def test_agent_health(self, session: aiohttp.ClientSession, agent: AgentTest) -> bool:
        """Test if agent health endpoint is responding"""
        try:
            url = f"{self.base_url}:{agent.port}{agent.health_endpoint}"
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    logger.info(f"✓ {agent.name} health check passed")
                    return True
                else:
                    logger.error(f"✗ {agent.name} health check failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"✗ {agent.name} health check failed: {str(e)}")
            return False
            
    async def test_agent_functionality(self, session: aiohttp.ClientSession, agent: AgentTest) -> Dict[str, Any]:
        """Test agent functionality with a simple task"""
        result = {
            "agent": agent.name,
            "status": "unknown",
            "response_time": 0,
            "error": None,
            "response": None
        }
        
        try:
            start_time = time.time()
            url = f"{self.base_url}:{agent.port}{agent.test_endpoint}"
            
            logger.info(f"Testing {agent.name} functionality...")
            
            timeout = aiohttp.ClientTimeout(total=agent.timeout)
            async with session.post(
                url, 
                json=agent.test_payload, 
                timeout=timeout
            ) as response:
                
                response_time = time.time() - start_time
                result["response_time"] = round(response_time, 2)
                
                if response.status == 200:
                    response_data = await response.json()
                    result["response"] = response_data
                    
                    # Check if response indicates success
                    if (isinstance(response_data, dict) and 
                        response_data.get("status") in ["success", "completed", "healthy"]):
                        result["status"] = "passed"
                        logger.info(f"✓ {agent.name} test PASSED ({response_time:.2f}s)")
                    else:
                        result["status"] = "partial"
                        logger.warning(f"⚠ {agent.name} test PARTIAL ({response_time:.2f}s)")
                else:
                    result["status"] = "failed"
                    result["error"] = f"HTTP {response.status}"
                    logger.error(f"✗ {agent.name} test FAILED: HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            result["status"] = "timeout"
            result["error"] = f"Timeout after {agent.timeout}s"
            logger.error(f"✗ {agent.name} test TIMEOUT")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"✗ {agent.name} test FAILED: {str(e)}")
            
        return result
        
    async def optimize_ollama_for_testing(self):
        """Switch Ollama to use the fastest model for testing"""
        logger.info("Optimizing Ollama configuration for testing...")
        
        try:
            # Test if lightweight model responds faster
            async with aiohttp.ClientSession() as session:
                test_payload = {
                    "model": "llama3.2:1b",
                    "prompt": "Hi",
                    "stream": False
                }
                
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=test_payload,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        logger.info("✓ Ollama responding with lightweight model")
                        return True
                    else:
                        logger.warning("⚠ Ollama slow even with lightweight model")
                        return False
                        
        except Exception as e:
            logger.error(f"✗ Ollama optimization check failed: {str(e)}")
            return False
            
    async def run_comprehensive_tests(self):
        """Run all agent tests systematically"""
        logger.info("Starting comprehensive SutazAI agent testing...")
        logger.info("=" * 60)
        
        # First optimize Ollama
        ollama_ready = await self.optimize_ollama_for_testing()
        if not ollama_ready:
            logger.warning("Ollama performance issues detected - tests may be slow")
            
        async with aiohttp.ClientSession() as session:
            # Phase 1: Health checks
            logger.info("Phase 1: Testing agent health endpoints...")
            healthy_agents = []
            
            for agent in self.agent_tests:
                if await self.test_agent_health(session, agent):
                    healthy_agents.append(agent)
                    
            logger.info(f"Health check results: {len(healthy_agents)}/{len(self.agent_tests)} agents healthy")
            
            if not healthy_agents:
                logger.error("No agents are healthy - aborting tests")
                return
                
            # Phase 2: Functionality tests
            logger.info("\nPhase 2: Testing agent functionality...")
            
            for agent in healthy_agents:
                test_result = await self.test_agent_functionality(session, agent)
                self.results["tests"].append(test_result)
                
                # Update summary
                if test_result["status"] == "passed":
                    self.results["summary"]["passed"] += 1
                elif test_result["status"] == "timeout":
                    self.results["summary"]["timeout"] += 1
                else:
                    self.results["summary"]["failed"] += 1
                    
        # Generate final report
        self.generate_test_report()
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("SUTAZAI AI AGENTS - COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        total_tests = len(self.results["tests"])
        passed = self.results["summary"]["passed"]
        failed = self.results["summary"]["failed"] 
        timeout = self.results["summary"]["timeout"]
        
        print(f"Test Results: {passed} PASSED | {failed} FAILED | {timeout} TIMEOUT | {total_tests} TOTAL")
        print("-" * 80)
        
        for test in self.results["tests"]:
            status_symbol = {
                "passed": "✓",
                "failed": "✗", 
                "timeout": "⏱",
                "partial": "⚠"
            }.get(test["status"], "?")
            
            print(f"{status_symbol} {test['agent']:25} | {test['status']:8} | {test['response_time']:6.2f}s")
            
            if test["error"]:
                print(f"    Error: {test['error']}")
                
        print("-" * 80)
        
        if passed == total_tests:
            print("🎉 ALL AGENTS WORKING PERFECTLY!")
        elif passed > 0:
            print(f"⚠ {passed}/{total_tests} agents working - some need attention")
        else:
            print("❌ ALL AGENTS FAILED - system needs debugging")
            
        print("=" * 80)
        
        # Save detailed report
        report_file = f"/opt/sutazaiapp/logs/agent_test_report_{int(time.time())}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Detailed report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

async def main():
    """Main execution function"""
    tester = SutazAIAgentTester()
    await tester.run_comprehensive_tests()

if __name__ == "__main__":
    asyncio.run(main())