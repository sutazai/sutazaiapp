#!/usr/bin/env python3
"""
Purpose: Test complete Ollama integration with all 131 agents
Usage: python test-ollama-integration.py [--full] [--agents AGENT1,AGENT2]
Requirements: httpx, asyncio, pyyaml
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.core.ollama_integration import OllamaIntegration, OllamaConfig
from agents.core.base_agent_v2 import BaseAgentV2


class OllamaIntegrationTester:
    """Tests the complete Ollama integration across all agents"""
    
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:9005")
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": 0,
            "tested": 0,
            "passed": 0,
            "failed": 0,
            "agents": {},
            "models": defaultdict(lambda: {"tested": 0, "passed": 0, "failed": 0}),
            "performance": {},
            "errors": []
        }
        
    async def test_ollama_connectivity(self) -> bool:
        """Test basic Ollama connectivity"""
        print("Testing Ollama connectivity...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    print(f"✅ Ollama is running with {len(models)} models")
                    self.results["ollama_models"] = [m.get("name") for m in models]
                    return True
                else:
                    print(f"❌ Ollama returned status {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            self.results["errors"].append(f"Ollama connectivity: {str(e)}")
            return False
            
    async def test_model_availability(self) -> Dict[str, bool]:
        """Test if required models are available"""
        print("\nTesting model availability...")
        required_models = [
            OllamaConfig.DEFAULT_MODEL,
            OllamaConfig.SONNET_MODEL,
            OllamaConfig.OPUS_MODEL
        ]
        
        results = {}
        async with OllamaIntegration() as ollama:
            for model in required_models:
                try:
                    available = await ollama.ensure_model_available(model)
                    results[model] = available
                    status = "✅" if available else "❌"
                    print(f"{status} Model {model}: {'Available' if available else 'Not found'}")
                except Exception as e:
                    results[model] = False
                    print(f"❌ Model {model}: Error - {e}")
                    
        self.results["model_availability"] = results
        return results
        
    async def test_agent_ollama_integration(self, agent_name: str) -> Dict[str, Any]:
        """Test a single agent's Ollama integration"""
        result = {
            "agent": agent_name,
            "model": OllamaConfig.get_model_for_agent(agent_name),
            "status": "untested",
            "response_time": None,
            "error": None
        }
        
        try:
            # Get the configured model for this agent
            model = OllamaConfig.get_model_for_agent(agent_name)
            config = OllamaConfig.get_model_config(agent_name)
            
            # Test with OllamaIntegration
            async with OllamaIntegration() as ollama:
                start_time = time.time()
                
                # Simple test prompt
                test_prompt = f"Hello, I am the {agent_name} agent. Please respond with a brief greeting."
                
                response = await ollama.generate(
                    prompt=test_prompt,
                    model=model,
                    temperature=config.get("temperature", 0.7),
                    max_tokens=100  # Small for testing
                )
                
                response_time = time.time() - start_time
                
                if response:
                    result["status"] = "passed"
                    result["response_time"] = response_time
                    result["response_preview"] = response[:100] + "..." if len(response) > 100 else response
                else:
                    result["status"] = "failed"
                    result["error"] = "No response from Ollama"
                    
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
        
    async def test_concurrent_agents(self, agent_names: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Test multiple agents concurrently"""
        print(f"\nTesting {len(agent_names)} agents with max {max_concurrent} concurrent...")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def test_with_limit(agent_name: str):
            async with semaphore:
                return await self.test_agent_ollama_integration(agent_name)
                
        results = await asyncio.gather(*[test_with_limit(name) for name in agent_names])
        return results
        
    async def test_model_switching(self) -> Dict[str, Any]:
        """Test model switching performance"""
        print("\nTesting model switching performance...")
        
        models = [
            OllamaConfig.DEFAULT_MODEL,
            OllamaConfig.SONNET_MODEL,
            OllamaConfig.OPUS_MODEL
        ]
        
        results = {"switching_times": {}}
        
        async with OllamaIntegration() as ollama:
            for i in range(len(models)):
                current_model = models[i]
                next_model = models[(i + 1) % len(models)]
                
                # Use current model
                start = time.time()
                await ollama.generate(
                    prompt="Test prompt",
                    model=current_model,
                    max_tokens=50
                )
                
                # Switch to next model
                switch_start = time.time()
                await ollama.generate(
                    prompt="Test prompt",
                    model=next_model,
                    max_tokens=50
                )
                switch_time = time.time() - switch_start
                
                results["switching_times"][f"{current_model}_to_{next_model}"] = switch_time
                print(f"  {current_model} → {next_model}: {switch_time:.2f}s")
                
        return results
        
    async def run_integration_tests(self, full_test: bool = False, specific_agents: Optional[List[str]] = None):
        """Run complete integration tests"""
        print("=" * 60)
        print("SutazAI Ollama Integration Test Suite")
        print("=" * 60)
        
        # Test Ollama connectivity
        if not await self.test_ollama_connectivity():
            print("\n❌ Cannot proceed without Ollama connectivity")
            return self.results
            
        # Test model availability
        model_status = await self.test_model_availability()
        if not any(model_status.values()):
            print("\n❌ No required models available")
            return self.results
            
        # Determine which agents to test
        if specific_agents:
            agents_to_test = specific_agents
        elif full_test:
            # Test all agents
            agents_to_test = list(OllamaConfig.AGENT_MODELS.keys())
        else:
            # Test sample from each category
            agents_to_test = []
            
            # Sample Opus agents
            opus_agents = [a for a, m in OllamaConfig.AGENT_MODELS.items() 
                          if m == OllamaConfig.OPUS_MODEL]
            agents_to_test.extend(opus_agents[:3])
            
            # Sample Sonnet agents  
            sonnet_agents = [a for a, m in OllamaConfig.AGENT_MODELS.items()
                           if m == OllamaConfig.SONNET_MODEL]
            agents_to_test.extend(sonnet_agents[:5])
            
            # Sample default agents
            default_agents = [a for a, m in OllamaConfig.AGENT_MODELS.items()
                            if m == OllamaConfig.DEFAULT_MODEL]
            agents_to_test.extend(default_agents[:2])
            
        self.results["total_agents"] = len(OllamaConfig.AGENT_MODELS)
        self.results["tested"] = len(agents_to_test)
        
        print(f"\nTesting {len(agents_to_test)} agents...")
        
        # Test agents
        agent_results = await self.test_concurrent_agents(agents_to_test)
        
        # Process results
        for result in agent_results:
            agent_name = result["agent"]
            model = result["model"]
            status = result["status"]
            
            self.results["agents"][agent_name] = result
            self.results["models"][model]["tested"] += 1
            
            if status == "passed":
                self.results["passed"] += 1
                self.results["models"][model]["passed"] += 1
                status_icon = "✅"
            else:
                self.results["failed"] += 1
                self.results["models"][model]["failed"] += 1
                status_icon = "❌"
                
            response_time = result.get("response_time", 0)
            if response_time:
                print(f"{status_icon} {agent_name} ({model}): {response_time:.2f}s")
            else:
                error = result.get("error", "Unknown error")
                print(f"{status_icon} {agent_name} ({model}): {error}")
                
        # Test model switching if not full test
        if not full_test:
            switch_results = await self.test_model_switching()
            self.results["performance"]["model_switching"] = switch_results
            
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Total Agents: {self.results['total_agents']}")
        print(f"Tested: {self.results['tested']}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Success Rate: {(self.results['passed'] / self.results['tested'] * 100):.1f}%")
        
        print("\nModel Performance:")
        for model, stats in self.results["models"].items():
            if stats["tested"] > 0:
                success_rate = (stats["passed"] / stats["tested"] * 100)
                print(f"  {model}: {stats['passed']}/{stats['tested']} ({success_rate:.1f}%)")
                
        # Calculate average response times
        response_times = defaultdict(list)
        for agent_data in self.results["agents"].values():
            if agent_data["response_time"]:
                model = agent_data["model"]
                response_times[model].append(agent_data["response_time"])
                
        if response_times:
            print("\nAverage Response Times:")
            for model, times in response_times.items():
                avg_time = sum(times) / len(times)
                print(f"  {model}: {avg_time:.2f}s")
                
        return self.results
        
    def save_results(self, filename: Optional[str] = None):
        """Save test results to file"""
        if not filename:
            filename = f"ollama_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = Path("/opt/sutazaiapp/tests/results") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\nResults saved to: {filepath}")
        

async def main():
    parser = argparse.ArgumentParser(description="Test Ollama integration")
    parser.add_argument("--full", action="store_true", help="Test all agents (default: sample)")
    parser.add_argument("--agents", help="Comma-separated list of specific agents to test")
    parser.add_argument("--save", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    tester = OllamaIntegrationTester()
    
    # Parse specific agents if provided
    specific_agents = None
    if args.agents:
        specific_agents = [a.strip() for a in args.agents.split(',')]
        
    # Run tests
    results = await tester.run_integration_tests(
        full_test=args.full,
        specific_agents=specific_agents
    )
    
    # Save results if requested
    if args.save:
        tester.save_results()
        
    # Exit with appropriate code
    if results["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())