#!/usr/bin/env python3
"""
SutazAI Complete AGI/ASI System Test Suite
Tests all components, agents, and integrations
"""

import httpx
import asyncio
import json
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

class AGISystemTester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "core_services": {},
            "ai_agents": {},
            "model_tests": {},
            "integration_tests": {},
            "performance_tests": {},
            "errors": []
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_core_services(self) -> Dict:
        """Test core infrastructure services"""
        print("\n🔍 Testing Core Services...")
        
        services = {
            "Backend API": {
                "url": "http://localhost:8000/health",
                "expected_status": 200
            },
            "Frontend UI": {
                "url": "http://localhost:8501",
                "expected_status": 200
            },
            "Ollama": {
                "url": "http://localhost:11434/api/tags",
                "expected_status": 200,
                "validate_response": lambda r: "models" in r
            },
            "LiteLLM Proxy": {
                "url": "http://localhost:4000/health",
                "expected_status": 200
            },
            "ChromaDB": {
                "url": "http://localhost:8001/api/v1/heartbeat",
                "expected_status": 200
            },
            "Qdrant": {
                "url": "http://localhost:6333/healthz",
                "expected_status": 200
            },
            "PostgreSQL": {
                "url": "http://localhost:8000/api/v1/db/health",
                "expected_status": 200
            },
            "Redis": {
                "url": "http://localhost:8000/api/v1/cache/health",
                "expected_status": 200
            },
            "Service Hub": {
                "url": "http://localhost:8114/health",
                "expected_status": 200
            }
        }
        
        for name, config in services.items():
            try:
                response = await self.client.get(config["url"])
                success = response.status_code == config["expected_status"]
                
                if success and "validate_response" in config:
                    try:
                        data = response.json()
                        success = config["validate_response"](data)
                    except:
                        success = False
                
                self.results["core_services"][name] = {
                    "status": "✅ Healthy" if success else "❌ Failed",
                    "response_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
                print(f"  {name}: {'✅' if success else '❌'}")
                
            except Exception as e:
                self.results["core_services"][name] = {
                    "status": "❌ Unreachable",
                    "error": str(e)
                }
                print(f"  {name}: ❌ ({str(e)[:50]})")
        
        return self.results["core_services"]
    
    async def test_ai_agents(self) -> Dict:
        """Test all AI agent endpoints"""
        print("\n🤖 Testing AI Agents...")
        
        agents = {
            "AutoGPT": {
                "url": "http://localhost:8080/health",
                "test_endpoint": "/api/agent/status"
            },
            "CrewAI": {
                "url": "http://localhost:8096/health",
                "test_endpoint": "/api/crews"
            },
            "Aider": {
                "url": "http://localhost:8095/health",
                "test_endpoint": "/api/status"
            },
            "GPT-Engineer": {
                "url": "http://localhost:8097/health",
                "test_endpoint": "/api/projects"
            },
            "LlamaIndex": {
                "url": "http://localhost:8098/health",
                "test_endpoint": "/api/indices"
            },
            "LocalAGI": {
                "url": "http://localhost:8103/health",
                "test_endpoint": "/api/agents"
            },
            "AutoGen": {
                "url": "http://localhost:8104/health",
                "test_endpoint": "/api/assistants"
            },
            "AgentZero": {
                "url": "http://localhost:8105/health",
                "test_endpoint": "/api/status"
            },
            "BigAGI": {
                "url": "http://localhost:8106",
                "test_endpoint": None
            },
            "Dify": {
                "url": "http://localhost:8107",
                "test_endpoint": None
            },
            "OpenDevin": {
                "url": "http://localhost:8108/health",
                "test_endpoint": "/generate"
            },
            "FinRobot": {
                "url": "http://localhost:8109/health",
                "test_endpoint": "/api/analyze"
            },
            "LangFlow": {
                "url": "http://localhost:8090",
                "test_endpoint": None
            },
            "Flowise": {
                "url": "http://localhost:8099",
                "test_endpoint": None
            },
            "n8n": {
                "url": "http://localhost:5678",
                "test_endpoint": None
            },
            "Code Improver": {
                "url": "http://localhost:8113/health",
                "test_endpoint": "/status"
            }
        }
        
        for name, config in agents.items():
            try:
                response = await self.client.get(config["url"])
                success = response.status_code in [200, 301, 302]
                
                self.results["ai_agents"][name] = {
                    "status": "✅ Running" if success else "❌ Failed",
                    "response_code": response.status_code,
                    "health_check": success
                }
                
                # Test additional endpoint if available
                if success and config.get("test_endpoint"):
                    try:
                        test_url = config["url"].replace("/health", "") + config["test_endpoint"]
                        test_response = await self.client.get(test_url)
                        self.results["ai_agents"][name]["test_endpoint"] = test_response.status_code == 200
                    except:
                        self.results["ai_agents"][name]["test_endpoint"] = False
                
                print(f"  {name}: {'✅' if success else '❌'}")
                
            except Exception as e:
                self.results["ai_agents"][name] = {
                    "status": "❌ Unreachable",
                    "error": str(e)[:100]
                }
                print(f"  {name}: ❌ ({str(e)[:50]})")
        
        return self.results["ai_agents"]
    
    async def test_ollama_models(self) -> Dict:
        """Test Ollama model availability and functionality"""
        print("\n🧠 Testing Ollama Models...")
        
        try:
            # Get available models
            response = await self.client.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                self.results["model_tests"]["error"] = "Failed to get model list"
                return self.results["model_tests"]
            
            models = response.json().get("models", [])
            self.results["model_tests"]["available_models"] = [m["name"] for m in models]
            self.results["model_tests"]["model_count"] = len(models)
            
            print(f"  Found {len(models)} models")
            
            # Test primary models
            required_models = [
                "deepseek-r1:8b",
                "qwen2.5:3b",
                "codellama:7b",
                "nomic-embed-text"
            ]
            
            available_model_names = [m["name"] for m in models]
            for model in required_models:
                if model in available_model_names:
                    # Test model generation
                    try:
                        test_response = await self.client.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": model,
                                "prompt": "Say 'Hello, SutazAI!'",
                                "stream": False
                            }
                        )
                        success = test_response.status_code == 200
                        self.results["model_tests"][model] = "✅ Working" if success else "❌ Failed"
                        print(f"  {model}: {'✅' if success else '❌'}")
                    except:
                        self.results["model_tests"][model] = "❌ Error"
                        print(f"  {model}: ❌")
                else:
                    self.results["model_tests"][model] = "⚠️ Not Found"
                    print(f"  {model}: ⚠️ Not Found")
            
        except Exception as e:
            self.results["model_tests"]["error"] = str(e)
            print(f"  Error testing models: {str(e)}")
        
        return self.results["model_tests"]
    
    async def test_litellm_proxy(self) -> Dict:
        """Test LiteLLM proxy OpenAI compatibility"""
        print("\n🔄 Testing LiteLLM Proxy...")
        
        try:
            # Test chat completion endpoint
            response = await self.client.post(
                "http://localhost:4000/v1/chat/completions",
                headers={
                    "Authorization": "Bearer sk-local",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4",
                    "messages": [
                        {"role": "user", "content": "Say 'LiteLLM proxy working!'"}
                    ],
                    "max_tokens": 50
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    self.results["integration_tests"]["litellm_proxy"] = {
                        "status": "✅ Working",
                        "model_mapping": "gpt-4 → deepseek-r1:8b",
                        "response": data["choices"][0]["message"]["content"][:50]
                    }
                    print("  ✅ OpenAI API compatibility working")
                else:
                    self.results["integration_tests"]["litellm_proxy"] = {
                        "status": "❌ Invalid Response"
                    }
                    print("  ❌ Invalid response format")
            else:
                self.results["integration_tests"]["litellm_proxy"] = {
                    "status": "❌ Failed",
                    "error": response.status_code
                }
                print(f"  ❌ Failed with status {response.status_code}")
                
        except Exception as e:
            self.results["integration_tests"]["litellm_proxy"] = {
                "status": "❌ Error",
                "error": str(e)
            }
            print(f"  ❌ Error: {str(e)[:50]}")
    
    async def test_service_orchestration(self) -> Dict:
        """Test service hub orchestration capabilities"""
        print("\n🎭 Testing Service Orchestration...")
        
        try:
            # Test service listing
            response = await self.client.get("http://localhost:8114/services")
            if response.status_code == 200:
                services = response.json()
                self.results["integration_tests"]["service_hub"] = {
                    "status": "✅ Working",
                    "registered_services": services.get("total", 0),
                    "services": services.get("services", [])[:5]  # First 5 services
                }
                print(f"  ✅ Service Hub managing {services.get('total', 0)} services")
            
            # Test orchestration
            orchestration_response = await self.client.post(
                "http://localhost:8114/orchestrate",
                json={
                    "task_type": "analysis",
                    "task_data": {"query": "Test orchestration"},
                    "agents": ["crewai", "autogen"]
                }
            )
            
            if orchestration_response.status_code == 200:
                self.results["integration_tests"]["orchestration"] = {
                    "status": "✅ Working",
                    "test": "Multi-agent orchestration successful"
                }
                print("  ✅ Multi-agent orchestration working")
            
        except Exception as e:
            self.results["integration_tests"]["service_hub"] = {
                "status": "❌ Error",
                "error": str(e)
            }
            print(f"  ❌ Error: {str(e)[:50]}")
    
    async def test_vector_databases(self) -> Dict:
        """Test vector database functionality"""
        print("\n📊 Testing Vector Databases...")
        
        # Test ChromaDB
        try:
            response = await self.client.get("http://localhost:8001/api/v1/collections")
            if response.status_code == 200:
                self.results["integration_tests"]["chromadb"] = "✅ Working"
                print("  ✅ ChromaDB")
            else:
                self.results["integration_tests"]["chromadb"] = "❌ Failed"
                print("  ❌ ChromaDB")
        except:
            self.results["integration_tests"]["chromadb"] = "❌ Unreachable"
            print("  ❌ ChromaDB (unreachable)")
        
        # Test Qdrant
        try:
            response = await self.client.get("http://localhost:6333/collections")
            if response.status_code == 200:
                self.results["integration_tests"]["qdrant"] = "✅ Working"
                print("  ✅ Qdrant")
            else:
                self.results["integration_tests"]["qdrant"] = "❌ Failed"
                print("  ❌ Qdrant")
        except:
            self.results["integration_tests"]["qdrant"] = "❌ Unreachable"
            print("  ❌ Qdrant (unreachable)")
    
    async def test_monitoring_stack(self) -> Dict:
        """Test monitoring services"""
        print("\n📈 Testing Monitoring Stack...")
        
        monitoring = {
            "Prometheus": "http://localhost:9090/-/healthy",
            "Grafana": "http://localhost:3000/api/health",
            "Loki": "http://localhost:3100/ready"
        }
        
        for name, url in monitoring.items():
            try:
                response = await self.client.get(url)
                success = response.status_code in [200, 204]
                self.results["integration_tests"][name.lower()] = "✅ Working" if success else "❌ Failed"
                print(f"  {name}: {'✅' if success else '❌'}")
            except:
                self.results["integration_tests"][name.lower()] = "❌ Unreachable"
                print(f"  {name}: ❌")
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("\n" + "="*60)
        report.append("🔍 SutazAI AGI/ASI System Test Report")
        report.append("="*60)
        report.append(f"Timestamp: {self.results['timestamp']}")
        
        # Core Services Summary
        core_healthy = sum(1 for s in self.results["core_services"].values() 
                          if "✅" in str(s.get("status", "")))
        report.append(f"\n📦 Core Services: {core_healthy}/{len(self.results['core_services'])}")
        
        # AI Agents Summary
        agents_running = sum(1 for s in self.results["ai_agents"].values() 
                            if "✅" in str(s.get("status", "")))
        report.append(f"🤖 AI Agents: {agents_running}/{len(self.results['ai_agents'])}")
        
        # Models Summary
        model_count = self.results["model_tests"].get("model_count", 0)
        report.append(f"🧠 Ollama Models: {model_count} available")
        
        # Integration Tests
        integrations_working = sum(1 for k, v in self.results["integration_tests"].items() 
                                  if "✅" in str(v))
        report.append(f"🔗 Integrations: {integrations_working}/{len(self.results['integration_tests'])}")
        
        # Detailed Results
        report.append("\n" + "-"*60)
        report.append("Detailed Results:")
        report.append("-"*60)
        
        # Failed Services
        failed_services = []
        for category in ["core_services", "ai_agents"]:
            for name, status in self.results[category].items():
                if "❌" in str(status.get("status", "")):
                    failed_services.append(f"  - {name}: {status.get('error', 'Failed')}")
        
        if failed_services:
            report.append("\n❌ Failed Services:")
            report.extend(failed_services)
        else:
            report.append("\n✅ All services healthy!")
        
        # Save detailed results
        with open("test_results_detailed.json", "w") as f:
            json.dump(self.results, f, indent=2)
        report.append(f"\n📝 Detailed results saved to: test_results_detailed.json")
        
        # Overall Status
        total_services = len(self.results["core_services"]) + len(self.results["ai_agents"])
        total_healthy = core_healthy + agents_running
        health_percentage = (total_healthy / total_services * 100) if total_services > 0 else 0
        
        report.append("\n" + "="*60)
        if health_percentage >= 80:
            report.append(f"🟢 SYSTEM STATUS: HEALTHY ({health_percentage:.1f}%)")
        elif health_percentage >= 60:
            report.append(f"🟡 SYSTEM STATUS: PARTIAL ({health_percentage:.1f}%)")
        else:
            report.append(f"🔴 SYSTEM STATUS: CRITICAL ({health_percentage:.1f}%)")
        report.append("="*60)
        
        return "\n".join(report)
    
    async def run_all_tests(self):
        """Run all system tests"""
        print("="*60)
        print("🚀 Starting SutazAI AGI/ASI System Tests")
        print("="*60)
        
        # Run tests in sequence to avoid overwhelming the system
        await self.test_core_services()
        await asyncio.sleep(2)  # Brief pause between test categories
        
        await self.test_ai_agents()
        await asyncio.sleep(2)
        
        await self.test_ollama_models()
        await asyncio.sleep(1)
        
        await self.test_litellm_proxy()
        await asyncio.sleep(1)
        
        await self.test_service_orchestration()
        await asyncio.sleep(1)
        
        await self.test_vector_databases()
        await asyncio.sleep(1)
        
        await self.test_monitoring_stack()
        
        # Generate and display report
        report = self.generate_report()
        print(report)
        
        # Cleanup
        await self.client.aclose()
        
        # Return exit code based on health
        total_services = len(self.results["core_services"]) + len(self.results["ai_agents"])
        healthy_services = sum(1 for cat in ["core_services", "ai_agents"] 
                             for s in self.results[cat].values() 
                             if "✅" in str(s.get("status", "")))
        
        return 0 if healthy_services / total_services >= 0.8 else 1

async def main():
    tester = AGISystemTester()
    return await tester.run_all_tests()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)