#!/usr/bin/env python3
"""
SutazAI AGI/ASI System Validation Script
Validates all components are properly configured and running
"""

import httpx
import asyncio
import json
from typing import Dict, List, Tuple
from datetime import datetime

class SystemValidator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "core_services": {},
            "ai_agents": {},
            "models": {},
            "features": {},
            "summary": {}
        }
    
    async def validate_core_services(self) -> Dict:
        """Validate core infrastructure services"""
        print("🔍 Validating Core Services...")
        
        services = {
            "Backend API": "http://localhost:8000/health",
            "Frontend UI": "http://localhost:8501",
            "PostgreSQL": "http://localhost:5432",  # Check via backend
            "Redis": "http://localhost:6379",  # Check via backend
            "ChromaDB": "http://localhost:8001/api/v1/heartbeat",
            "Qdrant": "http://localhost:6333/healthz",
            "Ollama": "http://localhost:11434/api/tags",
            "Neo4j": "http://localhost:7474"
        }
        
        for name, url in services.items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    self.results["core_services"][name] = {
                        "status": "✅ Healthy" if response.status_code == 200 else "⚠️ Unhealthy",
                        "response_code": response.status_code
                    }
            except Exception as e:
                self.results["core_services"][name] = {
                    "status": "❌ Unreachable",
                    "error": str(e)[:50]
                }
        
        return self.results["core_services"]
    
    async def validate_ai_agents(self) -> Dict:
        """Validate AI agent services"""
        print("\n🤖 Validating AI Agents...")
        
        agents = {
            # Core Agents
            "CrewAI": "http://localhost:8096/health",
            "Aider": "http://localhost:8095/health",
            "GPT-Engineer": "http://localhost:8097/health",
            "LlamaIndex": "http://localhost:8098/health",
            "AutoGPT": "http://localhost:8080/health",
            "LangFlow": "http://localhost:8090",
            "FlowiseAI": "http://localhost:8099",
            
            # Enhanced Agents
            "LocalAGI": "http://localhost:8103/health",
            "AutoGen": "http://localhost:8104/health",
            "AgentZero": "http://localhost:8105/health",
            "BigAGI": "http://localhost:8106",
            "Dify": "http://localhost:8107",
            "OpenDevin": "http://localhost:8108",
            "FinRobot": "http://localhost:8109/health",
            "RealtimeSTT": "http://localhost:8110/health",
            "Code Improver": "http://localhost:8113/health",
            "Service Hub": "http://localhost:8114/health",
            
            # ML Frameworks
            "PyTorch": "http://localhost:8089",
            "TensorFlow": "http://localhost:8088",
            "JAX": "http://localhost:8089"
        }
        
        for name, url in agents.items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    self.results["ai_agents"][name] = {
                        "status": "✅ Running" if response.status_code == 200 else "⚠️ Issue",
                        "response_code": response.status_code
                    }
            except Exception as e:
                self.results["ai_agents"][name] = {
                    "status": "❌ Not Running",
                    "error": str(e)[:50]
                }
        
        return self.results["ai_agents"]
    
    async def validate_models(self) -> Dict:
        """Validate available AI models"""
        print("\n🧠 Validating AI Models...")
        
        try:
            async with httpx.AsyncClient() as client:
                # Check Ollama models
                response = await client.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self.results["models"]["ollama"] = {
                        "count": len(models),
                        "models": [m["name"] for m in models]
                    }
                
                response = await client.get("http://localhost:4000/model/info")
                if response.status_code == 200:
                        "status": "✅ Proxy Active",
                        "models_mapped": ["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"]
                    }
        except Exception as e:
            self.results["models"]["error"] = str(e)
        
        return self.results["models"]
    
    async def validate_features(self) -> Dict:
        """Validate system features"""
        print("\n🚀 Validating System Features...")
        
        features = {
            "Service Communication Hub": {
                "check": "http://localhost:8114/services",
                "expected": "services"
            },
            "Code Improvement System": {
                "check": "http://localhost:8113/status",
                "expected": "status"
            },
            "Multi-Agent Orchestration": {
                "check": "http://localhost:8114/health",
                "expected": "services"
            },
            "Monitoring Stack": {
                "check": "http://localhost:9090/-/healthy",
                "expected": "Prometheus"
            },
            "Workflow Automation": {
                "check": "http://localhost:5678",
                "expected": "n8n"
            }
        }
        
        for feature, config in features.items():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(config["check"], timeout=5.0)
                    self.results["features"][feature] = "✅ Active" if response.status_code == 200 else "⚠️ Issue"
            except:
                self.results["features"][feature] = "❌ Inactive"
        
        return self.results["features"]
    
    def generate_summary(self):
        """Generate validation summary"""
        # Count statuses
        core_healthy = sum(1 for s in self.results["core_services"].values() 
                          if "✅" in str(s.get("status", "")))
        agents_running = sum(1 for s in self.results["ai_agents"].values() 
                            if "✅" in str(s.get("status", "")))
        features_active = sum(1 for s in self.results["features"].values() 
                             if "✅" in s)
        
        self.results["summary"] = {
            "core_services": f"{core_healthy}/{len(self.results['core_services'])}",
            "ai_agents": f"{agents_running}/{len(self.results['ai_agents'])}",
            "models": len(self.results["models"].get("ollama", {}).get("models", [])),
            "features": f"{features_active}/{len(self.results['features'])}",
            "overall_health": "🟢 Healthy" if core_healthy >= 5 and agents_running >= 10 else "🟡 Partial"
        }
    
    async def validate_all(self):
        """Run all validations"""
        print("═" * 60)
        print("🔍 SutazAI AGI/ASI System Validation")
        print("═" * 60)
        
        await self.validate_core_services()
        await self.validate_ai_agents()
        await self.validate_models()
        await self.validate_features()
        self.generate_summary()
        
        # Print results
        print("\n📊 Validation Results:")
        print("─" * 60)
        
        print("\n🏗️ Core Services:")
        for service, status in self.results["core_services"].items():
            print(f"  {service}: {status['status']}")
        
        print(f"\n🤖 AI Agents ({self.results['summary']['ai_agents']} running):")
        for agent, status in self.results["ai_agents"].items():
            if "✅" in status['status']:
                print(f"  {agent}: {status['status']}")
        
        print(f"\n🧠 Models Available: {self.results['summary']['models']}")
        if "ollama" in self.results["models"]:
            for model in self.results["models"]["ollama"]["models"][:5]:
                print(f"  • {model}")
        
        print(f"\n🚀 Features ({self.results['summary']['features']} active):")
        for feature, status in self.results["features"].items():
            print(f"  {feature}: {status}")
        
        print("\n" + "═" * 60)
        print(f"Overall System Health: {self.results['summary']['overall_health']}")
        print("═" * 60)
        
        # Save detailed results
        with open("system_validation_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\n📝 Detailed results saved to: system_validation_results.json")
        
        return self.results

async def main():
    validator = SystemValidator()
    results = await validator.validate_all()
    
    # Return exit code based on health
    if "🟢" in results["summary"]["overall_health"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)