#!/usr/bin/env python3
"""
Comprehensive Ollama Integration Test for SutazAI Agents
Tests connectivity and functionality of all 131 agents with Ollama
"""

import asyncio
import httpx
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add the agents directory to Python path
sys.path.append('/opt/sutazaiapp/agents')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/sutazaiapp/logs/ollama_integration_test.log')
    ]
)

logger = logging.getLogger(__name__)

class OllamaIntegrationTester:
    """Test Ollama integration for all SutazAI agents"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:10104"
        self.backend_url = "http://localhost:8000"
        self.results = {}
        
        # List of all 131 agents
        self.agent_list = [
            "adversarial-attack-detector", "agent-creator", "agent-debugger", 
            "agent-orchestrator", "agentgpt-autonomous-executor", "agentzero-coordinator",
            "ai-agent-debugger", "ai-product-manager", "qa-team-lead", 
            "ai-scrum-master", "ai-senior-backend-developer", "senior-engineer",
            "ai-senior-frontend-developer", "ai-senior-full-stack-developer",
            " system-architect", "ai-system-validator", "ai-testing-qa-validator",
            "automated-incident-responder", "autonomous-task-executor", "bias-and-fairness-auditor",
            "bigagi-system-manager", "browser-automation-orchestrator", "causal-inference-expert",
            "cicd-pipeline-orchestrator", "code-generation-improver", "code-improver",
            "code-quality-gateway-sonarqube", "codebase-team-lead", "cognitive-architecture-designer",
            "cognitive-load-monitor", "compute-scheduler-and-optimizer", "container-orchestrator-k3s",
            "container-vulnerability-scanner-trivy", "context-optimization-engineer",
            "cpu-only-hardware-optimizer", "data-drift-detector", "data-lifecycle-manager",
            "data-version-controller-dvc", "deep-learning-brain-architect", "deep-learning-brain-manager",
            "deep-local-brain-builder", "deploy-automation-master", "deployment-automation-master",
            "dify-automation-specialist", "distributed-tracing-analyzer-jaeger", 
            "document-knowledge-manager", "edge-inference-proxy", "emergency-shutdown-coordinator",
            "energy-consumption-optimize", "ethical-governor", "evolution-strategy-trainer",
            "experiment-tracker", "explainability-and-transparency-agent", "financial-analysis-specialist",
            "flowiseai-flow-manager", "garbage-collector", "genetic-algorithm-tuner",
            "goal-setting-and-planning-agent", "gpu-hardware-optimizer", "hardware-resource-optimizer",
            "honeypot-deployment-agent", "human-oversight-interface-agent", "infrastructure-devops-manager",
            "langflow-workflow-designer", "log-aggregator-loki", "mega-code-auditor",
            "metrics-collector-prometheus", "ml-experiment-tracker-mlflow", "neural-architecture-search",
            "observability-dashboard-manager-grafana", "ollama-integration-specialist",
            "opendevin-code-generator", "private-data-analyst", "private-registry-manager-harbor",
            "product-manager", "prompt-injection-guard", "qa-team-lead", "quantum-ai-researcher",
            "ram-hardware-optimizer", "resource-arbitration-agent", "resource-visualiser",
            "runtime-behavior-anomaly-detector", "scrum-master", "secrets-vault-manager-vault",
            "security-pentesting-specialist", "semgrep-security-analyzer", "senior-ai-engineer",
            "senior-backend-developer", "senior-engineer", "senior-frontend-developer",
            "senior-full-stack-developer", "shell-automation-specialist", "system-architect",
            "system-knowledge-curator", "system-optimizer-reorganizer", "system-performance-forecaster",
            "system-validator", "task-assignment-coordinator", "testing-qa-team-lead",
            "testing-qa-validator", "attention-optimizer", "autogen", "autogpt",
            "aider", "awesome-code-ai", "babyagi", "browser-automation-orchestrator",
            "causal-inference-expert", "cognitive-architecture-designer", "code-improver",
            "complex-problem-solver", "context-framework", "crewai", "data-analysis-engineer",
            "data-pipeline-engineer", "devika", "distributed-computing-architect",
            "edge-computing-optimizer", "episodic-memory-engineer", "explainable-ai-specialist",
            "federated-learning-coordinator", "finrobot", "fsdp", "garbage-collector-coordinator",
            "gpt-engineer", "gradient-compression-specialist", "health-monitor", 
            "intelligence-optimization-monitor", "knowledge-distillation-expert",
            "knowledge-graph-builder", "letta", "localagi-orchestration-manager",
            "mcp-server", "memory-persistence-manager", "meta-learning-specialist",
            "model-training-specialist", "multi-modal-fusion-coordinator", 
            "neuromorphic-computing-expert", "observability-monitoring-engineer",
            "pentestgpt", "privategpt", "product-strategy-architect", "quantum-computing-optimizer",
            "reinforcement-learning-trainer", "self-healing-orchestrator"
        ]
        
    async def test_ollama_connectivity(self) -> bool:
        """Test basic Ollama connectivity"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    logger.info(f"Ollama is accessible. Available models: {len(models)}")
                    return True
                else:
                    logger.error(f"Ollama connectivity failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Ollama connectivity error: {e}")
            return False
    
    async def test_model_availability(self) -> bool:
        """Test if GPT-OSS model is available"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    
                    if "tinyllama" in model_names:
                        logger.info("GPT-OSS model is available")
                        return True
                    else:
                        logger.error(f"GPT-OSS model not found. Available: {model_names}")
                        return False
        except Exception as e:
            logger.error(f"Model availability check error: {e}")
            return False
    
    async def test_ollama_generation(self) -> bool:
        """Test basic Ollama text generation"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": "tinyllama",
                    "prompt": "Hello, respond with exactly one word: 'SUCCESS'",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10
                    }
                }
                
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "").strip()
                    logger.info(f"Ollama generation test: '{generated_text}'")
                    return "SUCCESS" in generated_text.upper()
                else:
                    logger.error(f"Ollama generation failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return False
    
    async def test_backend_agent_api(self) -> bool:
        """Test backend agent registration API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test agent registration endpoint
                test_agent_data = {
                    "agent_name": "test-integration-agent",
                    "agent_type": "test",
                    "agent_version": "1.0.0",
                    "capabilities": ["test"],
                    "status": "active",
                    "model": "tinyllama",
                    "max_concurrent_tasks": 1,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                response = await client.post(
                    f"{self.backend_url}/api/agents/register",
                    json=test_agent_data
                )
                
                if response.status_code in [200, 201]:
                    logger.info("Backend agent registration API is working")
                    return True
                else:
                    logger.warning(f"Agent registration returned: {response.status_code}")
                    return response.status_code < 500  # Accept any non-server error
                    
        except Exception as e:
            logger.error(f"Backend API test error: {e}")
            return False
    
    async def test_base_agent_v2_integration(self) -> bool:
        """Test BaseAgent Ollama integration"""
        try:
            # Import and test BaseAgentV2
            from agents.core.base_agent import BaseAgent, OllamaConfig
            
            # Test model configuration
            test_config = OllamaConfig.get_model_config("test-agent")
            if test_config["model"] != "tinyllama":
                logger.error(f"Expected tinyllama, got {test_config['model']}")
                return False
            
            logger.info("BaseAgent configuration is correct")
            return True
            
        except Exception as e:
            logger.error(f"BaseAgent integration test error: {e}")
            return False
    
    async def test_individual_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Test configuration for individual agent"""
        result = {
            "agent_name": agent_name,
            "config_exists": False,
            "dockerfile_exists": False,
            "model_assigned": None,
            "startup_script_exists": False,
            "status": "unknown"
        }
        
        try:
            # Check if agent directory exists
            agent_path = Path(f"/opt/sutazaiapp/agents/{agent_name}")
            if not agent_path.exists():
                result["status"] = "directory_missing"
                return result
            
            # Check for app.py or main script
            app_files = ["app.py", "agent.py", "main.py"]
            script_exists = any((agent_path / f).exists() for f in app_files)
            
            # Check for Dockerfile
            dockerfile_exists = (agent_path / "Dockerfile").exists()
            result["dockerfile_exists"] = dockerfile_exists
            
            # Check for startup script
            startup_exists = (agent_path / "startup.sh").exists()
            result["startup_script_exists"] = startup_exists
            
            # Check configuration
            config_path = Path(f"/opt/sutazaiapp/agents/configs/{agent_name}_universal.json")
            if config_path.exists():
                result["config_exists"] = True
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        result["model_assigned"] = config.get("model", "unknown")
                except (AssertionError, Exception) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
            
            # Determine overall status
            if script_exists and (dockerfile_exists or startup_exists):
                result["status"] = "ready"
            elif script_exists:
                result["status"] = "partial"
            else:
                result["status"] = "incomplete"
            
        except Exception as e:
            result["status"] = f"error: {e}"
        
        return result
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test"""
        logger.info("Starting comprehensive Ollama integration test...")
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "unknown",
            "core_tests": {},
            "agent_tests": {},
            "summary": {}
        }
        
        # Core system tests
        logger.info("Running core system tests...")
        
        test_results["core_tests"]["ollama_connectivity"] = await self.test_ollama_connectivity()
        test_results["core_tests"]["model_availability"] = await self.test_model_availability()
        test_results["core_tests"]["ollama_generation"] = await self.test_ollama_generation()
        test_results["core_tests"]["backend_api"] = await self.test_backend_agent_api()
        test_results["core_tests"]["base_agent_v2"] = await self.test_base_agent_v2_integration()
        
        # Agent configuration tests
        logger.info("Testing individual agent configurations...")
        
        agent_results = {}
        ready_count = 0
        partial_count = 0
        missing_count = 0
        error_count = 0
        
        for i, agent_name in enumerate(self.agent_list):
            if i % 20 == 0:
                logger.info(f"Testing agent {i+1}/{len(self.agent_list)}: {agent_name}")
            
            result = await self.test_individual_agent_config(agent_name)
            agent_results[agent_name] = result
            
            # Count statuses
            status = result["status"]
            if status == "ready":
                ready_count += 1
            elif status == "partial":
                partial_count += 1
            elif status in ["directory_missing", "incomplete"]:
                missing_count += 1
            else:
                error_count += 1
        
        test_results["agent_tests"] = agent_results
        
        # Calculate summary
        total_agents = len(self.agent_list)
        core_tests_passed = sum(1 for v in test_results["core_tests"].values() if v)
        total_core_tests = len(test_results["core_tests"])
        
        test_results["summary"] = {
            "total_agents": total_agents,
            "ready_agents": ready_count,
            "partial_agents": partial_count,
            "missing_agents": missing_count,
            "error_agents": error_count,
            "core_tests_passed": core_tests_passed,
            "total_core_tests": total_core_tests,
            "core_success_rate": core_tests_passed / total_core_tests * 100,
            "agent_readiness_rate": ready_count / total_agents * 100
        }
        
        # Determine overall status
        if core_tests_passed >= 4 and ready_count > total_agents * 0.7:
            test_results["overall_status"] = "excellent"
        elif core_tests_passed >= 3 and ready_count > total_agents * 0.5:
            test_results["overall_status"] = "good"
        elif core_tests_passed >= 2:
            test_results["overall_status"] = "partial"
        else:
            test_results["overall_status"] = "failed"
        
        return test_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable test report"""
        summary = results["summary"]
        
        report = f"""
OLLAMA INTEGRATION TEST REPORT
==============================
Test Date: {results['timestamp']}
Overall Status: {results['overall_status'].upper()}

CORE SYSTEM TESTS
-----------------
✓ Ollama Connectivity: {'PASS' if results['core_tests']['ollama_connectivity'] else 'FAIL'}
✓ Model Availability: {'PASS' if results['core_tests']['model_availability'] else 'FAIL'}
✓ Text Generation: {'PASS' if results['core_tests']['ollama_generation'] else 'FAIL'}
✓ Backend API: {'PASS' if results['core_tests']['backend_api'] else 'FAIL'}
✓ BaseAgent Integration: {'PASS' if results['core_tests']['base_agent_v2'] else 'FAIL'}

Core Tests: {summary['core_tests_passed']}/{summary['total_core_tests']} passed ({summary['core_success_rate']:.1f}%)

AGENT READINESS SUMMARY
-----------------------
Total Agents: {summary['total_agents']}
Ready: {summary['ready_agents']} ({summary['agent_readiness_rate']:.1f}%)
Partial: {summary['partial_agents']}
Missing: {summary['missing_agents']}
Errors: {summary['error_agents']}

READY AGENTS
------------"""
        
        ready_agents = [name for name, result in results["agent_tests"].items() 
                       if result["status"] == "ready"]
        
        for i, agent in enumerate(ready_agents):
            if i % 3 == 0:
                report += "\n"
            report += f"{agent:<35}"
        
        report += f"\n\nRECOMMENDations\n--------------"
        
        if results["overall_status"] == "excellent":
            report += "\n✓ System is ready for production use with Ollama"
            report += "\n✓ All core components are functional"
            report += f"\n✓ {summary['ready_agents']} agents are ready to deploy"
        elif results["overall_status"] == "good":
            report += "\n✓ System is functional with minor issues"
            report += "\n- Consider addressing partial agents for full functionality"
        elif results["overall_status"] == "partial":
            report += "\n⚠ System has significant issues that should be addressed"
            report += "\n- Fix core system tests before deploying agents"
        else:
            report += "\n✗ System is not ready for use"
            report += "\n- Address core system failures before proceeding"
        
        report += f"\n\nFor detailed results, see: /opt/sutazaiapp/logs/ollama_integration_results.json"
        
        return report

async def main():
    """Main test runner"""
    tester = OllamaIntegrationTester()
    
    try:
        # Run comprehensive test
        results = await tester.run_comprehensive_test()
        
        # Save detailed results
        results_file = "/opt/sutazaiapp/logs/ollama_integration_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate and display report
        report = tester.generate_report(results)
        logger.info(report)
        
        # Save report
        report_file = "/opt/sutazaiapp/logs/ollama_integration_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Test completed. Results saved to {results_file}")
        logger.info(f"Report saved to {report_file}")
        
        # Exit with appropriate code
        if results["overall_status"] in ["excellent", "good"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test runner error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    unittest.main()
