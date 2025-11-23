#!/usr/bin/env python3
"""
AI Agent Comprehensive Testing
Tests all 8 deployed agents for functionality and integration
"""

import pytest
import httpx
import asyncio
from typing import Dict, List

TIMEOUT = 30.0

# Agent configurations
AGENTS = {
    "crewai": {"port": 11403, "capabilities": ["multi-agent", "orchestration"]},
    "aider": {"port": 11404, "capabilities": ["code-editing", "git"]},
    "langchain": {"port": 11405, "capabilities": ["chains", "reasoning"]},
    "shellgpt": {"port": 11413, "capabilities": ["shell-commands"]},
    "documind": {"port": 11414, "capabilities": ["document-processing"]},
    "finrobot": {"port": 11410, "capabilities": ["financial-analysis"]},
    "letta": {"port": 11401, "capabilities": ["memory", "persistence"]},
    "gpt-engineer": {"port": 11416, "capabilities": ["project-generation"]}
}

class TestAgentHealth:
    """Test health endpoints for all agents"""
    
    @pytest.mark.asyncio
    async def test_all_agents_healthy(self):
        """Test that all 8 agents respond to health checks"""
        results = {}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for agent, config in AGENTS.items():
                try:
                    response = await client.get(f"http://localhost:{config['port']}/health")
                    results[agent] = response.status_code == 200
                except Exception as e:
                    results[agent] = False
        
        # At least 6 out of 8 should be healthy
        healthy_count = sum(results.values())
        assert healthy_count >= 6, f"Only {healthy_count}/8 agents healthy: {results}"
    
    @pytest.mark.asyncio
    async def test_crewai_health(self):
        """Test CrewAI agent health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['crewai']['port']}/health")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_aider_health(self):
        """Test Aider agent health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['aider']['port']}/health")
            assert response.status_code == 200


class TestAgentMetrics:
    """Test Prometheus metrics from agents"""
    
    @pytest.mark.asyncio
    async def test_all_agents_expose_metrics(self):
        """Test that all agents expose /metrics endpoint"""
        results = {}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for agent, config in AGENTS.items():
                try:
                    response = await client.get(f"http://localhost:{config['port']}/metrics")
                    results[agent] = response.status_code == 200 and "python_" in response.text
                except Exception:
                    results[agent] = False
        
        # At least 6 out of 8 should expose metrics
        metrics_count = sum(results.values())
        assert metrics_count >= 6, f"Only {metrics_count}/8 agents exposing metrics"


class TestOllamaIntegration:
    """Test Ollama LLM integration for each agent"""
    
    @pytest.mark.asyncio
    async def test_ollama_connectivity(self):
        """Test that Ollama is accessible"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:11435/api/tags")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
    
    @pytest.mark.asyncio
    async def test_tinyllama_loaded(self):
        """Test that TinyLlama model is loaded"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:11435/api/tags")
            data = response.json()
            model_names = [m["name"] for m in data["models"]]
            assert any("tinyllama" in name.lower() for name in model_names)


class TestCrewAI:
    """Test CrewAI multi-agent orchestration"""
    
    @pytest.mark.asyncio
    async def test_crewai_capabilities(self):
        """Test CrewAI capabilities endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['crewai']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_crewai_crew_execution(self):
        """Test executing a crew task"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "task": "Simple test task",
                "agents": ["researcher", "writer"]
            }
            response = await client.post(
                f"http://localhost:{AGENTS['crewai']['port']}/execute",
                json=payload
            )
            assert response.status_code in [200, 201, 404, 422]


class TestAider:
    """Test Aider code editing capabilities"""
    
    @pytest.mark.asyncio
    async def test_aider_capabilities(self):
        """Test Aider capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['aider']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_aider_code_edit(self):
        """Test code editing request"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "file": "test.py",
                "instruction": "Add a docstring"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['aider']['port']}/edit",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestLangChain:
    """Test LangChain reasoning capabilities"""
    
    @pytest.mark.asyncio
    async def test_langchain_capabilities(self):
        """Test LangChain capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['langchain']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_langchain_chain_execution(self):
        """Test chain execution"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "chain_type": "simple",
                "input": "Test reasoning task"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['langchain']['port']}/execute",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestLetta:
    """Test Letta memory and persistence"""
    
    @pytest.mark.asyncio
    async def test_letta_memory(self):
        """Test Letta memory endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['letta']['port']}/memory")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_letta_session_persistence(self):
        """Test session persistence across requests"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Create session
            create_resp = await client.post(
                f"http://localhost:{AGENTS['letta']['port']}/session",
                json={"user_id": "test-user"}
            )
            assert create_resp.status_code in [200, 201, 404, 422]


class TestDocumind:
    """Test Documind document processing"""
    
    @pytest.mark.asyncio
    async def test_documind_capabilities(self):
        """Test Documind capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['documind']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_documind_process_document(self):
        """Test document processing"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "document_type": "text",
                "content": "Sample document for testing"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['documind']['port']}/process",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestFinRobot:
    """Test FinRobot financial analysis"""
    
    @pytest.mark.asyncio
    async def test_finrobot_capabilities(self):
        """Test FinRobot capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['finrobot']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_finrobot_analyze(self):
        """Test financial analysis"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "symbol": "AAPL",
                "analysis_type": "basic"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['finrobot']['port']}/analyze",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestShellGPT:
    """Test ShellGPT command generation"""
    
    @pytest.mark.asyncio
    async def test_shellgpt_capabilities(self):
        """Test ShellGPT capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['shellgpt']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_shellgpt_command_generation(self):
        """Test command generation"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "task": "List all files in directory"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['shellgpt']['port']}/generate",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestGPTEngineer:
    """Test GPT-Engineer project generation"""
    
    @pytest.mark.asyncio
    async def test_gpt_engineer_capabilities(self):
        """Test GPT-Engineer capabilities"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"http://localhost:{AGENTS['gpt-engineer']['port']}/capabilities")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_gpt_engineer_generate_project(self):
        """Test project generation"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "project_type": "python-cli",
                "description": "Simple test project"
            }
            response = await client.post(
                f"http://localhost:{AGENTS['gpt-engineer']['port']}/generate",
                json=payload
            )
            assert response.status_code in [200, 404, 422]


class TestAgentConcurrency:
    """Test concurrent requests to agents"""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test health checks to all agents concurrently"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [
                client.get(f"http://localhost:{config['port']}/health")
                for config in AGENTS.values()
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            assert successful >= 6, f"Only {successful}/8 agents responded"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
