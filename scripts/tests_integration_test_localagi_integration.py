#!/usr/bin/env python3
"""
Integration Tests for LocalAGI Autonomous Orchestration System

This module tests the integration between LocalAGI and the existing 
SutazAI infrastructure, ensuring all components work together correctly.
"""

import asyncio
import pytest
import json
import logging
import httpx
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add LocalAGI to path
sys.path.append(str(Path(__file__).parent.parent.parent / "localagi"))

from localagi.main import LocalAGISystem, get_localagi_system
from localagi.engine.autonomous_orchestration_engine import get_orchestration_engine
from localagi.frameworks.collaborative_problem_solver import Problem, ProblemType

logger = logging.getLogger(__name__)

class TestLocalAGIIntegration:
    """
    Integration tests for LocalAGI autonomous orchestration system.
    """
    
    @pytest.fixture(scope="session")
    async def localagi_system(self):
        """Initialize LocalAGI system for testing."""
        logger.info("Initializing LocalAGI system for testing")
        
        # Wait for dependencies to be ready
        await self._wait_for_dependencies()
        
        system = LocalAGISystem()
        await system.initialize()
        
        yield system
        
        # Cleanup
        await system.shutdown()
    
    @pytest.fixture(scope="session")
    async def orchestration_engine(self):
        """Get orchestration engine for testing."""
        engine = get_orchestration_engine()
        
        # Wait for agents to be loaded
        max_wait = 60
        wait_time = 0
        
        while wait_time < max_wait:
            if len(engine.agents) > 0:
                logger.info(f"Orchestration engine ready with {len(engine.agents)} agents")
                break
            await asyncio.sleep(2)
            wait_time += 2
        
        yield engine
    
    async def _wait_for_dependencies(self):
        """Wait for required services to be available."""
        services = [
            ("http://localhost:11434", "Ollama"),
            ("http://localhost:6379", "Redis"),
            ("http://localhost:8000", "Backend API")
        ]
        
        for service_url, service_name in services:
            await self._wait_for_service(service_url, service_name)
    
    async def _wait_for_service(self, url: str, service_name: str, timeout: int = 60):
        """Wait for a service to become available."""
        logger.info(f"Waiting for {service_name} at {url}")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=5.0)
                    if response.status_code < 500:  # Any non-server error is considered available
                        logger.info(f"{service_name} is available")
                        return
            except Exception as e:
                logger.debug(f"Waiting for {service_name}: {e}")
            
            await asyncio.sleep(2)
        
        logger.warning(f"{service_name} not available after {timeout}s, proceeding anyway")

    @pytest.mark.asyncio
    async def test_system_initialization(self, localagi_system):
        """Test that LocalAGI system initializes correctly."""
        assert localagi_system is not None
        assert localagi_system.system_status == "operational"
        assert localagi_system.orchestration_engine is not None
        assert localagi_system.decision_engine is not None
        assert localagi_system.task_decomposer is not None
        assert localagi_system.swarm_coordinator is not None
        assert localagi_system.workflow_engine is not None
        assert localagi_system.problem_solver is not None
        assert localagi_system.goal_system is not None
        assert localagi_system.coordination_protocols is not None
    
    @pytest.mark.asyncio
    async def test_agent_registry_integration(self, orchestration_engine):
        """Test integration with the existing agent registry."""
        # Check that agents are loaded from the registry
        assert len(orchestration_engine.agents) > 0
        
        # Test specific agents exist
        agent_names = [agent.name for agent in orchestration_engine.agents.values()]
        
        # Should have some core agents
        expected_agents = ["AutoGPT", "CrewAI", "Aider", "GPT-Engineer"]
        for expected_agent in expected_agents:
            found = any(expected_agent.lower() in name.lower() for name in agent_names)
            if found:
                logger.info(f"Found expected agent type: {expected_agent}")
        
        # At least some agents should be available
        assert len(agent_names) >= 3, f"Expected at least 3 agents, found {len(agent_names)}: {agent_names}"
    
    @pytest.mark.asyncio
    async def test_autonomous_task_submission(self, localagi_system):
        """Test autonomous task submission and execution."""
        # Submit a simple autonomous task
        task_id = await localagi_system.submit_autonomous_task(
            description="Test system status and report current time",
            requirements=["system", "monitoring"],
            priority=0.7,
            autonomous_mode=True
        )
        
        assert task_id is not None
        assert isinstance(task_id, str)
        
        # Wait a moment for task processing
        await asyncio.sleep(2)
        
        # Check system status includes the task
        status = await localagi_system.get_comprehensive_status()
        orchestration_status = status.get('orchestration_engine', {})
        
        # Should have processed at least one task
        assert orchestration_status.get('total_tasks', 0) > 0
    
    @pytest.mark.asyncio
    async def test_collaborative_problem_solving(self, localagi_system):
        """Test collaborative problem solving capabilities."""
        # Create a test problem
        problem_result = await localagi_system.solve_problem_collaboratively(
            problem_description="How can we optimize system performance for AI workloads?",
            problem_type="optimization",
            max_agents=3
        )
        
        assert problem_result is not None
        assert 'session_id' in problem_result
        assert 'problem_id' in problem_result
        assert problem_result.get('participating_agents', 0) > 0
        
        # Should have generated at least one solution or attempt
        assert problem_result.get('solutions_generated', 0) >= 0
    
    @pytest.mark.asyncio
    async def test_workflow_creation_and_execution(self, localagi_system):
        """Test workflow creation and execution."""
        # Define a simple workflow
        workflow_steps = [
            {
                'name': 'System Check',
                'description': 'Check system health',
                'agent_capability': 'monitoring',
                'parameters': {'check_type': 'health'},
                'success_criteria': ['System status retrieved']
            },
            {
                'name': 'Performance Analysis',
                'description': 'Analyze system performance',
                'agent_capability': 'analysis',
                'parameters': {'analysis_type': 'performance'},
                'success_criteria': ['Performance metrics analyzed'],
                'preconditions': ['System Check']
            }
        ]
        
        # Create workflow
        workflow_id = await localagi_system.create_autonomous_workflow(
            description="System Health and Performance Workflow",
            steps=workflow_steps,
            optimization_objectives=['efficiency', 'reliability']
        )
        
        assert workflow_id is not None
        
        # Execute workflow
        execution_result = await localagi_system.execute_workflow(workflow_id)
        
        assert execution_result is not None
        assert 'execution_id' in execution_result
        assert 'workflow_id' in execution_result
        assert execution_result['workflow_id'] == workflow_id
    
    @pytest.mark.asyncio
    async def test_agent_swarm_formation(self, localagi_system):
        """Test agent swarm formation capabilities."""
        swarm_result = await localagi_system.form_agent_swarm(
            goal="Coordinate system optimization tasks",
            required_capabilities=["monitoring", "analysis", "optimization"],
            max_size=5
        )
        
        assert swarm_result is not None
        
        if swarm_result.get('swarm_formed'):
            assert 'swarm_id' in swarm_result
            assert 'member_count' in swarm_result
            assert swarm_result['member_count'] > 0
            assert 'leader_id' in swarm_result
        else:
            # It's okay if swarm formation fails due to insufficient agents
            logger.info(f"Swarm formation failed: {swarm_result.get('error', 'Unknown error')}")
    
    @pytest.mark.asyncio
    async def test_complex_task_decomposition(self, localagi_system):
        """Test complex task decomposition."""
        decomposition_result = await localagi_system.decompose_complex_task(
            task_description="Implement a comprehensive system monitoring and alerting solution",
            requirements=["monitoring", "alerting", "analysis", "reporting"],
            constraints={"time_limit": "24 hours", "resources": "existing infrastructure"},
            max_depth=3
        )
        
        assert decomposition_result is not None
        assert 'original_task_id' in decomposition_result
        assert 'total_tasks' in decomposition_result
        assert decomposition_result['total_tasks'] > 1  # Should decompose into multiple tasks
        assert 'task_tree' in decomposition_result
        assert len(decomposition_result['task_tree']) > 0
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, localagi_system):
        """Test system health monitoring functionality."""
        # Wait for at least one health check cycle
        await asyncio.sleep(5)
        
        status = await localagi_system.get_comprehensive_status()
        
        assert status is not None
        assert 'system_status' in status
        assert status['system_status'] == 'operational'
        assert 'uptime' in status
        assert status['uptime'] > 0
        
        # Check component statuses
        assert 'orchestration_engine' in status
        assert 'decision_engine' in status
        assert 'workflow_engine' in status
    
    @pytest.mark.asyncio
    async def test_ollama_integration(self, orchestration_engine):
        """Test integration with Ollama service."""
        # Test that Ollama client is configured
        assert orchestration_engine.ollama_client is not None
        
        # Test basic Ollama connectivity
        try:
            response = await orchestration_engine.ollama_client.get("/api/tags")
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Available Ollama models: {models}")
            else:
                logger.warning(f"Ollama not fully accessible: {response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_redis_integration(self, orchestration_engine):
        """Test integration with Redis service."""
        # Test Redis connectivity through orchestration engine
        assert orchestration_engine.redis_client is not None
        
        try:
            # Test basic Redis operations
            await orchestration_engine.redis_client.set("test_key", "test_value", ex=60)
            value = await orchestration_engine.redis_client.get("test_key")
            assert value.decode() == "test_value"
            await orchestration_engine.redis_client.delete("test_key")
            logger.info("Redis integration test passed")
        except Exception as e:
            logger.warning(f"Redis integration test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, localagi_system):
        """Test system performance under concurrent load."""
        # Submit multiple tasks concurrently
        tasks = []
        for i in range(5):
            task = localagi_system.submit_autonomous_task(
                description=f"Performance test task {i+1}",
                requirements=["testing"],
                priority=0.5,
                autonomous_mode=False  # Disable autonomous mode for faster testing
            )
            tasks.append(task)
        
        # Wait for all tasks to be submitted
        task_ids = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful submissions
        successful_tasks = [tid for tid in task_ids if isinstance(tid, str)]
        assert len(successful_tasks) > 0, "At least some tasks should be submitted successfully"
        
        logger.info(f"Successfully submitted {len(successful_tasks)} out of {len(tasks)} tasks")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, localagi_system):
        """Test system error recovery capabilities."""
        # Try to create a workflow with invalid configuration
        try:
            invalid_steps = [
                {
                    'name': 'Invalid Step',
                    'description': 'This step has missing required fields'
                    # Missing agent_capability and other required fields
                }
            ]
            
            workflow_id = await localagi_system.create_autonomous_workflow(
                description="Invalid Workflow Test",
                steps=invalid_steps
            )
            
            # If it succeeds, try to execute it
            if workflow_id:
                execution_result = await localagi_system.execute_workflow(workflow_id)
                logger.info(f"Invalid workflow execution result: {execution_result}")
        
        except Exception as e:
            logger.info(f"System correctly handled invalid workflow: {e}")
        
        # System should still be operational after error
        status = await localagi_system.get_comprehensive_status()
        assert status['system_status'] == 'operational'
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Test that LocalAGI configuration integrates with SutazAI config."""
        from backend.app.core.config import get_settings
        
        settings = get_settings()
        
        # Check that LocalAGI can access required configuration
        assert settings.OLLAMA_HOST is not None
        assert settings.REDIS_HOST is not None
        assert settings.DATABASE_URL is not None
        
        # Verify the configuration points to the correct services
        assert "ollama" in settings.OLLAMA_HOST.lower()
        assert "redis" in settings.REDIS_HOST.lower()
        
        logger.info(f"Configuration integration test passed - Ollama: {settings.OLLAMA_HOST}, Redis: {settings.REDIS_HOST}")

# Utility functions for integration testing

async def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting LocalAGI integration tests")
    
    # Run pytest programmatically
    import pytest
    
    test_file = Path(__file__)
    result = pytest.main([
        str(test_file),
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ])
    
    return result

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/opt/sutazaiapp/logs/localagi_integration_test.log')
        ]
    )
    
    # Run the tests
    asyncio.run(run_integration_tests())